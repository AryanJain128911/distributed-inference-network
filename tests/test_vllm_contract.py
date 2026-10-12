"""Synthetic vLLM HTTP shapes, not a vLLM runtime or NVIDIA compatibility test.

Fixtures follow vLLM's documented Chat Completions protocol (reviewed 2026-10-09):
https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/
They demonstrate the existing parser's framing reuse and the need for translation
at a future backend boundary. They perform no model execution or GPU operations.
"""

import json
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, StreamingResponse

from inference_network.errors import NetworkError
from inference_network.registry import Reservation
from inference_network.schemas import ChatRequest, TokenUsage
from inference_network.streaming import iter_sse
from inference_network.transport import ProviderTransport

MODEL = "test-vllm-model"
TEXT_PARTS = ("Hello ", "世界!")
# Synthetic engine counters deliberately differ from the two text increments.
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
REQUEST = ChatRequest.model_validate(
    {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]}
)


@pytest.fixture
def fake_vllm():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        payload = await request.json()
        identity = {"id": "chatcmpl-engine-owned", "created": 123, "model": payload["model"]}
        if not payload.get("stream", False):
            return JSONResponse(
                {
                    **identity,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(TEXT_PARTS)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": USAGE,
                }
            )

        async def stream():
            choices = [
                {"delta": {"role": "assistant", "content": ""}, "finish_reason": None},
                *({"delta": {"content": text}, "finish_reason": None} for text in TEXT_PARTS),
                {"delta": {}, "finish_reason": "stop"},
            ]
            chunks = [{"choices": [{"index": 0, **choice}], "usage": None} for choice in choices]
            if payload.get("stream_options", {}).get("include_usage"):
                chunks.append({"choices": [], "usage": USAGE})
            for chunk in chunks:
                body = {**identity, "object": "chat.completion.chunk", **chunk}
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    # A test-only path alias isolates payload incompatibility from the different
    # engine/provider paths. It deliberately performs no response translation.
    app.add_api_route("/internal/chat/completions", complete, methods=["POST"])
    return app


async def test_fake_vllm_json_has_text_and_separate_engine_usage(fake_vllm):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(fake_vllm), base_url="http://127.0.0.1:9000"
    ) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST.model_dump())
    assert response.status_code == 200
    completion = response.json()
    assert completion["choices"][0]["message"]["content"] == "".join(TEXT_PARTS)
    assert TokenUsage.model_validate(completion["usage"]).completion_tokens == 7
    assert "metadata" not in completion


@pytest.mark.parametrize("include_usage", [False, True])
async def test_bounded_sse_parser_reads_fake_vllm_text_and_optional_usage(fake_vllm, include_usage):
    payload = REQUEST.model_dump()
    payload.update(stream=True, stream_options={"include_usage": include_usage})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(fake_vllm), base_url="http://127.0.0.1:9000"
    ) as client:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            events = [event async for event in iter_sse(response.aiter_bytes(chunk_size=7))]
    assert events[-1].data == "[DONE]"
    chunks = [json.loads(event.data) for event in events[:-1]]
    texts = [
        choice["delta"]["content"]
        for chunk in chunks
        for choice in chunk["choices"]
        if choice["delta"].get("content")
    ]
    assert texts == list(TEXT_PARTS)
    usage_chunks = [chunk for chunk in chunks if not chunk["choices"]]
    if include_usage:
        assert len(usage_chunks) == 1
        usage = TokenUsage.model_validate(usage_chunks[0]["usage"])
        assert usage.completion_tokens == 7
        assert usage.completion_tokens != len(texts)
    else:
        assert usage_chunks == []


@pytest.mark.parametrize("stream", [False, True])
async def test_raw_vllm_payload_is_not_a_din_provider_response(fake_vllm, stream):
    reservation = Reservation(uuid4(), "provider-a", uuid4(), "http://127.0.0.1:9000", False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(fake_vllm)) as client:
        transport = ProviderTransport(client)
        with pytest.raises(NetworkError) as failure:
            if stream:
                async for _ in transport.stream(reservation, REQUEST, "chatcmpl-din-owned"):
                    pytest.fail("Untranslated engine output must not be forwarded")
            else:
                await transport.dispatch(reservation, REQUEST, "chatcmpl-din-owned")
    assert failure.value.status == 502
    assert failure.value.code == "invalid_provider_response"
