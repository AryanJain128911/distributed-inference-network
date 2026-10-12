"""Streaming scheduling and replay boundaries, using synthetic provider wire data."""

import asyncio
import json
from contextlib import aclosing
from uuid import uuid4

import httpx
import pytest

from inference_network.errors import NetworkError
from inference_network.orchestration import Coordinator
from inference_network.registry import ProviderRegistry
from inference_network.scheduler import CapacityAware, RoundRobin
from inference_network.schemas import ChatRequest, ProviderRegistration
from inference_network.transport import ProviderTransport


def advertisement(name, port, *, model="alpha", streaming=True):
    return ProviderRegistration(
        provider_id=name,
        instance_id=uuid4(),
        endpoint=f"http://127.0.0.1:{port}",
        models=(model,),
        gpu_type="SIMULATED",
        vram_gb=1,
        max_concurrency=1,
        streaming=streaming,
    )


def stream_request(include_usage=False):
    return ChatRequest.model_validate(
        {
            "model": "alpha",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": include_usage},
        }
    )


def events(request, *, error=False):
    base = {
        "id": request.headers["X-Request-ID"],
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "alpha",
        "usage": None,
        "metadata": {"provider_id": "a" if request.url.port == 8101 else "b", "simulated": True},
    }

    def chunk(delta, reason=None):
        return (
            "data: "
            + json.dumps(
                {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
            )
            + "\n\n"
        )

    start = chunk({"role": "assistant"}) + chunk({"content": "[SIMULATED] hi"})
    if error:
        return (
            start + 'event: error\ndata: {"error":{"code":"backend_error",'
            '"type":"network_error","message":"Failed"}}\n\n'
        ).encode()
    return (
        start
        + chunk({}, "stop")
        + "data: "
        + json.dumps({**base, "choices": []})
        + "\n\ndata: [DONE]\n\n"
    ).encode()


@pytest.mark.parametrize("scheduler", [RoundRobin, CapacityAware])
def test_streaming_eligibility_preserves_model_capacity_and_readiness(scheduler):
    registry = ProviderRegistry(scheduler(), 10)
    registry.register(advertisement("a", 8101, streaming=False))
    registry.register(advertisement("b", 8102, model="other"))
    registry.register(advertisement("c", 8103).model_copy(update={"ready": False}))
    registry.register(advertisement("d", 8104))
    selected = registry.reserve("alpha", streaming=True)
    assert selected.provider_id == "d"
    with pytest.raises(NetworkError) as failure:
        registry.reserve("alpha", streaming=True)
    assert failure.value.code == "no_eligible_provider"
    registry.release(selected, succeeded=True)
    ordinary = registry.reserve("alpha")
    assert ordinary.provider_id == "a"
    registry.release(ordinary, succeeded=True)


@pytest.mark.parametrize("include_usage", [False, True])
async def test_stream_safe_connect_retry_and_usage_filter(include_usage):
    registry = ProviderRegistry(RoundRobin(), 10)
    registry.register(advertisement("a", 8101))
    registry.register(advertisement("b", 8102))
    calls = []

    def handle(request):
        calls.append(request.url.port)
        assert json.loads(request.content)["stream_options"]["include_usage"] is True
        if request.url.port == 8101:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=events(request)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        coordinator = Coordinator(registry, ProviderTransport(client), 2, 2)
        chunks = [chunk async for chunk in coordinator.stream(stream_request(include_usage))]
    assert calls == [8101, 8102]
    assert len(chunks) == (4 if include_usage else 3)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert len({chunk.id for chunk in chunks}) == 1
    assert all(row["in_flight"] == 0 for row in registry.snapshot())
    assert [row["completed_requests"] for row in registry.snapshot()] == [0, 1]
    assert [row["failed_requests"] for row in registry.snapshot()] == [1, 0]


@pytest.mark.parametrize("failure", ["status", "read", "active"])
async def test_stream_never_replays_ambiguous_or_active_failure(failure):
    registry = ProviderRegistry(RoundRobin(), 10)
    registry.register(advertisement("a", 8101))
    registry.register(advertisement("b", 8102))
    calls = []

    def handle(request):
        calls.append(request.url.port)
        if failure == "read":
            raise httpx.ReadError("interrupted", request=request)
        return httpx.Response(
            503 if failure == "status" else 200,
            headers={"content-type": "text/event-stream"},
            content=events(request, error=True),
        )

    chunks = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        coordinator = Coordinator(registry, ProviderTransport(client), 2, 2)
        with pytest.raises(NetworkError):
            async for chunk in coordinator.stream(stream_request()):
                chunks.append(chunk)
    assert calls == [8101]
    assert len(chunks) == (2 if failure == "active" else 0)
    assert not any(chunk.choices[0].finish_reason for chunk in chunks)
    assert all(row["in_flight"] == 0 for row in registry.snapshot())


async def test_stream_close_after_first_chunk_closes_http_and_releases_reservation():
    registry = ProviderRegistry(RoundRobin(), 10)
    registry.register(advertisement("a", 8101))
    closed = asyncio.Event()

    class PartialStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    body = b""

    def handle(request):
        nonlocal body
        body = events(request).split(b"\n\n")[0] + b"\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=PartialStream()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        coordinator = Coordinator(registry, ProviderTransport(client), 2, 2)
        async with aclosing(coordinator.stream(stream_request())) as chunks:
            await anext(chunks)
            assert registry.snapshot()[0]["in_flight"] == 1
        assert closed.is_set()
        assert registry.snapshot()[0]["in_flight"] == 0
        assert registry.snapshot()[0]["failed_requests"] == 1
