"""Bounded streaming client shared by the terminal and LAN verification tools."""

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from inference_network.errors import NetworkError
from inference_network.schemas import ChatCompletionChunk, ChatRequest, TokenUsage
from inference_network.streaming import iter_sse


@dataclass(frozen=True)
class StreamResult:
    content: str
    provider_id: str
    model: str
    simulated: bool
    usage: TokenUsage | None
    request_id: str
    elapsed_seconds: float
    first_text_seconds: float | None
    finish_reason: str


async def stream_request(
    client: httpx.AsyncClient,
    request: ChatRequest,
    on_text: Callable[[str], None] | None = None,
    path: str = "/v1/chat/completions",
    *,
    timeout_seconds: float = 65,
) -> StreamResult:
    """Never replay requests; interrupted or malformed termination is a failure."""
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise ValueError("Stream timeout must be positive, finite and at most 120 seconds")
    async with asyncio.timeout(timeout_seconds):
        return await _stream_request(client, request, on_text, path)


async def _stream_request(
    client: httpx.AsyncClient,
    request: ChatRequest,
    on_text: Callable[[str], None] | None,
    path: str,
) -> StreamResult:
    started = time.perf_counter()
    payload = request.model_dump(mode="json", exclude_none=True)
    payload.update(stream=True, stream_options={"include_usage": True})
    content: list[str] = []
    size = 0
    identity: tuple[str, int, str, bool] | None = None
    first_text: float | None = None
    usage: TokenUsage | None = None
    finish = ""
    phase = "role"
    async with client.stream(
        "POST",
        path,
        json=payload,
        headers={"Accept": "text/event-stream", "Accept-Encoding": "identity"},
    ) as response:
        if response.status_code != 200:
            # Do not echo a peer's arbitrary response body, URLs, or credentials.
            messages = {
                401: "Pairing credentials were rejected",
                403: "This paired role is not allowed to perform this operation",
                404: "The requested model is unavailable",
                429: "Provider capacity is full; try again after an active request finishes",
                503: "No ready provider has capacity for this model; retry later",
            }
            code = "request_rejected"
            body = bytearray()
            if response.headers.get("content-encoding", "identity").lower() == "identity":
                async for part in response.aiter_bytes():
                    body.extend(part)
                    if len(body) > 65536:
                        break
            if body and len(body) <= 65536:
                try:
                    error = json.loads(body)
                    remote_code = error["error"]["code"]
                    if (response.status_code, remote_code) in {
                        (503, "no_eligible_provider"),
                        (429, "provider_capacity"),
                        (404, "model_not_found"),
                    }:
                        code = remote_code
                except (ValueError, KeyError, TypeError):
                    pass
            raise NetworkError(
                response.status_code,
                code,
                messages.get(response.status_code, "Inference failed; check provider status"),
            )
        if (
            response.headers.get("content-type", "").split(";", 1)[0] != "text/event-stream"
            or response.headers.get("content-encoding", "identity") != "identity"
        ):
            raise ValueError("Expected an uncompressed event stream")
        async for event in iter_sse(response.aiter_bytes()):
            if event.event == "error":
                raise NetworkError(502, "stream_failed", "Inference failed during streaming")
            if event.event != "message" or phase == "done":
                raise ValueError("Unexpected stream event")
            if event.data == "[DONE]":
                if phase != "termination":
                    raise ValueError("Incomplete stream termination")
                phase = "done"
                continue
            chunk = ChatCompletionChunk.model_validate_json(event.data)
            observed = (
                chunk.id,
                chunk.created,
                chunk.metadata.provider_id,
                chunk.metadata.simulated,
            )
            if identity is None:
                identity = observed
            if observed != identity or chunk.model != request.model:
                raise ValueError("Stream identity changed")
            if not chunk.choices:
                if phase != "usage" or (chunk.metadata.simulated and chunk.usage is not None):
                    raise ValueError("Unexpected usage event")
                usage = chunk.usage
                phase = "termination"
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if phase == "role":
                if delta.role != "assistant" or delta.content or choice.finish_reason:
                    raise ValueError("Expected assistant role")
                phase = "text"
            elif phase == "text" and choice.finish_reason is None:
                if delta.role is not None or not delta.content:
                    raise ValueError("Expected generated text")
                size += len(delta.content)
                if size > 8192:
                    raise ValueError("Generated text exceeded the limit")
                if first_text is None:
                    first_text = time.perf_counter() - started
                content.append(delta.content)
                if on_text is not None:
                    on_text(delta.content)
            elif phase == "text" and choice.finish_reason:
                if delta.role is not None or delta.content is not None or not content:
                    raise ValueError("Invalid finish event")
                finish = choice.finish_reason
                phase = "usage"
            else:
                raise ValueError("Unexpected stream order")
    if phase != "done" or identity is None:
        raise ValueError("Stream disconnected without successful termination")
    return StreamResult(
        "".join(content),
        identity[2],
        request.model,
        identity[3],
        usage,
        identity[0],
        time.perf_counter() - started,
        first_text,
        finish,
    )
