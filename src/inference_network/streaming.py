"""Bounded SSE framing and the ASGI lifetime of one streamed response."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from inference_network.errors import NetworkError
from inference_network.schemas import ChatCompletionChunk

DONE_EVENT = b"data: [DONE]\n\n"
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_STREAM_EVENTS = 8196


@dataclass(frozen=True)
class SSEEvent:
    data: str
    event: str = "message"


def encode_chunk(chunk: ChatCompletionChunk) -> bytes:
    # Omit absent delta keys, but retain finish_reason:null on ordinary choices.
    payload = chunk.model_dump(mode="json")
    for choice in payload["choices"]:
        choice["delta"] = {
            key: value for key, value in choice["delta"].items() if value is not None
        }
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


def encode_error(error: NetworkError) -> bytes:
    payload = {"error": {"message": error.message, "type": "network_error", "code": error.code}}
    return f"event: error\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


async def iter_sse(
    source: AsyncIterator[bytes],
    *,
    max_event_bytes: int = 65536,
    max_total_bytes: int = MAX_STREAM_BYTES,
    max_events: int = MAX_STREAM_EVENTS,
) -> AsyncIterator[SSEEvent]:
    """Decode LF, CRLF, or CR events across arbitrary byte/UTF-8 boundaries.

    EOF is not an event delimiter: incomplete frames are protocol errors. Limits
    include ignored comments and fields so they cannot bypass the wire budget.
    """
    buffer = bytearray()
    data: list[str] = []
    event_type = "message"
    total_bytes = event_bytes = event_count = 0
    async for chunk in source:
        total_bytes += len(chunk)
        if total_bytes > max_total_bytes:
            raise ValueError("Stream exceeded the wire size limit")
        buffer.extend(chunk)
        while buffer:
            cr = buffer.find(b"\r")
            lf = buffer.find(b"\n")
            separators = [offset for offset in (cr, lf) if offset >= 0]
            if not separators:
                break
            end = min(separators)
            if buffer[end] == 13 and end + 1 == len(buffer):
                # A CRLF pair may itself be split across network reads.
                break
            width = 2 if buffer[end : end + 2] == b"\r\n" else 1
            event_bytes += end + width
            if event_bytes > max_event_bytes:
                raise ValueError("SSE event exceeded the size limit")
            line = bytes(buffer[:end]).decode("utf-8")
            del buffer[: end + width]
            if not line:
                event_count += 1
                if event_count > max_events:
                    raise ValueError("Stream exceeded the event count limit")
                if data:
                    yield SSEEvent("\n".join(data), event_type)
                data = []
                event_type = "message"
                event_bytes = 0
            elif not line.startswith(":"):
                field, separator, value = line.partition(":")
                if separator and value.startswith(" "):
                    value = value[1:]
                if field == "data":
                    data.append(value)
                elif field == "event":
                    event_type = value or "message"
        if event_bytes + len(buffer) > max_event_bytes:
            raise ValueError("SSE event exceeded the size limit")
    # A final CR is a complete newline; it is enough to delimit an empty line.
    if buffer == b"\r":
        event_count += 1
        if event_count > max_events:
            raise ValueError("Stream exceeded the event count limit")
        if data:
            yield SSEEvent("\n".join(data), event_type)
        buffer.clear()
        data = []
        event_bytes = 0
    if buffer or data or event_bytes:
        raise ValueError("Provider stream ended inside an SSE event")


async def close_stream(iterator: AsyncIterator[Any]) -> None:
    """Keep cleanup alive through repeated cancellation, then preserve cancellation."""
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    cleanup = asyncio.create_task(close())
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    await cleanup
    if cancelled:
        raise asyncio.CancelledError


class EventStreamResponse(Response):
    """Own generation, first-event prefetch, backpressure, and disconnect cleanup."""

    media_type = "text/event-stream"

    def __init__(
        self,
        iterator: AsyncIterator[ChatCompletionChunk],
        timeout_seconds: float,
        timeout_code: str = "request_timeout",
    ) -> None:
        super().__init__(headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        # Streaming responses have no known content length.
        self.raw_headers = [
            (key, value) for key, value in self.raw_headers if key != b"content-length"
        ]
        self._iterator = iterator
        self._timeout = timeout_seconds
        self._timeout_code = timeout_code
        self._started = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def disconnected() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        work = asyncio.create_task(self._stream(send))
        watcher = asyncio.create_task(disconnected())
        try:
            completed, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
            if work in completed:
                await work
            else:
                await watcher
        finally:
            for task in (work, watcher):
                if not task.done():
                    task.cancel()
            # Cancellation must not detach the generation cleanup from its owner.
            cleanup = asyncio.gather(work, watcher, return_exceptions=True)
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

    async def _stream(self, send: Send) -> None:
        try:
            try:
                async with asyncio.timeout(self._timeout):
                    try:
                        first = await anext(self._iterator)
                    except StopAsyncIteration as exc:
                        raise NetworkError(
                            502, "invalid_provider_response", "Empty inference stream"
                        ) from exc
                    # Once sending headers starts, ASGI cannot guarantee that a
                    # stalled/cancelled send has not already committed them.
                    self._started = True
                    await send(
                        {"type": "http.response.start", "status": 200, "headers": self.raw_headers}
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": encode_chunk(first),
                            "more_body": True,
                        }
                    )
                    async for chunk in self._iterator:
                        await send(
                            {
                                "type": "http.response.body",
                                "body": encode_chunk(chunk),
                                "more_body": True,
                            }
                        )
                    await send(
                        {"type": "http.response.body", "body": DONE_EVENT, "more_body": False}
                    )
            except TimeoutError as exc:
                raise NetworkError(
                    504, self._timeout_code, "Overall inference deadline exceeded"
                ) from exc
        except NetworkError as exc:
            if not self._started:
                raise
            # A stalled client must not retain execution indefinitely merely to
            # receive its error. Timeout notification has a short bounded grace.
            try:
                async with asyncio.timeout(0.1):
                    await send(
                        {
                            "type": "http.response.body",
                            "body": encode_error(exc),
                            "more_body": False,
                        }
                    )
            except (TimeoutError, OSError):
                pass
        except OSError:
            # ASGI servers may report a disconnect from send before receive.
            pass
        finally:
            await close_stream(self._iterator)
