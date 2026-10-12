import asyncio
import gzip
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from starlette.requests import Request
from starlette.types import Message, Scope

from inference_network.config import ProviderSettings
from inference_network.errors import NetworkError
from inference_network.http_safety import BodyLimitMiddleware, run_while_connected
from inference_network.registry import Reservation
from inference_network.schemas import ChatRequest, ProviderRegistration
from inference_network.transport import ProviderTransport


def http_scope() -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }


class RecordingApp:
    def __init__(self) -> None:
        self.called = False
        self.body = b""

    async def __call__(self, scope, receive, send) -> None:
        self.called = True
        self.body = (await receive())["body"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def test_chunked_body_without_content_length_cannot_bypass_limit() -> None:
    app = RecordingApp()
    middleware = BodyLimitMiddleware(app, max_body_bytes=16, body_timeout_seconds=1)
    messages = iter(
        [
            {"type": "http.request", "body": b"a" * 8, "more_body": True},
            {"type": "http.request", "body": b"b" * 8, "more_body": True},
            {"type": "http.request", "body": b"c", "more_body": False},
        ]
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return next(messages)

    async def send(message: Message) -> None:
        sent.append(message)

    await middleware(http_scope(), receive, send)
    assert app.called is False
    assert sent[0]["status"] == 413
    assert b"payload_too_large" in sent[1]["body"]


async def test_body_at_exact_limit_is_forwarded_once_without_modification() -> None:
    app = RecordingApp()
    middleware = BodyLimitMiddleware(app, max_body_bytes=16, body_timeout_seconds=1)
    messages = iter(
        [
            {"type": "http.request", "body": b"a" * 8, "more_body": True},
            {"type": "http.request", "body": b"b" * 8, "more_body": False},
        ]
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return next(messages)

    async def send(message: Message) -> None:
        sent.append(message)

    await middleware(http_scope(), receive, send)
    assert app.body == b"aaaaaaaa" + b"bbbbbbbb"
    assert sent[0]["status"] == 200


async def test_stalled_body_deadline_cancels_receive_without_entering_app() -> None:
    app = RecordingApp()
    middleware = BodyLimitMiddleware(app, max_body_bytes=16, body_timeout_seconds=0.01)
    cancelled = asyncio.Event()
    sent: list[Message] = []

    async def receive() -> Message:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("Unreachable receive completion")

    async def send(message: Message) -> None:
        sent.append(message)

    async with asyncio.timeout(3):
        await middleware(http_scope(), receive, send)
    assert cancelled.is_set()
    assert app.called is False
    assert sent[0]["status"] == 408
    assert b"body_timeout" in sent[1]["body"]


async def test_disconnect_during_body_upload_discards_partial_request() -> None:
    app = RecordingApp()
    middleware = BodyLimitMiddleware(app, max_body_bytes=16, body_timeout_seconds=1)
    messages = iter(
        [
            {"type": "http.request", "body": b"partial", "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return next(messages)

    async def send(message: Message) -> None:
        sent.append(message)

    await middleware(http_scope(), receive, send)
    assert app.called is False
    assert sent == []


async def test_error_response_send_timeout_does_not_start_a_second_response() -> None:
    app = RecordingApp()
    middleware = BodyLimitMiddleware(app, max_body_bytes=16, body_timeout_seconds=1)
    statuses = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"x" * 17, "more_body": False}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
            return
        raise TimeoutError("Client did not consume the error response")

    with pytest.raises(TimeoutError):
        await middleware(http_scope(), receive, send)
    assert statuses == [413]
    assert app.called is False


async def test_failed_operation_cancels_and_collects_disconnect_watcher() -> None:
    watcher_cancelled = asyncio.Event()

    async def receive() -> Message:
        try:
            await asyncio.Event().wait()
        finally:
            watcher_cancelled.set()
        raise AssertionError("Unreachable receive completion")

    async def fail() -> None:
        raise ValueError("Operation failed")

    with pytest.raises(ValueError, match="Operation failed"):
        await run_while_connected(fail(), Request(http_scope(), receive))
    assert watcher_cancelled.is_set()


@pytest.mark.parametrize("provider_id", [".", ".."])
def test_provider_identity_cannot_be_an_http_path_dot_segment(provider_id) -> None:
    with pytest.raises(ValidationError):
        ProviderSettings(provider_id=provider_id)
    with pytest.raises(ValidationError):
        ProviderRegistration(
            provider_id=provider_id,
            instance_id=uuid4(),
            endpoint="http://127.0.0.1:8101",
            models=("simulated-echo-v1",),
            gpu_type="SIMULATED GPU",
            vram_gb=24,
            max_concurrency=1,
        )


async def test_compressed_provider_response_is_rejected_before_decompression() -> None:
    class CompressedStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.was_read = False
            self.was_closed = False

        async def __aiter__(self):
            self.was_read = True
            yield gzip.compress(b"x" * 100000)

        async def aclose(self) -> None:
            self.was_closed = True

    stream = CompressedStream()

    def provider(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        transport = ProviderTransport(client)
        reservation = Reservation(uuid4(), "provider-a", uuid4(), "http://127.0.0.1:8101", True)
        request = ChatRequest.model_validate(
            {"model": "simulated-echo-v1", "messages": [{"role": "user", "content": "test"}]}
        )
        with pytest.raises(NetworkError) as error:
            await transport.dispatch(reservation, request, "chatcmpl-test")
    assert error.value.code == "invalid_provider_response"
    assert stream.was_read is False
    assert stream.was_closed is True
