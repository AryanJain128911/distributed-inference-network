"""Provider streaming lifecycle with a controllable, explicitly simulated backend."""

import asyncio
import json

import httpx
import pytest

from inference_network.backend import BackendChunk, BackendResult
from inference_network.config import ProviderSettings
from inference_network.errors import NetworkError
from inference_network.provider import create_app

PAYLOAD = {
    "model": "simulated-echo-v1",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
}


class ControlledBackend:
    def __init__(self, failure=None):
        self.failure = failure
        self.started = asyncio.Event()
        self.stopping = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.allow_cleanup.set()
        self.stopped = False

    async def generate(self, request):
        return BackendResult(content="[SIMULATED] hello", is_simulated=True)

    async def stream(self, request):
        self.started.set()
        try:
            if self.failure == "before":
                raise NetworkError(422, "prompt_too_long", "Too many tokens")
            yield BackendChunk(content="[SIMULATED] hello", is_simulated=True)
            if self.failure == "after":
                raise RuntimeError("sensitive backend detail")
            if self.failure == "wait":
                await asyncio.Event().wait()
            yield BackendChunk(is_simulated=True, finish_reason="stop")
        finally:
            self.stopping.set()
            await self.allow_cleanup.wait()
            self.stopped = True


@pytest.mark.parametrize("failure,status", [("before", 422), ("after", 200)])
async def test_provider_errors_obey_commitment_boundary_and_close_backend(failure, status):
    backend = ControlledBackend(failure)
    app = create_app(ProviderSettings(delay_seconds=0), backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post("/internal/chat/completions", json=PAYLOAD)
    assert response.status_code == status
    assert backend.stopped
    assert app.state.runtime.active_requests == 0
    assert "sensitive backend detail" not in response.text
    if failure == "after":
        assert "event: error" in response.text and "backend_error" in response.text
        assert "[DONE]" not in response.text
        assert '"finish_reason":"stop"' not in response.text
    else:
        assert response.json()["error"]["code"] == "prompt_too_long"


async def test_provider_timeout_cancels_backend_and_never_signals_success():
    backend = ControlledBackend("wait")
    app = create_app(ProviderSettings(delay_seconds=0, request_timeout_seconds=0.02), backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post("/internal/chat/completions", json=PAYLOAD)
    assert response.status_code == 200
    assert "provider_timeout" in response.text and "event: error" in response.text
    assert "[DONE]" not in response.text
    assert backend.stopped and app.state.runtime.active_requests == 0


async def test_disconnect_keeps_provider_slot_until_backend_cleanup_finishes():
    backend = ControlledBackend("wait")
    backend.allow_cleanup.clear()
    app = create_app(ProviderSettings(delay_seconds=0, max_concurrency=1), backend)
    disconnected = asyncio.Event()
    content_sent = asyncio.Event()
    body_delivered = False

    async def receive():
        nonlocal body_delivered
        if not body_delivered:
            body_delivered = True
            return {
                "type": "http.request",
                "body": json.dumps(PAYLOAD).encode(),
                "more_body": False,
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if b"[SIMULATED]" in message.get("body", b""):
            content_sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/internal/chat/completions",
        "raw_path": b"/internal/chat/completions",
        "query_string": b"",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8101),
    }
    serving = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(content_sent.wait(), 1)
        assert app.state.runtime.active_requests == 1
        disconnected.set()
        await asyncio.wait_for(backend.stopping.wait(), 1)
        assert not backend.stopped and not serving.done()
        assert app.state.runtime.active_requests == 1
        with pytest.raises(NetworkError) as full:
            app.state.runtime.admit(PAYLOAD["model"])
        assert full.value.code == "provider_capacity"
    finally:
        backend.allow_cleanup.set()
        await asyncio.wait_for(serving, 1)
    assert backend.stopped and app.state.runtime.active_requests == 0
