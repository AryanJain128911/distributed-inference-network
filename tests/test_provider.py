import asyncio
import json

import httpx
import pytest
from starlette.types import Message

from inference_network.backend import BackendResult, SimulatedEchoBackend
from inference_network.config import ProviderSettings
from inference_network.provider import ProviderRuntime, create_app, maintain_presence
from inference_network.schemas import ChatRequest, SimulationControl

PAYLOAD = {
    "model": "simulated-echo-v1",
    "messages": [{"role": "user", "content": "local simulation"}],
}
COMPLETIONS = "/internal/chat/completions"


class GatedBackend:
    def __init__(self, expected_concurrency: int = 1) -> None:
        self.expected_concurrency = expected_concurrency
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.running = 0
        self.peak_running = 0

    async def generate(self, request: ChatRequest) -> BackendResult:
        self.running += 1
        self.peak_running = max(self.peak_running, self.running)
        if self.running == self.expected_concurrency:
            self.started.set()
        try:
            await self.release.wait()
            return BackendResult("[SIMULATED] gate released", is_simulated=True)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.running -= 1


async def wait_for(event: asyncio.Event) -> None:
    async with asyncio.timeout(3):
        await event.wait()


async def test_simulator_echoes_latest_user_deterministically_and_bounds_content() -> None:
    backend = SimulatedEchoBackend()
    request = ChatRequest.model_validate(
        {
            "model": "simulated-echo-v1",
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "user", "content": "new"},
                {"role": "assistant", "content": "not the user"},
            ],
        }
    )
    first = await backend.generate(request)
    assert first == await backend.generate(request)
    assert first.content == "[SIMULATED simulated-echo-v1] new"
    assert first.is_simulated is True
    assert first.usage is None
    longest = ChatRequest.model_validate(
        {"model": request.model, "messages": [{"role": "user", "content": "x" * 8192}]}
    )
    assert len((await backend.generate(longest)).content) == 8192


async def test_provider_returns_contract_and_preserves_request_identity() -> None:
    app = create_app(ProviderSettings(delay_seconds=0))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            COMPLETIONS, json=PAYLOAD, headers={"X-Request-ID": "chatcmpl-test-identity"}
        )
        assert response.status_code == 200
        assert response.json()["id"] == "chatcmpl-test-identity"
        assert response.json()["metadata"] == {"provider_id": "provider-a", "simulated": True}
        assert response.json()["usage"] is None
        assert (await client.get("/health")).json()["active_requests"] == 0


async def test_provider_rejects_excess_concurrency_without_queueing() -> None:
    backend = GatedBackend(expected_concurrency=2)
    app = create_app(ProviderSettings(delay_seconds=0, max_concurrency=2), backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        pending = [asyncio.create_task(client.post(COMPLETIONS, json=PAYLOAD)) for _ in range(2)]
        try:
            await wait_for(backend.started)
            excess = await asyncio.gather(
                *(client.post(COMPLETIONS, json=PAYLOAD) for _ in range(20))
            )
            assert all(response.status_code == 429 for response in excess)
            assert (await client.get("/health")).json()["active_requests"] == 2
        finally:
            backend.release.set()
            responses = await asyncio.gather(*pending)
        assert all(response.status_code == 200 for response in responses)
        assert backend.peak_running == 2
        assert (await client.get("/health")).json()["active_requests"] == 0


@pytest.mark.parametrize(
    ("mode", "code"), [("error", "simulated_provider_error"), ("not_ready", "provider_not_ready")]
)
async def test_controlled_failure_and_readiness_recover_without_capacity_leaks(mode, code) -> None:
    app = create_app(ProviderSettings(delay_seconds=0, max_concurrency=1))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        assert (await client.put("/simulation", json={"mode": mode})).status_code == 200
        failed = await client.post(COMPLETIONS, json=PAYLOAD)
        assert failed.status_code == 503
        assert failed.json()["error"]["code"] == code
        assert (await client.get("/health")).json()["active_requests"] == 0
        await client.put("/simulation", json={"mode": "healthy", "delay_seconds": 0})
        assert (await client.post(COMPLETIONS, json=PAYLOAD)).status_code == 200


async def test_timeout_cancels_backend_and_releases_capacity() -> None:
    backend = GatedBackend()
    app = create_app(
        ProviderSettings(delay_seconds=0, max_concurrency=1, request_timeout_seconds=0.03), backend
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post(COMPLETIONS, json=PAYLOAD)
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "provider_timeout"
        assert backend.cancelled.is_set()
        assert backend.running == 0
        assert (await client.get("/health")).json()["active_requests"] == 0
        backend.release.set()
        assert (await client.post(COMPLETIONS, json=PAYLOAD)).status_code == 200


async def test_caller_cancellation_releases_capacity() -> None:
    backend = GatedBackend()
    app = create_app(ProviderSettings(delay_seconds=0, max_concurrency=1), backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        pending = asyncio.create_task(client.post(COMPLETIONS, json=PAYLOAD))
        await wait_for(backend.started)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert backend.cancelled.is_set()
        assert (await client.get("/health")).json()["active_requests"] == 0


async def test_http_disconnect_cancels_work_and_releases_capacity() -> None:
    backend = GatedBackend()
    app = create_app(ProviderSettings(delay_seconds=0, max_concurrency=1), backend)
    messages: asyncio.Queue[Message] = asyncio.Queue()
    messages.put_nowait({"type": "http.request", "body": json.dumps(PAYLOAD).encode()})
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": COMPLETIONS,
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "server": ("127.0.0.1", 8101),
        "client": ("127.0.0.1", 9000),
    }
    pending = asyncio.create_task(app(scope, messages.get, send))
    await wait_for(backend.started)
    messages.put_nowait({"type": "http.disconnect"})
    async with asyncio.timeout(3):
        await pending
    assert sent[0]["status"] == 499
    assert backend.cancelled.is_set()
    assert backend.running == 0
    assert app.state.runtime.active_requests == 0


async def test_backend_errors_are_sanitized_and_release_capacity(caplog) -> None:
    class FailingBackend:
        async def generate(self, request: ChatRequest) -> BackendResult:
            raise RuntimeError("sensitive prompt should never appear in logs")

    app = create_app(ProviderSettings(delay_seconds=0), FailingBackend())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post(COMPLETIONS, json=PAYLOAD)
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "backend_error"
        assert "sensitive prompt" not in response.text
        assert "sensitive prompt" not in caplog.text
        assert (await client.get("/health")).json()["active_requests"] == 0


async def test_provider_rejects_unsupported_models_and_invalid_controls() -> None:
    app = create_app(ProviderSettings(delay_seconds=0))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        unsupported = await client.post(COMPLETIONS, json={**PAYLOAD, "model": "unknown-model"})
        assert unsupported.status_code == 404
        assert unsupported.json()["error"]["code"] == "unsupported_model"
        assert (await client.put("/simulation", json={"delay_seconds": -1})).status_code == 422
        assert (
            await client.post(COMPLETIONS, json={**PAYLOAD, "stream": "invalid"})
        ).status_code == 422


async def test_presence_retries_and_reregisters_after_coordinator_restart() -> None:
    runtime = ProviderRuntime(
        ProviderSettings(delay_seconds=0, heartbeat_interval_seconds=0.001), SimulatedEchoBackend()
    )
    registrations = []
    heartbeats = []
    recovered = asyncio.Event()
    registration_attempts = 0

    async def coordinator(request: httpx.Request) -> httpx.Response:
        nonlocal registration_attempts
        body = json.loads(request.content)
        if request.url.path == "/providers/register":
            registration_attempts += 1
            if registration_attempts == 1:
                raise httpx.ConnectError("Coordinator is unavailable", request=request)
            registrations.append(body)
            return httpx.Response(200, json={"accepted": True, "provider_id": "provider-a"})
        heartbeats.append(body)
        if len(heartbeats) == 1:
            return httpx.Response(404, json={"error": "Coordinator restarted"})
        recovered.set()
        return httpx.Response(200, json={"accepted": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(coordinator), base_url="http://127.0.0.1:8000"
    ) as client:
        presence = asyncio.create_task(maintain_presence(runtime, client))
        try:
            await wait_for(recovered)
        finally:
            presence.cancel()
            with pytest.raises(asyncio.CancelledError):
                await presence
    assert registration_attempts == 3
    assert len(registrations) == 2
    assert {body["instance_id"] for body in registrations + heartbeats} == {
        str(runtime.instance_id)
    }
    assert [body["sequence"] for body in heartbeats] == [1, 2]
    assert runtime.registered is True


async def test_lifespan_sends_shutdown_readiness_and_closes_presence(monkeypatch) -> None:
    heartbeat_received = asyncio.Event()
    heartbeats = []

    async def coordinator(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/heartbeat"):
            heartbeats.append(json.loads(request.content))
            heartbeat_received.set()
        return httpx.Response(200, json={"accepted": True, "provider_id": "provider-a"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "inference_network.provider.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(coordinator), **kwargs),
    )
    app = create_app(ProviderSettings(delay_seconds=0, heartbeat_interval_seconds=0.001))
    async with app.router.lifespan_context(app):
        await wait_for(heartbeat_received)
        assert app.state.runtime.ready is True
    assert app.state.runtime.ready is False
    assert heartbeats[-1]["ready"] is False
    assert heartbeats[-1]["sequence"] > heartbeats[0]["sequence"]


@pytest.mark.parametrize(
    "invalid_acknowledgement",
    [{"accepted": "true"}, {"accepted": True, "provider_id": "different-provider"}],
)
async def test_presence_deadline_and_invalid_acknowledgement_allow_retry(
    invalid_acknowledgement,
) -> None:
    runtime = ProviderRuntime(
        ProviderSettings(heartbeat_interval_seconds=0.001, request_timeout_seconds=0.01),
        SimulatedEchoBackend(),
    )
    registration_attempts = 0
    stalled_call_cancelled = asyncio.Event()
    recovered = asyncio.Event()

    async def coordinator(request: httpx.Request) -> httpx.Response:
        nonlocal registration_attempts
        if request.url.path.endswith("/heartbeat"):
            recovered.set()
            return httpx.Response(200, json={"accepted": True})
        registration_attempts += 1
        if registration_attempts == 1:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stalled_call_cancelled.set()
                raise
        if registration_attempts == 2:
            return httpx.Response(200, json=invalid_acknowledgement)
        return httpx.Response(200, json={"accepted": True, "provider_id": "provider-a"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(coordinator), base_url="http://127.0.0.1:8000"
    ) as client:
        presence = asyncio.create_task(maintain_presence(runtime, client))
        try:
            await wait_for(recovered)
        finally:
            presence.cancel()
            with pytest.raises(asyncio.CancelledError):
                await presence
    assert stalled_call_cancelled.is_set()
    assert registration_attempts == 3
    assert runtime.registered is True


async def test_lost_registration_ack_and_readiness_change_preserve_retry_identity() -> None:
    runtime = ProviderRuntime(
        ProviderSettings(heartbeat_interval_seconds=0.001), SimulatedEchoBackend()
    )
    registrations = []
    registration_retried = asyncio.Event()
    readiness_received = asyncio.Event()

    async def coordinator(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/heartbeat"):
            assert body["ready"] is False
            readiness_received.set()
            return httpx.Response(200, json={"accepted": True})
        registrations.append(body)
        if len(registrations) == 1:
            runtime.simulation = SimulationControl(mode="not_ready")
            raise httpx.ReadError("Registration accepted but acknowledgement lost", request=request)
        registration_retried.set()
        if body != registrations[0]:
            return httpx.Response(409, json={"error": "Immutable registration changed"})
        return httpx.Response(200, json={"accepted": True, "provider_id": "provider-a"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(coordinator), base_url="http://127.0.0.1:8000"
    ) as client:
        presence = asyncio.create_task(maintain_presence(runtime, client))
        try:
            await wait_for(registration_retried)
            assert registrations[0] == registrations[1]
            await wait_for(readiness_received)
        finally:
            presence.cancel()
            with pytest.raises(asyncio.CancelledError):
                await presence
    assert runtime.registered is True
    assert runtime.ready is False
