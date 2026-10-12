"""Provider lifecycle/HTTP contracts using an explicitly fake MLX model loader."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from inference_network.backend import BackendResult
from inference_network.config import MLX_MODEL_ID, ProviderSettings
from inference_network.provider import create_app
from inference_network.schemas import ChatRequest, SimulationControl, TokenUsage

COMPLETIONS = "/internal/chat/completions"


def test_mlx_environment_defaults_advertise_only_the_model_that_will_load(monkeypatch) -> None:
    monkeypatch.setenv("DIN_BACKEND", "mlx")
    settings = ProviderSettings.from_env()
    assert settings.models == (MLX_MODEL_ID,)
    assert settings.max_concurrency == 1
    assert settings.delay_seconds == 0


@pytest.mark.parametrize(
    "override",
    [{"models": ("simulated-echo-v1",)}, {"max_concurrency": 2}, {"delay_seconds": 0.1}],
)
def test_mlx_configuration_rejects_false_model_identity_or_unsupported_execution(override) -> None:
    with pytest.raises(ValidationError):
        ProviderSettings(backend="mlx", **override)


class FakeMLXBackend:
    """No model execution: isolate loading, registration, and API behavior."""

    device_name = "Apple M4 (test device)"
    memory_gb = 16.0

    def __init__(self) -> None:
        self.load_started = asyncio.Event()
        self.allow_load = asyncio.Event()
        self.allow_load.set()
        self.loaded = False
        self.closed = False
        self.load_error = False
        self.generation_error = False
        self.requests: list[ChatRequest] = []

    async def load(self) -> None:
        self.load_started.set()
        await self.allow_load.wait()
        if self.load_error:
            raise RuntimeError("sensitive model loader detail")
        self.loaded = True

    async def close(self) -> None:
        self.closed = True
        self.loaded = False

    async def generate(self, request: ChatRequest) -> BackendResult:
        assert self.loaded
        self.requests.append(request)
        if self.generation_error:
            raise RuntimeError("sensitive model generation detail")
        return BackendResult(
            content="Model output supplied by the test double.",
            is_simulated=False,
            usage=TokenUsage(prompt_tokens=15, completion_tokens=9, total_tokens=24),
            finish_reason="length",
        )


async def wait_for(event: asyncio.Event) -> None:
    async with asyncio.timeout(3):
        await event.wait()


@pytest.fixture
def real_provider(monkeypatch):
    backend = FakeMLXBackend()
    requests = []
    registered = asyncio.Event()
    settings = ProviderSettings(backend="mlx", heartbeat_interval_seconds=0.001)

    async def coordinator(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/providers/register":
            assert backend.loaded
            registered.set()
        return httpx.Response(200, json={"accepted": True, "provider_id": settings.provider_id})

    client_type = httpx.AsyncClient
    monkeypatch.setattr("inference_network.provider.MLXBackend", lambda settings: backend)
    monkeypatch.setattr(
        "inference_network.provider.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(coordinator), **kwargs),
    )
    app = create_app(settings)
    return app, backend, requests, registered, client_type


async def test_real_provider_does_not_register_or_accept_requests_until_model_loaded(
    real_provider,
) -> None:
    app, backend, requests, registered, client_type = real_provider
    backend.allow_load.clear()
    lifespan = app.router.lifespan_context(app)
    startup = asyncio.create_task(lifespan.__aenter__())
    try:
        await wait_for(backend.load_started)
        assert not requests
        assert not app.state.runtime.ready
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            app.state.runtime.registration()
        async with client_type(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            assert (await client.get("/health")).json()["ready"] is False
            response = await client.post(
                COMPLETIONS,
                json={
                    "model": app.state.runtime.settings.models[0],
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "provider_not_ready"
        backend.allow_load.set()
        await startup
        await wait_for(registered)
        registration = requests[0][1]
        assert registration["models"] == list(app.state.runtime.settings.models)
        assert registration["ready"] is True
        assert registration["simulated"] is False
        assert registration["gpu_type"] == backend.device_name
        assert registration["vram_gb"] == backend.memory_gb
        assert registration["memory_kind"] == "unified_memory"
    finally:
        backend.allow_load.set()
        await startup
        await lifespan.__aexit__(None, None, None)
    assert backend.closed
    assert not app.state.runtime.ready
    assert requests[-1][0].endswith("/heartbeat")
    assert requests[-1][1]["ready"] is False


async def test_failed_model_load_never_registers_and_closes_resources(
    real_provider, caplog
) -> None:
    app, backend, requests, _, _ = real_provider
    backend.load_error = True
    with pytest.raises(RuntimeError, match="MLX model loading failed") as failure:
        async with app.router.lifespan_context(app):
            pytest.fail("Failed model loading must prevent startup")
    assert backend.closed
    assert not requests
    assert not app.state.runtime.ready
    assert "sensitive model loader detail" not in str(failure.value)
    assert "sensitive model loader detail" not in caplog.text
    assert "provider_model_load_failed" in caplog.text


async def test_cancelled_model_load_closes_resources_without_registration(real_provider) -> None:
    app, backend, requests, _, _ = real_provider
    backend.allow_load.clear()
    lifespan = app.router.lifespan_context(app)
    startup = asyncio.create_task(lifespan.__aenter__())
    await wait_for(backend.load_started)
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    assert backend.closed
    assert not requests
    assert not app.state.runtime.ready


async def test_real_provider_returns_backend_usage_and_rejects_simulation_controls(
    real_provider,
) -> None:
    app, backend, _, _, client_type = real_provider
    model = app.state.runtime.settings.models[0]
    payload = {"model": model, "messages": [{"role": "user", "content": "Hello"}]}
    async with (
        app.router.lifespan_context(app),
        client_type(transport=httpx.ASGITransport(app), base_url="http://testserver") as client,
    ):
        control = await client.put("/simulation", json={"mode": "error", "delay_seconds": 60})
        assert control.status_code == 404
        assert control.json()["error"]["code"] == "simulation_unavailable"
        # Even stale simulation state must not alter real inference behavior.
        app.state.runtime.simulation = SimulationControl(mode="error", delay_seconds=60)
        response = await client.post(COMPLETIONS, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == model
        assert body["metadata"] == {"provider_id": "provider-a", "simulated": False}
        assert (
            body["choices"][0]["message"]["content"] == "Model output supplied by the test double."
        )
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"] == {"prompt_tokens": 15, "completion_tokens": 9, "total_tokens": 24}
        health = (await client.get("/health")).json()
        assert health["backend"] == "mlx"
        assert health["simulated"] is False
        assert health["models"] == [model]
        assert health["active_requests"] == 0
        assert (
            await client.post(COMPLETIONS, json={**payload, "model": "wrong"})
        ).status_code == 404
        assert (await client.post(COMPLETIONS, json={**payload, "messages": []})).status_code == 422
        assert (
            await client.post(COMPLETIONS, json={**payload, "stream": "invalid"})
        ).status_code == 422
        assert len(backend.requests) == 1
    assert backend.closed


async def test_real_backend_request_failure_releases_capacity_and_allows_recovery(
    real_provider, caplog
) -> None:
    app, backend, _, _, client_type = real_provider
    payload = {
        "model": app.state.runtime.settings.models[0],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    async with (
        app.router.lifespan_context(app),
        client_type(transport=httpx.ASGITransport(app), base_url="http://testserver") as client,
    ):
        backend.generation_error = True
        failed = await client.post(COMPLETIONS, json=payload)
        assert failed.status_code == 500
        assert failed.json()["error"]["code"] == "backend_error"
        assert app.state.runtime.active_requests == 0
        assert "sensitive model generation detail" not in failed.text
        assert "sensitive model generation detail" not in caplog.text
        backend.generation_error = False
        assert (await client.post(COMPLETIONS, json=payload)).status_code == 200
        assert app.state.runtime.active_requests == 0
    assert backend.closed
