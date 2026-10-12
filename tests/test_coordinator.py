import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_network.config import CoordinatorSettings, ProviderSettings
from inference_network.coordinator import create_app
from inference_network.errors import NetworkError
from inference_network.orchestration import Coordinator
from inference_network.registry import ProviderRegistry
from inference_network.scheduler import RoundRobin
from inference_network.schemas import ChatRequest, ProviderRegistration
from inference_network.transport import ProviderTransport

CHAT = {"model": "simulated-echo-v1", "messages": [{"role": "user", "content": "test"}]}


def advertisement(name="a", port=8101):
    return ProviderRegistration(
        provider_id=name,
        instance_id=uuid4(),
        endpoint=f"http://127.0.0.1:{port}",
        models=("simulated-echo-v1",),
        gpu_type="SIMULATED GPU",
        vram_gb=24,
        max_concurrency=1,
    )


def completion(request):
    return {
        "id": request.headers["X-Request-ID"],
        "object": "chat.completion",
        "created": 1,
        "model": "simulated-echo-v1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "[SIMULATED] test"},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
        "metadata": {"provider_id": "a" if request.url.port == 8101 else "b", "simulated": True},
    }


def test_startup_registration_models_and_duplicate_errors():
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/v1/models").json() == {"object": "list", "data": []}
        provider = advertisement()
        body = provider.model_dump(mode="json")
        assert client.post("/providers/register", json=body).status_code == 200
        assert client.post("/providers/register", json=body).status_code == 200
        body["instance_id"] = str(uuid4())
        assert client.post("/providers/register", json=body).status_code == 409
        assert client.get("/v1/models").json()["data"][0]["id"] == CHAT["model"]
        rows = client.get("/providers").json()["providers"]
        assert rows[0]["simulated"] is True
        assert "instance_id" not in rows[0]
        heartbeat = {"instance_id": str(provider.instance_id), "sequence": 1, "ready": False}
        assert client.post("/providers/a/heartbeat", json=heartbeat).status_code == 200
        assert client.get("/v1/models").json()["data"] == []
        assert client.post("/providers/a/heartbeat", json=heartbeat).status_code == 409
        assert client.post("/providers/missing/heartbeat", json=heartbeat).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**CHAT, "stream": "invalid"},
        {**CHAT, "temperature": 1},
        {**CHAT, "messages": []},
        {**CHAT, "messages": [{"role": "user", "content": "x" * 8193}]},
        {**CHAT, "messages": [{"role": "tool", "content": "private-test-value"}]},
    ],
)
def test_invalid_input_is_rejected_without_echoing_body(payload):
    with TestClient(create_app()) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert "private-test-value" not in response.text


def test_unsupported_models_payload_limit_and_browser_writes():
    with TestClient(create_app()) as client:
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 404
        assert client.post("/v1/chat/completions", content=b"x" * 65537).status_code == 413
        assert (
            client.post(
                "/v1/chat/completions", json=CHAT, headers={"Origin": "https://example.com"}
            ).status_code
            == 403
        )
        assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 400


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8101",
        "http://localhost:8101",
        "http://example.com:8101",
        "http://169.254.169.254:8101",
        "http://user:password@127.0.0.1:8101",
        "http://127.0.0.1:8101/path",
        "http://127.0.0.1:8101?x=1",
        "http://127.0.0.1:80",
        "http://127.0.0.1:8101\n",
        " http://127.0.0.1:8101",
        "http://127.0.0.1:\t8101",
    ],
)
def test_registration_disallows_non_loopback_or_ambiguous_origins(endpoint):
    with pytest.raises(ValidationError):
        ProviderRegistration.model_validate({**advertisement().model_dump(), "endpoint": endpoint})


def test_endpoint_normalization_prevents_duplicate_registration_aliases():
    first = ProviderRegistration.model_validate(
        {**advertisement().model_dump(), "endpoint": "HTTP://127.0.0.1:08101/"}
    )
    assert first.endpoint == "http://127.0.0.1:8101"
    registry = ProviderRegistry(RoundRobin(), 10)
    registry.register(first)
    with pytest.raises(NetworkError) as error:
        registry.register(advertisement("b", 8101))
    assert error.value.code == "endpoint_conflict"


def test_environment_settings_validate_and_derive_endpoint(monkeypatch):
    monkeypatch.setenv("DIN_PORT", "8123")
    monkeypatch.setenv("DIN_MODELS", "alpha,beta")
    settings = ProviderSettings.from_env()
    assert settings.endpoint == "http://127.0.0.1:8123"
    assert settings.models == ("alpha", "beta")
    monkeypatch.setenv("DIN_HOST", "0.0.0.0")
    with pytest.raises(ValidationError):
        CoordinatorSettings.from_env()


async def run_completion(handler, *, attempts=2, timeout_seconds=1, registry=None):
    registry = registry or ProviderRegistry(RoundRobin(), 10)
    if not registry.snapshot():
        registry.register(advertisement())
        registry.register(advertisement("b", 8102))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        coordinator = Coordinator(registry, ProviderTransport(client), attempts, timeout_seconds)
        return await coordinator.complete(ChatRequest.model_validate(CHAT))


async def test_safe_connection_failure_retries_on_another_provider_and_releases():
    calls = []
    registry = ProviderRegistry(RoundRobin(), 10)

    def handle(request):
        calls.append(request.url.port)
        if request.url.port == 8101:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=completion(request))

    result = await run_completion(handle, registry=registry)
    assert result.metadata.provider_id == "b"
    assert calls == [8101, 8102]
    assert all(row["in_flight"] == 0 for row in registry.snapshot())
    assert [row["healthy"] for row in registry.snapshot()] == [False, True]


@pytest.mark.parametrize(
    "failure",
    [
        "read_timeout",
        "read_error",
        "write_error",
        "server_error",
        "invalid_json",
        "mismatched_model",
        "too_large",
        "redirect",
    ],
)
async def test_ambiguous_or_invalid_responses_are_not_replayed(failure):
    calls = []
    registry = ProviderRegistry(RoundRobin(), 10)

    def handle(request):
        calls.append(request.url.port)
        if failure == "read_timeout":
            raise httpx.ReadTimeout("private-test-value", request=request)
        if failure == "read_error":
            raise httpx.ReadError("private-test-value", request=request)
        if failure == "write_error":
            raise httpx.WriteError("private-test-value", request=request)
        if failure == "server_error":
            return httpx.Response(503, text="private-test-value")
        if failure == "invalid_json":
            return httpx.Response(200, text="private-test-value")
        if failure == "too_large":
            return httpx.Response(200, content=b"x" * 65537)
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "http://example.com"})
        body = completion(request)
        body["model"] = "different"
        return httpx.Response(200, json=body)

    with pytest.raises(NetworkError) as error:
        await run_completion(handle, registry=registry)
    assert error.value.status == (504 if failure == "read_timeout" else 502)
    assert "private-test-value" not in str(error.value)
    assert calls == [8101]
    assert all(row["in_flight"] == 0 for row in registry.snapshot())


async def test_retry_limit_and_connect_timeout():
    calls = []

    def handle(request):
        calls.append(request.url.port)
        raise httpx.ConnectTimeout("cannot connect", request=request)

    with pytest.raises(NetworkError) as error:
        await run_completion(handle, attempts=1)
    assert error.value.code == "provider_connect_failed"
    assert calls == [8101]


async def test_cancellation_and_overall_deadline_release_capacity():
    entered = asyncio.Event()
    registry = ProviderRegistry(RoundRobin(), 10)

    async def handle(request):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(run_completion(handle, registry=registry, timeout_seconds=10))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(row["in_flight"] == 0 for row in registry.snapshot())
    with pytest.raises(NetworkError) as error:
        await run_completion(handle, registry=registry, timeout_seconds=0.01)
    assert error.value.code == "request_timeout"
    assert all(row["in_flight"] == 0 for row in registry.snapshot())
    assert [row["healthy"] for row in registry.snapshot()] == [True, False]


@pytest.mark.parametrize(
    "simulated,usage",
    [(False, None), (False, {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})],
)
async def test_transport_accepts_backend_independent_contract_using_synthetic_response(
    simulated, usage
):
    registry = ProviderRegistry(RoundRobin(), 10)
    registry.register(advertisement().model_copy(update={"simulated": simulated}))

    def handle(request):
        body = completion(request)
        body["metadata"]["simulated"] = simulated
        body["usage"] = usage
        return httpx.Response(200, json=body)

    result = await run_completion(handle, registry=registry)
    assert result.metadata.simulated is False
    assert (result.usage.model_dump() if result.usage else None) == usage


@pytest.mark.parametrize(
    "change",
    [
        "simulation_flag",
        "simulated_tokens",
        "wrong_provider",
        "wrong_request_id",
        "inconsistent_tokens",
        "wrong_role",
    ],
)
async def test_transport_rejects_inconsistent_metadata_and_usage(change):
    def handle(request):
        body = completion(request)
        if change == "simulation_flag":
            body["metadata"]["simulated"] = False
        elif change == "simulated_tokens":
            body["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        elif change == "inconsistent_tokens":
            body["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 5}
        elif change == "wrong_provider":
            body["metadata"]["provider_id"] = "wrong"
        elif change == "wrong_role":
            body["choices"][0]["message"]["role"] = "user"
        else:
            body["id"] = "unrelated-response"
        return httpx.Response(200, json=body)

    with pytest.raises(NetworkError) as error:
        await run_completion(handle)
    assert error.value.code == "invalid_provider_response"


async def test_successful_coordinator_response_is_validated_and_usage_is_not_fabricated():
    result = await run_completion(lambda request: httpx.Response(200, json=completion(request)))
    assert result.choices[0].message.content == "[SIMULATED] test"
    assert result.usage is None
    assert result.metadata.simulated is True
    assert json.loads(result.model_dump_json())["object"] == "chat.completion"


async def test_nonstream_dispatch_omits_new_optional_fields_for_older_providers():
    def handle(request):
        payload = json.loads(request.content)
        # V0.1.1's extra=forbid rejects even a null stream_options field.
        assert set(payload) == {"model", "messages", "stream"}
        assert payload["stream"] is False
        return httpx.Response(200, json=completion(request))

    assert (await run_completion(handle)).choices[0].message.content == "[SIMULATED] test"
