import asyncio
import json
import stat
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from inference_network.backend import SimulatedEchoBackend
from inference_network.config import CoordinatorSettings, ProviderSettings
from inference_network.coordinator import create_app as coordinator_app
from inference_network.errors import NetworkError
from inference_network.pairing import (
    CoordinatorProfile,
    ProviderProfile,
    SenderProfile,
    authorization_header,
    generate_profiles,
    load_profile,
    save_profile,
)
from inference_network.provider import ProviderRuntime, maintain_presence
from inference_network.provider import create_app as provider_app
from inference_network.registry import Reservation
from inference_network.schemas import ChatRequest
from inference_network.transport import ProviderTransport

CHAT = {"model": "simulated-echo-v1", "messages": [{"role": "user", "content": "test"}]}


@pytest.fixture
def profiles(tmp_path):
    paths = generate_profiles(
        tmp_path / "network",
        ssh_host="192.168.1.20",
        ssh_user="owner",
        providers={"studio": 18101, "air": 18102},
    )
    return paths, {name: load_profile(path) for name, path in paths.items()}


def advertisement(profile):
    return {
        "provider_id": profile.provider_id,
        "instance_id": str(uuid4()),
        "endpoint": profile.endpoint,
        "models": ["simulated-echo-v1"],
        "gpu_type": "SIMULATED GPU",
        "vram_gb": 24,
        "max_concurrency": 1,
    }


def test_profiles_have_private_permissions_distinct_tokens_and_redacted_repr(profiles):
    paths, loaded = profiles
    coordinator = loaded["coordinator"]
    assert isinstance(coordinator, CoordinatorProfile)
    assert isinstance(loaded["sender"], SenderProfile)
    assert isinstance(loaded["provider-studio"], ProviderProfile)
    assert stat.S_IMODE(paths["coordinator"].parent.stat().st_mode) == 0o700
    secrets = [coordinator.sender_token.get_secret_value()]
    secrets.extend(access.token.get_secret_value() for access in coordinator.providers)
    assert len(set(secrets)) == 3
    for path in paths.values():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for profile in loaded.values():
        assert all(secret not in repr(profile) for secret in secrets)
    with pytest.raises(FileExistsError):
        save_profile(paths["coordinator"], coordinator)


@pytest.mark.parametrize("host", ["8.8.8.8", "127.0.0.1", "169.254.1.2", "::1", "host.local"])
def test_pairing_rejects_non_private_ipv4_without_creating_files(tmp_path, host):
    directory = tmp_path / "network"
    with pytest.raises(ValueError):
        generate_profiles(directory, ssh_host=host, ssh_user="owner")
    assert not directory.exists()


def test_profile_loader_rejects_symlinks_public_permissions_and_invalid_secret(profiles, tmp_path):
    paths, _ = profiles
    link = tmp_path / "linked.json"
    link.symlink_to(paths["sender"])
    with pytest.raises(ValueError):
        load_profile(link)
    paths["sender"].chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        load_profile(paths["sender"])
    paths["sender"].chmod(0o600)
    data = json.loads(paths["sender"].read_text())
    data["token"] = "private-invalid-test-secret"
    paths["sender"].write_text(json.dumps(data))
    with pytest.raises(ValueError) as error:
        load_profile(paths["sender"])
    assert "private-invalid-test-secret" not in str(error.value)


def test_explicit_lan_mode_requires_matching_role_identity_origin_and_port(profiles):
    paths, loaded = profiles
    with pytest.raises(ValidationError):
        CoordinatorSettings(lan_mode=True)
    with pytest.raises(ValidationError):
        CoordinatorSettings(credentials_file=paths["coordinator"])
    with pytest.raises(ValidationError):
        CoordinatorSettings(lan_mode=True, credentials_file=paths["sender"])
    with pytest.raises(ValidationError):
        CoordinatorSettings(lan_mode=True, credentials_file=paths["coordinator"], port=8001)
    with pytest.raises(ValidationError):
        ProviderSettings(endpoint="http://127.0.0.1:18101")
    options = {
        "lan_mode": True,
        "credentials_file": paths["provider-studio"],
        "provider_id": "studio",
        "endpoint": loaded["provider-studio"].endpoint,
    }
    assert ProviderSettings(**options).port == 8101
    for field, value in [("provider_id", "air"), ("endpoint", "http://127.0.0.1:18102")]:
        with pytest.raises(ValidationError):
            ProviderSettings(**{**options, field: value})
    with pytest.raises(ValidationError):
        ProviderSettings(**options, host="0.0.0.0")


@pytest.mark.parametrize("path", ["/health", "/providers", "/v1/models", "/docs", "/openapi.json"])
def test_lan_routes_require_authentication_including_health_and_schema(profiles, path):
    paths, _ = profiles
    settings = CoordinatorSettings(lan_mode=True, credentials_file=paths["coordinator"])
    with TestClient(coordinator_app(settings)) as client:
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_sender_can_read_and_request_but_cannot_register_or_heartbeat(profiles):
    paths, loaded = profiles
    settings = CoordinatorSettings(lan_mode=True, credentials_file=paths["coordinator"])
    headers = authorization_header(loaded["sender"].token)
    with TestClient(coordinator_app(settings), headers=headers) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/providers").json() == {"providers": []}
        assert client.get("/v1/models").json()["data"] == []
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 404
        assert (
            client.post(
                "/providers/register", json=advertisement(loaded["provider-studio"])
            ).status_code
            == 403
        )
        assert client.post("/providers/studio/heartbeat", json={}).status_code == 403


def test_provider_token_is_bound_to_registration_identity_endpoint_and_heartbeat(profiles):
    paths, loaded = profiles
    settings = CoordinatorSettings(lan_mode=True, credentials_file=paths["coordinator"])
    headers = authorization_header(loaded["provider-studio"].token)
    body = advertisement(loaded["provider-studio"])
    with TestClient(coordinator_app(settings), headers=headers) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/providers").status_code == 403
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 403
        for field, value in [("provider_id", "air"), ("endpoint", "http://127.0.0.1:18102")]:
            assert (
                client.post("/providers/register", json={**body, field: value}).status_code == 403
            )
        assert client.post("/providers/register", json=body).status_code == 200
        heartbeat = {"instance_id": body["instance_id"], "sequence": 1, "ready": True}
        assert client.post("/providers/air/heartbeat", json=heartbeat).status_code == 403
        assert client.post("/providers/studio/heartbeat", json=heartbeat).status_code == 200


def test_auth_precedes_body_validation_and_rejects_duplicate_authorization(profiles):
    paths, loaded = profiles
    settings = CoordinatorSettings(lan_mode=True, credentials_file=paths["coordinator"])
    token = authorization_header(loaded["sender"].token)["Authorization"]
    with TestClient(coordinator_app(settings)) as client:
        assert client.post("/v1/chat/completions", content=b"x" * 70000).status_code == 401
        duplicate = [("Authorization", token), ("Authorization", token)]
        assert client.get("/health", headers=duplicate).status_code == 401
        response = client.get("/health", headers={"Authorization": "Bearer secret-invalid"})
        assert "secret-invalid" not in response.text


async def test_provider_admission_is_protected_and_only_its_token_is_accepted(profiles):
    paths, loaded = profiles
    settings = ProviderSettings(
        lan_mode=True,
        credentials_file=paths["provider-studio"],
        provider_id="studio",
        endpoint=loaded["provider-studio"].endpoint,
        delay_seconds=0,
    )
    app = provider_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8101"
    ) as client:
        assert (await client.get("/health")).status_code == 401
        for name in ["sender", "provider-air"]:
            response = await client.post(
                "/internal/chat/completions",
                json=CHAT,
                headers=authorization_header(loaded[name].token),
            )
            assert response.status_code == 401
        assert app.state.runtime.active_requests == 0
        response = await client.post(
            "/internal/chat/completions",
            json=CHAT,
            headers=authorization_header(loaded["provider-studio"].token),
        )
        assert response.status_code == 200
        assert response.json()["metadata"]["provider_id"] == "studio"


async def test_transport_sends_endpoint_specific_token_and_refuses_unapproved_origin(profiles):
    _, loaded = profiles
    profile = loaded["provider-studio"]
    calls = []

    def handler(request):
        calls.append(request)
        expected = authorization_header(profile.token)["Authorization"]
        assert request.headers["Authorization"] == expected
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = ProviderTransport(client, provider_tokens={profile.endpoint: profile.token})
        request = ChatRequest.model_validate(CHAT)
        approved = Reservation(uuid4(), "studio", uuid4(), profile.endpoint, True)
        with pytest.raises(NetworkError, match="Provider rejected"):
            await transport.dispatch(approved, request, "test")
        rejected = Reservation(uuid4(), "other", uuid4(), "http://127.0.0.1:19999", True)
        with pytest.raises(NetworkError, match="Provider endpoint not approved"):
            await transport.dispatch(rejected, request, "test")
        stream = transport.stream(rejected, request, "test")
        with pytest.raises(NetworkError, match="Provider endpoint not approved"):
            await anext(stream)
    assert len(calls) == 1


def test_duplicate_authorities_are_rejected(profiles):
    _, loaded = profiles
    profile = loaded["coordinator"]
    for updated in [
        {"providers": (profile.providers[0], profile.providers[0])},
        {"sender_token": profile.providers[0].token},
        {"sender_token": SecretStr("short")},
    ]:
        with pytest.raises(ValidationError):
            CoordinatorProfile.model_validate({**profile.model_dump(), **updated})


async def test_provider_presence_sends_its_token_for_registration_heartbeat_and_shutdown(
    profiles, monkeypatch
):
    paths, loaded = profiles
    profile = loaded["provider-studio"]
    heartbeat_received = asyncio.Event()
    calls = []

    def coordinator(request):
        assert (
            request.headers["Authorization"] == authorization_header(profile.token)["Authorization"]
        )
        calls.append(json.loads(request.content))
        if request.url.path.endswith("/heartbeat"):
            heartbeat_received.set()
        return httpx.Response(200, json={"accepted": True, "provider_id": "studio"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "inference_network.provider.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(coordinator), **kwargs),
    )
    settings = ProviderSettings(
        lan_mode=True,
        credentials_file=paths["provider-studio"],
        provider_id="studio",
        endpoint=profile.endpoint,
        heartbeat_interval_seconds=0.001,
    )
    app = provider_app(settings)
    async with app.router.lifespan_context(app):
        async with asyncio.timeout(3):
            await heartbeat_received.wait()
        assert app.state.runtime.connected is True
    assert app.state.runtime.connected is False
    assert calls[-1]["ready"] is False


async def test_presence_connection_status_tracks_failure_and_recovery_without_losing_identity():
    runtime = ProviderRuntime(
        ProviderSettings(heartbeat_interval_seconds=0.001), SimulatedEchoBackend()
    )
    recovered = asyncio.Event()
    calls = 0

    def coordinator(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert runtime.connected is False
            assert request.url.path == "/providers/register"
            return httpx.Response(200, json={"accepted": True, "provider_id": "provider-a"})
        if calls == 2:
            assert runtime.connected is True
            raise httpx.ConnectError("Network disconnected", request=request)
        if calls == 3:
            assert runtime.connected is False
            assert runtime.registered is True
            assert request.url.path.endswith("/heartbeat")
        else:
            assert runtime.connected is True
            recovered.set()
        return httpx.Response(200, json={"accepted": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(coordinator), base_url="http://127.0.0.1:8000"
    ) as client:
        task = asyncio.create_task(maintain_presence(runtime, client))
        try:
            async with asyncio.timeout(3):
                await recovered.wait()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
