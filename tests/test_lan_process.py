"""Authenticated local subprocess evidence; this does not prove cross-Mac SSH or MLX."""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import httpx
import pytest

from inference_network.config import MLX_MODEL_ID, CoordinatorSettings, ProviderSettings
from inference_network.launcher import ServiceProcess
from inference_network.pairing import authorization_header, generate_profiles, load_profile

CHAT = {
    "model": "simulated-echo-v1",
    "messages": [{"role": "user", "content": "authenticated local process test"}],
    "stream": True,
    "stream_options": {"include_usage": True},
}


def free_ports() -> list[int]:
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for listener in sockets:
            listener.bind(("127.0.0.1", 0))
        return [listener.getsockname()[1] for listener in sockets]


def eventually(condition: Callable[[], bool], timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("Authenticated process state did not converge")


def stream_data(response: httpx.Response) -> Iterator[dict | str]:
    assert response.status_code == 200
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            yield data if data == "[DONE]" else json.loads(data)


def first_text(events: Iterator[dict | str]) -> dict:
    for event in events:
        if isinstance(event, dict) and event.get("choices"):
            if event["choices"][0]["delta"].get("content"):
                return event
    raise AssertionError("Stream ended without generated text")


@pytest.mark.integration
def test_authenticated_local_providers_clients_capacity_cancellation_and_recovery(tmp_path):
    coordinator_port, studio_port, air_port = free_ports()
    paths = generate_profiles(
        tmp_path / "trust",
        ssh_host="192.168.1.20",
        ssh_user="owner",
        coordinator_port=coordinator_port,
        providers={"studio": studio_port, "air": air_port},
    )
    profiles = {name: load_profile(path) for name, path in paths.items()}
    sender_headers = authorization_header(profiles["sender"].token)
    coordinator = CoordinatorSettings(
        port=coordinator_port,
        lan_mode=True,
        credentials_file=paths["coordinator"],
        heartbeat_ttl_seconds=0.5,
    )
    with ExitStack() as stack:
        service = stack.enter_context(ServiceProcess("coordinator", coordinator, sender_headers))
        providers = {}
        for name, port in (("studio", studio_port), ("air", air_port)):
            profile = profiles[f"provider-{name}"]
            settings = ProviderSettings(
                port=port,
                provider_id=name,
                endpoint=profile.endpoint,
                coordinator_url=service.url,
                lan_mode=True,
                credentials_file=paths[f"provider-{name}"],
                heartbeat_interval_seconds=0.05,
                max_concurrency=1,
                delay_seconds=0.01,
            )
            providers[name] = stack.enter_context(
                ServiceProcess("provider", settings, authorization_header(profile.token))
            )
        clients = [
            stack.enter_context(
                httpx.Client(
                    base_url=service.url, headers=sender_headers, timeout=3, trust_env=False
                )
            )
            for _ in range(2)
        ]

        def states():
            response = clients[0].get("/providers")
            assert response.status_code == 200
            return {item["provider_id"]: item for item in response.json()["providers"]}

        eventually(lambda: len(states()) == 2 and all(row["online"] for row in states().values()))
        with httpx.Client(base_url=service.url, timeout=3, trust_env=False) as unauthenticated:
            assert unauthenticated.get("/health").status_code == 401
            assert unauthenticated.post("/v1/chat/completions", json=CHAT).status_code == 401

        long_request = {**CHAT, "messages": [{"role": "user", "content": "x" * 4000}]}
        with ExitStack() as streams:
            selected = set()
            readers = []
            for client in clients:
                response = streams.enter_context(
                    client.stream("POST", "/v1/chat/completions", json=long_request)
                )
                events = stream_data(response)
                readers.append(events)
                event = first_text(events)
                selected.add(event["metadata"]["provider_id"])
                assert event["metadata"]["simulated"] is True
            assert selected == {"studio", "air"}
            assert all(row["in_flight"] == 1 for row in states().values())
            rejected = clients[0].post("/v1/chat/completions", json=CHAT)
            assert rejected.status_code == 503
            assert rejected.json()["error"]["code"] == "no_eligible_provider"
            for name, provider in providers.items():
                headers = authorization_header(profiles[f"provider-{name}"].token)
                assert clients[0].get(f"{provider.url}/health").status_code == 401
                response = clients[0].get(f"{provider.url}/health", headers=headers)
                assert response.json()["active_requests"] == 1
        eventually(lambda: all(row["in_flight"] == 0 for row in states().values()))
        assert sum(row["completed_requests"] for row in states().values()) == 0

        def complete(client):
            with client.stream("POST", "/v1/chat/completions", json=CHAT) as response:
                events = list(stream_data(response))
            assert events[-1] == "[DONE]"
            assert events[-2]["usage"] is None
            return first_text(iter(events))["metadata"]["provider_id"]

        with ThreadPoolExecutor(max_workers=2) as workers:
            completed = list(workers.map(complete, clients))
        assert set(completed) == {"studio", "air"}
        eventually(lambda: all(row["in_flight"] == 0 for row in states().values()))
        assert sum(row["completed_requests"] for row in states().values()) == 2

        providers["studio"].close()
        eventually(lambda: not states()["studio"]["online"])
        restarted = stack.enter_context(
            ServiceProcess("provider", providers["studio"].settings, providers["studio"].headers)
        )
        eventually(lambda: states()["studio"]["online"] and states()["studio"]["ready"])
        assert restarted.process is not None and restarted.process.poll() is None


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.skipif(os.getenv("DIN_RUN_MLX_TESTS") != "1", reason="Set DIN_RUN_MLX_TESTS=1")
def test_interactive_paired_provider_runs_cached_mlx_smoke_and_stops_on_sigterm(tmp_path):
    """Real MLX through the actual launcher on one Mac, without SSH or downloading."""
    coordinator_port, provider_port, _ = free_ports()
    paths = generate_profiles(
        tmp_path / "trust",
        ssh_host="192.168.1.20",
        ssh_user="owner",
        coordinator_port=coordinator_port,
        providers={"air-real": provider_port},
    )
    sender = load_profile(paths["sender"])
    headers = authorization_header(sender.token)
    settings = CoordinatorSettings(
        port=coordinator_port,
        lan_mode=True,
        credentials_file=paths["coordinator"],
        request_timeout_seconds=60,
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("DIN_")}
    environment.update(HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    report_path = tmp_path / "local-auth-smoke.json"
    log_path = tmp_path / "interactive-provider.log"
    with (
        ServiceProcess("coordinator", settings, headers) as coordinator,
        httpx.Client(
            base_url=coordinator.url, headers=headers, timeout=3, trust_env=False
        ) as client,
        log_path.open("w") as log,
    ):
        launcher = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "inference_network",
                "interactive",
                "--lan",
                "--local-coordinator",
                "--profile",
                str(paths["provider-air-real"]),
            ],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            text=True,
        )
        try:
            assert launcher.stdin is not None
            launcher.stdin.write("2\n1\n")
            launcher.stdin.flush()

            def provider_ready():
                assert launcher.poll() is None, log_path.read_text()
                rows = client.get("/providers").json()["providers"]
                return len(rows) == 1 and rows[0]["online"] and rows[0]["ready"]

            eventually(provider_ready, timeout=60)
            provider = client.get("/providers").json()["providers"][0]
            assert provider["provider_id"] == "air-real"
            assert provider["models"] == [MLX_MODEL_ID]
            assert provider["simulated"] is False
            assert provider["streaming"] is True
            assert provider["memory_kind"] == "unified_memory"
            assert provider["max_concurrency"] == 1
            smoke = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.lan_smoke",
                    "--coordinator-url",
                    coordinator.url,
                    "--credentials-file",
                    str(paths["sender"]),
                    "--expected-provider",
                    "air-real",
                    "--output",
                    str(report_path),
                    "--host-label",
                    "Single local Mac; actual interactive MLX provider",
                    "--network-note",
                    "Authenticated loopback only; no SSH or physical two-host verification",
                ],
                env=environment,
                text=True,
                capture_output=True,
                timeout=70,
            )
            assert smoke.returncode == 0, smoke.stdout + smoke.stderr
            report = json.loads(report_path.read_text())
            assert report["physical_two_host_verification"] is False
            assert report["validation_problems"] == []
            assert report["summary"]["completed"] == 1
            observation = report["observations"][0]
            assert observation["model"] == MLX_MODEL_ID
            assert observation["provider_id"] == "air-real"
            assert observation["simulated"] is False
            assert observation["done_received"] is True
            assert observation["first_text_seconds"] > 0
            assert observation["prompt_tokens"] > 0
            assert observation["completion_tokens"] > 0
            assert observation["total_tokens"] == (
                observation["prompt_tokens"] + observation["completion_tokens"]
            )
            print(f"Local authenticated real MLX smoke evidence: {report_path}")
        finally:
            if launcher.poll() is None:
                launcher.terminate()
            try:
                launcher.wait(timeout=15)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(timeout=3)
                raise
            if launcher.stdin is not None:
                launcher.stdin.close()
        assert launcher.returncode == 0, log_path.read_text()
        output = log_path.read_text()
        assert "Pinned model is cached; no download needed." in output
        assert "Download from Hugging Face now?" not in output
        assert "owned services and tunnels have been closed" in output
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", provider_port)) != 0
        eventually(lambda: not client.get("/providers").json()["providers"][0]["ready"])
