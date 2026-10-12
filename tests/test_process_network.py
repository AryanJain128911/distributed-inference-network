"""Acceptance checks against independent local processes and real HTTP sockets."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest

from inference_network.config import MLX_MODEL_ID
from inference_network.launcher import LocalNetwork

pytestmark = pytest.mark.integration
CHAT = {"model": "simulated-echo-v1", "messages": [{"role": "user", "content": "process test"}]}


def unused_ports() -> tuple[int, int, int]:
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for listener in sockets:
            listener.bind(("127.0.0.1", 0))
        ports = [listener.getsockname()[1] for listener in sockets]
    return ports[0], ports[1], ports[2]


def eventually(condition: Callable[[], bool], timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.025)
    raise AssertionError("Condition did not become true before the deadline")


def provider_states(client: httpx.Client) -> dict[str, dict[str, Any]]:
    response = client.get("/providers")
    response.raise_for_status()
    return {provider["provider_id"]: provider for provider in response.json()["providers"]}


def configure(client: httpx.Client, endpoint: str, mode: str = "healthy", delay: float = 0) -> None:
    response = client.put(f"{endpoint}/simulation", json={"mode": mode, "delay_seconds": delay})
    response.raise_for_status()


@pytest.fixture
def network() -> Iterator[LocalNetwork]:
    coordinator, first, second = unused_ports()
    with LocalNetwork(
        coordinator,
        (first, second),
        environment={
            "DIN_HEARTBEAT_INTERVAL_SECONDS": "0.1",
            "DIN_HEARTBEAT_TTL_SECONDS": "0.6",
            "DIN_REQUEST_TIMEOUT_SECONDS": "1",
            "DIN_MAX_CONCURRENCY": "1",
            "DIN_STRATEGY": "round_robin",
        },
    ) as running:
        yield running
    assert all(process.poll() is not None for process in running.processes.values())
    assert_ports_closed((coordinator, first, second))


def assert_ports_closed(ports: tuple[int, int, int]) -> None:
    for port in ports:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_real_http_routing_and_process_recovery(network: LocalNetwork) -> None:
    with httpx.Client(base_url=network.coordinator_url, timeout=3, trust_env=False) as client:
        assert client.get("/health").json()["service"] == "coordinator"
        assert {model["id"] for model in client.get("/v1/models").json()["data"]} == {
            "simulated-echo-v1"
        }
        assert len({process.pid for process in network.processes.values()}) == 3
        responses = [client.post("/v1/chat/completions", json=CHAT) for _ in range(4)]
        assert all(response.status_code == 200 for response in responses)
        outputs = [response.json() for response in responses]
        assert [output["metadata"]["provider_id"] for output in outputs] == [
            "provider-a",
            "provider-b",
            "provider-a",
            "provider-b",
        ]
        assert all(
            output["metadata"]["simulated"] and output["usage"] is None for output in outputs
        )
        assert outputs[0]["choices"][0]["message"]["content"] == (
            "[SIMULATED simulated-echo-v1] process test"
        )
        unsupported = client.post("/v1/chat/completions", json=dict(CHAT, model="unavailable"))
        assert unsupported.status_code == 404
        assert unsupported.json()["error"]["code"] == "model_not_found"
        assert (
            client.post("/v1/chat/completions", json=dict(CHAT, stream="invalid")).status_code
            == 422
        )

        stopped = network.processes["provider-a"]
        stopped.kill()
        stopped.wait(timeout=3)
        eventually(lambda: not provider_states(client)["provider-a"]["online"])
        healthy_response = client.post("/v1/chat/completions", json=CHAT)
        assert healthy_response.status_code == 200
        assert healthy_response.json()["metadata"]["provider_id"] == "provider-b"

        network.restart_provider("provider-a")
        network.wait_until_ready()
        assert network.processes["provider-a"].pid != stopped.pid
        recovered = [client.post("/v1/chat/completions", json=CHAT) for _ in range(4)]
        assert all(response.status_code == 200 for response in recovered)
        assert {response.json()["metadata"]["provider_id"] for response in recovered} == {
            "provider-a",
            "provider-b",
        }


def test_real_capacity_errors_timeouts_and_released_reservations(network: LocalNetwork) -> None:
    with httpx.Client(base_url=network.coordinator_url, timeout=3, trust_env=False) as client:
        for endpoint in network.provider_urls.values():
            configure(client, endpoint, delay=0.5)
        with ThreadPoolExecutor(max_workers=2) as requests:
            pending = [
                requests.submit(client.post, "/v1/chat/completions", json=CHAT) for _ in range(2)
            ]
            eventually(
                lambda: sum(item["in_flight"] for item in provider_states(client).values()) == 2
            )
            overloaded = client.post("/v1/chat/completions", json=CHAT)
            assert overloaded.status_code == 503
            assert overloaded.json()["error"]["code"] == "no_eligible_provider"
            for endpoint in network.provider_urls.values():
                assert client.get(f"{endpoint}/health").json()["active_requests"] <= 1
            assert all(future.result().status_code == 200 for future in pending)
        assert all(item["in_flight"] == 0 for item in provider_states(client).values())

        for endpoint in network.provider_urls.values():
            configure(client, endpoint, mode="error")
        failed = client.post("/v1/chat/completions", json=CHAT)
        assert failed.status_code == 502
        assert failed.json()["error"]["code"] == "provider_error"
        assert all(item["in_flight"] == 0 for item in provider_states(client).values())

        for endpoint in network.provider_urls.values():
            configure(client, endpoint, delay=2)
        eventually(lambda: all(item["healthy"] for item in provider_states(client).values()))
        timed_out = client.post("/v1/chat/completions", json=CHAT)
        assert timed_out.status_code == 504
        assert timed_out.json()["error"]["code"] in {"request_timeout", "provider_timeout"}
        assert all(item["in_flight"] == 0 for item in provider_states(client).values())
        eventually(
            lambda: all(
                client.get(f"{endpoint}/health").json()["active_requests"] == 0
                for endpoint in network.provider_urls.values()
            )
        )
        for endpoint in network.provider_urls.values():
            configure(client, endpoint)
        eventually(lambda: all(item["healthy"] for item in provider_states(client).values()))
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_launcher_rejects_occupied_ports_without_starting_children() -> None:
    coordinator, first, second = unused_ports()
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", first))
        occupied.listen()
        network = LocalNetwork(coordinator, (first, second))
        with pytest.raises(RuntimeError, match="unavailable"), network:
            pytest.fail("Startup should not succeed with an occupied port")
        assert network.processes == {}


def test_launcher_cleans_up_after_partial_startup_failure() -> None:
    coordinator, first, second = unused_ports()
    network = LocalNetwork(coordinator, (first, second), environment={"DIN_MAX_CONCURRENCY": "0"})
    with pytest.raises(RuntimeError, match="exited unexpectedly"), network:
        pytest.fail("Invalid provider configuration should prevent startup")
    assert len(network.processes) == 3
    assert all(process.poll() is not None for process in network.processes.values())
    assert_ports_closed((coordinator, first, second))


def test_cli_reports_invalid_configuration_without_echoing_sensitive_values() -> None:
    secret = "fakeSecret94"
    result = subprocess.run(
        [sys.executable, "-m", "inference_network", "provider"],
        env=dict(os.environ, DIN_COORDINATOR_URL=f"http://user:{secret}@127.0.0.1:8000"),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert "coordinator_url" in result.stderr
    assert secret not in result.stdout + result.stderr


def test_cli_signal_shutdown_releases_all_listening_ports() -> None:
    ports = unused_ports()
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "inference_network",
            "demo",
            "--coordinator-port",
            str(ports[0]),
            "--provider-ports",
            str(ports[1]),
            str(ports[2]),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{ports[0]}", timeout=0.5, trust_env=False
        ) as client:

            def ready() -> bool:
                assert supervisor.poll() is None, "Network supervisor exited during startup"
                try:
                    states = provider_states(client)
                    return len(states) == 2 and all(item["online"] for item in states.values())
                except httpx.HTTPError:
                    return False

            eventually(ready, timeout=15)
            private_content = "process-test-private-content-94"
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simulated-echo-v1",
                    "messages": [{"role": "user", "content": private_content}],
                },
            )
            assert response.status_code == 200
        supervisor.send_signal(signal.SIGTERM)
        output, _ = supervisor.communicate(timeout=12)
        assert supervisor.returncode == 0, output
        events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        completed = [event for event in events if event.get("event") == "dispatch_completed"]
        assert len(completed) == 1
        assert completed[0]["provider_id"] in {"provider-a", "provider-b"}
        assert completed[0]["request_id"] == response.json()["id"]
        assert private_content not in output
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.communicate(timeout=12)
    assert_ports_closed(ports)


@pytest.mark.real_model
@pytest.mark.skipif(os.getenv("DIN_RUN_MLX_TESTS") != "1", reason="Set DIN_RUN_MLX_TESTS=1")
def test_cached_mlx_model_generates_through_real_coordinator_and_cleans_up() -> None:
    ports = unused_ports()
    network = LocalNetwork(
        ports[0],
        (ports[1], ports[2]),
        startup_timeout=60,
        environment={
            "DIN_BACKEND": "mlx",
            "DIN_REQUEST_TIMEOUT_SECONDS": "60",
            "HF_HUB_OFFLINE": "1",
        },
    )
    with (
        network,
        httpx.Client(base_url=network.coordinator_url, timeout=60, trust_env=False) as client,
    ):
        assert len({process.pid for process in network.processes.values()}) == 3
        states = provider_states(client)
        assert len(states) == 2
        for state in states.values():
            assert state["ready"] and state["online"]
            assert state["models"] == [MLX_MODEL_ID]
            assert state["simulated"] is False
            assert state["memory_kind"] == "unified_memory"
            assert state["gpu_type"].startswith("Apple")
        assert client.get("/health").json()["simulation"] is False
        models = client.get("/v1/models").json()["data"]
        assert [model["id"] for model in models] == [MLX_MODEL_ID]
        assert models[0]["owned_by"] == "local-network"

        prompts = [
            "Explain what machine learning is in two sentences.",
            "What is the capital of France? Answer in one sentence.",
        ]
        outputs = []
        for prompt in prompts:
            response = client.post(
                "/v1/chat/completions",
                json={"model": MLX_MODEL_ID, "messages": [{"role": "user", "content": prompt}]},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["model"] == MLX_MODEL_ID
            assert body["metadata"]["simulated"] is False
            assert body["metadata"]["provider_id"] in states
            content = body["choices"][0]["message"]["content"]
            assert content.strip() and content != prompt and "[SIMULATED" not in content
            assert body["choices"][0]["finish_reason"] in {"stop", "length"}
            usage = body["usage"]
            assert usage["prompt_tokens"] > 0
            assert 0 < usage["completion_tokens"] <= 128
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            outputs.append(body)
        assert (
            outputs[0]["choices"][0]["message"]["content"]
            != outputs[1]["choices"][0]["message"]["content"]
        )
        assert {body["metadata"]["provider_id"] for body in outputs} == set(states)
        assert sum(state["completed_requests"] for state in provider_states(client).values()) == 2

        payload = {"model": MLX_MODEL_ID, "messages": [{"role": "user", "content": "hello"}]}
        assert (
            client.post("/v1/chat/completions", json=dict(payload, model="wrong")).status_code
            == 404
        )
        assert (
            client.post("/v1/chat/completions", json=dict(payload, messages=[])).status_code == 422
        )
        assert (
            client.post("/v1/chat/completions", json=dict(payload, stream="invalid")).status_code
            == 422
        )
        endpoint = network.provider_urls["provider-a"]
        assert client.put(f"{endpoint}/simulation", json={"mode": "error"}).status_code == 404
        # Stay under the shared character bound while exceeding the model token bound.
        too_long = dict(payload, messages=[{"role": "user", "content": "x " * 2200}])
        rejected = client.post(f"{endpoint}/internal/chat/completions", json=too_long)
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "prompt_too_long"
        assert client.get(f"{endpoint}/health").json()["active_requests"] == 0
        assert client.post("/v1/chat/completions", json=payload).status_code == 200
        assert all(state["in_flight"] == 0 for state in provider_states(client).values())
    # Uvicorn re-raises SIGTERM after completing its graceful lifespan shutdown.
    assert all(process.returncode in (0, -signal.SIGTERM) for process in network.processes.values())
    assert_ports_closed(ports)


@pytest.mark.real_model
@pytest.mark.skipif(os.getenv("DIN_RUN_MLX_TESTS") != "1", reason="Set DIN_RUN_MLX_TESTS=1")
def test_mlx_missing_snapshot_fails_startup_and_releases_processes(tmp_path) -> None:
    ports = unused_ports()
    network = LocalNetwork(
        ports[0],
        (ports[1], ports[2]),
        startup_timeout=60,
        environment={
            "DIN_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_CACHE": str(tmp_path / "empty-model-cache"),
        },
    )
    with pytest.raises(RuntimeError, match="exited unexpectedly"), network:
        pytest.fail("An absent pinned snapshot must prevent startup")
    assert all(process.poll() is not None for process in network.processes.values())
    assert_ports_closed(ports)


def stream_events(response: httpx.Response) -> Iterator[tuple[str, Any]]:
    """Read complete SSE frames independently of HTTP/network chunk boundaries."""
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    event = "message"
    data: list[str] = []
    for line in response.iter_lines():
        if not line:
            if data:
                value = "\n".join(data)
                yield event, value if value == "[DONE]" else json.loads(value)
            event, data = "message", []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif line.startswith("event:"):
            event = line[6:].strip()
    assert not data, "Stream ended in an unterminated SSE frame"


def assert_successful_stream(
    events: list[tuple[str, Any]], model: str, simulated: bool, *, include_usage: bool = False
) -> dict[str, Any]:
    assert events[-1] == ("message", "[DONE]")
    assert all(event == "message" and isinstance(data, dict) for event, data in events[:-1])
    chunks = [data for _, data in events[:-1]]
    first = chunks[0]
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["choices"][0]["delta"].get("content") in (None, "")
    assert first["choices"][0]["finish_reason"] is None
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert len({chunk["created"] for chunk in chunks}) == 1
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == model
        assert chunk["metadata"] == first["metadata"]
        assert chunk["metadata"]["simulated"] is simulated
        for choice in chunk["choices"]:
            assert choice["index"] == 0
            assert "message" not in choice
            assert "delta" in choice
    content_chunks = chunks[1 : -2 if include_usage else -1]
    assert len(content_chunks) > 1
    for chunk in content_chunks:
        choice = chunk["choices"][0]
        assert choice["delta"]["content"]
        assert choice["delta"].get("role") is None
        assert choice["finish_reason"] is None
        assert chunk.get("usage") is None
    terminal = chunks[-2 if include_usage else -1]
    assert terminal["choices"][0]["finish_reason"] in {"stop", "length"}
    assert not terminal["choices"][0]["delta"].get("content")
    usage = None
    if include_usage:
        assert chunks[-1]["choices"] == []
        usage = chunks[-1]["usage"]
        if simulated:
            assert usage is None
        else:
            assert usage["prompt_tokens"] > 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    return {
        "content": "".join(chunk["choices"][0]["delta"]["content"] for chunk in content_chunks),
        "provider_id": first["metadata"]["provider_id"],
        "usage": usage,
        "finish_reason": terminal["choices"][0]["finish_reason"],
        "content_chunks": len(content_chunks),
    }


def first_content(events: Iterator[tuple[str, Any]]) -> dict[str, Any]:
    for event, data in events:
        assert event == "message" and isinstance(data, dict)
        if data["choices"] and data["choices"][0]["delta"].get("content"):
            return data
    raise AssertionError("Stream ended without generated content")


def assert_capacity_released(client: httpx.Client, network: LocalNetwork) -> None:
    eventually(lambda: all(state["in_flight"] == 0 for state in provider_states(client).values()))
    eventually(
        lambda: all(
            client.get(f"{endpoint}/health").json()["active_requests"] == 0
            for endpoint in network.provider_urls.values()
        )
    )


@pytest.mark.parametrize("include_usage", [False, True])
def test_simulator_stream_protocol_reconstruction_and_round_robin(
    network: LocalNetwork, include_usage: bool
) -> None:
    payload = dict(CHAT, stream=True)
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    with httpx.Client(base_url=network.coordinator_url, timeout=3, trust_env=False) as client:
        assert all(state["streaming"] for state in provider_states(client).values())
        outputs = []
        for _ in range(2):
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                outputs.append(
                    assert_successful_stream(
                        list(stream_events(response)),
                        CHAT["model"],
                        True,
                        include_usage=include_usage,
                    )
                )
        assert [output["provider_id"] for output in outputs] == ["provider-a", "provider-b"]
        nonstream = client.post("/v1/chat/completions", json=dict(CHAT, stream=False))
        assert nonstream.status_code == 200
        assert all(
            output["content"] == nonstream.json()["choices"][0]["message"]["content"]
            for output in outputs
        )
        assert_capacity_released(client, network)


def test_concurrent_stream_admission_and_client_disconnect(network: LocalNetwork) -> None:
    payload = dict(CHAT, stream=True, messages=[{"role": "user", "content": "x" * 4000}])
    with httpx.Client(base_url=network.coordinator_url, timeout=3, trust_env=False) as client:
        for endpoint in network.provider_urls.values():
            configure(client, endpoint, delay=0.01)
        with ExitStack() as streams:
            selected = set()
            readers = []
            for _ in range(2):
                response = streams.enter_context(
                    client.stream("POST", "/v1/chat/completions", json=payload)
                )
                events = stream_events(response)
                readers.append(events)
                selected.add(first_content(events)["metadata"]["provider_id"])
            assert selected == {"provider-a", "provider-b"}
            assert all(state["in_flight"] == 1 for state in provider_states(client).values())
            overloaded = client.post("/v1/chat/completions", json=payload)
            assert overloaded.status_code == 503
            assert overloaded.json()["error"]["code"] == "no_eligible_provider"
            for endpoint in network.provider_urls.values():
                assert client.get(f"{endpoint}/health").json()["active_requests"] == 1
                direct = client.post(f"{endpoint}/internal/chat/completions", json=payload)
                assert direct.status_code == 429
                assert direct.json()["error"]["code"] == "provider_capacity"
        assert_capacity_released(client, network)
        assert sum(state["completed_requests"] for state in provider_states(client).values()) == 0
        for endpoint in network.provider_urls.values():
            configure(client, endpoint)
        with client.stream(
            "POST", "/v1/chat/completions", json=dict(CHAT, stream=True)
        ) as response:
            assert_successful_stream(list(stream_events(response)), CHAT["model"], True)


def test_stream_timeout_after_content_reports_error_without_success(network: LocalNetwork) -> None:
    payload = dict(CHAT, stream=True, messages=[{"role": "user", "content": "x" * 4000}])
    with httpx.Client(base_url=network.coordinator_url, timeout=3, trust_env=False) as client:
        for endpoint in network.provider_urls.values():
            configure(client, endpoint, delay=0.05)
        started = time.monotonic()
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            events = list(stream_events(response))
        assert time.monotonic() - started < 2.5
        assert any(
            data.get("choices", [{}])[0].get("delta", {}).get("content")
            for event, data in events
            if event == "message"
        )
        assert events[-1][0] == "error"
        assert events[-1][1]["error"]["code"] in {"request_timeout", "provider_timeout"}
        assert all(data != "[DONE]" for _, data in events)
        assert all(
            data["choices"][0]["finish_reason"] is None
            for event, data in events
            if event == "message"
        )
        assert_capacity_released(client, network)
        states = provider_states(client)
        assert sum(state["completed_requests"] for state in states.values()) == 0
        assert sum(state["failed_requests"] for state in states.values()) == 1


def test_provider_killed_during_stream_is_not_replayed(network: LocalNetwork) -> None:
    payload = dict(CHAT, stream=True, messages=[{"role": "user", "content": "x" * 4000}])
    with httpx.Client(base_url=network.coordinator_url, timeout=3, trust_env=False) as client:
        for endpoint in network.provider_urls.values():
            configure(client, endpoint, delay=0.02)
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            events = stream_events(response)
            selected = first_content(events)["metadata"]["provider_id"]
            process = network.processes[selected]
            process.kill()
            process.wait(timeout=3)
            tail = list(events)
        assert tail[-1][0] == "error"
        assert all(data != "[DONE]" for _, data in tail)
        assert all(
            data["choices"][0]["finish_reason"] is None
            for event, data in tail
            if event == "message"
        )
        states = provider_states(client)
        survivor = next(provider_id for provider_id in states if provider_id != selected)
        assert states[survivor]["completed_requests"] == 0
        assert states[survivor]["failed_requests"] == 0
        assert all(state["in_flight"] == 0 for state in states.values())
        eventually(lambda: not provider_states(client)[selected]["online"])
        configure(client, network.provider_urls[survivor])
        with client.stream(
            "POST", "/v1/chat/completions", json=dict(CHAT, stream=True)
        ) as response:
            output = assert_successful_stream(list(stream_events(response)), CHAT["model"], True)
        assert output["provider_id"] == survivor


def test_stream_precommit_errors_and_safe_connection_retry(network: LocalNetwork) -> None:
    payload = dict(CHAT, stream=True)
    with httpx.Client(base_url=network.coordinator_url, timeout=3, trust_env=False) as client:
        for endpoint in network.provider_urls.values():
            configure(client, endpoint, mode="error")
        rejected = client.post("/v1/chat/completions", json=payload)
        assert rejected.status_code == 502
        assert rejected.headers["content-type"].startswith("application/json")
        assert rejected.json()["error"]["code"] == "provider_error"
        assert_capacity_released(client, network)
        assert sum(state["failed_requests"] for state in provider_states(client).values()) == 1
        for endpoint in network.provider_urls.values():
            configure(client, endpoint)
        eventually(lambda: all(state["healthy"] for state in provider_states(client).values()))
        # The failed first request selected A, so B is next. Its registration is
        # still fresh when connection refusal proves this attempt executed no work.
        process = network.processes["provider-b"]
        process.kill()
        process.wait(timeout=3)
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            output = assert_successful_stream(list(stream_events(response)), CHAT["model"], True)
        assert output["provider_id"] == "provider-a"
        states = provider_states(client)
        assert states["provider-b"]["failed_requests"] == 1
        assert all(state["in_flight"] == 0 for state in states.values())


@pytest.mark.real_model
@pytest.mark.skipif(os.getenv("DIN_RUN_MLX_TESTS") != "1", reason="Set DIN_RUN_MLX_TESTS=1")
def test_mlx_stream_arrives_during_generation_matches_nonstream_and_cancels(tmp_path) -> None:
    # Observe the installed vendor generator in actual provider subprocesses.
    # No synthetic tokens, sleeps, or production telemetry are involved.
    (tmp_path / "sitecustomize.py").write_text("""
import contextlib
import json
import os
import sys
import time
from pathlib import Path
if "provider" in sys.argv:
    import mlx_lm
    original = mlx_lm.stream_generate
    def observed(*args, **kwargs):
        text = []
        finished_ns = None
        first_text_ns = None
        try:
            with contextlib.closing(original(*args, **kwargs)) as generator:
                for response in generator:
                    if response.text and first_text_ns is None:
                        first_text_ns = time.monotonic_ns()
                    text.append(response.text)
                    if response.finish_reason is not None:
                        finished_ns = time.monotonic_ns()
                    yield response
        finally:
            path = Path(os.environ["DIN_TEST_STREAM_EVIDENCE"]) / f"{os.getpid()}.jsonl"
            with path.open("a") as out:
                out.write(json.dumps({"text": "".join(text), "finished_ns": finished_ns,
                    "first_text_ns": first_text_ns}) + "\\n")
    mlx_lm.stream_generate = observed
""")
    ports = unused_ports()
    network = LocalNetwork(
        ports[0],
        (ports[1], ports[2]),
        startup_timeout=60,
        environment={
            "DIN_BACKEND": "mlx",
            "DIN_REQUEST_TIMEOUT_SECONDS": "60",
            "DIN_MAX_OUTPUT_TOKENS": "512",
            "DIN_TEST_STREAM_EVIDENCE": str(tmp_path),
            "PYTHONPATH": str(tmp_path) + os.pathsep + str(Path("src").resolve()),
            "HF_HUB_OFFLINE": "1",
        },
    )
    payload = {
        "model": MLX_MODEL_ID,
        "messages": [{"role": "user", "content": "Explain how computers work in ten paragraphs."}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with (
        network,
        httpx.Client(base_url=network.coordinator_url, timeout=60, trust_env=False) as client,
    ):
        assert len({process.pid for process in network.processes.values()}) == 3
        assert all(state["streaming"] for state in provider_states(client).values())
        events = []
        started_ns = time.monotonic_ns()
        first_received_ns = None
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            received_content = 0
            for event in stream_events(response):
                events.append(event)
                _, data = event
                if isinstance(data, dict) and data.get("choices"):
                    if data["choices"][0]["delta"].get("content"):
                        if first_received_ns is None:
                            first_received_ns = time.monotonic_ns()
                        received_content += 1
                        if received_content == 2:
                            provider_id = data["metadata"]["provider_id"]
                            endpoint = network.provider_urls[provider_id]
                            # Capacity remains owned during active streaming.
                            assert client.get(f"{endpoint}/health").json()["active_requests"] == 1
                            state = provider_states(client)[provider_id]
                            assert state["in_flight"] == 1
                            assert state["completed_requests"] == 0
        output = assert_successful_stream(events, MLX_MODEL_ID, False, include_usage=True)
        provider_pid = network.processes[output["provider_id"]].pid
        evidence = json.loads((tmp_path / f"{provider_pid}.jsonl").read_text().splitlines()[0])
        assert evidence["text"] == output["content"]
        assert first_received_ns is not None
        assert evidence["first_text_ns"] <= first_received_ns < evidence["finished_ns"]
        print(
            json.dumps(
                {
                    "real_stream_evidence": {
                        "content_chunks": output["content_chunks"],
                        "usage": output["usage"],
                        "first_client_text_seconds": (first_received_ns - started_ns) / 1e9,
                        "vendor_finished_seconds": (evidence["finished_ns"] - started_ns) / 1e9,
                        "client_text_matches_vendor": True,
                    }
                }
            )
        )
        assert 1 < output["usage"]["completion_tokens"] <= 512
        assert "[SIMULATED" not in output["content"]
        assert_capacity_released(client, network)
        nonstream_payload = {
            key: value for key, value in payload.items() if key != "stream_options"
        }
        nonstream = client.post("/v1/chat/completions", json=dict(nonstream_payload, stream=False))
        assert nonstream.status_code == 200, nonstream.text
        assert output["content"] == nonstream.json()["choices"][0]["message"]["content"]
        assert output["usage"] == nonstream.json()["usage"]
        assert output["finish_reason"] == nonstream.json()["choices"][0]["finish_reason"]

        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            reader = stream_events(response)
            first = first_content(reader)
            endpoint = network.provider_urls[first["metadata"]["provider_id"]]
            assert client.get(f"{endpoint}/health").json()["active_requests"] == 1
        assert_capacity_released(client, network)
        assert sum(state["completed_requests"] for state in provider_states(client).values()) == 2
        assert sum(state["failed_requests"] for state in provider_states(client).values()) == 1
        recovery = client.post(
            "/v1/chat/completions",
            json={"model": MLX_MODEL_ID, "messages": [{"role": "user", "content": "Say hello."}]},
        )
        assert recovery.status_code == 200
        assert_capacity_released(client, network)
    assert all(process.returncode in (0, -signal.SIGTERM) for process in network.processes.values())
    assert_ports_closed(ports)
