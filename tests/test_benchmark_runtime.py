"""Benchmark process ownership and metadata tests; no actual MLX execution."""

import json
import os
import signal
import socket
import subprocess
import sys

import pytest

from inference_network.config import MLX_MODEL_ID, CoordinatorSettings, ProviderSettings
from scripts import benchmark_runtime
from scripts.benchmark_runtime import BenchmarkNetwork, environment_metadata, rss_bytes


@pytest.fixture
def fake_services(tmp_path, monkeypatch):
    """Exercise real sockets/processes/signals while replacing only the CLI program."""
    service = tmp_path / "fake_service.py"
    service.write_text("""
import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

role, provider_id, endpoint, model, mode = sys.argv[1:]
if role == "provider" and mode == "exit":
    raise SystemExit(17)
if role == "provider" and mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = dict(provider_id=provider_id, endpoint=endpoint, models=[model],
                       ready=mode != "not_ready", online=True, healthy=True,
                       registered=True, backend="mlx", simulated=False, streaming=True)
        if self.path == "/providers":
            payload = {"providers": [payload]}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", int(os.environ["DIN_PORT"])), Handler).serve_forever()
""")
    original = subprocess.Popen
    calls = []

    def install(network, mode="ready"):
        def spawn(command, **kwargs):
            calls.append((command, kwargs["env"]))
            assert command[:3] == [sys.executable, "-m", "inference_network"]
            return original(
                [
                    sys.executable,
                    str(service),
                    command[-1],
                    network.settings.provider_id,
                    network.provider_url,
                    MLX_MODEL_ID,
                    mode,
                ],
                **kwargs,
            )

        monkeypatch.setattr(benchmark_runtime.subprocess, "Popen", spawn)
        return calls

    return install


def assert_reaped(network):
    assert network.processes
    assert all(process.poll() is not None for process in network.processes.values())
    assert len(network.shutdown_records) == len(network.processes)
    for url in (network.coordinator_url, network.provider_url):
        with socket.socket() as connection:
            connection.settimeout(0.2)
            assert connection.connect_ex(("127.0.0.1", int(url.rsplit(":", 1)[1]))) != 0


def test_network_owns_two_processes_and_preserves_validated_limits(
    tmp_path, monkeypatch, fake_services
) -> None:
    monkeypatch.setenv("DIN_BACKEND", "simulator")
    monkeypatch.setenv("DIN_HOST", "0.0.0.0")
    monkeypatch.setenv("DIN_MAX_CONCURRENCY", "64")
    monkeypatch.setenv("DIN_UNRECOGNIZED_SETTING", "inherited")
    settings = ProviderSettings(backend="mlx", max_output_tokens=64, request_timeout_seconds=60)
    network = BenchmarkNetwork(settings, tmp_path / "logs")
    calls = fake_services(network)
    with network:
        assert set(network.processes) == {"coordinator", "provider"}
        assert network.coordinator_url != network.provider_url
        assert all(value > 0 for value in network.startup_seconds.values())
        coordinator, provider = [env for _, env in calls]
        assert "DIN_BACKEND" not in coordinator
        for environment in (coordinator, provider):
            assert environment["DIN_HOST"] == "127.0.0.1"
            assert environment["DIN_REQUEST_TIMEOUT_SECONDS"] == "60.0"
            assert environment["HF_HUB_OFFLINE"] == "1"
            assert "DIN_UNRECOGNIZED_SETTING" not in environment
        assert provider["DIN_BACKEND"] == "mlx"
        assert provider["DIN_MAX_OUTPUT_TOKENS"] == "64"
        assert provider["DIN_MAX_CONCURRENCY"] == "1"
        assert provider["DIN_ENDPOINT"] == network.provider_url
        assert provider["DIN_COORDINATOR_URL"] == network.coordinator_url
    assert_reaped(network)
    assert all(not record["forced_kill"] for record in network.shutdown_records)
    assert all(path.is_file() for path in network.log_paths.values())
    network.close()
    assert len(network.shutdown_records) == 2


def test_spawned_services_reload_local_settings_without_pairing(
    tmp_path, monkeypatch, fake_services
) -> None:
    monkeypatch.setenv("DIN_CREDENTIALS_FILE", str(tmp_path / "unrelated-profile.json"))
    network = BenchmarkNetwork(ProviderSettings(backend="mlx"), tmp_path / "logs")
    calls = fake_services(network)
    with network:
        for command, environment in calls:
            settings_type = (
                CoordinatorSettings if command[-1] == "coordinator" else ProviderSettings
            )
            with monkeypatch.context() as process_environment:
                process_environment.setattr(os, "environ", environment)
                settings = settings_type.from_env()
            assert settings.credentials_file is None
            assert settings.lan_mode is False
    assert_reaped(network)


@pytest.mark.parametrize("mode,match", [("exit", "exited"), ("not_ready", "Timed out")])
def test_startup_failure_reaps_every_launched_process(tmp_path, fake_services, mode, match) -> None:
    network = BenchmarkNetwork(
        ProviderSettings(backend="mlx"), tmp_path / "logs", startup_timeout=1
    )
    fake_services(network, mode)
    with pytest.raises(RuntimeError, match=match), network:
        pytest.fail("Failed startup must not enter the benchmark")
    assert_reaped(network)


def test_interrupted_work_reaps_every_launched_process(tmp_path, fake_services) -> None:
    network = BenchmarkNetwork(ProviderSettings(backend="mlx"), tmp_path / "logs")
    fake_services(network)
    with pytest.raises(KeyboardInterrupt), network:
        raise KeyboardInterrupt
    assert_reaped(network)


def test_unresponsive_child_is_killed_reaped_and_reported(tmp_path, fake_services) -> None:
    network = BenchmarkNetwork(
        ProviderSettings(backend="mlx"), tmp_path / "logs", shutdown_timeout=0.1
    )
    fake_services(network, "ignore_term")
    with network:
        pass
    assert_reaped(network)
    provider = next(record for record in network.shutdown_records if record["role"] == "provider")
    assert provider["forced_kill"] is True
    assert provider["returncode"] == -signal.SIGKILL


def test_partial_launch_failure_reaps_previously_started_child(tmp_path, monkeypatch) -> None:
    network = BenchmarkNetwork(ProviderSettings(backend="mlx"), tmp_path / "logs")

    def spawn(role, settings):
        if role == "provider":
            raise OSError("Synthetic process creation failure")
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    monkeypatch.setattr(network, "_spawn", spawn)
    with pytest.raises(OSError, match="Synthetic process"), network:
        pytest.fail("Failed process creation must not enter the benchmark")
    assert_reaped(network)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 121])
def test_invalid_timeouts_are_rejected_before_launch(tmp_path, timeout) -> None:
    with pytest.raises(ValueError, match="startup_timeout"):
        BenchmarkNetwork(ProviderSettings(backend="mlx"), tmp_path, startup_timeout=timeout)


def test_simulation_cannot_be_mislabeled_as_real_benchmark(tmp_path) -> None:
    with pytest.raises(ValueError, match="MLX backend"):
        BenchmarkNetwork(ProviderSettings(), tmp_path)


def test_rss_is_bytes_and_missing_measurements_are_null(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_runtime, "_command_output", lambda command: "123" if command[-1] == "42" else None
    )
    assert rss_bytes({"known": 42, "missing": 43}) == {"known": 123 * 1024, "missing": None}
    with pytest.raises(ValueError, match="positive"):
        rss_bytes({"invalid": -1})


def test_rss_can_measure_the_current_process() -> None:
    value = rss_bytes({"self": os.getpid()})["self"]
    assert value is not None and value > 0


def test_environment_metadata_is_serializable_and_records_identity() -> None:
    metadata = environment_metadata(ProviderSettings(backend="mlx"))
    assert metadata["model"]["id"] == MLX_MODEL_ID
    assert len(metadata["model"]["revision"]) == 40
    assert metadata["python"]
    assert metadata["hardware"]["architecture"]
    assert metadata["dependencies"]["httpx"]
    assert "GPU memory" in metadata["memory_measurement"]
    json.dumps(metadata)
