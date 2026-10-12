"""Benchmark-only ownership of one coordinator and one existing MLX provider."""

import hashlib
import importlib.metadata
import math
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import httpx

from inference_network.config import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    CoordinatorSettings,
    ProviderSettings,
    Settings,
)

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkNetwork:
    """Start existing service entry points, wait for readiness, and reap every child.

    Ports are reserved together, then released immediately before each CLI starts.
    The CLI does not accept inherited sockets, so a competing bind can still fail
    startup. Readiness is bounded and verifies the expected provider identity.
    """

    def __init__(
        self,
        settings: ProviderSettings,
        output_dir: Path,
        *,
        startup_timeout: float = 60.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        if settings.backend != "mlx":
            raise ValueError("Inference benchmarks require the MLX backend")
        for name, value in (
            ("startup_timeout", startup_timeout),
            ("shutdown_timeout", shutdown_timeout),
        ):
            if not math.isfinite(value) or not 0 < value <= 120:
                raise ValueError(f"{name} must be greater than zero and at most 120 seconds")
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.coordinator_url = ""
        self.provider_url = ""
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.startup_seconds: dict[str, float] = {}
        self.shutdown_records: list[dict[str, object]] = []
        self.log_paths: dict[str, Path] = {}
        self._logs: list[BinaryIO] = []
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._started_at: dict[str, float] = {}

    def __enter__(self) -> "BenchmarkNetwork":
        if self.processes:
            raise RuntimeError("A benchmark network cannot be started twice")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            with ExitStack() as reservations:
                listeners = [reservations.enter_context(socket.socket()) for _ in range(2)]
                for listener in listeners:
                    listener.bind(("127.0.0.1", 0))
                coordinator_port, provider_port = [
                    listener.getsockname()[1] for listener in listeners
                ]
                self.coordinator_url = f"http://127.0.0.1:{coordinator_port}"
                self.provider_url = f"http://127.0.0.1:{provider_port}"
                shared = {
                    name: getattr(self.settings, name)
                    for name in Settings.model_fields
                    if name != "port"
                }
                coordinator = CoordinatorSettings(**shared, port=coordinator_port)
                provider = ProviderSettings.model_validate(
                    dict(
                        self.settings.model_dump(),
                        port=provider_port,
                        endpoint=self.provider_url,
                        coordinator_url=self.coordinator_url,
                    )
                )
                for role, listener, settings in zip(
                    ("coordinator", "provider"), listeners, (coordinator, provider), strict=True
                ):
                    listener.close()
                    self._started_at[role] = time.perf_counter()
                    self.processes[role] = self._spawn(role, settings)
                self.wait_until_ready()
            self.startup_seconds["network_process_to_ready"] = time.perf_counter() - started
            return self
        except BaseException as error:
            try:
                self.close()
            except RuntimeError as cleanup_error:
                error.add_note(str(cleanup_error))
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except RuntimeError as cleanup_error:
            if exception is None:
                raise
            exception.add_note(str(cleanup_error))

    def _spawn(self, role: str, settings: Settings) -> subprocess.Popen[bytes]:
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("DIN_")
        }
        environment.update(
            HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_OFFLINE="1"
        )
        for key, value in settings.model_dump().items():
            if value is None:
                continue
            environment[f"DIN_{key.upper()}"] = (
                ",".join(value) if isinstance(value, tuple) else str(value)
            )
        path = self.output_dir / f"{role}.log"
        log = path.open("xb")
        self.log_paths[role] = path
        self._logs.append(log)
        return subprocess.Popen(
            [sys.executable, "-m", "inference_network", role],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def check_running(self) -> None:
        for role, process in self.processes.items():
            if process.poll() is not None:
                raise RuntimeError(
                    f"{role} exited with status {process.returncode}; "
                    f"see {self.log_paths.get(role)}"
                )

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout
        with httpx.Client(timeout=0.25, trust_env=False) as client:
            while time.monotonic() < deadline:
                self.check_running()
                try:
                    registered = client.get(f"{self.coordinator_url}/providers")
                    if registered.status_code == 200:
                        self.startup_seconds.setdefault(
                            "coordinator_process_to_ready",
                            time.perf_counter() - self._started_at["coordinator"],
                        )
                        providers = registered.json().get("providers", [])
                        health = client.get(f"{self.provider_url}/health")
                        if health.status_code == 200 and self._is_ready(health.json(), providers):
                            self.check_running()
                            self.startup_seconds["provider_process_to_ready"] = (
                                time.perf_counter() - self._started_at["provider"]
                            )
                            return
                except httpx.HTTPError:
                    # Refused connections are expected until the processes bind.
                    pass
                time.sleep(0.025)
        raise RuntimeError(f"Timed out waiting for the benchmark provider; see {self.output_dir}")

    def _is_ready(self, health: dict[str, object], providers: list[dict[str, object]]) -> bool:
        if len(providers) != 1:
            return False
        provider = providers[0]
        return (
            health.get("ready") is True
            and health.get("registered") is True
            and health.get("backend") == "mlx"
            and health.get("simulated") is False
            and health.get("provider_id") == self.settings.provider_id
            and health.get("models") == list(self.settings.models)
            and provider.get("provider_id") == self.settings.provider_id
            and provider.get("endpoint") == self.provider_url
            and provider.get("models") == list(self.settings.models)
            and provider.get("simulated") is False
            and provider.get("streaming") is True
            and all(provider.get(field) is True for field in ("online", "ready", "healthy"))
        )

    def close(self) -> None:
        errors = []
        recorded = {record["pid"] for record in self.shutdown_records}
        # Stop the provider first so its shutdown heartbeat can reach the coordinator.
        for role, process in reversed(list(self.processes.items())):
            if process.pid in recorded:
                continue
            forced_kill = False
            try:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=self._shutdown_timeout)
                except subprocess.TimeoutExpired:
                    forced_kill = True
                    process.kill()
                    process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(f"{role}: {type(error).__name__}")
            finally:
                url = self.provider_url if role == "provider" else self.coordinator_url
                with socket.socket() as probe:
                    probe.settimeout(0.25)
                    listener_closed = (
                        probe.connect_ex(("127.0.0.1", int(url.rsplit(":", 1)[1]))) != 0
                    )
                self.shutdown_records.append(
                    dict(
                        role=role,
                        pid=process.pid,
                        returncode=process.poll(),
                        forced_kill=forced_kill,
                        listener_closed=listener_closed,
                    )
                )
        for log in self._logs:
            log.close()
        if errors:
            raise RuntimeError("Benchmark subprocess cleanup failed: " + "; ".join(errors))


def _command_output(command: list[str]) -> str | None:
    """Return unavailable metadata as null rather than inventing a measurement."""
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment_metadata(settings: ProviderSettings | None = None) -> dict[str, object]:
    """Collect reproducibility information without initializing MLX or the GPU."""
    dependencies: dict[str, str | None] = {}
    for name in (
        "nvidia-distributed-inference-prototype",
        "mlx",
        "mlx-metal",
        "mlx-lm",
        "fastapi",
        "httpx",
        "pydantic",
        "uvicorn",
        "huggingface-hub",
        "transformers",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    hardware: dict[str, object] = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        hardware["chip"] = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        hardware["model"] = _command_output(["sysctl", "-n", "hw.model"])
        memory = _command_output(["sysctl", "-n", "hw.memsize"])
        hardware["physical_memory_bytes"] = int(memory) if memory and memory.isdigit() else None
    status = _command_output(["git", "status", "--porcelain"])
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "hardware": hardware,
        "model": {"id": MLX_MODEL_ID, "revision": MLX_MODEL_REVISION},
        "model_revision": MLX_MODEL_REVISION,
        "dependencies": dependencies,
        "git": {
            "commit": _command_output(["git", "rev-parse", "HEAD"]),
            "dirty": status != "" if status is not None else None,
            "status": status,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [*ROOT.glob("src/inference_network/*.py"), *ROOT.glob("scripts/benchmark*.py")]
            )
        },
        "provider_settings": settings.model_dump(mode="json") if settings else None,
        "memory_measurement": "ps RSS in KiB, converted to bytes; not GPU memory or peak RSS",
    }


def rss_bytes(pids: dict[str, int]) -> dict[str, int | None]:
    """Take process RSS snapshots outside timed requests; missing processes are null."""
    result: dict[str, int | None] = {}
    for role, pid in pids.items():
        if pid <= 0:
            raise ValueError("RSS measurement requires positive process IDs")
        output = _command_output(["ps", "-o", "rss=", "-p", str(pid)])
        result[role] = int(output) * 1024 if output and output.isdigit() else None
    return result
