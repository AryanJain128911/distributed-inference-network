"""Bounded service startup and owned subprocess cleanup for interactive roles and demos."""

import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import FrameType, TracebackType

import httpx

from inference_network.config import CoordinatorSettings, ProviderSettings

PROVIDER_IDS = ("provider-a", "provider-b")


class ServiceProcess:
    """Own one existing service for the interactive launcher, including failed startup."""

    def __init__(
        self,
        role: str,
        settings: CoordinatorSettings | ProviderSettings,
        headers: Mapping[str, str] | None = None,
        startup_timeout: float = 120,
    ) -> None:
        if role not in {"provider", "coordinator"}:
            raise ValueError("Unknown service role")
        self.role = role
        self.settings = settings
        self.headers = dict(headers or {})
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.url = f"http://127.0.0.1:{settings.port}"

    def __enter__(self) -> "ServiceProcess":
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.settings.port))
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("DIN_")
        }
        for field, value in self.settings.model_dump().items():
            if value is None:
                continue
            if isinstance(value, tuple):
                rendered = ",".join(value)
            elif isinstance(value, (str, int, float, bool, Path)):
                rendered = str(value)
            else:
                raise ValueError("Unsupported service configuration")
            environment[f"DIN_{field.upper()}"] = rendered
        self.process = subprocess.Popen(
            [sys.executable, "-m", "inference_network", self.role],
            env=environment,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + self.startup_timeout
            with httpx.Client(timeout=0.5, trust_env=False, headers=self.headers) as client:
                while time.monotonic() < deadline:
                    self.check_running()
                    try:
                        response = client.get(f"{self.url}/health")
                        if response.status_code == 200 and response.json().get("status") == "ok":
                            return self
                        if response.status_code in {401, 403}:
                            raise RuntimeError("Service pairing credentials were rejected")
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
            raise RuntimeError("Service startup timed out; inspect the provider setup")
        except BaseException:
            self.close()
            raise

    def check_running(self) -> None:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"{self.role} stopped unexpectedly")

    def close(self) -> None:
        if self.process is not None:
            stop_processes([self.process])

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


class LocalNetwork:
    """Own exactly three service subprocesses; release them even if startup fails."""

    def __init__(
        self,
        coordinator_port: int = 8000,
        provider_ports: tuple[int, int] = (8101, 8102),
        startup_timeout: float = 15,
        environment: Mapping[str, str] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        CoordinatorSettings(port=coordinator_port)
        for provider_id, port in zip(PROVIDER_IDS, provider_ports, strict=True):
            ProviderSettings(
                provider_id=provider_id, port=port, endpoint=f"http://127.0.0.1:{port}"
            )
        if len({coordinator_port, *provider_ports}) != 3:
            raise ValueError("Coordinator and provider ports must be distinct")
        if not math.isfinite(startup_timeout) or not 0 < startup_timeout <= 120:
            raise ValueError("Startup timeout must be greater than 0 and at most 120 seconds")
        self.coordinator_port = coordinator_port
        self.provider_ports = provider_ports
        self.coordinator_url = f"http://127.0.0.1:{coordinator_port}"
        self.provider_urls = {
            provider_id: f"http://127.0.0.1:{port}"
            for provider_id, port in zip(PROVIDER_IDS, provider_ports, strict=True)
        }
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self._environment = os.environ.copy()
        self._environment.update(environment or {})
        self._startup_timeout = startup_timeout
        self._stop_event = stop_event or threading.Event()

    def __enter__(self) -> "LocalNetwork":
        try:
            self._check_ports()
            self.processes["coordinator"] = self._spawn("coordinator", self.coordinator_port)
            for provider_id, port in zip(PROVIDER_IDS, self.provider_ports, strict=True):
                self.processes[provider_id] = self._spawn("provider", port, provider_id)
            self.wait_until_ready()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _check_ports(self) -> None:
        for port in (self.coordinator_port, *self.provider_ports):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError as error:
                    raise RuntimeError(f"Local port {port} is unavailable") from error

    def _spawn(
        self, role: str, port: int, provider_id: str | None = None
    ) -> subprocess.Popen[bytes]:
        environment = dict(self._environment, DIN_PORT=str(port))
        if provider_id is not None:
            environment.update(
                DIN_PROVIDER_ID=provider_id,
                DIN_COORDINATOR_URL=self.coordinator_url,
                DIN_ENDPOINT=f"http://127.0.0.1:{port}",
            )
        return subprocess.Popen(
            [sys.executable, "-m", "inference_network", role],
            env=environment,
            # The supervisor alone handles terminal interrupts and forwards shutdown.
            start_new_session=True,
        )

    def check_running(self) -> None:
        for name, process in self.processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited unexpectedly with status {process.returncode}")

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout
        with httpx.Client(timeout=0.5, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    raise InterruptedError("Network startup interrupted")
                self.check_running()
                try:
                    response = client.get(f"{self.coordinator_url}/providers")
                    if response.status_code == 200:
                        providers = response.json().get("providers", [])
                        ready_ids = {
                            provider["provider_id"]
                            for provider in providers
                            if provider["online"] and provider["ready"] and provider["healthy"]
                        }
                        if set(PROVIDER_IDS) <= ready_ids:
                            return
                except httpx.HTTPError:
                    pass  # Startup connection failures are expected until listeners are ready.
                self._stop_event.wait(0.05)
        raise RuntimeError("Timed out waiting for both providers to register and become ready")

    def restart_provider(self, provider_id: str) -> None:
        """Restart a stopped provider; its new instance must register after lease expiry."""
        index = PROVIDER_IDS.index(provider_id)
        previous = self.processes[provider_id]
        if previous.poll() is None:
            raise RuntimeError(f"{provider_id} is still running")
        self.processes[provider_id] = self._spawn(
            "provider", self.provider_ports[index], provider_id
        )

    def close(self) -> None:
        stop_processes(list(reversed(self.processes.values())))


def run_network(
    coordinator_port: int = 8000,
    provider_ports: tuple[int, int] = (8101, 8102),
    startup_timeout: float = 15,
) -> int:
    stop_event = threading.Event()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        stop_event.set()

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with LocalNetwork(
            coordinator_port, provider_ports, startup_timeout, stop_event=stop_event
        ) as network:
            print(
                f"Local inference network ready: {network.coordinator_url} "
                f"(providers {provider_ports[0]}, {provider_ports[1]}). Ctrl+C stops all services.",
                flush=True,
            )
            while not stop_event.wait(0.2):
                network.check_running()
        return 0
    except InterruptedError:
        return 0
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
