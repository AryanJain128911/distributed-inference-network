"""Owned OpenSSH tunnels for explicitly selected private IPv4 LAN peers.

HTTP remains on loopback at both ends. SSH authenticates the remote machine and
the approved OS account; application headers authenticate the inference role.
"""

import json
import math
import os
import re
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx

from inference_network.pairing import validate_lan_host

_SSH_PATH = "/usr/bin/ssh"
_POLL_SECONDS = 0.05


class SshTunnelError(RuntimeError):
    """A sanitized tunnel failure suitable for the interactive terminal."""


def _is_lan_address(value: str) -> bool:
    try:
        validate_lan_host(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SshTunnelConfig:
    host: str
    user: str
    coordinator_port: int = 8000
    local_forward_port: int = 18000
    ssh_port: int = 22
    reverse_port: int | None = None
    provider_port: int | None = None
    known_hosts_file: Path | None = None
    startup_timeout_seconds: float = 10

    def __post_init__(self) -> None:
        if not _is_lan_address(self.host):
            raise ValueError("SSH host must be an explicit RFC1918 private IPv4 LAN address")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", self.user):
            raise ValueError("SSH user must be a valid account name")
        for name in (
            "coordinator_port",
            "local_forward_port",
            "ssh_port",
            "reverse_port",
            "provider_port",
        ):
            value = getattr(self, name)
            if value is None and name in {"reverse_port", "provider_port"}:
                continue
            minimum = 1 if name == "ssh_port" else 1024
            if type(value) is not int or not minimum <= value <= 65535:
                raise ValueError(f"{name} must be an integer from {minimum} to 65535")
        if (self.reverse_port is None) != (self.provider_port is None):
            raise ValueError("Provider and reverse ports must be supplied together")
        if self.reverse_port in {self.coordinator_port, self.ssh_port}:
            raise ValueError("Reverse port must differ from coordinator and SSH ports")
        if self.provider_port == self.local_forward_port:
            raise ValueError("Provider and local forward ports must differ")
        if not math.isfinite(self.startup_timeout_seconds) or not (
            0 < self.startup_timeout_seconds <= 120
        ):
            raise ValueError("SSH startup timeout must be greater than 0 and at most 120 seconds")
        if self.known_hosts_file is not None:
            path = str(self.known_hosts_file)
            # OpenSSH expands tokens and parses a whitespace-separated path list.
            # Reject those forms rather than accidentally trusting another file.
            if not self.known_hosts_file.is_absolute() or re.search(r'[\s%$"\\]', path):
                raise ValueError("Known-hosts file must be an absolute path without SSH expansions")
            if not self.known_hosts_file.is_file():
                raise ValueError("Known-hosts file must be an existing file")


def _ssh_environment() -> dict[str, str]:
    # In particular, exclude SSH_ASKPASS, SSH_SK_PROVIDER, LD_*, DYLD_*, and
    # proxy/config variables. The already approved local agent is never forwarded.
    environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C", "LC_ALL": "C"}
    for name in ("HOME", "SSH_AUTH_SOCK"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


class SshTunnel:
    """Own one foreground SSH process; never create keys or change OS settings."""

    def __init__(
        self, config: SshTunnelConfig, *, auth_headers: Mapping[str, str] | None = None
    ) -> None:
        self.config = config
        self.coordinator_url = f"http://127.0.0.1:{config.local_forward_port}"
        self._auth_headers = dict(auth_headers or {})
        self._process: subprocess.Popen[bytes] | None = None

    def command(self) -> list[str]:
        """Return the credential-free, shell-free OpenSSH argument vector."""
        config = self.config
        command = [_SSH_PATH, "-4", "-N", "-T", "-n", "-a", "-x", "-F", "none"]
        options = (
            "BatchMode=yes",
            "StrictHostKeyChecking=yes",
            "ExitOnForwardFailure=yes",
            "ConnectTimeout=5",
            "ConnectionAttempts=1",
            "ServerAliveInterval=5",
            "ServerAliveCountMax=2",
            "ForwardAgent=no",
            "ForwardX11=no",
            "PermitLocalCommand=no",
            "ProxyCommand=none",
            "ProxyJump=none",
            "ControlMaster=no",
            "ControlPath=none",
            "ControlPersist=no",
            "KnownHostsCommand=none",
            "UpdateHostKeys=no",
            "VerifyHostKeyDNS=no",
            "PasswordAuthentication=no",
            "KbdInteractiveAuthentication=no",
            "PreferredAuthentications=publickey",
            "AddKeysToAgent=no",
            "Tunnel=no",
            "LogLevel=ERROR",
        )
        for option in options:
            command.extend(("-o", option))
        if config.known_hosts_file is not None:
            command.extend(
                (
                    "-o",
                    f"UserKnownHostsFile={config.known_hosts_file}",
                    "-o",
                    "GlobalKnownHostsFile=none",
                )
            )
        command.extend(
            ("-L", f"127.0.0.1:{config.local_forward_port}:127.0.0.1:{config.coordinator_port}")
        )
        if config.reverse_port is not None:
            command.extend(
                ("-R", f"127.0.0.1:{config.reverse_port}:127.0.0.1:{config.provider_port}")
            )
        command.extend(("-p", str(config.ssh_port), "-l", config.user, config.host))
        return command

    def start(self) -> Self:
        if self._process is not None:
            raise SshTunnelError("SSH tunnel is already started")
        try:
            self._check_local_port()
            self._process = subprocess.Popen(
                self.command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_ssh_environment(),
                start_new_session=True,
            )
            self._wait_for_coordinator()
            self._check_reverse_is_private()
            self.check_alive()
            return self
        except OSError:
            self.close()
            raise SshTunnelError(
                "Could not start /usr/bin/ssh; verify OpenSSH is installed"
            ) from None
        except BaseException:
            self.close()
            raise

    def _check_local_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", self.config.local_forward_port))
            except OSError:
                raise SshTunnelError(
                    "Local SSH forward port is unavailable; choose another"
                ) from None

    def check_alive(self) -> None:
        if self._process is None or self._process.poll() is not None:
            raise SshTunnelError(
                "SSH tunnel is not running. Check the approved host key, key-based login, "
                "Remote Login permission, and forwarding ports; see the LAN setup guide."
            )

    def _wait_for_coordinator(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            while (remaining := deadline - time.monotonic()) > 0:
                self.check_alive()
                try:
                    with client.stream(
                        "GET",
                        f"{self.coordinator_url}/health",
                        headers={**self._auth_headers, "Accept-Encoding": "identity"},
                        timeout=min(0.5, remaining),
                    ) as response:
                        if response.status_code in {401, 403}:
                            raise SshTunnelError("Coordinator rejected the supplied pairing token")
                        if response.status_code == 200:
                            if response.headers.get("content-encoding", "identity") != "identity":
                                raise SshTunnelError("Compressed coordinator health is unsupported")
                            body = bytearray()
                            for chunk in response.iter_raw():
                                if time.monotonic() >= deadline:
                                    raise SshTunnelError("Coordinator health response timed out")
                                body.extend(chunk)
                                if len(body) > 4096:
                                    raise SshTunnelError(
                                        "Coordinator health response exceeded limit"
                                    )
                            try:
                                health = json.loads(body)
                            except (ValueError, UnicodeError):
                                raise SshTunnelError(
                                    "Invalid coordinator health response"
                                ) from None
                            if (
                                isinstance(health, dict)
                                and health.get("status") == "ok"
                                and (health.get("service") == "coordinator")
                            ):
                                self.check_alive()
                                return
                except httpx.HTTPError:
                    pass  # OpenSSH may still be authenticating or opening the forward.
                time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
        raise SshTunnelError("Timed out reaching the coordinator through SSH; check its local port")

    def _check_reverse_is_private(self) -> None:
        if self.config.reverse_port is None:
            return
        # GatewayPorts=yes on sshd overrides even an explicit loopback -R bind.
        # This guard catches reachability on the selected LAN interface. The owner
        # must also verify sshd's effective GatewayPorts=no configuration.
        try:
            connection = socket.create_connection(
                (self.config.host, self.config.reverse_port), timeout=0.5
            )
        except OSError:
            return
        connection.close()
        raise SshTunnelError(
            "Reverse SSH port is reachable on the LAN. Stop and verify the coordinator's "
            "effective sshd GatewayPorts=no setting before serving a provider."
        )

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        finally:
            self._process = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def discover_lan_addresses() -> tuple[str, ...]:
    """Read local interface addresses without DNS or any outbound socket probe."""
    commands = (
        ("/sbin/ifconfig", "-a"),
        ("/usr/sbin/ifconfig", "-a"),
        ("/sbin/ip", "-o", "-4", "addr", "show"),
        ("/usr/sbin/ip", "-o", "-4", "addr", "show"),
    )
    for command in commands:
        if not Path(command[0]).is_file():
            continue
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
                env=_ssh_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        candidates = re.findall(r"\binet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
        return tuple(sorted({value for value in candidates if _is_lan_address(value)}))
    return ()
