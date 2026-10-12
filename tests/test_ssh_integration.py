"""Opt-in real OpenSSH checks on one host, never evidence of two physical Macs.

Run: DIN_SSH_INTEGRATION=1 .venv/bin/pytest -q tests/test_ssh_integration.py
Creates only temporary synthetic keys, an isolated agent and a user-space sshd
on a private interface/high port. No system SSH configuration is changed.
"""

import getpass
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from inference_network.config import MLX_MODEL_ID, CoordinatorSettings, ProviderSettings
from inference_network.launcher import ServiceProcess
from inference_network.pairing import authorization_header, generate_profiles, load_profile
from inference_network.ssh_tunnel import (
    SshTunnel,
    SshTunnelConfig,
    SshTunnelError,
    discover_lan_addresses,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("DIN_SSH_INTEGRATION") != "1",
        reason="set DIN_SSH_INTEGRATION=1 to run an isolated local SSH daemon",
    ),
]
_AUTHORIZATION = "Bearer synthetic-ssh-test-token"


def _stop(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _free_ports(count):
    with ExitStack() as stack:
        listeners = [stack.enter_context(socket.socket()) for _ in range(count)]
        for listener in listeners:
            listener.bind(("127.0.0.1", 0))
        return [listener.getsockname()[1] for listener in listeners]


@pytest.fixture
def ssh_lab(tmp_path, monkeypatch, request):
    for program in ("ssh", "ssh-agent", "ssh-add", "ssh-keygen"):
        if not Path(f"/usr/bin/{program}").is_file():
            pytest.skip(f"OpenSSH {program} is unavailable")
    if not Path("/usr/sbin/sshd").is_file():
        pytest.skip("An installed OpenSSH server is required")
    addresses = discover_lan_addresses()
    if not addresses:
        pytest.skip("A private IPv4 local interface is required")
    host = addresses[0]
    ssh_port, forward_port, reverse_port = _free_ports(3)
    requests = []

    class CoordinatorHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != _AUTHORIZATION:
                self.send_error(401)
                return
            requests.append(self.path)
            body = json.dumps({"status": "ok", "service": "coordinator"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ExitStack() as stack:
        coordinator = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CoordinatorHandler)
        stack.callback(coordinator.server_close)
        threading.Thread(target=coordinator.serve_forever, daemon=True).start()
        stack.callback(coordinator.shutdown)
        for name in ("host_key", "client_key", "other_host_key"):
            subprocess.run(
                [
                    "/usr/bin/ssh-keygen",
                    "-q",
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-f",
                    str(tmp_path / name),
                ],
                check=True,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        authorized_keys = tmp_path / "authorized_keys"
        authorized_keys.write_text((tmp_path / "client_key.pub").read_text())
        known_hosts = tmp_path / "known_hosts"
        known_hosts.write_text(f"[{host}]:{ssh_port} " + (tmp_path / "host_key.pub").read_text())
        gateway_ports = getattr(request, "param", "no")
        server_config = tmp_path / "sshd_config"
        server_config.write_text(
            f"Port {ssh_port}\nListenAddress {host}\n"
            f"HostKey {tmp_path / 'host_key'}\nPidFile {tmp_path / 'pid'}\n"
            f"AuthorizedKeysFile {authorized_keys}\nAllowUsers {getpass.getuser()}\n"
            "PasswordAuthentication no\nKbdInteractiveAuthentication no\nUsePAM no\n"
            "AllowTcpForwarding yes\nAllowAgentForwarding no\nX11Forwarding no\n"
            "PermitTTY no\nForceCommand /usr/bin/false\n"
            f"GatewayPorts {gateway_ports}\nLogLevel ERROR\n"
        )
        validation = subprocess.run(
            ["/usr/sbin/sshd", "-t", "-f", str(server_config)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert validation.returncode == 0, validation.stderr
        daemon_log = stack.enter_context((tmp_path / "sshd.log").open("w+"))
        daemon = subprocess.Popen(
            ["/usr/sbin/sshd", "-D", "-e", "-f", str(server_config)],
            stdin=subprocess.DEVNULL,
            stdout=daemon_log,
            stderr=daemon_log,
        )
        stack.callback(_stop, daemon)
        # macOS limits Unix-domain socket paths to 104 bytes; pytest's full
        # temporary test name can exceed that before the socket filename.
        agent_directory = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="din-ssh-", dir="/tmp")
        )
        agent_socket = Path(agent_directory) / "agent.sock"
        agent = subprocess.Popen(
            ["/usr/bin/ssh-agent", "-D", "-a", str(agent_socket)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        stack.callback(_stop, agent)
        deadline = time.monotonic() + 5
        while not agent_socket.exists() and time.monotonic() < deadline:
            assert agent.poll() is None, "Temporary SSH agent exited before readiness"
            time.sleep(0.02)
        assert agent_socket.exists(), "Temporary SSH agent startup timed out"
        subprocess.run(
            ["/usr/bin/ssh-add", str(tmp_path / "client_key")],
            env={"PATH": "/usr/bin:/bin", "SSH_AUTH_SOCK": str(agent_socket)},
            check=True,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        monkeypatch.setenv("SSH_AUTH_SOCK", str(agent_socket))
        while time.monotonic() < deadline:
            assert daemon.poll() is None, "Temporary sshd exited before readiness"
            try:
                with socket.create_connection((host, ssh_port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            pytest.fail("Temporary SSH daemon startup timed out")
        yield SimpleNamespace(
            config=SshTunnelConfig(
                host=host,
                user=getpass.getuser(),
                ssh_port=ssh_port,
                coordinator_port=coordinator.server_port,
                local_forward_port=forward_port,
                reverse_port=reverse_port,
                provider_port=coordinator.server_port,
                known_hosts_file=known_hosts,
                startup_timeout_seconds=5,
            ),
            root=tmp_path,
            requests=requests,
            known_hosts=known_hosts,
            authorized_keys=authorized_keys,
        )


def test_real_ssh_authentication_forward_reverse_and_cleanup(ssh_lab):
    tunnel = SshTunnel(ssh_lab.config, auth_headers={"Authorization": _AUTHORIZATION})
    with tunnel:
        with httpx.Client(trust_env=False, timeout=2) as client:
            response = client.get(
                f"http://127.0.0.1:{ssh_lab.config.reverse_port}/health",
                headers={"Authorization": _AUTHORIZATION},
            )
        assert response.json() == {"status": "ok", "service": "coordinator"}
        assert ssh_lab.requests == ["/health", "/health"]
        with pytest.raises(OSError):
            socket.create_connection(
                (ssh_lab.config.host, ssh_lab.config.reverse_port), timeout=0.2
            )
    with pytest.raises(SshTunnelError, match="not running"):
        tunnel.check_alive()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", ssh_lab.config.local_forward_port), timeout=0.2)


@pytest.mark.parametrize("host_key", ["unknown", "changed"])
def test_real_ssh_rejects_untrusted_coordinator_before_http(ssh_lab, host_key):
    content = ""
    if host_key == "changed":
        content = (
            f"[{ssh_lab.config.host}]:{ssh_lab.config.ssh_port} "
            + (ssh_lab.root / "other_host_key.pub").read_text()
        )
    ssh_lab.known_hosts.write_text(content)
    with pytest.raises(SshTunnelError, match="not running"):
        SshTunnel(ssh_lab.config, auth_headers={"Authorization": _AUTHORIZATION}).start()
    assert ssh_lab.requests == []
    assert ssh_lab.known_hosts.read_text() == content


def test_real_ssh_rejects_unapproved_client_before_http(ssh_lab):
    ssh_lab.authorized_keys.write_text("")
    with pytest.raises(SshTunnelError, match="not running"):
        SshTunnel(ssh_lab.config, auth_headers={"Authorization": _AUTHORIZATION}).start()
    assert ssh_lab.requests == []


@pytest.mark.parametrize("ssh_lab", ["yes"], indirect=True)
def test_real_sshd_wildcard_reverse_policy_fails_closed(ssh_lab):
    with pytest.raises(SshTunnelError, match="GatewayPorts=no"):
        SshTunnel(ssh_lab.config, auth_headers={"Authorization": _AUTHORIZATION}).start()
    with pytest.raises(OSError):
        socket.create_connection((ssh_lab.config.host, ssh_lab.config.reverse_port), timeout=0.2)


def test_real_ssh_reverse_port_collision_fails_startup(ssh_lab):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", ssh_lab.config.reverse_port))
        occupied.listen()
        with pytest.raises(SshTunnelError, match="not running"):
            SshTunnel(ssh_lab.config, auth_headers={"Authorization": _AUTHORIZATION}).start()


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get("DIN_RUN_MLX_TESTS") != "1", reason="requires cached real MLX")
def test_real_model_registration_and_stream_through_actual_ssh(ssh_lab, tmp_path):
    """Exercise both encrypted hops and actual Qwen, still on ONE physical Mac."""
    coordinator_port, provider_port = _free_ports(2)
    paths = generate_profiles(
        tmp_path / "profiles",
        ssh_host=ssh_lab.config.host,
        ssh_user=ssh_lab.config.user,
        coordinator_port=coordinator_port,
        providers={"ssh-mlx": ssh_lab.config.reverse_port},
    )
    sender = load_profile(paths["sender"])
    provider = load_profile(paths["provider-ssh-mlx"])
    sender_headers = authorization_header(sender.token)
    provider_headers = authorization_header(provider.token)
    coordinator_settings = CoordinatorSettings(
        port=coordinator_port,
        lan_mode=True,
        credentials_file=paths["coordinator"],
        request_timeout_seconds=60,
    )
    config = replace(ssh_lab.config, coordinator_port=coordinator_port, provider_port=provider_port)
    with ExitStack() as stack:
        coordinator = stack.enter_context(
            ServiceProcess("coordinator", coordinator_settings, sender_headers)
        )
        tunnel = stack.enter_context(SshTunnel(config, auth_headers=provider_headers))
        settings = ProviderSettings(
            backend="mlx",
            port=provider_port,
            provider_id="ssh-mlx",
            endpoint=provider.endpoint,
            coordinator_url=tunnel.coordinator_url,
            lan_mode=True,
            credentials_file=paths["provider-ssh-mlx"],
            request_timeout_seconds=60,
        )
        stack.enter_context(ServiceProcess("provider", settings, provider_headers))
        client = stack.enter_context(
            httpx.Client(
                base_url=coordinator.url,
                headers=sender_headers,
                trust_env=False,
                timeout=5,
            )
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            rows = client.get("/providers").json()["providers"]
            if rows and rows[0]["online"] and rows[0]["ready"]:
                break
            time.sleep(0.05)
        else:
            pytest.fail("Provider did not register through its SSH forward")
        report_path = tmp_path / "local-ssh-mlx-smoke.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.lan_smoke",
                "--coordinator-url",
                coordinator.url,
                "--credentials-file",
                str(paths["sender"]),
                "--expected-provider",
                "ssh-mlx",
                "--output",
                str(report_path),
                "--host-label",
                "Single Apple Mac with actual OpenSSH",
                "--network-note",
                "Same physical host; authenticated SSH forward/reverse with real MLX",
            ],
            capture_output=True,
            text=True,
            timeout=70,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(report_path.read_text())
        assert report["unauthenticated_access_rejected"] is True
        assert report["validation_problems"] == []
        observation = report["observations"][0]
        assert observation["model"] == MLX_MODEL_ID
        assert observation["provider_id"] == "ssh-mlx"
        assert observation["simulated"] is False
        assert observation["completion_tokens"] > 0
        assert observation["done_received"] is True
        assert report["providers_after"][0]["in_flight"] == 0
        print(f"Single-host SSH and real MLX evidence: {report_path}")
