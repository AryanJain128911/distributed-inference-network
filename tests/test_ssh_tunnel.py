import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from inference_network import ssh_tunnel
from inference_network.ssh_tunnel import (
    SshTunnel,
    SshTunnelConfig,
    SshTunnelError,
    discover_lan_addresses,
)


class FakeProcess:
    def __init__(self, *, returncode=None, stubborn=False):
        self.returncode = returncode
        self.stubborn = stubborn
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("synthetic ssh", timeout)
        return self.returncode


@pytest.fixture
def config():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return SshTunnelConfig(host="192.168.1.40", user="owner", local_forward_port=port)


def patch_http(monkeypatch, handler):
    real_client = httpx.Client
    monkeypatch.setattr(
        ssh_tunnel.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )


def healthy_response(request):
    return httpx.Response(
        200,
        stream=httpx.ByteStream(json.dumps({"status": "ok", "service": "coordinator"}).encode()),
    )


@pytest.mark.parametrize(
    "host",
    [
        "8.8.8.8",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "192.0.2.1",
        "100.64.0.1",
        "169.254.1.2",
        "224.0.0.1",
        "coordinator.local",
        "192.168.1.2;id",
        "-oProxyCommand=id",
    ],
)
def test_only_literal_private_lan_targets_are_allowed(host):
    with pytest.raises(ValueError, match="private IPv4"):
        SshTunnelConfig(host=host, user="owner")


@pytest.mark.parametrize("host", ["10.0.0.1", "172.16.0.1", "172.31.255.254", "192.168.1.1"])
def test_private_lan_targets_are_accepted(host):
    assert SshTunnelConfig(host=host, user="owner").host == host


@pytest.mark.parametrize("user", ["-oProxyCommand=id", "owner@host", "owner\nroot", "", "a" * 65])
def test_ssh_account_cannot_change_command_arguments(user):
    with pytest.raises(ValueError, match="account name"):
        SshTunnelConfig(host="192.168.1.2", user=user)


@pytest.mark.parametrize(
    "changes",
    [
        {"local_forward_port": 0},
        {"local_forward_port": True},
        {"coordinator_port": 65536},
        {"ssh_port": 0},
        {"reverse_port": 18101},
        {"provider_port": 8101},
        {"reverse_port": 8000, "provider_port": 8101},
        {"reverse_port": 18101, "provider_port": 18000},
        {"startup_timeout_seconds": float("inf")},
        {"startup_timeout_seconds": 0},
    ],
)
def test_invalid_ports_and_unbounded_timeouts_are_rejected(changes):
    with pytest.raises(ValueError):
        SshTunnelConfig(host="192.168.1.2", user="owner", **changes)


def test_command_keeps_http_loopback_only_and_disables_external_configuration(tmp_path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("# synthetic pinned hosts\n")
    config = SshTunnelConfig(
        host="192.168.1.2",
        user="owner",
        reverse_port=18101,
        provider_port=8101,
        known_hosts_file=known_hosts,
    )
    tunnel = SshTunnel(config, auth_headers={"Authorization": "Bearer synthetic-secret"})
    command = tunnel.command()
    assert command[command.index("-L") + 1] == "127.0.0.1:18000:127.0.0.1:8000"
    assert command[command.index("-R") + 1] == "127.0.0.1:18101:127.0.0.1:8101"
    assert command[-3:] == ["-l", "owner", "192.168.1.2"]
    assert command[command.index("-F") + 1] == "none"
    assert {"-N", "-T", "-a", "-x", "-n"}.issubset(command)
    assert {
        "StrictHostKeyChecking=yes",
        "ExitOnForwardFailure=yes",
        "ForwardAgent=no",
        "BatchMode=yes",
        "ProxyCommand=none",
        "ProxyJump=none",
        "ControlMaster=no",
        "ControlPath=none",
        "KnownHostsCommand=none",
        "UpdateHostKeys=no",
        "PasswordAuthentication=no",
        "PermitLocalCommand=no",
        "GlobalKnownHostsFile=none",
    }.issubset(command)
    assert "synthetic-secret" not in repr(tunnel) + repr(config) + repr(command)


@pytest.mark.parametrize("name", ["two hosts", "expanded%h", "expanded${HOME}", 'quote"'])
def test_known_hosts_paths_cannot_expand_or_select_additional_files(tmp_path, name):
    path = tmp_path / name
    path.touch()
    with pytest.raises(ValueError, match="without SSH expansions"):
        SshTunnelConfig(host="192.168.1.2", user="owner", known_hosts_file=path)


def test_authenticated_health_and_owned_cleanup(monkeypatch, config):
    process = FakeProcess()
    launches = []

    def spawn(command, **kwargs):
        launches.append((command, kwargs))
        return process

    def health(request):
        assert request.url == f"http://127.0.0.1:{config.local_forward_port}/health"
        assert request.headers["Authorization"] == "Bearer synthetic-token"
        assert request.headers["Accept-Encoding"] == "identity"
        return healthy_response(request)

    monkeypatch.setenv("SSH_ASKPASS", "/untrusted/program")
    monkeypatch.setenv("SSH_SK_PROVIDER", "/untrusted/library")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/untrusted/library")
    monkeypatch.setattr(ssh_tunnel.subprocess, "Popen", spawn)
    patch_http(monkeypatch, health)
    with SshTunnel(config, auth_headers={"Authorization": "Bearer synthetic-token"}) as tunnel:
        tunnel.check_alive()
        assert not process.terminated
    assert process.terminated
    assert not process.killed
    assert len(launches) == 1
    assert (
        not {"SSH_ASKPASS", "SSH_SK_PROVIDER", "DYLD_INSERT_LIBRARIES"}
        & launches[0][1]["env"].keys()
    )
    assert launches[0][1]["stderr"] == subprocess.DEVNULL
    assert "synthetic-token" not in repr(launches)
    tunnel.close()


def test_live_ssh_exit_is_reported_without_subprocess_output(monkeypatch, config):
    process = FakeProcess(returncode=255)
    monkeypatch.setattr(ssh_tunnel.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(SshTunnelError, match="approved host key"):
        SshTunnel(config).start()


def test_busy_local_forward_fails_before_launch(monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        config = SshTunnelConfig(
            host="192.168.1.2",
            user="owner",
            local_forward_port=listener.getsockname()[1],
        )
        monkeypatch.setattr(
            ssh_tunnel.subprocess,
            "Popen",
            lambda *args, **kwargs: pytest.fail("must not spawn SSH when local port is occupied"),
        )
        with pytest.raises(SshTunnelError, match="port is unavailable"):
            SshTunnel(config).start()


def test_startup_timeout_kills_stubborn_process_with_bounded_wait(monkeypatch, config):
    process = FakeProcess(stubborn=True)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        ssh_tunnel,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    monkeypatch.setattr(ssh_tunnel.subprocess, "Popen", lambda *args, **kwargs: process)
    patch_http(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(SshTunnelError, match="Timed out"):
        SshTunnel(config).start()
    assert process.terminated and process.killed
    assert clock.now == pytest.approx(config.startup_timeout_seconds)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (lambda: httpx.Response(401), "pairing token"),
        (lambda: httpx.Response(200, stream=httpx.ByteStream(b"{")), "Invalid coordinator"),
        (lambda: httpx.Response(200, stream=httpx.ByteStream(b"x" * 4097)), "exceeded limit"),
        (lambda: httpx.Response(200, headers={"Content-Encoding": "gzip"}), "Compressed"),
    ],
)
def test_invalid_health_fails_closed_and_stops_ssh(monkeypatch, config, response, message):
    process = FakeProcess()
    monkeypatch.setattr(ssh_tunnel.subprocess, "Popen", lambda *args, **kwargs: process)
    patch_http(monkeypatch, lambda request: response())
    with pytest.raises(SshTunnelError, match=message):
        SshTunnel(config).start()
    assert process.terminated


def test_health_response_cannot_drip_beyond_startup_deadline(monkeypatch, config):
    process = FakeProcess()
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(ssh_tunnel, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(ssh_tunnel.subprocess, "Popen", lambda *args, **kwargs: process)

    class DrippingResponse(httpx.SyncByteStream):
        def __iter__(self):
            yield b"{"
            clock.now = config.startup_timeout_seconds + 1
            yield b"}"

    patch_http(monkeypatch, lambda request: httpx.Response(200, stream=DrippingResponse()))
    with pytest.raises(SshTunnelError, match="response timed out"):
        SshTunnel(config).start()
    assert process.terminated


def test_gateway_ports_exposure_closes_tunnel_before_provider_can_start(monkeypatch, config):
    process = FakeProcess()
    reverse = SshTunnelConfig(
        host=config.host,
        user=config.user,
        local_forward_port=config.local_forward_port,
        reverse_port=18101,
        provider_port=8101,
    )
    monkeypatch.setattr(ssh_tunnel.subprocess, "Popen", lambda *args, **kwargs: process)
    patch_http(monkeypatch, healthy_response)
    connections = []

    def exposed(address, timeout):
        connections.append((address, timeout))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(ssh_tunnel.socket, "create_connection", exposed)
    with pytest.raises(SshTunnelError, match="GatewayPorts=no"):
        SshTunnel(reverse).start()
    assert process.terminated
    assert connections == [((config.host, 18101), 0.5)]


def test_discovery_reads_local_interfaces_without_network_probes(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "/sbin/ifconfig")
    monkeypatch.setattr(
        ssh_tunnel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="inet 127.0.0.1\n inet 192.168.2.4 netmask\n inet 8.8.8.8\n"
            "inet 10.2.0.1 netmask\ninet 192.168.2.4 netmask\ninet6 ::1\n",
        ),
    )
    monkeypatch.setattr(
        ssh_tunnel.socket,
        "socket",
        lambda *args: pytest.fail("no network discovery probe"),
    )
    assert discover_lan_addresses() == ("10.2.0.1", "192.168.2.4")


def test_installed_openssh_accepts_security_and_forward_options(config):
    if not Path("/usr/bin/ssh").is_file():
        pytest.skip("OpenSSH client is not installed")
    command = SshTunnel(config).command()
    command.insert(1, "-G")  # Configuration expansion only: never contacts a host.
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    settings = dict(line.split(" ", 1) for line in result.stdout.splitlines())
    assert settings["stricthostkeychecking"] in {"yes", "true"}
    assert settings["sessiontype"] == "none"
    assert settings["exitonforwardfailure"] == "yes"
    assert settings["forwardagent"] == "no"
    assert settings["proxyjump"] == "none" if "proxyjump" in settings else True
    assert settings["localforward"] == (f"[127.0.0.1]:{config.local_forward_port} [127.0.0.1]:8000")
