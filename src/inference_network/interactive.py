"""Terminal interaction; service execution and request routing remain in their owners."""

import asyncio
import getpass
import signal
import socket
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import FrameType
from urllib.parse import urlsplit

import httpx

from inference_network.client import stream_request
from inference_network.config import MLX_MODEL_ID, CoordinatorSettings, ProviderSettings
from inference_network.errors import NetworkError
from inference_network.launcher import ServiceProcess
from inference_network.model_setup import prepare_model
from inference_network.pairing import (
    CoordinatorProfile,
    ProviderProfile,
    SenderProfile,
    authorization_header,
    generate_profiles,
    load_profile,
)
from inference_network.schemas import ChatRequest, validate_local_url
from inference_network.ssh_tunnel import SshTunnel, SshTunnelConfig, discover_lan_addresses


class _StopInteractive(Exception):
    """SIGTERM exits the owner; Ctrl+C may cancel only an active request."""


def _ask(prompt: str, default: str = "") -> str:
    answer = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return answer or default


def _choice(prompt: str, options: tuple[str, ...]) -> int:
    print(f"\n{prompt}")
    for index, option in enumerate(options, 1):
        print(f"{index}. {option}")
    while True:
        selected = _ask("Selection")
        if selected.isdecimal() and 1 <= int(selected) <= len(options):
            return int(selected)
        print("Choose one of the listed numbers.")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _profile_path(path: Path | None) -> Path:
    return (path or Path(_ask("Private pairing profile path"))).expanduser().absolute()


def _terminal_text(text: str) -> str:
    return "".join(
        character for character in text if character.isprintable() or character in "\n\t"
    )


def _display_providers(client: httpx.Client) -> list[str]:
    response = client.get("/providers")
    response.raise_for_status()
    providers = response.json()["providers"]
    print("\nProviders:")
    for provider in providers:
        available = provider["online"] and provider["ready"] and provider["healthy"]
        state = "ready" if available else "offline / not ready"
        print(
            _terminal_text(
                f"  {provider['provider_id']}: {state}; "
                f"{provider['in_flight']}/{provider['max_concurrency']} active; "
                f"{', '.join(provider['models'])}; {provider['gpu_type']}"
            )
        )
    if not providers:
        print("  None yet. Start a provider, then enter /providers to refresh.")
    models = client.get("/v1/models")
    models.raise_for_status()
    return [item["id"] for item in models.json()["data"]]


async def _send_prompt(url: str, headers: dict[str, str], model: str, prompt: str) -> None:
    request = ChatRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
    )
    async with httpx.AsyncClient(
        base_url=url, headers=headers, timeout=65, trust_env=False, follow_redirects=False
    ) as client:
        result = await stream_request(
            client, request, lambda text: print(_terminal_text(text), end="", flush=True)
        )
    print(f"\n[provider={result.provider_id}; model={result.model}; simulated={result.simulated}]")
    if result.usage is not None:
        print(
            f"Tokens: prompt {result.usage.prompt_tokens}, "
            f"completion {result.usage.completion_tokens}, total {result.usage.total_tokens}"
        )


def _chat(url: str, headers: dict[str, str]) -> None:
    print(f"\nConnected: {url}")
    print(
        "Enter a prompt, /providers, /models, /model MODEL_ID, or /quit. Ctrl+C cancels a stream."
    )
    selected = MLX_MODEL_ID
    with httpx.Client(base_url=url, headers=headers, timeout=5, trust_env=False) as client:
        models = _display_providers(client)
        if models:
            selected = models[0]
        while True:
            prompt = input(f"\n{selected}> ").strip()
            if prompt == "/quit":
                return
            if prompt in {"/providers", "/models"}:
                models = _display_providers(client)
                print("Available models: " + (", ".join(models) or "none"))
                continue
            if prompt.startswith("/model "):
                selected = prompt.split(maxsplit=1)[1]
                continue
            if not prompt:
                continue
            try:
                asyncio.run(_send_prompt(url, headers, selected, prompt))
            except KeyboardInterrupt:
                print("\nRequest cancelled. Provider cleanup may finish after cancellation.")
            except NetworkError as error:
                print(f"\n{error.message}")
            except (httpx.HTTPError, ValueError, TimeoutError):
                print("\nStream failed or disconnected. Refresh /providers before retrying.")


def _create_network(lan: bool, profile_path: Path | None) -> None:
    headers: dict[str, str] = {}
    if not lan:
        settings = CoordinatorSettings(
            port=int(_ask("Coordinator port", "8000")), request_timeout_seconds=60
        )
    else:
        if profile_path is None:
            addresses = discover_lan_addresses()
            print("Local private IPv4 addresses: " + (", ".join(addresses) or "none detected"))
            host = _ask("Coordinator LAN IPv4 address", addresses[0] if addresses else "")
            user = _ask("Coordinator macOS SSH username", getpass.getuser())
            port = int(_ask("Coordinator port", "8000"))
            default = str(Path.home() / ".din" / datetime.now().strftime("network-%Y%m%d-%H%M%S"))
            directory = Path(_ask("New private pairing directory", default))
            paths = generate_profiles(
                directory,
                ssh_host=host,
                ssh_user=user,
                coordinator_port=port,
                providers={"studio": 18101, "air": 18102},
            )
            profile_path = paths["coordinator"]
            print("\nPairing files created (credentials are never printed):")
            for name, path in paths.items():
                print(f"  {name}: {path}")
            print(
                f"SSH endpoint: {user}@{host}:22. Only SSH is reachable over the LAN.\n"
                "Approve SSH keys and verify the host fingerprint using docs/setup.md.\n"
                "Transfer only the recipient's profile over verified SSH (or AirDrop in person).\n"
                "Keep coordinator.json private. Never forward router ports."
            )
        profile_path = profile_path.expanduser().absolute()
        profile = load_profile(profile_path)
        if not isinstance(profile, CoordinatorProfile):
            raise ValueError("Create Network requires the owner's coordinator profile")
        headers = authorization_header(profile.sender_token)
        settings = CoordinatorSettings(
            port=profile.coordinator_port,
            lan_mode=True,
            credentials_file=profile_path,
            request_timeout_seconds=60,
        )
    with ServiceProcess("coordinator", settings, headers):
        _chat(f"http://127.0.0.1:{settings.port}", headers)


def _connect(
    stack: ExitStack,
    profile: SenderProfile | ProviderProfile,
    local: bool,
    provider_port: int | None = None,
) -> str:
    if local:
        return f"http://127.0.0.1:{profile.coordinator_port}"
    reverse = urlsplit(profile.endpoint).port if isinstance(profile, ProviderProfile) else None
    tunnel = stack.enter_context(
        SshTunnel(
            SshTunnelConfig(
                host=profile.ssh_host,
                user=profile.ssh_user,
                coordinator_port=profile.coordinator_port,
                local_forward_port=_free_port(),
                reverse_port=reverse,
                provider_port=provider_port,
            ),
            auth_headers=authorization_header(profile.token),
        )
    )
    print("Authenticated SSH tunnel connected; host key verification is enabled.")
    return tunnel.coordinator_url


def _join_network(lan: bool, path: Path | None, local: bool) -> None:
    with ExitStack() as stack:
        if lan:
            profile = load_profile(_profile_path(path))
            if not isinstance(profile, SenderProfile):
                raise ValueError("Join Network requires a sender profile")
            url = _connect(stack, profile, local)
            headers = authorization_header(profile.token)
        else:
            url = validate_local_url(_ask("Coordinator URL", "http://127.0.0.1:8000"))
            headers = {}
        _chat(url, headers)


def _provide(lan: bool, path: Path | None, local: bool) -> None:
    _choice("Select inference model:", ("Qwen2.5-0.5B-Instruct-4bit",))
    prepare_model()
    headers: dict[str, str] = {}
    with ExitStack() as stack:
        if lan:
            path = _profile_path(path)
            profile = load_profile(path)
            if not isinstance(profile, ProviderProfile):
                raise ValueError("Provider role requires its approved provider profile")
            advertised_port = urlsplit(profile.endpoint).port
            assert advertised_port is not None
            port = advertised_port if local else int(_ask("Provider local port", "8101"))
            url = _connect(stack, profile, local, None if local else port)
            headers = authorization_header(profile.token)
            settings = ProviderSettings(
                backend="mlx",
                provider_id=profile.provider_id,
                port=port,
                endpoint=profile.endpoint,
                coordinator_url=url,
                lan_mode=True,
                credentials_file=path,
                request_timeout_seconds=60,
            )
        else:
            port = int(_ask("Provider local port", "8101"))
            settings = ProviderSettings(
                backend="mlx",
                provider_id=_ask("Provider identity", "provider-mlx"),
                port=port,
                endpoint=f"http://127.0.0.1:{port}",
                coordinator_url=validate_local_url(
                    _ask("Coordinator URL", "http://127.0.0.1:8000")
                ),
                request_timeout_seconds=60,
            )
        print("Loading the pinned model on this Mac; waiting for readiness…")
        service = stack.enter_context(ServiceProcess("provider", settings, headers))
        print(f"Provider serving at {service.url}. Ctrl+C stops serving and unloads the model.")
        with httpx.Client(
            base_url=service.url, headers=headers, timeout=2, trust_env=False
        ) as client:
            previous: tuple[bool, bool, int] | None = None
            while True:
                service.check_running()
                response = client.get("/health")
                response.raise_for_status()
                health = response.json()
                state = (health["connected"], health["ready"], health["active_requests"])
                if state != previous:
                    print(
                        f"Coordinator connected={state[0]}; model ready={state[1]}; "
                        f"active requests={state[2]}"
                    )
                    previous = state
                time.sleep(0.25)


def run_interactive(lan: bool = False, profile: Path | None = None, local: bool = False) -> int:
    def stop(signum: int, frame: FrameType | None) -> None:
        raise _StopInteractive

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        print("Distributed Inference Network (DIN)")
        print("SSH LAN mode (explicit opt-in)" if lan else "Localhost mode (default)")
        role = _choice("Choose your role:", ("Sender", "Provider"))
        if role == 2:
            _provide(lan, profile, local)
        elif _choice("Sender:", ("Create a network", "Join an existing network")) == 1:
            _create_network(lan, profile)
        else:
            _join_network(lan, profile, local)
        return 0
    except (EOFError, KeyboardInterrupt, _StopInteractive):
        print("\nStopped; owned services and tunnels have been closed.")
        return 0
    except httpx.HTTPError:
        print("Connection failed. Check pairing, coordinator status, and the setup guide.")
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
