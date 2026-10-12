"""Explicit SSH-LAN trust profiles and application authorization.

HTTP remains loopback-only. SSH supplies transport authentication/encryption;
these independent bearer credentials authorize callers and approved provider origins.
"""

import ipaddress
import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
from starlette.types import ASGIApp, Receive, Scope, Send

from inference_network.errors import error_response
from inference_network.schemas import LocalURL, ProviderIdentifier


def validate_lan_host(value: str) -> str:
    """Use a literal RFC1918 IPv4 address, without DNS or public-address fallbacks."""
    address = ipaddress.IPv4Address(value)
    if not any(
        address in network
        for network in (
            ipaddress.IPv4Network("10.0.0.0/8"),
            ipaddress.IPv4Network("172.16.0.0/12"),
            ipaddress.IPv4Network("192.168.0.0/16"),
        )
    ):
        raise ValueError("SSH host must be a private LAN IPv4 address")
    return str(address)


def _validate_token(value: SecretStr) -> SecretStr:
    if re.fullmatch(r"[A-Za-z0-9_-]{43,128}", value.get_secret_value()) is None:
        raise ValueError("Pairing credentials must be securely generated URL-safe tokens")
    return value


Token = Annotated[SecretStr, AfterValidator(_validate_token)]
LANHost = Annotated[str, AfterValidator(validate_lan_host)]
Port = Annotated[int, Field(ge=1024, le=65535)]


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ProviderAccess(Profile):
    provider_id: ProviderIdentifier
    endpoint: LocalURL
    token: Token


class CoordinatorProfile(Profile):
    role: Literal["coordinator"] = "coordinator"
    coordinator_port: Port = 8000
    sender_token: Token
    providers: tuple[ProviderAccess, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_permissions(self) -> Self:
        identities = [access.provider_id for access in self.providers]
        endpoints = [access.endpoint for access in self.providers]
        tokens = [self.sender_token.get_secret_value()] + [
            access.token.get_secret_value() for access in self.providers
        ]
        if any(len(values) != len(set(values)) for values in (identities, endpoints, tokens)):
            raise ValueError("Provider identities, endpoints and all tokens must be distinct")
        if f"http://127.0.0.1:{self.coordinator_port}" in endpoints:
            raise ValueError("Provider endpoint cannot use the coordinator port")
        return self


class SenderProfile(Profile):
    role: Literal["sender"] = "sender"
    token: Token
    coordinator_port: Port = 8000
    ssh_host: LANHost
    ssh_user: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class ProviderProfile(Profile):
    role: Literal["provider"] = "provider"
    token: Token
    coordinator_port: Port = 8000
    ssh_host: LANHost
    ssh_user: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    provider_id: ProviderIdentifier
    endpoint: LocalURL

    @model_validator(mode="after")
    def separate_ports(self) -> Self:
        if self.endpoint == f"http://127.0.0.1:{self.coordinator_port}":
            raise ValueError("Provider endpoint cannot use the coordinator port")
        return self


TrustProfile = CoordinatorProfile | SenderProfile | ProviderProfile
_PROFILE_TYPES: dict[
    str, type[CoordinatorProfile] | type[SenderProfile] | type[ProviderProfile]
] = {
    "coordinator": CoordinatorProfile,
    "sender": SenderProfile,
    "provider": ProviderProfile,
}


def load_profile(path: Path) -> TrustProfile:
    """Read an owner-only regular file; never include secret input in errors."""
    try:
        descriptor = os.open(path.expanduser(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_size > 65536
            ):
                raise ValueError("Pairing file must be owner-only, regular and at most 64 KiB")
            data = json.loads(source.read(65537))
            if not isinstance(data, dict) or not isinstance(data.get("role"), str):
                raise ValueError("Pairing profile requires an explicit role")
            profile_type = _PROFILE_TYPES.get(data["role"])
            if profile_type is None:
                raise ValueError("Pairing profile role is unsupported")
            return profile_type.model_validate(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        raise ValueError(
            "Cannot load private pairing profile; check its format and permissions"
        ) from None


def save_profile(path: Path, profile: TrustProfile) -> None:
    """Create one new 0600 profile in an existing private directory, without overwrite."""
    path = path.expanduser()
    directory = path.parent.stat()
    if directory.st_uid != os.getuid() or stat.S_IMODE(directory.st_mode) & 0o077:
        raise ValueError("Pairing directory must be owned by you with mode 0700")
    payload = profile.model_dump(mode="python")

    # SecretStr stays redacted everywhere except this explicit private-file boundary.
    def reveal(value: object) -> object:
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        raise TypeError("Unsupported pairing profile value")

    encoded = json.dumps(payload, default=reveal, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(encoded)


def generate_profiles(
    directory: Path,
    *,
    ssh_host: str,
    ssh_user: str,
    coordinator_port: int = 8000,
    providers: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Create the owner's profile and separately shareable sender/provider invitations."""
    approved = providers if providers is not None else {"provider-studio": 18101}
    sender = SenderProfile(
        token=SecretStr(secrets.token_urlsafe(32)),
        coordinator_port=coordinator_port,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
    )
    invitations = [
        ProviderProfile(
            token=SecretStr(secrets.token_urlsafe(32)),
            coordinator_port=coordinator_port,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            provider_id=provider_id,
            endpoint=f"http://127.0.0.1:{port}",
        )
        for provider_id, port in approved.items()
    ]
    coordinator = CoordinatorProfile(
        coordinator_port=coordinator_port,
        sender_token=sender.token,
        providers=tuple(
            ProviderAccess(provider_id=item.provider_id, endpoint=item.endpoint, token=item.token)
            for item in invitations
        ),
    )
    directory = directory.expanduser()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    profiles: dict[str, TrustProfile] = {"coordinator": coordinator, "sender": sender}
    profiles.update({f"provider-{item.provider_id}": item for item in invitations})
    paths: dict[str, Path] = {}
    for name, profile in profiles.items():
        paths[name] = directory / f"{name}.json"
        save_profile(paths[name], profile)
    return paths


def authorization_header(token: SecretStr) -> dict[str, str]:
    return {"Authorization": f"Bearer {token.get_secret_value()}"}


class PairingMiddleware:
    """Authenticate before buffering request bodies, then enforce each role's routes."""

    def __init__(self, app: ASGIApp, profile: CoordinatorProfile | ProviderProfile) -> None:
        self.app = app
        self.profile = profile

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        values = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
        received = values[0] if len(values) == 1 else b""

        def matches(token: SecretStr) -> bool:
            expected = b"Bearer " + token.get_secret_value().encode("ascii")
            return secrets.compare_digest(received, expected)

        method, path = scope["method"], scope["path"]
        authenticated = False
        authorized = False
        if isinstance(self.profile, ProviderProfile):
            authenticated = authorized = matches(self.profile.token)
        elif matches(self.profile.sender_token):
            authenticated = True
            authorized = (method, path) in {
                ("GET", "/health"),
                ("GET", "/providers"),
                ("GET", "/v1/models"),
                ("POST", "/v1/chat/completions"),
            }
        else:
            for access in self.profile.providers:
                if matches(access.token):
                    authenticated = True
                    authorized = (method, path) in {
                        ("GET", "/health"),
                        ("POST", "/providers/register"),
                        ("POST", f"/providers/{access.provider_id}/heartbeat"),
                    }
                    scope.setdefault("state", {})["provider_access"] = access
                    break
        if not authenticated:
            await error_response(
                401, "authentication_required", "Valid pairing credentials required"
            )(scope, receive, send)
            return
        if not authorized:
            await error_response(403, "access_denied", "Pairing does not authorize this operation")(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)
