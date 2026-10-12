"""Validated development settings; environment variables use the DIN_ prefix."""

import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_network.pairing import CoordinatorProfile, ProviderProfile, load_profile
from inference_network.schemas import Identifier, LocalURL, ProviderIdentifier

MLX_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MLX_MODEL_REVISION = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    request_timeout_seconds: float = Field(default=5, gt=0, le=120, allow_inf_nan=False)
    connect_timeout_seconds: float = Field(default=0.5, gt=0, le=10, allow_inf_nan=False)
    max_body_bytes: int = Field(default=65536, ge=1024, le=1048576)
    body_timeout_seconds: float = Field(default=5, gt=0, le=30, allow_inf_nan=False)
    lan_mode: bool = False
    credentials_file: Path | None = None

    @model_validator(mode="after")
    def explicit_lan_credentials(self) -> Self:
        if self.lan_mode != (self.credentials_file is not None):
            raise ValueError("LAN mode and a private credentials file must be enabled together")
        return self

    @classmethod
    def from_env(cls) -> Self:
        values = {
            field: os.environ[f"DIN_{field.upper()}"]
            for field in cls.model_fields
            if f"DIN_{field.upper()}" in os.environ
        }
        return cls.model_validate(values)


class CoordinatorSettings(Settings):
    strategy: Literal["round_robin", "capacity_aware"] = "round_robin"
    heartbeat_ttl_seconds: float = Field(default=3, gt=0, le=300, allow_inf_nan=False)
    max_providers: int = Field(default=64, ge=1, le=256)
    max_attempts: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def paired_coordinator(self) -> Self:
        if self.credentials_file is not None:
            profile = load_profile(self.credentials_file)
            if not isinstance(profile, CoordinatorProfile) or profile.coordinator_port != self.port:
                raise ValueError("Coordinator configuration does not match its pairing profile")
        return self


class ProviderSettings(Settings):
    backend: Literal["simulator", "mlx"] = "simulator"
    port: int = Field(default=8101, ge=1024, le=65535)
    provider_id: ProviderIdentifier = "provider-a"
    coordinator_url: LocalURL = "http://127.0.0.1:8000"
    endpoint: LocalURL = "http://127.0.0.1:8101"
    models: tuple[Identifier, ...] = ("simulated-echo-v1",)
    gpu_type: str = "SIMULATED NVIDIA GPU (no physical device)"
    vram_gb: float = Field(default=24, gt=0, le=1024, allow_inf_nan=False)
    max_concurrency: int = Field(default=2, ge=1, le=64)
    heartbeat_interval_seconds: float = Field(default=0.5, gt=0, le=60, allow_inf_nan=False)
    delay_seconds: float = Field(default=0.05, ge=0, le=60, allow_inf_nan=False)
    max_output_tokens: int = Field(default=128, ge=1, le=512)
    max_prompt_tokens: int = Field(default=2048, ge=1, le=4096)

    @model_validator(mode="before")
    @classmethod
    def backend_defaults(cls, values: dict[str, object]) -> dict[str, object]:
        values = dict(values)
        if values.get("backend") == "mlx":
            values.setdefault("models", (MLX_MODEL_ID,))
            values.setdefault("max_concurrency", 1)
            values.setdefault("delay_seconds", 0)
        return values

    @classmethod
    def from_env(cls) -> Self:
        values: dict[str, object] = {
            field: os.environ[f"DIN_{field.upper()}"]
            for field in cls.model_fields
            if f"DIN_{field.upper()}" in os.environ
        }
        if "models" in values:
            values["models"] = tuple(str(values["models"]).split(","))
        if "endpoint" not in values:
            values["endpoint"] = f"http://127.0.0.1:{values.get('port', 8101)}"
        return cls.model_validate(values)

    @model_validator(mode="after")
    def consistent_endpoint(self) -> Self:
        if self.lan_mode:
            assert self.credentials_file is not None
            profile = load_profile(self.credentials_file)
            if (
                not isinstance(profile, ProviderProfile)
                or profile.provider_id != self.provider_id
                or profile.endpoint != self.endpoint
            ):
                raise ValueError("Provider configuration does not match its pairing profile")
        elif self.endpoint != f"http://127.0.0.1:{self.port}":
            raise ValueError("Provider endpoint must match its local listening port")
        if self.backend == "mlx":
            if self.models != (MLX_MODEL_ID,):
                raise ValueError("MLX mode serves only the pinned MLX_MODEL_ID")
            if self.max_concurrency != 1:
                raise ValueError("MLX mode requires max_concurrency=1")
            if self.delay_seconds != 0:
                raise ValueError("Artificial delays are only supported by the simulator")
        return self
