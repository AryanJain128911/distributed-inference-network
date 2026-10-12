"""Bounded wire contracts shared by coordinator and providers."""

from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._/-]+$")]


def validate_provider_identifier(value: str) -> str:
    if value in {".", ".."}:
        raise ValueError("Provider identifiers cannot be URL dot segments")
    return value


ProviderIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._-]+$"),
    AfterValidator(validate_provider_identifier),
]


def validate_local_url(value: str) -> str:
    """Numeric loopback only: no DNS, redirects, URL credentials, or arbitrary paths."""
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise ValueError("Endpoint cannot contain whitespace or control characters")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or not 1024 <= parsed.port <= 65535
    ):
        raise ValueError("Use an http://127.0.0.1:PORT origin with port 1024–65535")
    return f"http://127.0.0.1:{parsed.port}"


LocalURL = Annotated[str, AfterValidator(validate_local_url)]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(WireModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8192)


class StreamOptions(WireModel):
    include_usage: bool = False


class ChatRequest(WireModel):
    model: Identifier
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=32)
    stream: bool = False
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def require_user_message(self) -> "ChatRequest":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("At least one user message is required")
        if self.stream_options is not None and not self.stream:
            raise ValueError("Stream options require streaming")
        return self


class TokenUsage(WireModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def consistent_total(self) -> "TokenUsage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("Total tokens must equal prompt plus completion tokens")
        return self


class Choice(WireModel):
    index: Literal[0] = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ResponseMetadata(WireModel):
    provider_id: ProviderIdentifier
    simulated: bool


class ChatCompletion(WireModel):
    id: str = Field(min_length=1, max_length=128)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(ge=0)
    model: Identifier
    choices: tuple[Choice, ...] = Field(min_length=1, max_length=1)
    usage: TokenUsage | None = None
    metadata: ResponseMetadata


class ChatCompletionDelta(WireModel):
    role: Literal["assistant"] | None = None
    content: str | None = Field(default=None, max_length=8192)


class ChunkChoice(WireModel):
    index: Literal[0] = 0
    delta: ChatCompletionDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(WireModel):
    id: str = Field(min_length=1, max_length=128)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(ge=0)
    model: Identifier
    choices: tuple[ChunkChoice, ...] = Field(max_length=1)
    usage: TokenUsage | None = None
    metadata: ResponseMetadata

    @model_validator(mode="after")
    def usage_has_no_choices(self) -> "ChatCompletionChunk":
        if self.choices and self.usage is not None:
            raise ValueError("Only usage events have empty choices")
        if not self.choices and self.usage is None and not self.metadata.simulated:
            raise ValueError("Real inference usage events require token usage")
        return self


class ProviderRegistration(WireModel):
    provider_id: ProviderIdentifier
    instance_id: UUID
    models: tuple[Identifier, ...] = Field(min_length=1, max_length=16)
    gpu_type: str = Field(min_length=1, max_length=128)
    vram_gb: float = Field(gt=0, le=1024, allow_inf_nan=False)
    max_concurrency: int = Field(ge=1, le=64)
    ready: bool = True
    endpoint: LocalURL
    simulated: bool = True
    streaming: bool = False
    memory_kind: Literal["simulated_vram", "unified_memory"] = "simulated_vram"

    @model_validator(mode="after")
    def unique_models(self) -> "ProviderRegistration":
        if len(set(self.models)) != len(self.models):
            raise ValueError("Models must be unique")
        return self


class Heartbeat(WireModel):
    instance_id: UUID
    sequence: int = Field(ge=1)
    ready: bool


class SimulationControl(WireModel):
    mode: Literal["healthy", "error", "not_ready"] = "healthy"
    delay_seconds: float = Field(default=0, ge=0, le=60, allow_inf_nan=False)
