"""Inference contract and deterministic simulator; no GPU execution occurs here."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from inference_network.schemas import ChatRequest, TokenUsage


@dataclass(frozen=True)
class BackendResult:
    content: str
    is_simulated: bool
    usage: TokenUsage | None = None
    finish_reason: Literal["stop", "length"] = "stop"


@dataclass(frozen=True)
class BackendChunk:
    """Incremental text, or an empty terminal chunk after execution cleanup."""

    content: str = ""
    is_simulated: bool = False
    usage: TokenUsage | None = None
    finish_reason: Literal["stop", "length"] | None = None


class InferenceBackend(Protocol):
    """Implementations must cooperate with cancellation and release their resources."""

    async def generate(self, request: ChatRequest) -> BackendResult: ...


@runtime_checkable
class StreamingBackend(Protocol):
    """Closing the iterator must stop execution before returning capacity."""

    def stream(self, request: ChatRequest) -> AsyncGenerator[BackendChunk, None]: ...


class SimulatedEchoBackend:
    async def generate(self, request: ChatRequest) -> BackendResult:
        last_user_message = next(
            message.content for message in reversed(request.messages) if message.role == "user"
        )
        # Keep the complete marker even for the longest accepted input. Token counts
        # require a real tokenizer/model, so simulation reports no usage measurements.
        content = f"[SIMULATED {request.model}] {last_user_message}"[:8192]
        return BackendResult(content=content, is_simulated=True)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[BackendChunk, None]:
        # This deterministic split is simulation only, not model streaming evidence.
        result = await self.generate(request)
        for offset in range(0, len(result.content), 16):
            await asyncio.sleep(0)
            yield BackendChunk(content=result.content[offset : offset + 16], is_simulated=True)
        yield BackendChunk(is_simulated=True, finish_reason=result.finish_reason)
