"""Standalone provider API, bounded admission, and coordinator presence lifecycle."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, Request
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from inference_network.backend import (
    BackendChunk,
    BackendResult,
    InferenceBackend,
    SimulatedEchoBackend,
    StreamingBackend,
)
from inference_network.config import ProviderSettings
from inference_network.errors import NetworkError, install_error_handlers
from inference_network.http_safety import BodyLimitMiddleware, run_while_connected
from inference_network.mlx_backend import MLXBackend
from inference_network.pairing import (
    PairingMiddleware,
    ProviderProfile,
    authorization_header,
    load_profile,
)
from inference_network.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatMessage,
    ChatRequest,
    Choice,
    ChunkChoice,
    Heartbeat,
    ProviderRegistration,
    ResponseMetadata,
    SimulationControl,
)
from inference_network.streaming import EventStreamResponse

logger = logging.getLogger(__name__)
SHUTDOWN_HEARTBEAT_TIMEOUT_SECONDS = 1.0


class ProviderRuntime:
    """One process owns its identity, backend readiness, and local execution count."""

    def __init__(self, settings: ProviderSettings, backend: InferenceBackend) -> None:
        self.settings = settings
        self.backend = backend
        self.instance_id = uuid4()
        self.sequence = 0
        self.active_requests = 0
        self.shutting_down = False
        self.backend_ready = settings.backend == "simulator"
        self.simulation = SimulationControl(delay_seconds=settings.delay_seconds)
        self.registered = False
        self.connected = False
        self._registration: ProviderRegistration | None = None
        if self.backend_ready:
            self._registration = self._build_registration(settings.gpu_type, settings.vram_gb)

    def _build_registration(self, device_name: str, memory_gb: float) -> ProviderRegistration:
        # Preserve the registration payload across lost acknowledgements. Current
        # readiness belongs to sequenced heartbeats, not registration retries.
        simulated = self.settings.backend == "simulator"
        return ProviderRegistration(
            provider_id=self.settings.provider_id,
            instance_id=self.instance_id,
            models=self.settings.models,
            gpu_type=device_name,
            vram_gb=memory_gb,
            memory_kind="simulated_vram" if simulated else "unified_memory",
            max_concurrency=self.settings.max_concurrency,
            ready=self.ready,
            endpoint=self.settings.endpoint,
            simulated=simulated,
            streaming=isinstance(self.backend, StreamingBackend),
        )

    def mark_model_loaded(self, device_name: str, memory_gb: float) -> None:
        self.backend_ready = True
        self._registration = self._build_registration(device_name, memory_gb)

    @property
    def ready(self) -> bool:
        return (
            self.backend_ready
            and not self.shutting_down
            and (self.settings.backend != "simulator" or self.simulation.mode != "not_ready")
        )

    def registration(self) -> ProviderRegistration:
        if self._registration is None:
            raise RuntimeError("Model must be loaded before provider registration")
        return self._registration

    def heartbeat(self) -> Heartbeat:
        self.sequence += 1
        return Heartbeat(instance_id=self.instance_id, sequence=self.sequence, ready=self.ready)

    def admit(self, model: str) -> None:
        # No await occurs between checking and reserving: atomic within this app's
        # single event loop. The provider never queues excess or abandoned work.
        if model not in self.settings.models:
            raise NetworkError(404, "unsupported_model", "Provider does not serve this model")
        if not self.ready:
            raise NetworkError(503, "provider_not_ready", "Provider is not ready")
        if self.active_requests >= self.settings.max_concurrency:
            raise NetworkError(429, "provider_capacity", "Provider is at capacity")
        self.active_requests += 1

    async def generate(self, request: ChatRequest) -> BackendResult:
        await self._prepare_generation(self.simulation)
        return await self.backend.generate(request)

    async def _prepare_generation(self, simulation: SimulationControl) -> None:
        if self.settings.backend == "simulator":
            if simulation.delay_seconds:
                await asyncio.sleep(simulation.delay_seconds)
            if simulation.mode == "error":
                raise NetworkError(503, "simulated_provider_error", "Controlled simulated failure")

    async def stream(self, request: ChatRequest) -> AsyncGenerator[BackendChunk, None]:
        if not isinstance(self.backend, StreamingBackend):
            raise NetworkError(422, "streaming_unsupported", "Backend does not support streaming")
        simulation = self.simulation
        await self._prepare_generation(simulation)
        async with aclosing(self.backend.stream(request)) as chunks:
            async for chunk in chunks:
                yield chunk
                # Simulation pacing is explicit test behavior, never used by MLX.
                if self.settings.backend == "simulator" and simulation.delay_seconds:
                    await asyncio.sleep(simulation.delay_seconds)


async def stream_completion(
    runtime: ProviderRuntime, payload: ChatRequest, request_id: str
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Own provider admission until the backend iterator has stopped and closed."""
    runtime.admit(payload.model)
    created = int(time.time())
    metadata = ResponseMetadata(
        provider_id=runtime.settings.provider_id,
        simulated=runtime.settings.backend == "simulator",
    )

    def event(choice: ChunkChoice) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=request_id,
            created=created,
            model=payload.model,
            choices=(choice,),
            metadata=metadata,
        )

    try:
        terminal: BackendChunk | None = None
        async with aclosing(runtime.stream(payload)) as chunks:
            # Validate tokenized input and start actual execution before committing.
            first = await anext(chunks)
            yield event(ChunkChoice(delta=ChatCompletionDelta(role="assistant")))
            chunk = first
            while True:
                if chunk.is_simulated != metadata.simulated or terminal is not None:
                    raise NetworkError(500, "backend_error", "Invalid backend stream")
                if chunk.content:
                    yield event(ChunkChoice(delta=ChatCompletionDelta(content=chunk.content)))
                if chunk.finish_reason is not None:
                    terminal = chunk
                try:
                    chunk = await anext(chunks)
                except StopAsyncIteration:
                    break
        if terminal is None:
            raise NetworkError(500, "backend_error", "Backend stream ended without completion")
        yield event(ChunkChoice(delta=ChatCompletionDelta(), finish_reason=terminal.finish_reason))
        if payload.stream_options and payload.stream_options.include_usage:
            yield ChatCompletionChunk(
                id=request_id,
                created=created,
                model=payload.model,
                choices=(),
                usage=terminal.usage,
                metadata=metadata,
            )
    except NetworkError:
        raise
    except Exception as exc:
        logger.error(
            "provider_inference_failed",
            extra={"provider_id": runtime.settings.provider_id, "error_type": type(exc).__name__},
        )
        raise NetworkError(500, "backend_error", "Provider backend failed") from exc
    finally:
        runtime.active_requests -= 1


async def maintain_presence(runtime: ProviderRuntime, client: httpx.AsyncClient) -> None:
    """Reconnect periodically; each registration/heartbeat attempt has a deadline."""
    had_failure = False
    while True:
        try:
            async with asyncio.timeout(runtime.settings.request_timeout_seconds):
                is_registration = not runtime.registered
                if not is_registration:
                    response = await client.post(
                        f"/providers/{runtime.settings.provider_id}/heartbeat",
                        json=runtime.heartbeat().model_dump(mode="json"),
                    )
                    if response.status_code in (404, 409):
                        runtime.registered = False
                else:
                    response = await client.post(
                        "/providers/register", json=runtime.registration().model_dump(mode="json")
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("accepted") is not True:
                    raise ValueError("Coordinator did not acknowledge presence")
                if is_registration and payload.get("provider_id") != runtime.settings.provider_id:
                    raise ValueError("Coordinator acknowledged a different provider")
                runtime.registered = True
                runtime.connected = True
                if had_failure:
                    logger.info(
                        "provider_presence_restored",
                        extra={"provider_id": runtime.settings.provider_id},
                    )
                had_failure = False
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            runtime.connected = False
            if not had_failure:
                logger.warning(
                    "provider_presence_unavailable",
                    extra={
                        "provider_id": runtime.settings.provider_id,
                        "error_type": type(exc).__name__,
                    },
                )
            had_failure = True
        await asyncio.sleep(runtime.settings.heartbeat_interval_seconds)


async def _publish_shutdown(runtime: ProviderRuntime, client: httpx.AsyncClient) -> None:
    if not runtime.registered:
        return
    try:
        async with asyncio.timeout(SHUTDOWN_HEARTBEAT_TIMEOUT_SECONDS):
            response = await client.post(
                f"/providers/{runtime.settings.provider_id}/heartbeat",
                json=runtime.heartbeat().model_dump(mode="json"),
            )
            response.raise_for_status()
    except (httpx.HTTPError, TimeoutError):
        # The coordinator also expires heartbeats when this best-effort notice fails.
        logger.info(
            "provider_shutdown_notice_unavailable",
            extra={"provider_id": runtime.settings.provider_id},
        )


def create_app(
    settings: ProviderSettings | None = None, backend: InferenceBackend | None = None
) -> FastAPI:
    settings = settings or ProviderSettings.from_env()
    profile = load_profile(settings.credentials_file) if settings.credentials_file else None
    if profile is not None and not isinstance(profile, ProviderProfile):
        raise ValueError("Provider requires a provider pairing profile")
    managed_backend = (
        MLXBackend(settings) if backend is None and settings.backend == "mlx" else None
    )
    runtime = ProviderRuntime(settings, backend or managed_backend or SimulatedEchoBackend())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            if managed_backend is not None:
                try:
                    await managed_backend.load()
                    runtime.mark_model_loaded(
                        managed_backend.device_name, managed_backend.memory_gb
                    )
                except Exception as exc:
                    logger.error(
                        "provider_model_load_failed",
                        extra={
                            "provider_id": settings.provider_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise RuntimeError("MLX model loading failed; see docs/setup.md") from None
            async with httpx.AsyncClient(
                base_url=settings.coordinator_url,
                timeout=httpx.Timeout(
                    settings.request_timeout_seconds, connect=settings.connect_timeout_seconds
                ),
                follow_redirects=False,
                trust_env=False,
                headers=authorization_header(profile.token) if profile is not None else None,
            ) as client:
                presence = asyncio.create_task(maintain_presence(runtime, client))
                try:
                    yield
                finally:
                    runtime.shutting_down = True
                    presence.cancel()
                    await asyncio.gather(presence, return_exceptions=True)
                    await _publish_shutdown(runtime, client)
        finally:
            runtime.backend_ready = False
            runtime.connected = False
            if managed_backend is not None:
                await managed_backend.close()

    app = FastAPI(title="Local inference provider", lifespan=lifespan)
    app.state.runtime = runtime
    install_error_handlers(app)
    app.add_middleware(
        BodyLimitMiddleware,
        max_body_bytes=settings.max_body_bytes,
        body_timeout_seconds=settings.body_timeout_seconds,
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    if profile is not None:
        app.add_middleware(PairingMiddleware, profile=profile)

    @app.get("/health")
    async def health() -> dict[str, str | bool | int | tuple[str, ...]]:
        return {
            "status": "ok" if runtime.ready else "not_ready",
            "provider_id": settings.provider_id,
            "simulated": settings.backend == "simulator",
            "backend": settings.backend,
            "models": settings.models,
            "ready": runtime.ready,
            "active_requests": runtime.active_requests,
            "max_concurrency": settings.max_concurrency,
            "registered": runtime.registered,
            "connected": runtime.connected,
        }

    @app.put("/simulation", response_model=SimulationControl)
    async def configure_simulation(control: SimulationControl) -> SimulationControl:
        if settings.backend != "simulator":
            raise NetworkError(404, "simulation_unavailable", "Provider is running a real model")
        runtime.simulation = control
        return control

    @app.post("/internal/chat/completions", response_model=ChatCompletion)
    async def complete(
        payload: ChatRequest,
        request: Request,
        x_request_id: Annotated[
            str | None, Header(max_length=128, pattern=r"^[a-zA-Z0-9._-]+$")
        ] = None,
    ) -> ChatCompletion | Response:
        if payload.stream:
            return EventStreamResponse(
                stream_completion(runtime, payload, x_request_id or f"chatcmpl-{uuid4().hex}"),
                settings.request_timeout_seconds,
                timeout_code="provider_timeout",
            )
        runtime.admit(payload.model)
        try:
            async with asyncio.timeout(settings.request_timeout_seconds):
                result = await run_while_connected(runtime.generate(payload), request)
            return ChatCompletion(
                id=x_request_id or f"chatcmpl-{uuid4().hex}",
                created=int(time.time()),
                model=payload.model,
                choices=(
                    Choice(
                        message=ChatMessage(role="assistant", content=result.content),
                        finish_reason=result.finish_reason,
                    ),
                ),
                usage=result.usage,
                metadata=ResponseMetadata(
                    provider_id=settings.provider_id, simulated=result.is_simulated
                ),
            )
        except TimeoutError as exc:
            raise NetworkError(504, "provider_timeout", "Provider execution timed out") from exc
        except NetworkError:
            raise
        except Exception as exc:
            # Catch backend failures here, but log only their type to avoid exposing prompts.
            logger.error(
                "provider_inference_failed",
                extra={"provider_id": settings.provider_id, "error_type": type(exc).__name__},
            )
            raise NetworkError(500, "backend_error", "Provider backend failed") from exc
        finally:
            runtime.active_requests -= 1

    return app
