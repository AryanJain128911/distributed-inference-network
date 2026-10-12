"""FastAPI assembly and HTTP routes; scheduling and transport live outside the API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from inference_network.config import CoordinatorSettings
from inference_network.errors import NetworkError, install_error_handlers
from inference_network.http_safety import BodyLimitMiddleware, run_while_connected
from inference_network.orchestration import Coordinator
from inference_network.pairing import (
    CoordinatorProfile,
    PairingMiddleware,
    ProviderAccess,
    load_profile,
)
from inference_network.registry import ProviderRegistry
from inference_network.scheduler import CapacityAware, RoundRobin
from inference_network.schemas import ChatCompletion, ChatRequest, Heartbeat, ProviderRegistration
from inference_network.streaming import EventStreamResponse
from inference_network.transport import ProviderTransport


def create_app(settings: CoordinatorSettings | None = None) -> FastAPI:
    settings = settings or CoordinatorSettings.from_env()
    profile = load_profile(settings.credentials_file) if settings.credentials_file else None
    if profile is not None and not isinstance(profile, CoordinatorProfile):
        raise ValueError("Coordinator requires a coordinator pairing profile")
    scheduler = RoundRobin() if settings.strategy == "round_robin" else CapacityAware()
    registry = ProviderRegistry(scheduler, settings.heartbeat_ttl_seconds, settings.max_providers)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            settings.request_timeout_seconds, connect=settings.connect_timeout_seconds
        )
        async with httpx.AsyncClient(
            timeout=timeout, trust_env=False, follow_redirects=False
        ) as client:
            app.state.coordinator = Coordinator(
                registry,
                ProviderTransport(
                    client,
                    provider_tokens=(
                        {access.endpoint: access.token for access in profile.providers}
                        if profile is not None
                        else None
                    ),
                ),
                settings.max_attempts,
                settings.request_timeout_seconds,
            )
            yield

    app = FastAPI(
        title="Independent Distributed Inference Network", version="0.1.4", lifespan=lifespan
    )
    app.state.registry = registry
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
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "coordinator",
            "simulation": all(provider["simulated"] for provider in registry.snapshot()),
            "strategy": settings.strategy,
        }

    @app.post("/providers/register")
    async def register(registration: ProviderRegistration, request: Request) -> dict[str, object]:
        if profile is not None:
            access: ProviderAccess = request.state.provider_access
            if (
                registration.provider_id != access.provider_id
                or registration.endpoint != access.endpoint
            ):
                raise NetworkError(
                    403, "provider_not_approved", "Provider identity or endpoint not approved"
                )
        registry.register(registration)
        return {"provider_id": registration.provider_id, "accepted": True}

    @app.post("/providers/{provider_id}/heartbeat")
    async def heartbeat(provider_id: str, update: Heartbeat) -> dict[str, bool]:
        registry.heartbeat(provider_id, update)
        return {"accepted": True}

    @app.get("/providers")
    async def providers() -> dict[str, object]:
        return {"providers": registry.snapshot()}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "local-network"}
                for model in registry.models()
            ],
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletion)
    async def complete(payload: ChatRequest, request: Request) -> ChatCompletion | Response:
        coordinator: Coordinator = app.state.coordinator
        if payload.stream:
            return EventStreamResponse(
                coordinator.stream(payload), settings.request_timeout_seconds
            )
        return await run_while_connected(coordinator.complete(payload), request)

    return app
