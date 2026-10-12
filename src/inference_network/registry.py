"""Coordinator-owned, single-event-loop provider state and atomic admission."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

from inference_network.errors import NetworkError
from inference_network.scheduler import Candidate, Scheduler
from inference_network.schemas import Heartbeat, ProviderRegistration


@dataclass
class _Provider:
    registration: ProviderRegistration
    last_seen: float
    ready: bool
    sequence: int = 0
    healthy: bool = True
    in_flight: int = 0
    completed_requests: int = 0
    failed_requests: int = 0


@dataclass(frozen=True)
class Reservation:
    token: UUID
    provider_id: str
    instance_id: UUID
    endpoint: str
    simulated: bool


class ProviderRegistry:
    """No method awaits: selection + increment are atomic on one asyncio loop.

    Call only from async handlers on that loop. Multiple coordinator workers are
    unsupported; a shared transactional admission store would be needed first.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        heartbeat_ttl_seconds: float,
        max_providers: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._scheduler = scheduler
        self._ttl = heartbeat_ttl_seconds
        self._limit = max_providers
        self._clock = clock
        self._providers: dict[str, _Provider] = {}
        self._reservations: dict[UUID, Reservation] = {}

    def register(self, registration: ProviderRegistration) -> None:
        existing = self._providers.get(registration.provider_id)
        if existing is not None:
            if existing.registration.instance_id == registration.instance_id:
                if existing.registration != registration:
                    raise NetworkError(
                        409, "registration_conflict", "Capabilities are immutable per boot"
                    )
                # An idempotent retry cannot renew heartbeats or reset reservations.
                return
            if self._fresh(existing) or existing.in_flight:
                raise NetworkError(409, "identity_conflict", "Provider identity is already active")
        if any(
            item.registration.endpoint == registration.endpoint and key != registration.provider_id
            for key, item in self._providers.items()
        ):
            raise NetworkError(409, "endpoint_conflict", "Endpoint already belongs to a provider")
        if existing is None and len(self._providers) >= self._limit:
            raise NetworkError(503, "registry_full", "Provider registry is full")
        self._providers[registration.provider_id] = _Provider(
            registration=registration, last_seen=self._clock(), ready=registration.ready
        )

    def heartbeat(self, provider_id: str, heartbeat: Heartbeat) -> None:
        provider = self._providers.get(provider_id)
        if provider is None:
            raise NetworkError(404, "unknown_provider", "Register before sending heartbeats")
        if heartbeat.instance_id != provider.registration.instance_id:
            raise NetworkError(
                409, "identity_conflict", "Heartbeat belongs to another provider boot"
            )
        if heartbeat.sequence <= provider.sequence:
            raise NetworkError(409, "stale_heartbeat", "Heartbeat sequence must increase")
        provider.sequence = heartbeat.sequence
        provider.last_seen = self._clock()
        provider.ready = heartbeat.ready
        provider.healthy = heartbeat.ready

    def reserve(
        self, model: str, excluded: set[str] | None = None, *, streaming: bool = False
    ) -> Reservation:
        if not any(model in item.registration.models for item in self._providers.values()):
            raise NetworkError(
                404, "model_not_found", "No registered provider advertises this model"
            )
        candidates = tuple(
            Candidate(key, item.in_flight, item.registration.max_concurrency)
            for key, item in self._providers.items()
            if key not in (excluded or set())
            and model in item.registration.models
            and (not streaming or item.registration.streaming)
            and self._available(item)
            and item.in_flight < item.registration.max_concurrency
        )
        if not candidates:
            raise NetworkError(
                503, "no_eligible_provider", "No healthy provider has available capacity"
            )
        selected = self._providers[self._scheduler.choose(model, candidates)]
        registration = selected.registration
        reservation = Reservation(
            uuid4(),
            registration.provider_id,
            registration.instance_id,
            registration.endpoint,
            registration.simulated,
        )
        selected.in_flight += 1
        self._reservations[reservation.token] = reservation
        return reservation

    def release(self, reservation: Reservation, *, succeeded: bool) -> None:
        if self._reservations.pop(reservation.token, None) != reservation:
            raise RuntimeError("Reservation was already released or was never acquired")
        provider = self._providers[reservation.provider_id]
        provider.in_flight -= 1
        if succeeded:
            provider.completed_requests += 1
        else:
            provider.failed_requests += 1

    def mark_unhealthy(self, reservation: Reservation) -> None:
        provider = self._providers[reservation.provider_id]
        if provider.registration.instance_id == reservation.instance_id:
            provider.healthy = False

    def models(self) -> list[str]:
        return sorted(
            {
                model
                for item in self._providers.values()
                if self._available(item)
                for model in item.registration.models
            }
        )

    def snapshot(self) -> list[dict[str, object]]:
        return [
            {
                **item.registration.model_dump(mode="json", exclude={"instance_id"}),
                "ready": item.ready,
                "healthy": item.healthy,
                "online": self._fresh(item),
                "in_flight": item.in_flight,
                "heartbeat_age_seconds": round(self._clock() - item.last_seen, 3),
                "completed_requests": item.completed_requests,
                "failed_requests": item.failed_requests,
            }
            for _, item in sorted(self._providers.items())
        ]

    def _fresh(self, provider: _Provider) -> bool:
        return self._clock() - provider.last_seen < self._ttl

    def _available(self, provider: _Provider) -> bool:
        return self._fresh(provider) and provider.ready and provider.healthy
