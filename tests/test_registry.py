import asyncio
from uuid import uuid4

import pytest

from inference_network.errors import NetworkError
from inference_network.registry import ProviderRegistry
from inference_network.scheduler import CapacityAware, RoundRobin
from inference_network.schemas import Heartbeat, ProviderRegistration


def registration(name="a", port=8101, capacity=2, models=("simulated-echo-v1",)):
    return ProviderRegistration(
        provider_id=name,
        instance_id=uuid4(),
        models=models,
        gpu_type="SIMULATED GPU",
        vram_gb=24,
        max_concurrency=capacity,
        endpoint=f"http://127.0.0.1:{port}",
    )


@pytest.fixture
def clock():
    return [100.0]


@pytest.fixture
def registry(clock):
    return ProviderRegistry(RoundRobin(), 3, clock=lambda: clock[0])


def test_registration_idempotency_preserves_reservations_and_liveness(registry, clock):
    provider = registration()
    registry.register(provider)
    reservation = registry.reserve("simulated-echo-v1")
    clock[0] += 2
    registry.register(provider)
    assert registry.snapshot()[0]["in_flight"] == 1
    assert registry.snapshot()[0]["heartbeat_age_seconds"] == 2
    registry.release(reservation, succeeded=True)
    assert registry.snapshot()[0]["completed_requests"] == 1


def test_conflicting_identity_endpoint_and_capability_are_rejected(registry):
    provider = registration()
    registry.register(provider)
    for conflict in (
        registration(),
        registration(name="b"),
        provider.model_copy(update={"max_concurrency": 3}),
    ):
        with pytest.raises(NetworkError) as error:
            registry.register(conflict)
        assert error.value.status == 409


def test_exact_expiry_and_valid_heartbeat_recovery(registry, clock):
    provider = registration()
    registry.register(provider)
    clock[0] += 3
    assert registry.snapshot()[0]["online"] is False
    assert registry.models() == []
    with pytest.raises(NetworkError, match="No healthy"):
        registry.reserve("simulated-echo-v1")
    registry.heartbeat("a", Heartbeat(instance_id=provider.instance_id, sequence=1, ready=True))
    reservation = registry.reserve("simulated-echo-v1")
    assert registry.snapshot()[0]["online"] is True
    registry.release(reservation, succeeded=True)


def test_stale_or_foreign_heartbeat_cannot_restore_liveness(registry, clock):
    provider = registration()
    registry.register(provider)
    update = Heartbeat(instance_id=provider.instance_id, sequence=2, ready=True)
    registry.heartbeat("a", update)
    clock[0] += 4
    for invalid in (update, update.model_copy(update={"instance_id": uuid4(), "sequence": 3})):
        with pytest.raises(NetworkError):
            registry.heartbeat("a", invalid)
    assert registry.snapshot()[0]["online"] is False


def test_replacement_requires_expiry_and_no_outstanding_work(registry, clock):
    old = registration()
    registry.register(old)
    reservation = registry.reserve("simulated-echo-v1")
    clock[0] += 4
    replacement = registration()
    with pytest.raises(NetworkError):
        registry.register(replacement)
    registry.release(reservation, succeeded=False)
    registry.register(replacement)
    with pytest.raises(NetworkError):
        registry.heartbeat("a", Heartbeat(instance_id=old.instance_id, sequence=100, ready=True))
    assert registry.snapshot()[0]["in_flight"] == 0


def test_readiness_health_and_model_filtering(registry):
    a, b = registration(models=("alpha",)), registration("b", 8102, models=("beta",))
    registry.register(a)
    registry.register(b)
    assert registry.models() == ["alpha", "beta"]
    with pytest.raises(NetworkError) as missing:
        registry.reserve("absent")
    assert missing.value.code == "model_not_found"
    reservation = registry.reserve("beta")
    assert reservation.provider_id == "b"
    registry.mark_unhealthy(reservation)
    registry.release(reservation, succeeded=False)
    with pytest.raises(NetworkError, match="No healthy"):
        registry.reserve("beta")
    registry.heartbeat("b", Heartbeat(instance_id=b.instance_id, sequence=1, ready=True))
    reservation = registry.reserve("beta")
    registry.release(reservation, succeeded=True)
    registry.heartbeat("b", Heartbeat(instance_id=b.instance_id, sequence=2, ready=False))
    assert registry.models() == ["alpha"]


def test_round_robin_is_fair_and_skips_busy_providers(registry):
    registry.register(registration(capacity=1))
    registry.register(registration("b", 8102, capacity=1))
    selected = []
    for _ in range(6):
        reservation = registry.reserve("simulated-echo-v1")
        selected.append(reservation.provider_id)
        registry.release(reservation, succeeded=True)
    assert selected == ["a", "b", "a", "b", "a", "b"]
    first, second = registry.reserve("simulated-echo-v1"), registry.reserve("simulated-echo-v1")
    assert {first.provider_id, second.provider_id} == {"a", "b"}
    with pytest.raises(NetworkError, match="available capacity"):
        registry.reserve("simulated-echo-v1")
    registry.release(first, succeeded=True)
    third = registry.reserve("simulated-echo-v1")
    assert third.provider_id == first.provider_id
    registry.release(second, succeeded=True)
    registry.release(third, succeeded=True)


def test_capacity_aware_uses_normalized_load_and_round_robin_ties():
    registry = ProviderRegistry(CapacityAware(), 3)
    registry.register(registration(capacity=2))
    registry.register(registration("b", 8102, capacity=4))
    reservations = [registry.reserve("simulated-echo-v1") for _ in range(6)]
    # 0/2 vs 0/4 tie -> a; then b; then b at .25 vs a at .5.
    assert [item.provider_id for item in reservations] == ["a", "b", "b", "a", "b", "b"]
    assert [row["in_flight"] for row in registry.snapshot()] == [2, 4]
    for reservation in reservations:
        registry.release(reservation, succeeded=True)


async def test_simultaneous_admission_cannot_oversubscribe(registry):
    registry.register(registration(capacity=2))
    gate = asyncio.Event()
    reservations = []

    async def attempt():
        try:
            lease = registry.reserve("simulated-echo-v1")
        except NetworkError:
            return False
        reservations.append(lease)
        try:
            await gate.wait()
            return True
        finally:
            registry.release(lease, succeeded=True)

    tasks = [asyncio.create_task(attempt()) for _ in range(30)]
    await asyncio.sleep(0)  # Give each task its initial admission turn, no wall-clock delay.
    assert len(reservations) == 2
    assert registry.snapshot()[0]["in_flight"] == 2
    gate.set()
    assert sum(await asyncio.gather(*tasks)) == 2
    assert registry.snapshot()[0]["in_flight"] == 0


def test_double_release_is_rejected_and_does_not_reduce_capacity(registry):
    registry.register(registration())
    lease = registry.reserve("simulated-echo-v1")
    registry.release(lease, succeeded=True)
    with pytest.raises(RuntimeError):
        registry.release(lease, succeeded=True)
    assert registry.snapshot()[0]["in_flight"] == 0


def test_registry_size_is_bounded():
    registry = ProviderRegistry(RoundRobin(), 3, max_providers=1)
    registry.register(registration())
    with pytest.raises(NetworkError) as error:
        registry.register(registration("b", 8102))
    assert error.value.code == "registry_full"
