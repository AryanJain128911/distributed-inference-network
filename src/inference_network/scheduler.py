"""Selection policy only; reservations belong exclusively to ProviderRegistry."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    provider_id: str
    in_flight: int
    max_concurrency: int


class Scheduler(Protocol):
    def choose(self, model: str, candidates: tuple[Candidate, ...]) -> str: ...


class RoundRobin:
    def __init__(self) -> None:
        self._last_selected: dict[str, str] = {}

    def choose(self, model: str, candidates: tuple[Candidate, ...]) -> str:
        identifiers = sorted(candidate.provider_id for candidate in candidates)
        previous = self._last_selected.get(model, "")
        selected = next((name for name in identifiers if name > previous), identifiers[0])
        self._last_selected[model] = selected
        return selected


class CapacityAware:
    def __init__(self) -> None:
        self._ties = RoundRobin()

    def choose(self, model: str, candidates: tuple[Candidate, ...]) -> str:
        minimum = min(Fraction(item.in_flight, item.max_concurrency) for item in candidates)
        least_loaded = tuple(
            item for item in candidates if Fraction(item.in_flight, item.max_concurrency) == minimum
        )
        return self._ties.choose(model, least_loaded)
