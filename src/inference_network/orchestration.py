"""One request lifecycle: reserve, dispatch, record outcome, release."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from uuid import uuid4

from inference_network.errors import NetworkError
from inference_network.registry import ProviderRegistry
from inference_network.schemas import ChatCompletion, ChatCompletionChunk, ChatRequest
from inference_network.transport import ProviderTransport, SafeConnectFailure

logger = logging.getLogger(__name__)


class Coordinator:
    def __init__(
        self,
        registry: ProviderRegistry,
        transport: ProviderTransport,
        max_attempts: int,
        timeout_seconds: float,
    ) -> None:
        self._registry = registry
        self._transport = transport
        self._max_attempts = max_attempts
        self._timeout = timeout_seconds

    async def complete(self, request: ChatRequest) -> ChatCompletion:
        try:
            async with asyncio.timeout(self._timeout) as deadline:
                async with aclosing(
                    self._attempt(request, f"chatcmpl-{uuid4().hex}", deadline)
                ) as results:
                    result = await anext(results)
                    assert isinstance(result, ChatCompletion)
                    return result
        except TimeoutError as exc:
            raise NetworkError(
                504, "request_timeout", "Overall inference deadline exceeded"
            ) from exc

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        # The response owns the end-to-end deadline, including client backpressure.
        async with aclosing(self._attempt(request, f"chatcmpl-{uuid4().hex}")) as results:
            async for result in results:
                assert isinstance(result, ChatCompletionChunk)
                yield result

    async def _attempt(
        self, request: ChatRequest, request_id: str, deadline: asyncio.Timeout | None = None
    ) -> AsyncGenerator[ChatCompletion | ChatCompletionChunk, None]:
        excluded: set[str] = set()
        last_failure: SafeConnectFailure | None = None
        for attempt in range(self._max_attempts):
            try:
                reservation = self._registry.reserve(
                    request.model, excluded, streaming=request.stream
                )
            except NetworkError:
                if last_failure is not None:
                    raise last_failure from None
                raise
            succeeded = False
            committed = False
            started = time.monotonic()
            try:
                if request.stream:
                    async with aclosing(
                        self._transport.stream(reservation, request, request_id)
                    ) as chunks:
                        async for chunk in chunks:
                            if chunk.choices or (
                                request.stream_options and request.stream_options.include_usage
                            ):
                                committed = True
                                yield chunk
                    succeeded = True
                else:
                    result = await self._transport.dispatch(reservation, request, request_id)
                    succeeded = True
                    yield result
                return
            except SafeConnectFailure as exc:
                self._registry.mark_unhealthy(reservation)
                if committed:
                    raise
                excluded.add(reservation.provider_id)
                last_failure = exc
            except NetworkError:
                self._registry.mark_unhealthy(reservation)
                raise
            except asyncio.CancelledError:
                if deadline is not None and deadline.expired():
                    self._registry.mark_unhealthy(reservation)
                raise
            finally:
                # Synchronous release cannot itself be interrupted by cancellation.
                self._registry.release(reservation, succeeded=succeeded)
                logger.info(
                    "dispatch_completed",
                    extra={
                        "request_id": request_id,
                        "provider_id": reservation.provider_id,
                        "attempt": attempt + 1,
                        "succeeded": succeeded,
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                    },
                )
            if attempt + 1 < self._max_attempts:
                await asyncio.sleep(0.05 * (attempt + 1))
        assert last_failure is not None
        raise last_failure
