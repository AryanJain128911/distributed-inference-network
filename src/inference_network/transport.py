"""Bounded HTTP dispatch, with replay safety explicit at the connection boundary."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing

import httpx
from pydantic import SecretStr, ValidationError

from inference_network.errors import NetworkError
from inference_network.pairing import authorization_header
from inference_network.registry import Reservation
from inference_network.schemas import ChatCompletion, ChatCompletionChunk, ChatRequest
from inference_network.streaming import iter_sse


class SafeConnectFailure(NetworkError):
    """Connection was not established; no request was delivered by this attempt."""

    def __init__(self) -> None:
        super().__init__(
            502, "provider_connect_failed", "Could not connect to the selected provider"
        )


class ProviderTransport:
    def __init__(
        self,
        client: httpx.AsyncClient,
        max_response_bytes: int = 65536,
        *,
        provider_tokens: dict[str, SecretStr] | None = None,
    ) -> None:
        self._client = client
        self._max_response_bytes = max_response_bytes
        self._provider_tokens = provider_tokens

    def _headers(self, endpoint: str, request_id: str) -> dict[str, str]:
        headers = {"X-Request-ID": request_id, "Accept-Encoding": "identity"}
        if self._provider_tokens is not None:
            token = self._provider_tokens.get(endpoint)
            if token is None:
                raise NetworkError(403, "provider_not_approved", "Provider endpoint not approved")
            headers.update(authorization_header(token))
        return headers

    async def dispatch(
        self, reservation: Reservation, request: ChatRequest, request_id: str
    ) -> ChatCompletion:
        headers_received = False
        try:
            async with self._client.stream(
                "POST",
                f"{reservation.endpoint}/internal/chat/completions",
                json=request.model_dump(mode="json", exclude_none=True),
                headers=self._headers(reservation.endpoint, request_id),
            ) as response:
                headers_received = True
                if response.status_code != 200:
                    raise NetworkError(
                        502, "provider_error", "Provider rejected or failed the request"
                    )
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise NetworkError(
                        502,
                        "invalid_provider_response",
                        "Compressed provider responses are unsupported",
                    )
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > self._max_response_bytes:
                        raise NetworkError(
                            502,
                            "invalid_provider_response",
                            "Provider response exceeded the size limit",
                        )
                result = ChatCompletion.model_validate_json(content)
                if (
                    result.model != request.model
                    or result.id != request_id
                    or result.metadata.provider_id != reservation.provider_id
                    or result.choices[0].message.role != "assistant"
                    or result.metadata.simulated != reservation.simulated
                    or (result.metadata.simulated and result.usage is not None)
                ):
                    raise NetworkError(
                        502,
                        "invalid_provider_response",
                        "Provider response violated the request contract",
                    )
                return result
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if headers_received:
                raise NetworkError(
                    502, "invalid_provider_response", "Interrupted provider response"
                ) from exc
            raise SafeConnectFailure() from exc
        except httpx.TimeoutException as exc:
            raise NetworkError(
                504, "provider_timeout", "Provider response deadline exceeded"
            ) from exc
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise NetworkError(
                502, "invalid_provider_response", "Invalid or interrupted provider response"
            ) from exc

    async def stream(
        self, reservation: Reservation, request: ChatRequest, request_id: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        headers_received = False
        payload = request.model_dump(mode="json")
        # Usage is always checked inside the network, even when the caller does
        # not request its public event. Text chunks are never used as token counts.
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        try:
            async with self._client.stream(
                "POST",
                f"{reservation.endpoint}/internal/chat/completions",
                json=payload,
                headers={
                    **self._headers(reservation.endpoint, request_id),
                    "Accept": "text/event-stream",
                },
            ) as response:
                headers_received = True
                if response.status_code != 200:
                    raise NetworkError(
                        502, "provider_error", "Provider rejected or failed the request"
                    )
                if (
                    response.headers.get("content-encoding", "identity").lower() != "identity"
                    or response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    != "text/event-stream"
                ):
                    raise ValueError("Expected uncompressed event stream")
                async with aclosing(
                    self._validated_chunks(response.aiter_bytes(), reservation, request, request_id)
                ) as chunks:
                    async for chunk in chunks:
                        yield chunk
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if not headers_received:
                raise SafeConnectFailure() from exc
            raise NetworkError(
                502, "invalid_provider_response", "Interrupted provider stream"
            ) from exc
        except httpx.TimeoutException as exc:
            raise NetworkError(
                504, "provider_timeout", "Provider stream deadline exceeded"
            ) from exc
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise NetworkError(
                502, "invalid_provider_response", "Invalid or interrupted provider stream"
            ) from exc

    async def _validated_chunks(
        self,
        source: AsyncIterator[bytes],
        reservation: Reservation,
        request: ChatRequest,
        request_id: str,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        created: int | None = None
        text_length = 0
        phase = "role"
        finish: ChatCompletionChunk | None = None
        usage: ChatCompletionChunk | None = None
        done = False
        async for event in iter_sse(source, max_event_bytes=self._max_response_bytes):
            if done:
                raise ValueError("Event after stream termination")
            if event.event == "error":
                raise NetworkError(502, "provider_error", "Provider failed during generation")
            if event.event != "message":
                raise ValueError("Unsupported stream event")
            if event.data == "[DONE]":
                if finish is None or usage is None:
                    raise ValueError("Stream ended before finish and usage")
                done = True
                continue
            chunk = ChatCompletionChunk.model_validate_json(event.data)
            if created is None:
                created = chunk.created
            if (
                chunk.id != request_id
                or chunk.model != request.model
                or chunk.created != created
                or chunk.metadata.provider_id != reservation.provider_id
                or chunk.metadata.simulated != reservation.simulated
                or (reservation.simulated and chunk.usage is not None)
            ):
                raise ValueError("Provider stream identity or usage mismatch")
            if not chunk.choices:
                if phase != "usage":
                    raise ValueError("Out-of-order usage event")
                usage = chunk
                phase = "done"
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if phase == "role":
                if (
                    delta.role != "assistant"
                    or delta.content not in (None, "")
                    or choice.finish_reason is not None
                ):
                    raise ValueError("Expected initial assistant role event")
                phase = "content"
                yield chunk
            elif phase == "content" and choice.finish_reason is None:
                if delta.role is not None or not delta.content:
                    raise ValueError("Expected nonempty content delta")
                text_length += len(delta.content)
                if text_length > 8192:
                    raise ValueError("Provider stream exceeded the text limit")
                yield chunk
            elif phase == "content" and choice.finish_reason is not None:
                if delta.role is not None or delta.content is not None or not text_length:
                    raise ValueError("Invalid finish event")
                finish = chunk
                phase = "usage"
            else:
                raise ValueError("Out-of-order stream event")
        if not done or finish is None or usage is None:
            raise ValueError("Provider stream ended without successful termination")
        # Terminal success is withheld until framing, DONE, and EOF validate.
        # Content above is forwarded immediately; only these two events are held.
        yield finish
        yield usage
