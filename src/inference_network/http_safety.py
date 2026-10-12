"""Bound request bodies, reject browser-origin writes, and own disconnect cancellation."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from inference_network.errors import NetworkError, error_response

Result = TypeVar("Result")


async def _wait_for_disconnect(request: Request) -> None:
    # The API handler has already parsed the complete body. ASGI receive then
    # blocks until disconnect, without polling or racing another body reader.
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def run_while_connected(operation: Coroutine[Any, Any, Result], request: Request) -> Result:
    """Cancel and collect request work when its caller disconnects or is cancelled."""
    work = asyncio.create_task(operation)
    disconnection = asyncio.create_task(_wait_for_disconnect(request))
    try:
        completed, _ = await asyncio.wait(
            (work, disconnection), return_when=asyncio.FIRST_COMPLETED
        )
        if work in completed:
            return await work
        await disconnection
        raise NetworkError(499, "client_disconnected", "Client disconnected before completion")
    finally:
        for task in (work, disconnection):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, disconnection, return_exceptions=True)


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int, body_timeout_seconds: float) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.body_timeout_seconds = body_timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        if scope["method"] in {"POST", "PUT", "PATCH", "DELETE"} and b"origin" in headers:
            await error_response(403, "browser_origin_denied", "Use a local API client")(
                scope, receive, send
            )
            return
        body = bytearray()
        oversized = False
        try:
            async with asyncio.timeout(self.body_timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > self.max_body_bytes:
                        oversized = True
                        break
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await error_response(408, "body_timeout", "Request body deadline exceeded")(
                scope, receive, send
            )
            return
        if oversized:
            await error_response(413, "payload_too_large", "Request body is too large")(
                scope, receive, send
            )
            return
        delivered = False

        async def buffered_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, buffered_receive, send)
