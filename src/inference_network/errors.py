"""Domain errors and non-sensitive HTTP error envelopes."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse


class NetworkError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "network_error", "code": code}},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NetworkError)
    async def network_error_handler(request: Request, exc: NetworkError) -> JSONResponse:
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic errors may include the original prompt or body. Never echo them.
        return error_response(
            422, "invalid_request", "Request does not match the documented schema"
        )
