"""Stable JSON error contract for the API surface.

Every error response is ``{"error": {"code": ..., "message": ...}}`` with
stable machine-readable codes; stack traces and internal details never
reach clients (server-side logging keeps the traceback, redacted).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_INTERNAL_MESSAGE = "internal error — details are withheld"


def error_body(code: str, message: str) -> dict[str, Any]:
    """The one error shape every client sees."""
    return {"error": {"code": code, "message": message}}


class JsonError(Exception):
    """An error that maps directly onto the stable JSON contract."""

    def __init__(self, *, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def unauthorized() -> JsonError:
    return JsonError(
        status=401,
        code="unauthorized",
        message="missing or invalid bearer token",
    )


def rate_limited() -> JsonError:
    return JsonError(
        status=429,
        code="rate_limited",
        message="too many requests — slow down and retry later",
    )


def bad_request(message: str) -> JsonError:
    return JsonError(
        status=400,
        code="invalid_request",
        message=message,
    )


def internal_error() -> JsonError:
    return JsonError(
        status=500,
        code="internal_error",
        message=_INTERNAL_MESSAGE,
    )


def install_handlers(app: FastAPI) -> None:
    """Register the global handlers enforcing the error contract."""

    @app.exception_handler(JsonError)
    async def _json_error(_: Request, exc: JsonError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content=error_body(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Request bodies are client-controlled; echo only a generic shape.
        return JSONResponse(
            status_code=400,
            content=error_body("invalid_request", "request body failed validation"),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, JsonError):
            status, code, message = exc.status, exc.code, exc.message
        else:
            status, code, message = 500, "internal_error", _INTERNAL_MESSAGE
        logger.exception(
            "unhandled error on %s", redacted_url(request.url), exc_info=exc
        )
        return JSONResponse(status_code=status, content=error_body(code, message))


def redacted_url(url: Any) -> str:
    """Strip query strings (they may carry PII) before persisting a URL."""
    return str(getattr(url, "path", url))
