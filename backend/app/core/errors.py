"""Consistent, privacy-safe API error responses."""

import sqlite3

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.request_context import new_request_id
from backend.app.core.safe_logging import log_safe_exception
from backend.app.storage.repository import StorageError


class ErrorDetail(BaseModel):
    """Machine-readable and user-safe error detail."""

    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    """Standard API error envelope."""

    error: ErrorDetail


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str):
        return request_id
    request_id = new_request_id()
    request.state.request_id = request_id
    return request_id


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=_request_id(request),
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def http_exception_handler(
    request: Request,
    exception: StarletteHTTPException,
) -> JSONResponse:
    """Convert framework HTTP errors to the standard response envelope."""
    if exception.status_code >= 500 and isinstance(
        exception.__cause__,
        Exception,
    ):
        log_safe_exception(
            event="service_request_failed",
            request_id=_request_id(request),
            exception=exception.__cause__,
        )
    message = exception.detail if isinstance(exception.detail, str) else "Request failed"
    return _error_response(request, exception.status_code, "http_error", message)


async def validation_exception_handler(
    request: Request,
    _exception: RequestValidationError,
) -> JSONResponse:
    """Avoid echoing untrusted request contents in validation errors."""
    return _error_response(
        request,
        422,
        "validation_error",
        "Request validation failed",
    )


async def storage_exception_handler(
    request: Request,
    exception: StorageError | sqlite3.Error,
) -> JSONResponse:
    """Translate SQLite failures without exposing statements or local paths."""

    log_safe_exception(
        event="storage_operation_failed",
        request_id=_request_id(request),
        exception=exception,
    )
    return _error_response(
        request,
        503,
        "storage_error",
        "Storage service is unavailable",
    )


async def unexpected_exception_handler(
    request: Request,
    exception: Exception,
) -> JSONResponse:
    """Return a safe message without exposing internal paths or user data."""
    log_safe_exception(
        event="unexpected_request_failure",
        request_id=_request_id(request),
        exception=exception,
    )
    return _error_response(
        request,
        500,
        "internal_error",
        "Internal server error",
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach all common exception handlers to an application."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StorageError, storage_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(sqlite3.Error, storage_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_exception_handler)
