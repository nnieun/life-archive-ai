"""Consistent, privacy-safe API error responses."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException


class ErrorDetail(BaseModel):
    """Machine-readable and user-safe error detail."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Standard API error envelope."""

    error: ErrorDetail


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def http_exception_handler(
    _request: Request,
    exception: StarletteHTTPException,
) -> JSONResponse:
    """Convert framework HTTP errors to the standard response envelope."""
    message = exception.detail if isinstance(exception.detail, str) else "Request failed"
    return _error_response(exception.status_code, "http_error", message)


async def validation_exception_handler(
    _request: Request,
    _exception: RequestValidationError,
) -> JSONResponse:
    """Avoid echoing untrusted request contents in validation errors."""
    return _error_response(422, "validation_error", "Request validation failed")


async def unexpected_exception_handler(
    _request: Request,
    _exception: Exception,
) -> JSONResponse:
    """Return a safe message without exposing internal paths or user data."""
    return _error_response(500, "internal_error", "Internal server error")


def register_error_handlers(app: FastAPI) -> None:
    """Attach all common exception handlers to an application."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_exception_handler)
