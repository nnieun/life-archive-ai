"""HTTP request tracking without recording user-controlled content."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Final
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.app.core.safe_logging import configure_safe_logging, log_safe_exception

REQUEST_ID_HEADER: Final = "X-Request-ID"
_VALID_REQUEST_ID: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id(candidate: str | None = None) -> str:
    """Accept a safe caller ID or generate an opaque server ID."""

    if candidate and _VALID_REQUEST_ID.fullmatch(candidate):
        return candidate
    return uuid4().hex


class RequestContextMiddleware:
    """Attach a request ID and emit one privacy-safe completion event."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._logger = configure_safe_logging()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        header_value = dict(scope.get("headers", [])).get(b"x-request-id", b"")
        request_id = new_request_id(header_value.decode("ascii", errors="ignore"))
        scope.setdefault("state", {})["request_id"] = request_id
        status_code = 500
        started_at = perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        except Exception as exception:
            log_safe_exception(
                event="unexpected_request_failure",
                request_id=request_id,
                exception=exception,
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "Internal server error",
                        "request_id": request_id,
                    }
                },
            )
            await response(scope, receive, send_with_request_id)
        finally:
            route = getattr(scope.get("route"), "path", "<unmatched>")
            self._logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method", ""),
                    "route": route,
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
