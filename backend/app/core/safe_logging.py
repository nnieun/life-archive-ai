"""Privacy-safe structured logging for application events."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Final

LOGGER_NAME: Final = "life_archive"
_ALLOWED_FIELDS: Final = (
    "request_id",
    "method",
    "route",
    "status_code",
    "duration_ms",
    "error_type",
)
_SECRET_PATTERNS: Final = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
)


def mask_secrets(value: object) -> object:
    """Mask API credentials in values accepted by the structured logger."""

    if isinstance(value, str):
        masked = value
        for pattern in _SECRET_PATTERNS:
            replacement = "Bearer ***" if "Bearer" in pattern.pattern else "sk-***"
            masked = pattern.sub(replacement, masked)
        return masked
    return value


class SafeJsonFormatter(logging.Formatter):
    """Serialize an allowlisted event without paths, bodies, or stack traces."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": mask_secrets(record.getMessage()),
        }
        for field in _ALLOWED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = mask_secrets(value)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_safe_logging() -> logging.Logger:
    """Return the application's single configured structured logger."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if not any(getattr(handler, "_life_archive_safe", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(SafeJsonFormatter())
        handler._life_archive_safe = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_safe_exception(
    *,
    event: str,
    request_id: str,
    exception: Exception,
) -> None:
    """Log only the exception type, never its potentially sensitive message."""

    configure_safe_logging().error(
        event,
        extra={
            "request_id": request_id,
            "error_type": type(exception).__name__,
        },
    )
