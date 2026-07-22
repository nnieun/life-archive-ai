"""Typed HTTP client for the Life Archive AI backend."""

from typing import Final, Literal

import httpx
from pydantic import BaseModel, ValidationError

DEFAULT_API_URL: Final = "http://127.0.0.1:8000/api/v1"


class HealthStatus(BaseModel):
    """Validated health response returned by the backend."""

    status: Literal["ok"]
    service: str
    version: str


class ApiClientError(RuntimeError):
    """User-safe backend communication failure."""


class LifeArchiveApiClient:
    """Small synchronous client suitable for Streamlit reruns."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = f"{base_url.rstrip('/')}/"
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def get_health(self) -> HealthStatus:
        """Fetch and validate backend health without exposing response details."""
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.get("health")
                response.raise_for_status()
                return HealthStatus.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exception:
            raise ApiClientError("Backend health check failed") from exception
