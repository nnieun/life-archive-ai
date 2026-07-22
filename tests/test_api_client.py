"""Streamlit API client tests without a live backend."""

import httpx
import pytest

from frontend.api_client import ApiClientError, LifeArchiveApiClient


def test_health_client_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/health"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "service": "Life Archive AI",
                "version": "0.0.0",
            },
        )

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    assert client.get_health().status == "ok"


def test_health_client_returns_safe_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ApiClientError, match="Backend health check failed"):
        client.get_health()
