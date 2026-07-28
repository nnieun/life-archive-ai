"""Streamlit API client tests without a live backend."""

from base64 import b64decode

import httpx
import pytest

from frontend.api_client import (
    ApiClientError,
    LifeArchiveApiClient,
)


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


def test_upload_client_preserves_original_bytes() -> None:
    original = b"\xef\xbb\xbfmemory\r\ntext"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert b64decode(payload["content_base64"]) == original
        return httpx.Response(
            200,
            json={
                "transcript_id": "tr_001",
                "filename": "memory.txt",
                "segment_count": 1,
                "memory_count": 0,
                "indexed_memory_count": 0,
                "memory_ids": [],
            },
        )

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    assert client.ingest_transcript("memory.txt", original).transcript_id == "tr_001"
