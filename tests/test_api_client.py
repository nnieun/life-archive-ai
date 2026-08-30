"""Streamlit API client tests without a live backend."""

from base64 import b64decode

import httpx
import pytest

from frontend.api_client import (
    INGEST_TIMEOUT_SECONDS,
    ApiClientError,
    LifeArchiveApiClient,
    TranscriptDeletionResult,
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
        return httpx.Response(
            503,
            headers={"X-Request-ID": "req-health"},
            json={
                "error": {
                    "code": "storage_error",
                    "message": "Storage service is unavailable",
                    "request_id": "req-health",
                }
            },
        )

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ApiClientError, match="Backend health check failed") as captured:
        client.get_health()

    assert captured.value.status_code == 503
    assert captured.value.request_id == "req-health"
    assert captured.value.error_code == "storage_error"
    assert captured.value.user_message == "Storage service is unavailable"


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


def test_upload_client_preserves_conflict_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(409)

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ApiClientError) as captured:
        client.ingest_transcript("memory.txt", b"duplicate")

    assert captured.value.status_code == 409


def test_upload_client_uses_longer_timeout_for_model_processing() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200,
            json={
                "transcript_id": "tr_001",
                "filename": "memory.txt",
                "segment_count": 1,
                "memory_count": 1,
                "indexed_memory_count": 1,
                "memory_ids": ["mem_001"],
            },
        )

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    client.ingest_transcript("memory.txt", b"content")

    assert captured["timeout"] == {
        "connect": INGEST_TIMEOUT_SECONDS,
        "read": INGEST_TIMEOUT_SECONDS,
        "write": INGEST_TIMEOUT_SECONDS,
        "pool": INGEST_TIMEOUT_SECONDS,
    }


def test_delete_client_parses_transcript_deletion_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/transcripts/tr_001"
        return httpx.Response(
            200,
            json={
                "transcript_id": "tr_001",
                "deleted_segment_count": 1,
                "deleted_memory_count": 2,
                "deleted_vector_count": 2,
                "bm25_memory_count": 2,
                "invalidated_conversation_message_count": 0,
                "invalidated_autobiography_count": 0,
                "raw_file_deleted": False,
            },
        )

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    result = client.delete_transcript("tr_001")

    assert isinstance(result, TranscriptDeletionResult)
    assert result.deleted_memory_count == 2
    assert result.raw_file_deleted is False
