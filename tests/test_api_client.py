"""Streamlit API client tests without a live backend."""

from base64 import b64decode

import httpx
import pytest

from frontend.api_client import (
    INGEST_TIMEOUT_SECONDS,
    ApiClientError,
    LifeArchiveApiClient,
    MemoryGapReconstructionResult,
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


def _gap_json() -> dict[str, object]:
    return {
        "gap_id": "gap_001",
        "memory_id": "mem_001",
        "gap_type": "MISSING_LOCATION",
        "clue_text": "극장 이름이 기억나지 않는다.",
        "missing_field": "location_detail",
        "people": [],
        "confidence": 0.9,
        "importance_score": 0.8,
        "status": "OPEN",
    }


def _gap_candidate_json() -> dict[str, object]:
    return {
        "candidate_id": "gcan_001",
        "gap_id": "gap_001",
        "value": "아카데미극장",
        "explanation": "같은 시기 기록입니다.",
        "deterministic_score": 0.9,
        "llm_relation": "SUPPORTS",
        "supporting_source_ids": ["seg_001"],
        "status": "PROPOSED",
    }


def test_memory_gap_client_lists_and_reconstructs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/api/v1/memory-gaps"
            assert request.url.params["include_closed"] == "false"
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"gap": _gap_json(), "candidates": []},
                    ]
                },
            )
        assert request.url.path == "/api/v1/memory-gaps/gap_001/reconstruct"
        return httpx.Response(
            200,
            json={
                "gap": {**_gap_json(), "status": "CANDIDATE_FOUND"},
                "candidates": [_gap_candidate_json()],
                "searched_tools": ["search_memory"],
                "tool_call_count": 1,
                "message": "복원 후보를 찾았습니다.",
            },
        )

    client = LifeArchiveApiClient(transport=httpx.MockTransport(handler))

    assert client.list_memory_gaps()[0].gap.gap_id == "gap_001"
    reconstructed = client.reconstruct_memory_gap("gap_001")

    assert isinstance(reconstructed, MemoryGapReconstructionResult)
    assert reconstructed.candidates[0].value == "아카데미극장"


def test_memory_gap_client_sends_explicit_confirmation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert request.url.path == "/api/v1/memory-gaps/gap_001/resolve"
        assert payload == {
            "candidate_id": "gcan_001",
            "user_confirmed": True,
        }
        return httpx.Response(
            200,
            json={
                "gap": {**_gap_json(), "status": "RESOLVED"},
                "candidate": {**_gap_candidate_json(), "status": "ACCEPTED"},
                "resolved_memory_id": "mem_corrected",
            },
        )

    result = LifeArchiveApiClient(
        transport=httpx.MockTransport(handler)
    ).resolve_memory_gap(
        "gap_001",
        "gcan_001",
        user_confirmed=True,
    )

    assert result.resolved_memory_id == "mem_corrected"


def test_autobiography_gap_preflight_parses_related_gaps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert request.url.path == "/api/v1/autobiographies/gap-check"
        assert payload["request"] == "영화에 대한 이야기를 써 줘"
        return httpx.Response(
            200,
            json={
                "important_unresolved_gaps": [_gap_json()],
                "retrieved_memory_ids": ["mem_001"],
                "requires_gap_confirmation": True,
            },
        )

    result = LifeArchiveApiClient(
        transport=httpx.MockTransport(handler)
    ).check_autobiography_gaps(
        title="나의 기억",
        request="영화에 대한 이야기를 써 줘",
        target_period=None,
        target_topics=["영화"],
        chapter_count=1,
    )

    assert result.requires_gap_confirmation is True
    assert result.important_unresolved_gaps[0].gap_id == "gap_001"
