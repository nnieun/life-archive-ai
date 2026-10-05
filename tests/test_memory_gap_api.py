"""Memory-gap API contracts and granular error mapping."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from backend.app.api.memories import get_memory_repository
from backend.app.api.memory_gaps import (
    get_memory_gap_reconstruction_service,
    get_memory_gap_resolution_service,
)
from backend.app.main import app
from backend.app.models.gap import (
    MemoryGapCandidateCreate,
    MemoryGapCandidateRelation,
    MemoryGapCreate,
    MemoryGapReconstructionResult,
    MemoryGapSourceType,
    MemoryGapStatus,
    MemoryGapType,
)
from backend.app.models.memory import DatePrecision
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.gap_reconstruction import (
    MemoryGapCandidateOutputError,
    MemoryGapResolutionService,
)
from backend.app.storage.models import (
    MemoryCreate,
    MemorySourceCreate,
    TranscriptSegmentCreate,
)

CONTENT = "2001년 대구 동성로 아카데미극장에서 친구와 영화를 봤다."


def _seed_gap(repository):
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_gap_api",
            filename="gap-api.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 8, 31, tzinfo=UTC),
            content_hash=hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
            raw_content=CONTENT,
            normalized_content=CONTENT,
        )
    )
    repository.create_segment(
        TranscriptSegmentCreate(
            segment_id="seg_gap_api",
            transcript_id="tr_gap_api",
            chunk_index=0,
            content=CONTENT,
            start_offset=0,
            end_offset=len(CONTENT),
        )
    )
    repository.create_memory(
        MemoryCreate(
            memory_id="mem_gap_api",
            transcript_id="tr_gap_api",
            title="친구와 영화 관람",
            summary="2001년 대구 동성로에서 친구와 영화를 봤다.",
            people=["친구"],
            location="대구 동성로",
            event_date="2001",
            date_precision=DatePrecision.YEAR,
            confidence=0.8,
            uncertainty_notes="극장 이름을 확인해야 한다.",
        )
    )
    repository.create_memory_source(
        MemorySourceCreate(
            memory_source_id="src_gap_api",
            memory_id="mem_gap_api",
            transcript_id="tr_gap_api",
            segment_id="seg_gap_api",
            start_offset=0,
            end_offset=len(CONTENT),
        )
    )
    gap = repository.create_memory_gap(
        MemoryGapCreate(
            gap_id="gap_api",
            memory_id="mem_gap_api",
            gap_type=MemoryGapType.MISSING_LOCATION,
            clue_text="극장 이름은 기억나지 않는다.",
            missing_field="location_detail",
            period_start="2001",
            period_end="2001",
            location="대구 동성로",
            people=["친구"],
            confidence=0.9,
            importance_score=0.9,
            status=MemoryGapStatus.CANDIDATE_FOUND,
            source_type=MemoryGapSourceType.MEMORY,
            source_id="mem_gap_api",
        )
    )
    candidate = repository.create_memory_gap_candidate(
        MemoryGapCandidateCreate(
            candidate_id="gcan_gap_api",
            gap_id=gap.gap_id,
            value="아카데미극장",
            explanation="원문에 같은 시기와 장소가 기록되어 있습니다.",
            deterministic_score=0.9,
            llm_relation=MemoryGapCandidateRelation.SUPPORTS,
            supporting_source_ids=["seg_gap_api"],
        )
    )
    return gap, candidate


def _client(repository, *, reconstruction_service=None) -> TestClient:
    app.dependency_overrides[get_memory_repository] = lambda: repository
    app.dependency_overrides[get_memory_gap_resolution_service] = (
        lambda: MemoryGapResolutionService(repository)
    )
    if reconstruction_service is not None:
        app.dependency_overrides[get_memory_gap_reconstruction_service] = (
            lambda: reconstruction_service
        )
    return TestClient(app)


def test_gap_list_clue_and_dismiss_contracts(sqlite_repository) -> None:
    gap, _candidate = _seed_gap(sqlite_repository)
    client = _client(sqlite_repository)
    try:
        listed = client.get("/api/v1/memory-gaps")
        clue = client.post(
            f"/api/v1/memory-gaps/{gap.gap_id}/clues",
            json={"clue": "극장 옆에 백화점이 있었다."},
        )
        duplicate = client.post(
            f"/api/v1/memory-gaps/{gap.gap_id}/clues",
            json={"clue": "극장 옆에 백화점이 있었다."},
        )
        dismissed = client.post(
            f"/api/v1/memory-gaps/{gap.gap_id}/dismiss"
        )
        active = client.get("/api/v1/memory-gaps")
        history = client.get("/api/v1/memory-gaps?include_closed=true")
    finally:
        app.dependency_overrides.clear()

    assert listed.status_code == 200
    assert listed.json()["items"][0]["gap"]["gap_type"] == "MISSING_LOCATION"
    assert listed.json()["items"][0]["candidates"][0]["value"] == "아카데미극장"
    assert clue.status_code == 200
    assert clue.json()["gap"]["status"] == "OPEN"
    assert clue.json()["gap"]["user_clues"] == ["극장 옆에 백화점이 있었다."]
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "memory_gap_clue_duplicate"
    assert dismissed.status_code == 200
    assert dismissed.json()["gap"]["status"] == "DISMISSED"
    assert active.json()["items"] == []
    assert len(history.json()["items"]) == 1


def test_resolve_endpoint_requires_confirmation_then_returns_new_memory_id(
    sqlite_repository,
) -> None:
    gap, candidate = _seed_gap(sqlite_repository)
    client = _client(sqlite_repository)
    try:
        rejected = client.post(
            f"/api/v1/memory-gaps/{gap.gap_id}/resolve",
            json={
                "candidate_id": candidate.candidate_id,
                "user_confirmed": False,
            },
        )
        resolved = client.post(
            f"/api/v1/memory-gaps/{gap.gap_id}/resolve",
            json={
                "candidate_id": candidate.candidate_id,
                "user_confirmed": True,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert rejected.status_code == 428
    assert rejected.json()["error"]["code"] == (
        "memory_gap_confirmation_required"
    )
    assert resolved.status_code == 200
    body = resolved.json()
    assert body["gap"]["status"] == "RESOLVED"
    assert body["candidate"]["status"] == "ACCEPTED"
    assert body["resolved_memory_id"].startswith("mem_")
    corrected = sqlite_repository.get_memory(body["resolved_memory_id"])
    assert corrected.location == "대구 동성로 · 아카데미극장"


class StubReconstructionService:
    def __init__(self, result: MemoryGapReconstructionResult) -> None:
        self.result = result

    def reconstruct(self, gap_id: str) -> MemoryGapReconstructionResult:
        assert gap_id == self.result.gap.gap_id
        return self.result


class InvalidCandidateService:
    def reconstruct(self, _gap_id: str) -> MemoryGapReconstructionResult:
        raise MemoryGapCandidateOutputError("private model output")


def test_reconstruct_endpoint_and_specific_candidate_output_error(
    sqlite_repository,
) -> None:
    gap, candidate = _seed_gap(sqlite_repository)
    result = MemoryGapReconstructionResult(
        gap=gap,
        candidates=[candidate],
        searched_tools=["search_memory"],
        tool_call_count=1,
        message="복원 후보를 찾았습니다.",
    )
    client = _client(
        sqlite_repository,
        reconstruction_service=StubReconstructionService(result),
    )
    try:
        success = client.post(
            f"/api/v1/memory-gaps/{gap.gap_id}/reconstruct"
        )
        app.dependency_overrides[get_memory_gap_reconstruction_service] = (
            lambda: InvalidCandidateService()
        )
        invalid = client.post(
            f"/api/v1/memory-gaps/{gap.gap_id}/reconstruct"
        )
    finally:
        app.dependency_overrides.clear()

    assert success.status_code == 200
    assert success.json()["searched_tools"] == ["search_memory"]
    assert invalid.status_code == 503
    assert invalid.json()["error"]["code"] == (
        "memory_gap_candidate_output_invalid"
    )
    assert "private" not in invalid.text
