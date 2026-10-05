"""Deterministic, non-mutating memory-gap detection tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.models.gap import MemoryGapStatus, MemoryGapType
from backend.app.models.memory import DatePrecision
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.gap_detection import MemoryGapDetectionService
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import (
    MemoryCreate,
    MemorySourceCreate,
    TranscriptSegmentCreate,
)
from backend.app.storage.repository import SQLiteRepository


@pytest.fixture
def gap_storage(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "gaps.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    yield repository
    database.close()


def _create_memory(
    repository: SQLiteRepository,
    *,
    suffix: str,
    title: str,
    summary: str,
    people: list[str] | None = None,
    location: str | None = None,
    event_date: str | None = None,
    date_precision: DatePrecision = DatePrecision.UNKNOWN,
    confidence: float = 0.9,
    uncertainty_notes: str | None = None,
):
    transcript_id = f"tr_{suffix}"
    segment_id = f"seg_{suffix}"
    memory_id = f"mem_{suffix}"
    repository.create_transcript(
        LoadedTranscript(
            transcript_id=transcript_id,
            filename=f"{suffix}.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 8, 31, tzinfo=UTC),
            content_hash=hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
            raw_content=summary,
            normalized_content=summary,
        )
    )
    repository.create_segment(
        TranscriptSegmentCreate(
            segment_id=segment_id,
            transcript_id=transcript_id,
            chunk_index=0,
            content=summary,
            start_offset=0,
            end_offset=len(summary),
        )
    )
    memory = repository.create_memory(
        MemoryCreate(
            memory_id=memory_id,
            transcript_id=transcript_id,
            title=title,
            summary=summary,
            people=people or [],
            location=location,
            event_date=event_date,
            date_precision=date_precision,
            confidence=confidence,
            uncertainty_notes=uncertainty_notes,
        )
    )
    repository.create_memory_source(
        MemorySourceCreate(
            memory_source_id=f"src_{suffix}",
            memory_id=memory_id,
            transcript_id=transcript_id,
            segment_id=segment_id,
            start_offset=0,
            end_offset=len(summary),
        )
    )
    return memory


def test_specific_unknown_place_creates_one_idempotent_gap(gap_storage) -> None:
    memory = _create_memory(
        gap_storage,
        suffix="place",
        title="동성로 영화관",
        summary=(
            "2001년 대구 동성로에서 친구들과 영화를 봤지만 "
            "극장 이름은 기억나지 않는다."
        ),
        people=["친구"],
        location="대구 동성로",
        event_date="2001",
        date_precision=DatePrecision.YEAR,
        confidence=0.8,
        uncertainty_notes="극장 이름을 정확히 기억하지 못한다.",
    )
    detector = MemoryGapDetectionService(gap_storage)

    first = detector.detect_for_memories([memory])
    second = detector.detect_for_memories([memory])

    assert len(first) == 1
    assert second == first
    assert len(gap_storage.list_memory_gaps()) == 1
    gap = first[0]
    assert gap.gap_type is MemoryGapType.MISSING_LOCATION
    assert gap.missing_field == "location_detail"
    assert gap.status is MemoryGapStatus.OPEN
    assert gap.memory_id == memory.memory_id
    assert gap_storage.get_memory(memory.memory_id) == memory


def test_missing_person_and_date_are_detected_without_guessing_values(
    gap_storage,
) -> None:
    memory = _create_memory(
        gap_storage,
        suffix="unknowns",
        title="어릴 때의 만남",
        summary="어릴 때 누군가와 함께 오래 이야기했다.",
    )

    gaps = MemoryGapDetectionService(gap_storage).detect_for_memories([memory])

    assert {gap.gap_type for gap in gaps} == {
        MemoryGapType.MISSING_PERSON,
        MemoryGapType.MISSING_DATE,
    }
    assert all(gap.period_start is None for gap in gaps)
    assert all(gap.people == [] for gap in gaps)


def test_complete_high_confidence_memory_does_not_create_gap(gap_storage) -> None:
    memory = _create_memory(
        gap_storage,
        suffix="complete",
        title="대학교 졸업식",
        summary="2010년 서울에서 가족과 대학교 졸업식에 참석했다.",
        people=["가족"],
        location="서울",
        event_date="2010",
        date_precision=DatePrecision.YEAR,
        confidence=0.95,
    )

    gaps = MemoryGapDetectionService(gap_storage).detect_for_memories([memory])

    assert gaps == []


def test_same_titled_memories_with_different_dates_create_conflict_gap(
    gap_storage,
) -> None:
    _create_memory(
        gap_storage,
        suffix="conflicta",
        title="첫 직장 입사",
        summary="2010년 서울의 첫 직장에 입사했다.",
        location="서울",
        event_date="2010",
        date_precision=DatePrecision.YEAR,
    )
    second = _create_memory(
        gap_storage,
        suffix="conflictb",
        title="첫 직장 입사",
        summary="2011년 서울의 첫 직장에 입사했다.",
        location="서울",
        event_date="2011",
        date_precision=DatePrecision.YEAR,
    )

    gaps = MemoryGapDetectionService(gap_storage).detect_for_memories([second])

    assert len(gaps) == 1
    assert gaps[0].gap_type is MemoryGapType.CONFLICTING_FACT
    assert gaps[0].missing_field == "event_date"
    assert gaps[0].importance_score == 1.0


def test_single_source_medium_confidence_memory_is_weak_provenance(
    gap_storage,
) -> None:
    memory = _create_memory(
        gap_storage,
        suffix="weak",
        title="짧은 기억",
        summary="작은 선물을 받았다.",
        confidence=0.6,
    )

    gaps = MemoryGapDetectionService(gap_storage).detect_for_memories([memory])

    assert len(gaps) == 1
    assert gaps[0].gap_type is MemoryGapType.WEAK_PROVENANCE
