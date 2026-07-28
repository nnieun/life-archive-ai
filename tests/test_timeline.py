"""Temporary SQLite timeline service and API tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.timeline import get_timeline_service
from backend.app.main import app
from backend.app.models.memory import DatePrecision
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.timeline import TimelineService
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import (
    MemoryCreate,
    MemorySourceCreate,
    MemoryStatus,
    TranscriptSegmentCreate,
)
from backend.app.storage.repository import SQLiteRepository


@pytest.fixture
def timeline_storage(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "timeline.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    content = "가" * 200
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_001",
            filename="private.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 7, 27, tzinfo=UTC),
            content_hash="d" * 64,
            raw_content=content,
            normalized_content=content,
        )
    )
    repository.create_segments(
        [
            TranscriptSegmentCreate(
                segment_id="seg_001",
                transcript_id="tr_001",
                chunk_index=0,
                content=content,
                start_offset=0,
                end_offset=len(content),
            )
        ]
    )
    yield repository
    database.close()


def _add_memory(
    repository: SQLiteRepository,
    memory_id: str,
    *,
    title: str,
    event_date: str | None,
    precision: DatePrecision,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    supersedes_memory_id: str | None = None,
    with_source: bool = True,
) -> None:
    repository.create_memory(
        MemoryCreate(
            memory_id=memory_id,
            transcript_id="tr_001",
            title=title,
            summary=f"{title}에 대한 기억",
            people=[],
            location=None,
            event_date=event_date,
            date_precision=precision,
            emotion=None,
            confidence=0.8,
            uncertainty_notes=(
                "정확한 시점은 불확실함"
                if precision is DatePrecision.APPROXIMATE
                else None
            ),
            status=status,
            supersedes_memory_id=supersedes_memory_id,
        )
    )
    if with_source:
        repository.create_memory_source(
            MemorySourceCreate(
                memory_source_id=f"src_{memory_id}",
                memory_id=memory_id,
                transcript_id="tr_001",
                segment_id="seg_001",
                start_offset=0,
                end_offset=20,
            )
        )


def test_timeline_sorts_date_precisions_and_separates_unknown(
    timeline_storage,
) -> None:
    repository = timeline_storage
    _add_memory(
        repository,
        "mem_exact",
        title="정확한 시각",
        event_date="2012-05-03T10:30:00+09:00",
        precision=DatePrecision.EXACT,
    )
    _add_memory(
        repository,
        "mem_month",
        title="월 단위 기억",
        event_date="2005-08",
        precision=DatePrecision.MONTH,
    )
    _add_memory(
        repository,
        "mem_year",
        title="연도 단위 기억",
        event_date="2001",
        precision=DatePrecision.YEAR,
    )
    _add_memory(
        repository,
        "mem_approximate",
        title="대략적인 기억",
        event_date="1990년대 초",
        precision=DatePrecision.APPROXIMATE,
    )
    _add_memory(
        repository,
        "mem_unknown",
        title="날짜 없는 기억",
        event_date=None,
        precision=DatePrecision.UNKNOWN,
    )

    result = TimelineService(repository).get_timeline()

    assert [event.memory_id for event in result.events] == [
        "mem_approximate",
        "mem_year",
        "mem_month",
        "mem_exact",
    ]
    assert result.events[0].date_label == "1990년대 초 (추정)"
    assert result.events[2].date_label == "2005-08"
    assert [event.memory_id for event in result.undated_events] == [
        "mem_unknown"
    ]
    assert result.undated_events[0].date_label == "날짜 미상"
    assert result.total == 5


def test_timeline_prefers_corrected_and_excludes_deleted(timeline_storage) -> None:
    repository = timeline_storage
    _add_memory(
        repository,
        "mem_original",
        title="원래 기억",
        event_date="2010",
        precision=DatePrecision.YEAR,
    )
    _add_memory(
        repository,
        "mem_corrected",
        title="정정된 기억",
        event_date="2011",
        precision=DatePrecision.YEAR,
        status=MemoryStatus.CORRECTED,
        supersedes_memory_id="mem_original",
    )
    _add_memory(
        repository,
        "mem_deleted",
        title="삭제된 기억",
        event_date="2012",
        precision=DatePrecision.YEAR,
        status=MemoryStatus.DELETED,
    )

    result = TimelineService(repository).get_timeline()

    assert [event.memory_id for event in result.events] == ["mem_corrected"]
    assert result.events[0].status is MemoryStatus.CORRECTED


def test_same_date_events_have_deterministic_order_and_citations(
    timeline_storage,
) -> None:
    repository = timeline_storage
    _add_memory(
        repository,
        "mem_b",
        title="두 번째 제목",
        event_date="2015-01-02",
        precision=DatePrecision.DAY,
    )
    _add_memory(
        repository,
        "mem_a",
        title="첫 번째 제목",
        event_date="2015-01-02",
        precision=DatePrecision.DAY,
    )

    events = TimelineService(repository).get_timeline().events

    assert [event.memory_id for event in events] == ["mem_a", "mem_b"]
    assert events[0].citations[0].transcript_id == "tr_001"
    assert events[0].citations[0].segment_id == "seg_001"
    assert events[0].citations[0].start_offset == 0
    assert events[0].citations[0].end_offset == 20


def test_date_range_uses_partial_date_intervals_and_hides_unknown(
    timeline_storage,
) -> None:
    repository = timeline_storage
    _add_memory(
        repository,
        "mem_year",
        title="2012년 기억",
        event_date="2012",
        precision=DatePrecision.YEAR,
    )
    _add_memory(
        repository,
        "mem_month",
        title="2013년 2월 기억",
        event_date="2013-02",
        precision=DatePrecision.MONTH,
    )
    _add_memory(
        repository,
        "mem_unknown",
        title="날짜 미상",
        event_date=None,
        precision=DatePrecision.UNKNOWN,
    )
    service = TimelineService(repository)

    result = service.get_timeline(
        start_date=date(2012, 6, 1),
        end_date=date(2012, 6, 30),
    )

    assert [event.memory_id for event in result.events] == ["mem_year"]
    assert result.undated_events == []
    assert result.start_date == date(2012, 6, 1)
    assert result.end_date == date(2012, 6, 30)


def test_unparseable_approximate_date_is_preserved_as_undated(
    timeline_storage,
) -> None:
    repository = timeline_storage
    _add_memory(
        repository,
        "mem_approximate",
        title="어린 시절 기억",
        event_date="어린 시절",
        precision=DatePrecision.APPROXIMATE,
    )

    result = TimelineService(repository).get_timeline()

    assert result.events == []
    assert result.undated_events[0].memory_id == "mem_approximate"
    assert result.undated_events[0].event_date == "어린 시절"


def test_memory_without_source_is_not_returned(timeline_storage) -> None:
    repository = timeline_storage
    _add_memory(
        repository,
        "mem_untraceable",
        title="출처 없는 기억",
        event_date="2020",
        precision=DatePrecision.YEAR,
        with_source=False,
    )

    result = TimelineService(repository).get_timeline()

    assert result.total == 0


def test_timeline_rejects_reversed_date_range(timeline_storage) -> None:
    with pytest.raises(ValueError, match="start_date"):
        TimelineService(timeline_storage).get_timeline(
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 1),
        )


def test_timeline_api_returns_filtered_events(timeline_storage) -> None:
    repository = timeline_storage
    _add_memory(
        repository,
        "mem_2012",
        title="2012년 기억",
        event_date="2012",
        precision=DatePrecision.YEAR,
    )
    service = TimelineService(repository)
    app.dependency_overrides[get_timeline_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/api/v1/timeline",
            json={
                "start_date": "2012-01-01",
                "end_date": "2012-12-31",
            },
        )
        invalid = client.post(
            "/api/v1/timeline",
            json={
                "start_date": "2013-01-01",
                "end_date": "2012-12-31",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["events"][0]["memory_id"] == "mem_2012"
    assert response.json()["events"][0]["citations"][0]["memory_id"] == "mem_2012"
    assert response.json()["undated_events"] == []
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
