"""Focused validation regressions for memory dates and citations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.models.memory import DatePrecision, ExtractedMemory
from backend.app.storage.models import CitationRecord, MemoryCreate


@pytest.mark.parametrize(
    ("event_date", "precision", "uncertainty"),
    [
        ("2020-01-02T03:04:05+09:00", DatePrecision.EXACT, None),
        ("2020-01-02", DatePrecision.DAY, None),
        ("2020-01", DatePrecision.MONTH, None),
        ("2020", DatePrecision.YEAR, None),
        ("2020년 무렵", DatePrecision.APPROXIMATE, "연도는 대략적이다."),
        (None, DatePrecision.UNKNOWN, None),
    ],
)
def test_all_supported_date_precisions_validate(
    event_date: str | None,
    precision: DatePrecision,
    uncertainty: str | None,
) -> None:
    memory = ExtractedMemory(
        title="합성 기억",
        summary="검증을 위한 합성 기억이다.",
        people=[],
        location=None,
        event_date=event_date,
        date_precision=precision,
        emotion=None,
        confidence=0.9,
        evidence_start_offset=0,
        evidence_end_offset=5,
        uncertainty_notes=uncertainty,
    )

    assert memory.date_precision is precision


def test_memory_and_citation_reject_inconsistent_ranges() -> None:
    with pytest.raises(ValidationError, match="event_date"):
        MemoryCreate(
            memory_id="mem_invalid",
            transcript_id="tr_invalid",
            title="잘못된 기억",
            summary="날짜 정밀도와 값이 일치하지 않는다.",
            people=[],
            event_date="2020",
            date_precision=DatePrecision.UNKNOWN,
            confidence=0.9,
        )

    with pytest.raises(ValidationError, match="end_offset"):
        CitationRecord(
            memory_id="mem_invalid",
            transcript_id="tr_invalid",
            start_offset=10,
            end_offset=9,
        )
