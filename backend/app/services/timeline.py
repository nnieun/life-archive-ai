"""Chronological memory timeline built directly from SQLite."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime

from backend.app.models.memory import DatePrecision
from backend.app.models.timeline import TimelineEvent, TimelineResult
from backend.app.storage.models import CitationRecord, MemoryRecord, MemoryStatus
from backend.app.storage.repository import SQLiteRepository

_YEAR_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_DECADE_PATTERN = re.compile(r"(?<!\d)(\d{4})\s*년대")
_PRECISION_ORDER = {
    DatePrecision.EXACT: 0,
    DatePrecision.DAY: 1,
    DatePrecision.MONTH: 2,
    DatePrecision.YEAR: 3,
    DatePrecision.APPROXIMATE: 4,
    DatePrecision.UNKNOWN: 5,
}


@dataclass(frozen=True)
class _DatedEvent:
    event: TimelineEvent
    start: date
    end: date


class TimelineService:
    """Build a citation-bearing timeline without LangGraph or model calls."""

    def __init__(self, repository: SQLiteRepository) -> None:
        self._repository = repository

    def get_timeline(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> TimelineResult:
        """Return resolved memories in chronological and unknown-date groups."""

        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start_date must not follow end_date")

        memories = _prefer_corrections(self._repository.list_memories())
        dated: list[_DatedEvent] = []
        undated: list[TimelineEvent] = []
        for memory in memories:
            citations = self._citations(memory)
            if not citations:
                continue
            event = _timeline_event(memory, citations)
            interval = _date_interval(memory)
            if interval is None:
                if start_date is None and end_date is None:
                    undated.append(event)
                continue
            interval_start, interval_end = interval
            if start_date is not None and interval_end < start_date:
                continue
            if end_date is not None and interval_start > end_date:
                continue
            dated.append(
                _DatedEvent(
                    event=event,
                    start=interval_start,
                    end=interval_end,
                )
            )

        dated.sort(
            key=lambda item: (
                item.start,
                _PRECISION_ORDER[item.event.date_precision],
                item.event.event_date or "",
                item.event.memory_id,
            )
        )
        undated.sort(
            key=lambda event: (
                event.title.casefold(),
                event.memory_id,
            )
        )
        return TimelineResult(
            events=[item.event for item in dated],
            undated_events=undated,
            start_date=start_date,
            end_date=end_date,
        )

    def _citations(self, memory: MemoryRecord) -> list[CitationRecord]:
        return [
            CitationRecord(
                memory_id=source.memory_id,
                transcript_id=source.transcript_id,
                segment_id=source.segment_id,
                start_offset=source.start_offset,
                end_offset=source.end_offset,
            )
            for source in self._repository.list_memory_sources(memory.memory_id)
        ]


def _prefer_corrections(memories: list[MemoryRecord]) -> list[MemoryRecord]:
    """Remove active/corrected rows superseded by another visible correction."""

    superseded_ids = {
        memory.supersedes_memory_id
        for memory in memories
        if memory.status is MemoryStatus.CORRECTED
        and memory.supersedes_memory_id is not None
    }
    return [
        memory
        for memory in memories
        if memory.memory_id not in superseded_ids
    ]


def _timeline_event(
    memory: MemoryRecord,
    citations: list[CitationRecord],
) -> TimelineEvent:
    return TimelineEvent(
        memory_id=memory.memory_id,
        transcript_id=memory.transcript_id,
        title=memory.title,
        description=memory.summary,
        event_date=memory.event_date,
        date_precision=memory.date_precision,
        date_label=_date_label(memory),
        people=memory.people,
        location=memory.location,
        emotion=memory.emotion,
        confidence=memory.confidence,
        uncertainty_notes=memory.uncertainty_notes,
        status=memory.status,
        citations=citations,
    )


def _date_label(memory: MemoryRecord) -> str:
    if memory.date_precision is DatePrecision.UNKNOWN:
        return "날짜 미상"
    if memory.date_precision is DatePrecision.APPROXIMATE:
        return f"{memory.event_date} (추정)"
    return memory.event_date or "날짜 미상"


def _date_interval(memory: MemoryRecord) -> tuple[date, date] | None:
    value = memory.event_date
    if value is None or memory.date_precision is DatePrecision.UNKNOWN:
        return None
    try:
        if memory.date_precision is DatePrecision.EXACT:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            return parsed, parsed
        if memory.date_precision is DatePrecision.DAY:
            parsed = date.fromisoformat(value)
            return parsed, parsed
        if memory.date_precision is DatePrecision.MONTH:
            year, month = (int(part) for part in value.split("-"))
            return (
                date(year, month, 1),
                date(year, month, calendar.monthrange(year, month)[1]),
            )
        if memory.date_precision is DatePrecision.YEAR:
            year = int(value)
            return date(year, 1, 1), date(year, 12, 31)
        return _approximate_interval(value)
    except (TypeError, ValueError):
        return None


def _approximate_interval(value: str) -> tuple[date, date] | None:
    decade_match = _DECADE_PATTERN.search(value)
    if decade_match:
        year = int(decade_match.group(1))
        return date(year, 1, 1), date(year + 9, 12, 31)
    years = [int(year) for year in _YEAR_PATTERN.findall(value)]
    if not years:
        return None
    return date(min(years), 1, 1), date(max(years), 12, 31)
