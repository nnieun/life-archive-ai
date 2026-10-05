"""Deterministic memory-gap detection over validated stored memories."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from backend.app.models.gap import (
    MemoryGapCreate,
    MemoryGapRecord,
    MemoryGapSourceType,
    MemoryGapType,
)
from backend.app.models.memory import DatePrecision
from backend.app.storage.models import MemoryRecord
from backend.app.storage.repository import SQLiteRepository

_UNKNOWN_WORDS = (
    r"기억(?:이\s*)?나지|기억하지\s*못|모르|알\s*수\s*없|"
    r"불확실|미상|정확하지\s*않"
)
_SPECIFIC_LOCATION_UNKNOWN = re.compile(
    rf"(?:극장|영화관|가게|식당|카페|학교|병원|역|장소|곳|상호|지명)"
    rf".{{0,24}}(?:이름|명칭).{{0,24}}(?:{_UNKNOWN_WORDS})"
)
_LOCATION_CONTEXT = re.compile(
    r"어디|장소|근처|동네|지역|학교|회사|병원|극장|영화관|역|집|공원|"
    r"가게|식당|카페|에서|으로\s*갔|에\s*갔"
)
_SPECIFIC_PERSON_UNKNOWN = re.compile(
    rf"(?:사람|친구|동료|선생|가족|남자|여자)(?:의|\s)*"
    rf"(?:이름|누구).{{0,24}}(?:{_UNKNOWN_WORDS})"
)
_PERSON_CONTEXT = re.compile(
    r"누구|누군가|그\s*사람|친구|동료|선생|가족|부모|아버지|어머니|"
    r"형|누나|언니|오빠|동생|함께"
)
_DATE_UNKNOWN = re.compile(
    rf"(?:날짜|연도|몇\s*년|언제|시기).{{0,24}}(?:{_UNKNOWN_WORDS})"
)
_DATE_CONTEXT = re.compile(
    r"그때|당시|언젠가|어릴\s*때|시절|쯤|무렵|봄|여름|가을|겨울|"
    r"몇\s*년|년도|날짜|언제"
)
_UNCERTAINTY = re.compile(
    rf"(?:{_UNKNOWN_WORDS})|아마|아마도|확실하지|잘\s*모르|같기도|듯하"
)
_FOOD_PLACE_CLUE = re.compile(
    r"(?:떡볶이|분식|김밥|라면|순대).{0,32}(?:먹|집|가게|포장)",
)


class MemoryGapDetector(Protocol):
    def detect_for_memories(
        self,
        memories: Sequence[MemoryRecord],
    ) -> list[MemoryGapRecord]:
        """Detect and persist actionable gaps without changing memories."""


@dataclass(frozen=True)
class _GapSignal:
    gap_type: MemoryGapType
    missing_field: str | None
    confidence: float
    importance_score: float


class MemoryGapDetectionService:
    """Create conservative, idempotent gaps from grounded memory fields."""

    def __init__(self, repository: SQLiteRepository) -> None:
        self._repository = repository

    def detect_for_memories(
        self,
        memories: Sequence[MemoryRecord],
    ) -> list[MemoryGapRecord]:
        if not memories:
            return []
        all_memories = self._repository.list_memories()
        records: list[MemoryGapRecord] = []
        for memory in memories:
            source_count = len(
                self._repository.list_memory_sources(memory.memory_id)
            )
            for signal in self._signals(memory, all_memories, source_count):
                gap_id = _stable_gap_id(
                    memory.memory_id,
                    signal.gap_type,
                    signal.missing_field,
                )
                existing = self._repository.get_memory_gap(gap_id)
                if existing is not None:
                    records.append(existing)
                    continue
                clue_text = memory.summary.strip()
                if memory.uncertainty_notes:
                    clue_text = (
                        f"{clue_text}\n불확실한 점: "
                        f"{memory.uncertainty_notes.strip()}"
                    )
                records.append(
                    self._repository.create_memory_gap(
                        MemoryGapCreate(
                            gap_id=gap_id,
                            memory_id=memory.memory_id,
                            gap_type=signal.gap_type,
                            clue_text=clue_text,
                            missing_field=signal.missing_field,
                            period_start=memory.event_date,
                            period_end=memory.event_date,
                            location=memory.location,
                            people=memory.people,
                            confidence=signal.confidence,
                            importance_score=signal.importance_score,
                            source_type=MemoryGapSourceType.MEMORY,
                            source_id=memory.memory_id,
                        )
                    )
                )
        return records

    def detect_for_memory_ids(self, memory_ids: Sequence[str]) -> list[MemoryGapRecord]:
        memories = [
            memory
            for memory_id in memory_ids
            if (memory := self._repository.get_memory(memory_id)) is not None
        ]
        return self.detect_for_memories(memories)

    @staticmethod
    def _signals(
        memory: MemoryRecord,
        all_memories: Sequence[MemoryRecord],
        source_count: int,
    ) -> list[_GapSignal]:
        text = "\n".join(
            value
            for value in (memory.title, memory.summary, memory.uncertainty_notes)
            if value
        )
        signals: list[_GapSignal] = []

        if _FOOD_PLACE_CLUE.search(text):
            signals.append(
                _GapSignal(
                    MemoryGapType.MISSING_LOCATION,
                    "food_place_name",
                    0.78,
                    0.82,
                )
            )

        if _SPECIFIC_LOCATION_UNKNOWN.search(text):
            signals.append(
                _GapSignal(
                    MemoryGapType.MISSING_LOCATION,
                    "location_detail",
                    0.92,
                    0.9,
                )
            )
        elif memory.location is None and _LOCATION_CONTEXT.search(text):
            signals.append(
                _GapSignal(MemoryGapType.MISSING_LOCATION, "location", 0.72, 0.8)
            )

        if _SPECIFIC_PERSON_UNKNOWN.search(text):
            signals.append(
                _GapSignal(
                    MemoryGapType.MISSING_PERSON,
                    "person_detail",
                    0.9,
                    0.85,
                )
            )
        elif not memory.people and _PERSON_CONTEXT.search(text):
            signals.append(
                _GapSignal(MemoryGapType.MISSING_PERSON, "people", 0.68, 0.75)
            )

        if memory.event_date is None and (
            _DATE_UNKNOWN.search(text) or _DATE_CONTEXT.search(text)
        ):
            signals.append(
                _GapSignal(MemoryGapType.MISSING_DATE, "event_date", 0.7, 0.8)
            )
        elif memory.date_precision is DatePrecision.APPROXIMATE:
            signals.append(
                _GapSignal(MemoryGapType.MISSING_DATE, "event_date", 0.85, 0.85)
            )

        if _has_date_conflict(memory, all_memories):
            signals.append(
                _GapSignal(
                    MemoryGapType.CONFLICTING_FACT,
                    "event_date",
                    0.95,
                    1.0,
                )
            )

        has_actionable_signal = bool(signals)
        if not has_actionable_signal and (
            memory.uncertainty_notes is not None
            or memory.confidence < 0.5
            or _UNCERTAINTY.search(text)
        ):
            signals.append(
                _GapSignal(MemoryGapType.UNCERTAIN_EVENT, None, 0.75, 0.65)
            )
        elif (
            not has_actionable_signal
            and source_count <= 1
            and 0.5 <= memory.confidence < 0.7
        ):
            signals.append(
                _GapSignal(MemoryGapType.WEAK_PROVENANCE, None, 0.65, 0.55)
            )

        deduplicated: list[_GapSignal] = []
        seen: set[tuple[MemoryGapType, str | None]] = set()
        for signal in signals:
            key = (signal.gap_type, signal.missing_field)
            if key not in seen:
                seen.add(key)
                deduplicated.append(signal)
        return deduplicated


def _has_date_conflict(
    memory: MemoryRecord,
    all_memories: Sequence[MemoryRecord],
) -> bool:
    if memory.event_date is None:
        return False
    title = _normalized_title(memory.title)
    return any(
        other.memory_id != memory.memory_id
        and other.event_date is not None
        and other.event_date != memory.event_date
        and _normalized_title(other.title) == title
        for other in all_memories
    )


def _normalized_title(value: str) -> str:
    return " ".join(value.casefold().split())


def _stable_gap_id(
    memory_id: str,
    gap_type: MemoryGapType,
    missing_field: str | None,
) -> str:
    identity = f"{memory_id}:{gap_type.value}:{missing_field or ''}".encode("utf-8")
    return f"gap_{hashlib.sha256(identity).hexdigest()[:24]}"
