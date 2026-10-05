"""Append-only memory corrections that keep the original evidence traceable."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from backend.app.models.memory import MemoryCorrection
from backend.app.storage.models import (
    MemoryCreate,
    MemoryRecord,
    MemorySourceCreate,
    MemoryStatus,
)
from backend.app.storage.repository import SQLiteRepository

HUMAN_CORRECTION_CONFIDENCE = 1.0


class MemoryCorrectionError(RuntimeError):
    """Base class for correction failures."""


class MemoryNotFoundError(MemoryCorrectionError):
    """The memory to correct does not exist or was deleted."""


class MemoryAlreadyCorrectedError(MemoryCorrectionError):
    """The memory was already replaced by a live correction."""


class UntraceableMemoryError(MemoryCorrectionError):
    """The memory carries no transcript source to inherit."""


@dataclass(frozen=True)
class PreparedMemoryCorrection:
    """Validated append-only correction ready for one atomic repository write."""

    memory: MemoryCreate
    sources: tuple[MemorySourceCreate, ...]


class MemoryCorrectionService:
    """Replace a memory by appending a correction instead of editing in place."""

    def __init__(self, repository: SQLiteRepository) -> None:
        self._repository = repository

    def correct_memory(
        self,
        memory_id: str,
        correction: MemoryCorrection,
    ) -> MemoryRecord:
        """Store a corrected copy that supersedes ``memory_id``.

        The original row stays in SQLite so the extraction remains auditable;
        ``list_memories`` is what hides it from timeline, retrieval and search
        once the correction exists.
        """

        prepared = self.prepare_memory_correction(memory_id, correction)
        return self._repository.create_memory_correction(
            prepared.memory,
            list(prepared.sources),
        )

    def prepare_memory_correction(
        self,
        memory_id: str,
        correction: MemoryCorrection,
    ) -> PreparedMemoryCorrection:
        """Validate and build a correction without writing it.

        Gap resolution uses this form so the correction and candidate status
        transition can be committed in the same SQLite transaction.
        """

        original = self._repository.get_memory(memory_id)
        if original is None:
            raise MemoryNotFoundError("Memory to correct was not found")
        if self._repository.get_correction_of(memory_id) is not None:
            raise MemoryAlreadyCorrectedError("Memory was already corrected")
        sources = self._repository.list_memory_sources(memory_id)
        if not sources:
            raise UntraceableMemoryError("Memory has no transcript source to inherit")

        changed = correction.model_fields_set
        corrected_id = f"mem_{uuid4().hex[:24]}"

        def value(field: str) -> object:
            return (
                getattr(correction, field)
                if field in changed
                else getattr(original, field)
            )

        record = MemoryCreate(
            memory_id=corrected_id,
            transcript_id=original.transcript_id,
            title=value("title"),
            summary=value("summary"),
            people=value("people"),
            location=value("location"),
            event_date=value("event_date"),
            date_precision=value("date_precision"),
            emotion=value("emotion"),
            confidence=HUMAN_CORRECTION_CONFIDENCE,
            uncertainty_notes=value("uncertainty_notes"),
            status=MemoryStatus.CORRECTED,
            supersedes_memory_id=memory_id,
        )
        return PreparedMemoryCorrection(
            memory=record,
            sources=tuple(
                MemorySourceCreate(
                    memory_source_id=f"src_{uuid4().hex[:24]}",
                    memory_id=corrected_id,
                    transcript_id=source.transcript_id,
                    segment_id=source.segment_id,
                    start_offset=source.start_offset,
                    end_offset=source.end_offset,
                )
                for source in sources
            ),
        )
