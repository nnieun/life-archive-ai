"""Structured, evidence-checked memory extraction and persistence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from backend.app.models.memory import ExtractedMemory, MemoryExtractionBatch
from backend.app.prompts.extraction import (
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    build_memory_extraction_input,
)
from backend.app.storage.models import (
    MemoryCreate,
    MemoryRecord,
    MemorySourceCreate,
    TranscriptSegmentRecord,
)
from backend.app.storage.repository import SQLiteRepository

DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"


class StructuredMemoryModel(Protocol):
    """Minimal interface implemented by LangChain and test doubles."""

    def invoke(self, input: object) -> object:
        """Return a structured extraction response."""


class MemoryExtractionError(RuntimeError):
    """Base class for privacy-safe extraction failures."""


class MemoryExtractionOutputError(MemoryExtractionError):
    """The model output or evidence range failed validation."""


class MemoryExtractionRefusalError(MemoryExtractionError):
    """The model refused the extraction request."""


class ExtractionSegmentNotFoundError(MemoryExtractionError):
    """The requested active SQLite segment does not exist."""


def build_openai_memory_model(
    model_name: str = DEFAULT_OPENAI_MODEL,
) -> StructuredMemoryModel:
    """Configure OpenAI native Structured Outputs through LangChain."""
    model = ChatOpenAI(model=model_name)
    return model.with_structured_output(
        MemoryExtractionBatch,
        method="json_schema",
        include_raw=True,
        strict=True,
    )


def _parse_model_output(output: object) -> MemoryExtractionBatch:
    if isinstance(output, MemoryExtractionBatch):
        return output
    if not isinstance(output, Mapping):
        raise MemoryExtractionOutputError("Model returned an invalid output envelope")

    parsing_error = output.get("parsing_error")
    if parsing_error is not None:
        raise MemoryExtractionOutputError(
            "Model output did not match the memory schema"
        ) from parsing_error

    parsed = output.get("parsed")
    if parsed is None:
        raw = output.get("raw")
        additional_kwargs = getattr(raw, "additional_kwargs", {})
        refusal = (
            additional_kwargs.get("refusal")
            if isinstance(additional_kwargs, Mapping)
            else None
        )
        if refusal:
            raise MemoryExtractionRefusalError("Model refused memory extraction")
        raise MemoryExtractionOutputError("Model returned no parsed memories")

    try:
        return (
            parsed
            if isinstance(parsed, MemoryExtractionBatch)
            else MemoryExtractionBatch.model_validate(parsed)
        )
    except ValidationError as exception:
        raise MemoryExtractionOutputError(
            "Model output did not match the memory schema"
        ) from exception


def _validate_evidence(
    candidate: ExtractedMemory,
    segment: TranscriptSegmentRecord,
) -> None:
    if candidate.evidence_end_offset > len(segment.content):
        raise MemoryExtractionOutputError(
            "Memory evidence falls outside the transcript segment"
        )
    evidence = segment.content[
        candidate.evidence_start_offset : candidate.evidence_end_offset
    ]
    if not evidence.strip():
        raise MemoryExtractionOutputError("Memory evidence must contain source text")


def _stable_id(prefix: str, values: Sequence[object]) -> str:
    identity = ":".join(str(value) for value in values).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(identity).hexdigest()[:24]}"


def _storage_items(
    batch: MemoryExtractionBatch,
    segment: TranscriptSegmentRecord,
) -> list[tuple[MemoryCreate, MemorySourceCreate]]:
    event_dates_by_evidence: dict[tuple[int, int, str], set[str]] = {}
    for candidate in batch.memories:
        if candidate.event_date is None:
            continue
        evidence_key = (
            candidate.evidence_start_offset,
            candidate.evidence_end_offset,
            candidate.title.casefold(),
        )
        event_dates_by_evidence.setdefault(evidence_key, set()).add(
            candidate.event_date
        )
    if any(len(event_dates) > 1 for event_dates in event_dates_by_evidence.values()):
        raise MemoryExtractionOutputError(
            "Model returned conflicting event dates for the same evidence"
        )

    items: list[tuple[MemoryCreate, MemorySourceCreate]] = []
    for candidate_index, candidate in enumerate(batch.memories):
        _validate_evidence(candidate, segment)
        absolute_start = segment.start_offset + candidate.evidence_start_offset
        absolute_end = segment.start_offset + candidate.evidence_end_offset
        memory_id = _stable_id(
            "mem",
            (
                segment.transcript_id,
                segment.segment_id,
                candidate_index,
                absolute_start,
                absolute_end,
                candidate.title,
                candidate.summary,
            ),
        )
        source_id = _stable_id(
            "src",
            (memory_id, segment.segment_id, absolute_start, absolute_end),
        )
        items.append(
            (
                MemoryCreate(
                    memory_id=memory_id,
                    transcript_id=segment.transcript_id,
                    title=candidate.title,
                    summary=candidate.summary,
                    people=candidate.people,
                    location=candidate.location,
                    event_date=candidate.event_date,
                    date_precision=candidate.date_precision,
                    emotion=candidate.emotion,
                    confidence=candidate.confidence,
                    uncertainty_notes=candidate.uncertainty_notes,
                ),
                MemorySourceCreate(
                    memory_source_id=source_id,
                    memory_id=memory_id,
                    transcript_id=segment.transcript_id,
                    segment_id=segment.segment_id,
                    start_offset=absolute_start,
                    end_offset=absolute_end,
                ),
            )
        )
    return items


def extract_and_store_segment(
    repository: SQLiteRepository,
    model: StructuredMemoryModel,
    segment_id: str,
) -> list[MemoryRecord]:
    """Extract validated memories from one stored segment and save atomically."""
    segment = repository.get_segment(segment_id)
    if segment is None:
        raise ExtractionSegmentNotFoundError("Active transcript segment was not found")

    messages = [
        SystemMessage(content=MEMORY_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(
            content=build_memory_extraction_input(
                transcript_id=segment.transcript_id,
                segment_id=segment.segment_id,
                segment_start_offset=segment.start_offset,
                segment_content=segment.content,
            )
        ),
    ]
    try:
        batch = _parse_model_output(model.invoke(messages))
    except MemoryExtractionError:
        raise
    except Exception as exception:
        raise MemoryExtractionError("Memory extraction model call failed") from exception

    items = _storage_items(batch, segment)
    return repository.create_memories_with_sources(items)
