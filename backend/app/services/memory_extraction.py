"""Structured, evidence-checked memory extraction and persistence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from backend.app.models.memory import (
    DatePrecision,
    ExtractedMemory,
    LocalizedMemoryFields,
    MemoryExtractionBatch,
    MemoryExtractionProposal,
    MemoryExtractionProposalBatch,
    MemoryLocalizationBatch,
)
from backend.app.prompts.extraction import (
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    build_memory_extraction_input,
)
from backend.app.prompts.localization import (
    MEMORY_LOCALIZATION_SYSTEM_PROMPT,
    build_memory_localization_input,
)
from backend.app.storage.models import (
    MemoryCreate,
    MemoryRecord,
    MemorySourceCreate,
    TranscriptSegmentRecord,
)
from backend.app.storage.repository import SQLiteRepository

DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"

MemoryModelBatch = MemoryExtractionProposalBatch | MemoryExtractionBatch

_HANGUL_PATTERN = re.compile(r"[가-힣]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")


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
    *,
    api_key: str | None = None,
) -> StructuredMemoryModel:
    """Configure OpenAI native Structured Outputs through LangChain."""
    model_kwargs: dict[str, object] = {"model": model_name, "temperature": 0}
    if api_key:
        model_kwargs["api_key"] = api_key
    model = ChatOpenAI(**model_kwargs)
    return model.with_structured_output(
        MemoryExtractionProposalBatch,
        method="json_schema",
        include_raw=True,
        strict=True,
    )


def build_openai_memory_localization_model(
    model_name: str = DEFAULT_OPENAI_MODEL,
    *,
    api_key: str | None = None,
) -> StructuredMemoryModel:
    """Configure a separate schema that cannot modify evidence or dates."""

    model_kwargs: dict[str, object] = {"model": model_name, "temperature": 0}
    if api_key:
        model_kwargs["api_key"] = api_key
    model = ChatOpenAI(**model_kwargs)
    return model.with_structured_output(
        MemoryLocalizationBatch,
        method="json_schema",
        include_raw=True,
        strict=True,
    )


def _parse_model_output(output: object) -> MemoryModelBatch:
    if isinstance(output, (MemoryExtractionProposalBatch, MemoryExtractionBatch)):
        return output
    if not isinstance(output, Mapping):
        raise MemoryExtractionOutputError("Model returned an invalid output envelope")

    parsing_error = output.get("parsing_error")
    if parsing_error is not None:
        raise MemoryExtractionOutputError(
            "Model response could not be parsed as structured output"
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

    if isinstance(parsed, (MemoryExtractionProposalBatch, MemoryExtractionBatch)):
        return parsed
    try:
        return MemoryExtractionProposalBatch.model_validate(parsed)
    except ValidationError as proposal_exception:
        # Numeric-offset batches remain accepted for internal callers and test
        # doubles while production Structured Output uses evidence_text.
        try:
            return MemoryExtractionBatch.model_validate(parsed)
        except ValidationError:
            raise MemoryExtractionOutputError(
                "Model output did not match the memory schema"
            ) from proposal_exception


_RECOVERABLE_DATE_MESSAGES = frozenset(
    {
        "event_date does not match date_precision",
        "known date precision requires event_date",
        "unknown date precision requires a null event_date",
        "approximate dates require uncertainty_notes",
    }
)


def _is_recoverable_date_validation(exception: ValidationError) -> bool:
    """Return whether only model-supplied date metadata failed validation."""

    messages = {
        str(error.get("msg", "")).removeprefix("Value error, ")
        for error in exception.errors()
    }
    return bool(messages) and messages <= _RECOVERABLE_DATE_MESSAGES


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


def _normalize_evidence_offsets(
    candidate: ExtractedMemory,
    segment: TranscriptSegmentRecord,
) -> ExtractedMemory:
    """Normalize legacy numeric-offset batches used by internal callers."""

    start = candidate.evidence_start_offset
    end = candidate.evidence_end_offset
    segment_length = len(segment.content)
    if 0 <= start < end <= segment_length:
        return candidate
    if (
        segment.start_offset <= start < end <= segment.end_offset
    ):
        return candidate.model_copy(
            update={
                "evidence_start_offset": start - segment.start_offset,
                "evidence_end_offset": end - segment.start_offset,
            }
        )
    # Some legacy callers use one-based, inclusive-looking coordinates.
    # Convert only when the full range fits that coordinate system.
    if 1 <= start < end <= segment_length + 1:
        return candidate.model_copy(
            update={
                "evidence_start_offset": start - 1,
                "evidence_end_offset": end - 1,
            }
        )
    if (
        segment.start_offset + 1 <= start < end <= segment.end_offset + 1
    ):
        return candidate.model_copy(
            update={
                "evidence_start_offset": start - segment.start_offset - 1,
                "evidence_end_offset": end - segment.start_offset - 1,
            }
        )
    return candidate


def _resolve_evidence_text(
    proposal: MemoryExtractionProposal,
    segment: TranscriptSegmentRecord,
) -> ExtractedMemory:
    """Locate a verbatim model quote and derive trusted Python offsets."""

    evidence_text = proposal.evidence_text.strip()
    if not evidence_text:
        raise MemoryExtractionOutputError("Memory evidence must contain source text")
    start_offset = segment.content.find(evidence_text)
    if start_offset < 0:
        raise MemoryExtractionOutputError(
            "Memory evidence quote was not found in the transcript segment"
        )

    values = proposal.model_dump(exclude={"evidence_text"})
    values.update(
        {
            "evidence_start_offset": start_offset,
            "evidence_end_offset": start_offset + len(evidence_text),
        }
    )
    try:
        return ExtractedMemory.model_validate(values)
    except ValidationError as exception:
        if _is_recoverable_date_validation(exception):
            # A malformed model date must not discard an otherwise grounded
            # memory or abort the whole upload. Unknown is the conservative
            # fallback: it stores no unsupported date claim.
            values.update(
                {
                    "event_date": None,
                    "date_precision": DatePrecision.UNKNOWN,
                }
            )
            try:
                return ExtractedMemory.model_validate(values)
            except ValidationError:
                pass
        raise MemoryExtractionOutputError(
            "Model output did not match the memory schema"
        ) from exception


def _resolve_candidates(
    batch: MemoryModelBatch,
    segment: TranscriptSegmentRecord,
) -> list[ExtractedMemory]:
    candidates: list[ExtractedMemory] = []
    for raw_candidate in batch.memories:
        candidate = (
            _resolve_evidence_text(raw_candidate, segment)
            if isinstance(raw_candidate, MemoryExtractionProposal)
            else _normalize_evidence_offsets(raw_candidate, segment)
        )
        _validate_evidence(candidate, segment)
        candidates.append(candidate)
    return candidates


def _parse_localization_output(output: object) -> MemoryLocalizationBatch:
    """Validate the optional localizer without converting failures to ingest errors."""

    if isinstance(output, MemoryLocalizationBatch):
        return output
    if not isinstance(output, Mapping):
        raise ValueError("Localization returned an invalid output envelope")
    if output.get("parsing_error") is not None:
        raise ValueError("Localization response could not be parsed")
    parsed = output.get("parsed")
    if parsed is None:
        raise ValueError("Localization returned no parsed output")
    return MemoryLocalizationBatch.model_validate(parsed)


def _is_korean_language(language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().casefold().replace("_", "-")
    return (
        normalized in {"ko", "kor", "korean", "한국어"}
        or normalized.startswith("ko-")
    )


def _text_needs_korean_localization(value: str | None) -> bool:
    if value is None or not value.strip():
        return False
    if _LATIN_PATTERN.search(value):
        return True
    if _HANGUL_PATTERN.search(value):
        return False
    return any(character.isalpha() for character in value)


def _candidate_needs_korean_localization(candidate: ExtractedMemory) -> bool:
    values = (
        candidate.title,
        candidate.summary,
        *candidate.people,
        candidate.location,
        candidate.emotion,
        candidate.uncertainty_notes,
    )
    return any(_text_needs_korean_localization(value) for value in values)


def _localization_shape_matches(
    candidate: ExtractedMemory,
    localized: LocalizedMemoryFields,
    source_evidence: str,
) -> bool:
    """Reject localizer responses that add or remove optional facts."""

    if len(localized.people) != len(candidate.people):
        return False
    if any(
        (getattr(candidate, field) is None)
        != (getattr(localized, field) is None)
        for field in ("location", "emotion", "uncertainty_notes")
    ):
        return False
    if not (
        _HANGUL_PATTERN.search(localized.title)
        and _HANGUL_PATTERN.search(localized.summary)
    ):
        return False
    if not _HANGUL_PATTERN.search(source_evidence):
        return True

    translated_pairs = [
        (candidate.location, localized.location),
        (candidate.emotion, localized.emotion),
        (candidate.uncertainty_notes, localized.uncertainty_notes),
        *zip(candidate.people, localized.people, strict=True),
    ]
    return all(
        original is None
        or translated is None
        or not _text_needs_korean_localization(original)
        or _HANGUL_PATTERN.search(translated)
        for original, translated in translated_pairs
    )


def _localize_korean_display_fields(
    candidates: list[ExtractedMemory],
    segment: TranscriptSegmentRecord,
    language: str | None,
    localization_model: StructuredMemoryModel | None,
) -> list[ExtractedMemory]:
    """Localize display fields while keeping evidence and all facts immutable.

    Localization is deliberately best-effort. Extraction has already succeeded,
    so a refusal, network error, or invalid translation falls back to the original
    fields instead of failing the upload.
    """

    if localization_model is None or not _is_korean_language(language):
        return candidates

    requested_indexes = [
        index
        for index, candidate in enumerate(candidates)
        if _candidate_needs_korean_localization(candidate)
    ]
    if not requested_indexes:
        return candidates

    items: list[dict[str, object]] = []
    for index in requested_indexes:
        candidate = candidates[index]
        items.append(
            {
                "memory_index": index,
                "title": candidate.title,
                "summary": candidate.summary,
                "people": candidate.people,
                "location": candidate.location,
                "emotion": candidate.emotion,
                "uncertainty_notes": candidate.uncertainty_notes,
                "source_evidence": segment.content[
                    candidate.evidence_start_offset : candidate.evidence_end_offset
                ],
            }
        )

    messages = [
        SystemMessage(content=MEMORY_LOCALIZATION_SYSTEM_PROMPT),
        HumanMessage(content=build_memory_localization_input(items)),
    ]
    try:
        batch = _parse_localization_output(localization_model.invoke(messages))
    except Exception:
        return candidates

    localized_by_index = {
        localized.memory_index: localized for localized in batch.memories
    }
    if (
        len(localized_by_index) != len(batch.memories)
        or set(localized_by_index) != set(requested_indexes)
    ):
        return candidates
    if any(
        not _localization_shape_matches(
            candidates[index],
            localized_by_index[index],
            segment.content[
                candidates[index].evidence_start_offset :
                candidates[index].evidence_end_offset
            ],
        )
        for index in requested_indexes
    ):
        return candidates

    localized_candidates = list(candidates)
    for index in requested_indexes:
        localized = localized_by_index[index]
        localized_candidates[index] = candidates[index].model_copy(
            update={
                "title": localized.title,
                "summary": localized.summary,
                "people": localized.people,
                "location": localized.location,
                "emotion": localized.emotion,
                "uncertainty_notes": localized.uncertainty_notes,
            }
        )
    return localized_candidates


def _stable_id(prefix: str, values: Sequence[object]) -> str:
    identity = ":".join(str(value) for value in values).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(identity).hexdigest()[:24]}"


def _storage_items(
    batch: MemoryModelBatch,
    segment: TranscriptSegmentRecord,
    repository: SQLiteRepository,
    *,
    language: str | None = None,
    localization_model: StructuredMemoryModel | None = None,
) -> list[tuple[MemoryCreate, MemorySourceCreate]]:
    """Build storage items, dropping candidates chunk overlap already stored.

    Adjacent chunks share a ``chunk_overlap`` character window, so the same
    sentence can be handed to the model twice under two different segment_ids
    and come back as two structurally identical candidates. memory_id is a
    hash of (transcript_id, absolute evidence span, title, summary) with no
    segment_id or candidate index in it, so a duplicate here always hashes to
    an id that either repeats within this batch or already exists in SQLite
    from a previous chunk - either way it is dropped instead of stored twice.
    """

    candidates = _resolve_candidates(batch, segment)
    candidates = _localize_korean_display_fields(
        candidates,
        segment,
        language,
        localization_model,
    )
    event_dates_by_evidence: dict[tuple[int, int, str], set[str]] = {}
    for candidate in candidates:
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
    seen_memory_ids: set[str] = set()
    for candidate in candidates:
        absolute_start = segment.start_offset + candidate.evidence_start_offset
        absolute_end = segment.start_offset + candidate.evidence_end_offset
        memory_id = _stable_id(
            "mem",
            (
                segment.transcript_id,
                absolute_start,
                absolute_end,
                candidate.title,
                candidate.summary,
            ),
        )
        if memory_id in seen_memory_ids or repository.get_memory(memory_id):
            continue
        seen_memory_ids.add(memory_id)
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
    *,
    localization_model: StructuredMemoryModel | None = None,
) -> list[MemoryRecord]:
    """Extract validated memories from one stored segment and save atomically."""
    segment = repository.get_segment(segment_id)
    if segment is None:
        raise ExtractionSegmentNotFoundError("Active transcript segment was not found")

    extraction_input = build_memory_extraction_input(
        transcript_id=segment.transcript_id,
        segment_id=segment.segment_id,
        segment_content=segment.content,
    )
    messages = [
        SystemMessage(content=MEMORY_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=extraction_input),
    ]
    try:
        batch = _parse_model_output(model.invoke(messages))
    except MemoryExtractionError:
        raise
    except ValidationError as exception:
        raise MemoryExtractionOutputError(
            "Model output did not match the memory schema"
        ) from exception
    except Exception as exception:
        raise MemoryExtractionError("Memory extraction model call failed") from exception

    language: str | None = None
    if localization_model is not None:
        transcript = repository.get_transcript(segment.transcript_id)
        if transcript is not None:
            language = transcript.language
    items = _storage_items(
        batch,
        segment,
        repository,
        language=language,
        localization_model=localization_model,
    )
    return repository.create_memories_with_sources(items)
