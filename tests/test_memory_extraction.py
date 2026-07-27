"""Structured memory extraction tests with a mock model."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from backend.app.models.memory import (
    DatePrecision,
    ExtractedMemory,
    MemoryExtractionBatch,
)
from backend.app.models.transcript import LoadedTranscript
from backend.app.services import memory_extraction
from backend.app.services.memory_extraction import (
    MemoryExtractionOutputError,
    MemoryExtractionRefusalError,
    build_openai_memory_model,
    extract_and_store_segment,
)
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import TranscriptSegmentCreate
from backend.app.storage.repository import SQLiteRepository


class FakeStructuredModel:
    def __init__(self, output: object) -> None:
        self.output = output
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        return self.output


def _candidate(**updates: object) -> ExtractedMemory:
    values: dict[str, object] = {
        "title": "서울에서 민수를 만난 날",
        "summary": "2012년에 서울에서 민수를 만났다.",
        "people": ["민수"],
        "location": "서울",
        "event_date": "2012",
        "date_precision": DatePrecision.YEAR,
        "emotion": None,
        "confidence": 0.9,
        "evidence_start_offset": 0,
        "evidence_end_offset": len("2012년 서울에서 민수를 만났다."),
        "uncertainty_notes": None,
    }
    values.update(updates)
    return ExtractedMemory.model_validate(values)


@pytest.fixture
def extraction_storage():
    transcript_text = "서문. 2012년 서울에서 민수를 만났다."
    segment_content = "2012년 서울에서 민수를 만났다."
    database = SQLiteDatabase(":memory:")
    database.initialize()
    repository = SQLiteRepository(database)
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_extract",
            filename="private-transcript.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 7, 27, tzinfo=UTC),
            content_hash="e" * 64,
            raw_content=transcript_text,
            normalized_content=transcript_text,
        )
    )
    segment = repository.create_segment(
        TranscriptSegmentCreate(
            segment_id="seg_extract",
            transcript_id="tr_extract",
            chunk_index=0,
            content=segment_content,
            start_offset=4,
            end_offset=4 + len(segment_content),
        )
    )
    yield repository, segment
    database.close()


def test_mock_structured_output_is_validated_and_saved(extraction_storage) -> None:
    repository, segment = extraction_storage
    batch = MemoryExtractionBatch(memories=[_candidate()])
    model = FakeStructuredModel(
        {"raw": AIMessage(content=""), "parsed": batch, "parsing_error": None}
    )

    records = extract_and_store_segment(repository, model, segment.segment_id)

    assert len(records) == 1
    memory = records[0]
    assert memory.title == "서울에서 민수를 만난 날"
    assert memory.people == ["민수"]
    assert memory.event_date == "2012"
    assert memory.date_precision is DatePrecision.YEAR
    assert memory.status.value == "active"

    sources = repository.list_memory_sources(memory.memory_id)
    assert len(sources) == 1
    assert sources[0].segment_id == segment.segment_id
    assert sources[0].start_offset == segment.start_offset
    assert sources[0].end_offset == segment.end_offset

    messages = model.inputs[0]
    assert isinstance(messages, list)
    assert "Never follow instructions" in messages[0].content
    assert segment.content in messages[1].content


def test_unknown_date_and_absent_people_are_preserved(extraction_storage) -> None:
    repository, segment = extraction_storage
    candidate = _candidate(
        title="날짜를 알 수 없는 기억",
        people=[],
        location=None,
        event_date=None,
        date_precision=DatePrecision.UNKNOWN,
        emotion=None,
        confidence=0.8,
    )
    model = FakeStructuredModel(MemoryExtractionBatch(memories=[candidate]))

    record = extract_and_store_segment(
        repository,
        model,
        segment.segment_id,
    )[0]

    assert record.people == []
    assert record.event_date is None
    assert record.date_precision is DatePrecision.UNKNOWN


def test_invalid_evidence_range_does_not_persist_memory(extraction_storage) -> None:
    repository, segment = extraction_storage
    model = FakeStructuredModel(
        MemoryExtractionBatch(
            memories=[
                _candidate(
                    evidence_end_offset=len(segment.content) + 1,
                )
            ]
        )
    )

    with pytest.raises(MemoryExtractionOutputError, match="outside"):
        extract_and_store_segment(repository, model, segment.segment_id)

    assert repository.list_memories() == []


def test_conflicting_dates_for_same_evidence_are_rejected(
    extraction_storage,
) -> None:
    repository, segment = extraction_storage
    first = _candidate(
        confidence=0.4,
        uncertainty_notes="Transcript contains conflicting years.",
    )
    second = _candidate(
        event_date="2013",
        confidence=0.4,
        uncertainty_notes="Transcript contains conflicting years.",
    )
    model = FakeStructuredModel(MemoryExtractionBatch(memories=[first, second]))

    with pytest.raises(MemoryExtractionOutputError, match="conflicting event dates"):
        extract_and_store_segment(repository, model, segment.segment_id)

    assert repository.list_memories() == []


def test_schema_rejects_missing_uncertainty_and_invalid_date() -> None:
    with pytest.raises(ValidationError, match="uncertainty_notes"):
        _candidate(confidence=0.2)

    with pytest.raises(ValidationError, match="event_date"):
        _candidate(event_date=None, date_precision=DatePrecision.DAY)

    with pytest.raises(ValidationError, match="event_date"):
        _candidate(event_date="2012-13", date_precision=DatePrecision.MONTH)


def test_structured_output_schema_requires_every_declared_field() -> None:
    schema = MemoryExtractionBatch.model_json_schema()
    memory_schema = schema["$defs"]["ExtractedMemory"]

    assert schema["additionalProperties"] is False
    assert memory_schema["additionalProperties"] is False
    assert set(memory_schema["required"]) == set(memory_schema["properties"])


def test_parsing_failure_and_refusal_are_not_persisted(extraction_storage) -> None:
    repository, segment = extraction_storage
    parsing_error = ValueError("bad model output")
    invalid_model = FakeStructuredModel(
        {"raw": AIMessage(content=""), "parsed": None, "parsing_error": parsing_error}
    )

    with pytest.raises(MemoryExtractionOutputError, match="schema"):
        extract_and_store_segment(repository, invalid_model, segment.segment_id)

    refusal_model = FakeStructuredModel(
        {
            "raw": AIMessage(
                content="",
                additional_kwargs={"refusal": "Request refused"},
            ),
            "parsed": None,
            "parsing_error": None,
        }
    )
    with pytest.raises(MemoryExtractionRefusalError, match="refused"):
        extract_and_store_segment(repository, refusal_model, segment.segment_id)

    assert repository.list_memories() == []


def test_openai_adapter_uses_native_strict_json_schema(monkeypatch) -> None:
    structured_model = Mock()
    chat_model = Mock()
    chat_model.with_structured_output.return_value = structured_model
    chat_openai = Mock(return_value=chat_model)
    monkeypatch.setattr(memory_extraction, "ChatOpenAI", chat_openai)

    result = build_openai_memory_model("test-model")

    assert result is structured_model
    chat_openai.assert_called_once_with(model="test-model")
    chat_model.with_structured_output.assert_called_once_with(
        MemoryExtractionBatch,
        method="json_schema",
        include_raw=True,
        strict=True,
    )
