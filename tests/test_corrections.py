"""Append-only memory corrections and the readers that must agree on them."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.memories import get_correction_service, get_memory_repository
from backend.app.main import app
from backend.app.models.memory import DatePrecision, MemoryCorrection
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.corrections import (
    MemoryAlreadyCorrectedError,
    MemoryCorrectionService,
    MemoryNotFoundError,
    UntraceableMemoryError,
)
from backend.app.models.vector import MemoryVectorSearchHit
from backend.app.services.retrieval import BM25MemoryIndex, HybridMemoryRetriever
from backend.app.services.timeline import TimelineService
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import (
    MemoryCreate,
    MemorySourceCreate,
    MemoryStatus,
    TranscriptSegmentCreate,
)
from backend.app.storage.repository import SQLiteRepository

CONTENT = "2012년 서울에서 민수를 만났다."


@pytest.fixture
def correction_storage(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "corrections.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_001",
            filename="private.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 7, 27, tzinfo=UTC),
            content_hash="f" * 64,
            raw_content=CONTENT,
            normalized_content=CONTENT,
        )
    )
    repository.create_segment(
        TranscriptSegmentCreate(
            segment_id="seg_001",
            transcript_id="tr_001",
            chunk_index=0,
            content=CONTENT,
            start_offset=0,
            end_offset=len(CONTENT),
        )
    )
    repository.create_memory(
        MemoryCreate(
            memory_id="mem_original",
            transcript_id="tr_001",
            title="서울에서 민수를 만난 날",
            summary="2012년에 서울에서 민수를 만났다.",
            people=["민수"],
            location="서울",
            event_date="2012",
            date_precision=DatePrecision.YEAR,
            emotion="반가움",
            confidence=0.6,
            uncertainty_notes="연도가 불확실함",
        )
    )
    repository.create_memory_source(
        MemorySourceCreate(
            memory_source_id="src_original",
            memory_id="mem_original",
            transcript_id="tr_001",
            segment_id="seg_001",
            start_offset=0,
            end_offset=len(CONTENT),
        )
    )
    yield repository
    database.close()


def test_correction_replaces_the_original_for_every_reader(
    correction_storage,
) -> None:
    repository = correction_storage
    service = MemoryCorrectionService(repository)

    corrected = service.correct_memory(
        "mem_original",
        MemoryCorrection(event_date="2013", date_precision=DatePrecision.YEAR),
    )

    assert corrected.status is MemoryStatus.CORRECTED
    assert corrected.supersedes_memory_id == "mem_original"
    # One storage-level filter is what keeps these four readers from disagreeing.
    assert [memory.memory_id for memory in repository.list_memories()] == [
        corrected.memory_id
    ]
    timeline = TimelineService(repository).get_timeline()
    assert [event.memory_id for event in timeline.events] == [corrected.memory_id]
    index = BM25MemoryIndex(repository)
    index.rebuild_from_sqlite()
    assert [hit.memory_id for hit in index.search("민수", top_k=5)] == [
        corrected.memory_id
    ]


class StaleDenseIndex:
    """A Chroma collection that has not yet dropped the superseded vector."""

    def __init__(self, repository: SQLiteRepository, memory_id: str) -> None:
        self._repository = repository
        self._memory_id = memory_id

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[MemoryVectorSearchHit]:
        memory = self._repository.get_memory(self._memory_id)
        if memory is None:
            return []
        return [
            MemoryVectorSearchHit(
                memory_id=memory.memory_id,
                distance=0.1,
                memory=memory,
            )
        ]


def test_stale_dense_vector_for_a_superseded_memory_is_not_retrieved(
    correction_storage,
) -> None:
    repository = correction_storage
    corrected = MemoryCorrectionService(repository).correct_memory(
        "mem_original",
        MemoryCorrection(title="부산에서 민수를 만난 날"),
    )
    retriever = HybridMemoryRetriever(
        repository,
        StaleDenseIndex(repository, "mem_original"),
        BM25MemoryIndex(repository),
    )

    # Chroma only drops the old vector on the next sync, so the retriever has
    # to honour the correction on its own.
    assert [hit.memory_id for hit in retriever.search("민수", top_k=5)] == [
        corrected.memory_id
    ]


def test_original_stays_auditable_after_being_superseded(correction_storage) -> None:
    repository = correction_storage
    MemoryCorrectionService(repository).correct_memory(
        "mem_original",
        MemoryCorrection(title="부산에서 민수를 만난 날"),
    )

    original = repository.get_memory("mem_original")
    assert original is not None
    assert original.title == "서울에서 민수를 만난 날"
    assert original.memory_id in {
        memory.memory_id
        for memory in repository.list_memories(include_superseded=True)
    }


def test_omitted_fields_inherit_and_explicit_null_clears(correction_storage) -> None:
    repository = correction_storage
    corrected = MemoryCorrectionService(repository).correct_memory(
        "mem_original",
        MemoryCorrection(title="민수를 만난 날", emotion=None),
    )

    assert corrected.title == "민수를 만난 날"
    assert corrected.emotion is None
    assert corrected.people == ["민수"]
    assert corrected.location == "서울"
    assert corrected.event_date == "2012"
    assert corrected.date_precision is DatePrecision.YEAR
    assert corrected.confidence == 1.0


def test_correction_inherits_the_transcript_sources(correction_storage) -> None:
    repository = correction_storage
    corrected = MemoryCorrectionService(repository).correct_memory(
        "mem_original",
        MemoryCorrection(title="민수를 만난 날"),
    )

    sources = repository.list_memory_sources(corrected.memory_id)
    original_sources = repository.list_memory_sources("mem_original")
    assert len(sources) == 1
    assert sources[0].memory_source_id != original_sources[0].memory_source_id
    assert sources[0].segment_id == "seg_001"
    assert sources[0].start_offset == original_sources[0].start_offset
    assert sources[0].end_offset == original_sources[0].end_offset


def test_a_memory_can_only_be_corrected_once(correction_storage) -> None:
    repository = correction_storage
    service = MemoryCorrectionService(repository)
    service.correct_memory("mem_original", MemoryCorrection(title="첫 번째 정정"))

    with pytest.raises(MemoryAlreadyCorrectedError):
        service.correct_memory("mem_original", MemoryCorrection(title="두 번째 정정"))


def test_unknown_and_untraceable_memories_are_rejected(correction_storage) -> None:
    repository = correction_storage
    service = MemoryCorrectionService(repository)
    repository.create_memory(
        MemoryCreate(
            memory_id="mem_no_source",
            transcript_id="tr_001",
            title="출처 없는 기억",
            summary="출처가 없다.",
            people=[],
            confidence=0.9,
        )
    )

    with pytest.raises(MemoryNotFoundError):
        service.correct_memory("mem_missing", MemoryCorrection(title="정정"))
    with pytest.raises(UntraceableMemoryError):
        service.correct_memory("mem_no_source", MemoryCorrection(title="정정"))


def test_correction_model_rejects_empty_and_half_specified_dates() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        MemoryCorrection()
    with pytest.raises(ValueError, match="corrected together"):
        MemoryCorrection(event_date="2013")
    with pytest.raises(ValueError, match="does not match"):
        MemoryCorrection(event_date="2013-13", date_precision=DatePrecision.MONTH)


def test_correction_api_returns_citations_and_maps_failures(
    correction_storage,
) -> None:
    repository = correction_storage
    app.dependency_overrides[get_memory_repository] = lambda: repository
    app.dependency_overrides[get_correction_service] = lambda: MemoryCorrectionService(
        repository
    )
    client = TestClient(app)
    try:
        created = client.post(
            "/api/v1/memories/mem_original/corrections",
            json={"title": "부산에서 민수를 만난 날"},
        )
        conflict = client.post(
            "/api/v1/memories/mem_original/corrections",
            json={"title": "또 다른 정정"},
        )
        missing = client.post(
            "/api/v1/memories/mem_missing/corrections",
            json={"title": "정정"},
        )
        invalid = client.post(
            "/api/v1/memories/mem_original/corrections",
            json={"event_date": "2013"},
        )
        listed = client.get("/api/v1/memories")
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 201
    body = created.json()
    assert body["memory"]["title"] == "부산에서 민수를 만난 날"
    assert body["memory"]["supersedes_memory_id"] == "mem_original"
    assert body["citations"][0]["segment_id"] == "seg_001"
    assert conflict.status_code == 409
    assert missing.status_code == 404
    assert invalid.status_code == 422
    assert [view["memory"]["memory_id"] for view in listed.json()] == [
        body["memory"]["memory_id"]
    ]
