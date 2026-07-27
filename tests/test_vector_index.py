"""Temporary persistent-Chroma tests without real OpenAI calls."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.models.memory import DatePrecision
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import MemoryCreate, MemoryUpdate
from backend.app.storage.repository import SQLiteRepository


class FakeEmbeddings:
    """Small deterministic embedding provider for local tests."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("학교")) + 0.01,
            float(lowered.count("바다")) + 0.01,
            float(lowered.count("요리")) + 0.01,
        ]


@pytest.fixture
def vector_storage(tmp_path: Path):
    database_path = tmp_path / "memories.sqlite3"
    database = SQLiteDatabase(database_path)
    database.initialize()
    repository = SQLiteRepository(database)
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_001",
            filename="private.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 7, 27, tzinfo=UTC),
            content_hash="a" * 64,
            raw_content="원본",
            normalized_content="원본",
        )
    )
    embeddings = FakeEmbeddings()
    index = MemoryVectorIndex(
        repository,
        tmp_path / "chroma",
        embeddings=embeddings,
        embedding_version="fake-ko:v1",
    )
    yield repository, index, embeddings
    database.close()


def _memory(
    memory_id: str,
    *,
    title: str,
    summary: str,
) -> MemoryCreate:
    return MemoryCreate(
        memory_id=memory_id,
        transcript_id="tr_001",
        title=title,
        summary=summary,
        people=[],
        location=None,
        event_date=None,
        date_precision=DatePrecision.UNKNOWN,
        emotion=None,
        confidence=0.9,
    )


def test_indexes_memory_with_minimal_metadata_and_searches_by_id(
    vector_storage,
) -> None:
    repository, index, embeddings = vector_storage
    repository.create_memory(
        _memory(
            "mem_school",
            title="학교 운동회",
            summary="학교에서 친구들과 달리기를 했다.",
        )
    )
    repository.create_memory(
        _memory(
            "mem_beach",
            title="여름 바다",
            summary="가족과 바다에서 수영했다.",
        )
    )

    index.sync_from_sqlite()
    hits = index.similarity_search("학교 친구", top_k=1)
    metadata = index.get_metadata("mem_school")

    assert index.count == 2
    assert [hit.memory_id for hit in hits] == ["mem_school"]
    assert hits[0].memory == repository.get_memory("mem_school")
    assert metadata is not None
    assert set(metadata.model_dump()) == {
        "memory_id",
        "embedding_version",
        "content_hash",
    }
    assert metadata.memory_id == "mem_school"
    assert metadata.embedding_version == "fake-ko:v1"
    assert embeddings.query_calls == 1


def test_duplicate_index_is_skipped_without_embedding_again(vector_storage) -> None:
    repository, index, embeddings = vector_storage
    repository.create_memory(
        _memory("mem_001", title="학교", summary="학교에 입학했다.")
    )

    first = index.index_memory("mem_001")
    second = index.index_memory("mem_001")

    assert first.indexed is True
    assert second.indexed is False
    assert second.content_hash == first.content_hash
    assert embeddings.document_calls == 1
    assert index.count == 1


def test_changed_memory_is_stale_until_reindexed(vector_storage) -> None:
    repository, index, _embeddings = vector_storage
    repository.create_memory(
        _memory("mem_001", title="학교", summary="학교에 입학했다.")
    )
    original = index.index_memory("mem_001")
    repository.update_memory(
        "mem_001",
        MemoryUpdate(title="바다 여행", summary="바다에서 수영했다."),
    )

    assert index.similarity_search("학교", top_k=1) == []

    refreshed = index.index_memory("mem_001")
    hits = index.similarity_search("바다", top_k=1)

    assert refreshed.indexed is True
    assert refreshed.content_hash != original.content_hash
    assert index.count == 1
    assert [hit.memory_id for hit in hits] == ["mem_001"]
    assert hits[0].memory.title == "바다 여행"


def test_deleted_sqlite_memory_is_removed_from_chroma(vector_storage) -> None:
    repository, index, _embeddings = vector_storage
    repository.create_memory(
        _memory("mem_001", title="학교", summary="학교에 입학했다.")
    )
    index.index_memory("mem_001")

    repository.soft_delete_memory("mem_001")
    result = index.index_memory("mem_001")

    assert result.deleted is True
    assert index.count == 0
    assert index.get_metadata("mem_001") is None
    assert index.similarity_search("학교") == []


def test_rebuild_restores_deleted_chroma_vectors_from_sqlite(
    vector_storage,
) -> None:
    repository, index, _embeddings = vector_storage
    repository.create_memory(
        _memory("mem_school", title="학교", summary="학교 졸업식에 갔다.")
    )
    repository.create_memory(
        _memory("mem_cooking", title="요리", summary="가족과 요리를 배웠다.")
    )
    index.sync_from_sqlite()
    index.delete_memory("mem_school")
    index.delete_memory("mem_cooking")
    assert index.count == 0

    results = index.rebuild_from_sqlite()

    assert index.count == 2
    assert {result.memory_id for result in results} == {
        "mem_school",
        "mem_cooking",
    }
    assert all(result.indexed for result in results)
    assert index.similarity_search("요리", top_k=1)[0].memory_id == "mem_cooking"


def test_sync_removes_orphaned_vector_for_deleted_memory(vector_storage) -> None:
    repository, index, _embeddings = vector_storage
    repository.create_memory(
        _memory("mem_001", title="학교", summary="학교에 입학했다.")
    )
    index.index_memory("mem_001")
    repository.soft_delete_memory("mem_001")

    assert index.count == 1
    assert index.sync_from_sqlite() == []
    assert index.count == 0
