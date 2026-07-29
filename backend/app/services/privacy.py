"""Privacy-safe transcript deletion across source-of-truth and indexes."""

from __future__ import annotations

from typing import Protocol

from backend.app.models.privacy import TranscriptDeletionResult
from backend.app.storage.repository import SQLiteRepository


class VectorDeletionIndex(Protocol):
    def delete_memory(self, memory_id: str) -> bool:
        """Delete one disposable vector if it exists."""


class KeywordRebuildIndex(Protocol):
    @property
    def count(self) -> int:
        """Return active keyword documents."""

    def rebuild_from_sqlite(self) -> object:
        """Recreate the disposable keyword index from SQLite."""


class PrivacyDeletionError(RuntimeError):
    """Disposable index cleanup failed after SQLite was made safe."""


class TranscriptDeletionService:
    """Delete derived access while deliberately preserving raw originals."""

    def __init__(
        self,
        repository: SQLiteRepository,
        vector_index: VectorDeletionIndex,
        bm25_index: KeywordRebuildIndex,
    ) -> None:
        self._repository = repository
        self._vector_index = vector_index
        self._bm25_index = bm25_index

    def delete_transcript(self, transcript_id: str) -> TranscriptDeletionResult:
        """Logically delete SQLite data, then purge disposable indexes."""

        sqlite_result = self._repository.soft_delete_transcript_cascade(
            transcript_id
        )
        try:
            deleted_vector_count = sum(
                self._vector_index.delete_memory(memory_id)
                for memory_id in sqlite_result.memory_ids
            )
            self._bm25_index.rebuild_from_sqlite()
        except Exception as exception:
            raise PrivacyDeletionError(
                "SQLite deletion succeeded but index cleanup failed"
            ) from exception

        return TranscriptDeletionResult(
            transcript_id=transcript_id,
            deleted_segment_count=sqlite_result.deleted_segment_count,
            deleted_memory_count=sqlite_result.deleted_memory_count,
            deleted_vector_count=deleted_vector_count,
            bm25_memory_count=self._bm25_index.count,
            invalidated_conversation_message_count=(
                sqlite_result.invalidated_conversation_message_count
            ),
            invalidated_autobiography_count=(
                sqlite_result.invalidated_autobiography_count
            ),
            raw_file_deleted=False,
        )
