"""Privacy-safe transcript deletion across source-of-truth and indexes."""

from __future__ import annotations

from typing import Protocol

from backend.app.models.privacy import TranscriptDeletionResult
from backend.app.storage.repository import SQLiteRepository, StorageNotFoundError


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
        """Purge the vector index first, then commit SQLite, then rebuild BM25.

        Every step is ordered so a failure leaves a state the caller can retry.
        Purging vectors before the commit means a vector failure changes nothing
        in SQLite, and a commit failure only costs dense recall that the next
        ``sync_from_sqlite`` restores. Deleting SQLite first would strand the
        title and summary in Chroma with no way back in: the retry would find
        the transcript already gone and answer 404 forever.

        The whole call is idempotent, so a caller that saw an error can simply
        send the same DELETE again until it succeeds.
        """

        transcript = self._repository.get_transcript(
            transcript_id,
            include_deleted=True,
        )
        if transcript is None:
            raise StorageNotFoundError("Transcript was not found")

        memory_ids = [
            memory.memory_id
            for memory in self._repository.list_memories(
                transcript_id,
                include_deleted=True,
                include_superseded=True,
            )
        ]
        try:
            deleted_vector_count = sum(
                self._vector_index.delete_memory(memory_id)
                for memory_id in memory_ids
            )
        except Exception as exception:
            raise PrivacyDeletionError(
                "Vector purge failed before SQLite was changed"
            ) from exception

        deleted_segment_count = 0
        deleted_memory_count = 0
        dismissed_gap_count = 0
        invalidated_message_count = 0
        invalidated_autobiography_count = 0
        if transcript.deleted_at is None:
            sqlite_result = self._repository.soft_delete_transcript_cascade(
                transcript_id
            )
            deleted_segment_count = sqlite_result.deleted_segment_count
            deleted_memory_count = sqlite_result.deleted_memory_count
            dismissed_gap_count = sqlite_result.dismissed_gap_count
            invalidated_message_count = (
                sqlite_result.invalidated_conversation_message_count
            )
            invalidated_autobiography_count = (
                sqlite_result.invalidated_autobiography_count
            )

        try:
            self._bm25_index.rebuild_from_sqlite()
        except Exception as exception:
            raise PrivacyDeletionError(
                "SQLite deletion succeeded but keyword index cleanup failed"
            ) from exception

        return TranscriptDeletionResult(
            transcript_id=transcript_id,
            deleted_segment_count=deleted_segment_count,
            deleted_memory_count=deleted_memory_count,
            dismissed_gap_count=dismissed_gap_count,
            deleted_vector_count=deleted_vector_count,
            bm25_memory_count=self._bm25_index.count,
            invalidated_conversation_message_count=invalidated_message_count,
            invalidated_autobiography_count=invalidated_autobiography_count,
            raw_file_deleted=False,
        )
