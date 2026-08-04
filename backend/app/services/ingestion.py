"""Immutable TXT upload, extraction, and indexing orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from backend.app.models.ingestion import IngestionResult
from backend.app.models.transcript import TranscriptLoadRequest
from backend.app.services.chunking import chunk_and_store_transcript
from backend.app.services.memory_extraction import (
    StructuredMemoryModel,
    extract_and_store_segment,
)
from backend.app.services.transcript_loader import (
    TranscriptLoader,
    TranscriptLoadError,
)
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.repository import SQLiteRepository


class IngestionError(RuntimeError):
    """Base class for privacy-safe ingestion failures."""


class InvalidUploadError(IngestionError):
    """The upload name or content is not an acceptable TXT file."""


class UploadConflictError(IngestionError):
    """The immutable destination filename already exists."""


class TranscriptIngestionService:
    """Coordinate normal services while keeping business logic out of the UI."""

    def __init__(
        self,
        transcript_root: Path,
        repository: SQLiteRepository,
        extraction_model: StructuredMemoryModel,
        vector_index: MemoryVectorIndex,
    ) -> None:
        transcript_root.mkdir(parents=True, exist_ok=True)
        self._transcript_root = transcript_root.resolve(strict=True)
        self._repository = repository
        self._extraction_model = extraction_model
        self._vector_index = vector_index

    def ingest(
        self,
        *,
        filename: str,
        content: bytes,
        language: str | None = None,
        recorded_at: datetime | None = None,
    ) -> IngestionResult:
        """Persist a new raw file, extract memories, and refresh Chroma."""

        safe_name = self._validate_upload(filename, content)
        content_hash = sha256(content).hexdigest()
        known_hashes = {
            transcript.content_hash
            for transcript in self._repository.list_transcripts(
                include_deleted=False
            )
        }
        if content_hash in known_hashes:
            raise UploadConflictError("This transcript content already exists")
        target = self._transcript_root / safe_name
        try:
            with target.open("xb") as uploaded_file:
                uploaded_file.write(content)
        except FileExistsError as exception:
            raise UploadConflictError(
                "A transcript with this filename already exists"
            ) from exception
        except OSError as exception:
            raise IngestionError("Transcript upload could not be saved") from exception

        transcript_id: str | None = None
        try:
            loaded = TranscriptLoader(
                self._transcript_root,
                known_content_hashes=known_hashes,
            ).load(
                TranscriptLoadRequest(
                    source_path=target,
                    uploaded_at=datetime.now(UTC),
                    recorded_at=recorded_at,
                    language=language,
                )
            )
            self._repository.create_transcript(loaded)
            transcript_id = loaded.transcript_id
            chunks = chunk_and_store_transcript(
                self._repository,
                loaded.transcript_id,
            )
            memories = []
            for chunk in chunks:
                memories.extend(
                    extract_and_store_segment(
                        self._repository,
                        self._extraction_model,
                        chunk.segment_id,
                    )
                )
            try:
                index_results = [
                    self._vector_index.index_memory(memory.memory_id)
                    for memory in memories
                ]
            except Exception as exception:
                raise IngestionError("Memory index update failed") from exception
        except TranscriptLoadError as exception:
            self._cleanup_failed_upload(target, transcript_id)
            raise InvalidUploadError("TXT upload could not be processed") from exception
        except Exception as exception:
            self._cleanup_failed_upload(target, transcript_id)
            if isinstance(exception, IngestionError):
                raise
            raise IngestionError("Transcript upload could not be completed") from exception

        return IngestionResult(
            transcript_id=loaded.transcript_id,
            filename=loaded.filename,
            segment_count=len(chunks),
            memory_count=len(memories),
            indexed_memory_count=sum(
                result.indexed or result.content_hash is not None
                for result in index_results
            ),
            memory_ids=[memory.memory_id for memory in memories],
        )

    def _cleanup_failed_upload(self, target: Path, transcript_id: str | None) -> None:
        if transcript_id is not None:
            self._repository.delete_transcript(transcript_id)
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _validate_upload(filename: str, content: bytes) -> str:
        stripped_name = filename.strip()
        if (
            not stripped_name
            or Path(stripped_name).name != stripped_name
            or "/" in stripped_name
            or "\\" in stripped_name
            or Path(stripped_name).suffix.casefold() != ".txt"
        ):
            raise InvalidUploadError("Upload must be a plain TXT filename")
        if not content:
            raise InvalidUploadError("TXT upload must not be empty")
        try:
            decoded = content.decode(
                "utf-8-sig" if content.startswith(b"\xef\xbb\xbf") else "utf-8"
            )
        except UnicodeDecodeError as exception:
            raise InvalidUploadError("TXT upload must use UTF-8") from exception
        if not decoded.strip():
            raise InvalidUploadError("TXT upload must contain text")
        return stripped_name
