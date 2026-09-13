"""Immutable TXT/PDF upload, extraction, and indexing orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from backend.app.models.ingestion import IngestionResult
from backend.app.models.transcript import TranscriptLoadRequest
from backend.app.services.chunking import chunk_and_store_transcript
from backend.app.services.gap_detection import (
    MemoryGapDetectionService,
    MemoryGapDetector,
)
from backend.app.services.memory_extraction import (
    MemoryExtractionError,
    StructuredMemoryModel,
    extract_and_store_segment,
)
from backend.app.services.transcript_loader import (
    TranscriptLoader,
    TranscriptLoadError,
)
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.repository import SQLiteRepository, StorageError


class IngestionError(RuntimeError):
    """Base class for privacy-safe ingestion failures."""


class InvalidUploadError(IngestionError):
    """The upload name or content is not an acceptable TXT/PDF file."""


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
        localization_model: StructuredMemoryModel | None = None,
        gap_detector: MemoryGapDetector | None = None,
    ) -> None:
        transcript_root.mkdir(parents=True, exist_ok=True)
        self._transcript_root = transcript_root.resolve(strict=True)
        self._repository = repository
        self._extraction_model = extraction_model
        self._vector_index = vector_index
        self._localization_model = localization_model
        self._gap_detector = gap_detector or MemoryGapDetectionService(repository)

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
        if target.exists():
            try:
                same_content = target.read_bytes() == content
            except OSError as exception:
                raise UploadConflictError(
                    "A transcript with this filename already exists"
                ) from exception
            if not same_content:
                raise UploadConflictError(
                    "A transcript with this filename already exists"
                )
            # A deleted transcript keeps its immutable raw file. Re-uploading
            # the same content is therefore a new processing attempt, stored
            # under a collision-free filename instead of overwriting the raw file.
            target = self._next_reprocess_target(target)
            safe_name = target.name
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
        indexed_memory_ids: list[str] = []
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
            if (
                self._repository.get_transcript(
                    loaded.transcript_id,
                    include_deleted=True,
                )
                is not None
            ):
                # transcript_id is derived from content_hash, and a
                # soft-deleted row keeps that id forever for audit purposes
                # (docs/PRIVACY.md). known_hashes already proved no *active*
                # transcript has this content, so any collision here can
                # only be a deleted leftover - mint a fresh id instead of
                # colliding with it, rather than surfacing a raw SQLite
                # constraint error for a perfectly legal re-upload.
                loaded = loaded.model_copy(
                    update={
                        "transcript_id": f"{loaded.transcript_id}_{uuid4().hex[:8]}"
                    }
                )
            loaded = loaded.model_copy(
                update={
                    "source_path": str(
                        Path("data/raw/transcripts") / safe_name
                    ).replace("\\", "/")
                }
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
                        localization_model=self._localization_model,
                    )
                )
            gaps = self._gap_detector.detect_for_memories(memories)
            try:
                # Failed batches may have persisted vectors; register all IDs
                # before attempting writes so rollback also removes those.
                indexed_memory_ids.extend(memory.memory_id for memory in memories)
                index_results = self._vector_index.index_memories(indexed_memory_ids)
            except Exception as exception:
                raise IngestionError("Memory index update failed") from exception
        except TranscriptLoadError as exception:
            self._cleanup_failed_upload(target, transcript_id, indexed_memory_ids)
            raise InvalidUploadError("TXT upload could not be processed") from exception
        except MemoryExtractionError:
            self._cleanup_failed_upload(target, transcript_id, indexed_memory_ids)
            raise
        except StorageError:
            self._cleanup_failed_upload(target, transcript_id, indexed_memory_ids)
            raise
        except Exception as exception:
            self._cleanup_failed_upload(target, transcript_id, indexed_memory_ids)
            if isinstance(exception, IngestionError):
                raise
            raise IngestionError("Transcript upload could not be completed") from exception

        return IngestionResult(
            transcript_id=loaded.transcript_id,
            filename=loaded.filename,
            segment_count=len(chunks),
            memory_count=len(memories),
            gap_count=len(gaps),
            indexed_memory_count=sum(
                result.indexed or result.content_hash is not None
                for result in index_results
            ),
            memory_ids=[memory.memory_id for memory in memories],
            gap_ids=[gap.gap_id for gap in gaps],
        )

    @staticmethod
    def _next_reprocess_target(target: Path) -> Path:
        """Return a collision-free filename for reprocessing identical raw data."""

        stem = target.stem
        suffix = target.suffix
        index = 2
        while True:
            candidate = target.with_name(f"{stem} ({index}){suffix}")
            if not candidate.exists():
                return candidate
            index += 1

    def _cleanup_failed_upload(
        self,
        target: Path,
        transcript_id: str | None,
        indexed_memory_ids: list[str],
    ) -> None:
        """Undo everything a failed ingest may have done, including Chroma.

        Batch writes can fail after some vectors have reached Chroma.
        Once ``delete_transcript`` hard-deletes the SQLite rows there is no
        way to rediscover those memory_ids from the database, so the caller
        must hand back every attempted ID before cleanup runs.
        """
        for memory_id in indexed_memory_ids:
            try:
                self._vector_index.delete_memory(memory_id)
            except Exception:
                pass
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
            or Path(stripped_name).suffix.casefold() not in {".txt", ".pdf"}
        ):
            raise InvalidUploadError("Upload must be a plain TXT or PDF filename")
        if not content:
            raise InvalidUploadError("Upload must not be empty")
        if Path(stripped_name).suffix.casefold() == ".txt":
            try:
                decoded = content.decode(
                    "utf-8-sig" if content.startswith(b"\xef\xbb\xbf") else "utf-8"
                )
            except UnicodeDecodeError as exception:
                raise InvalidUploadError("TXT upload must use UTF-8") from exception
            if not decoded.strip():
                raise InvalidUploadError("TXT upload must contain text")
        return stripped_name
