"""Transcript ingestion and structured-memory read APIs."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from backend.app.core.config import get_settings
from backend.app.core.safe_logging import log_safe_exception
from backend.app.models.ingestion import IngestionResult
from backend.app.models.memory import MemoryCorrection
from backend.app.services.corrections import (
    MemoryAlreadyCorrectedError,
    MemoryCorrectionService,
    MemoryNotFoundError,
    UntraceableMemoryError,
)
from backend.app.services.ingestion import (
    IngestionError,
    InvalidUploadError,
    TranscriptIngestionService,
    UploadConflictError,
)
from backend.app.services.memory_extraction import (
    MemoryExtractionError,
    MemoryExtractionOutputError,
    MemoryExtractionRefusalError,
    build_openai_memory_localization_model,
    build_openai_memory_model,
)
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import MemoryRecord
from backend.app.storage.repository import SQLiteRepository, StorageError

router = APIRouter(tags=["memories"])


class IngestTranscriptRequest(BaseModel):
    """Base64 transport keeps the original TXT bytes unchanged."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=20_000_000)
    language: str | None = Field(default=None, max_length=32)
    recorded_at: AwareDatetime | None = None

    @field_validator("filename", "language")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value


class MemoryView(BaseModel):
    """Structured memory plus user-readable transcript source locations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory: MemoryRecord
    citations: list["MemorySourceView"]
    source_filename: str


class MemorySourceView(BaseModel):
    """A memory source with offsets and one-based line numbers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    transcript_id: str
    segment_id: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


def _memory_view(
    memory: MemoryRecord,
    repository: SQLiteRepository,
) -> MemoryView:
    transcript = repository.get_transcript(memory.transcript_id)
    if transcript is None:
        raise StorageError("Memory transcript was not found")
    return MemoryView(
        memory=memory,
        source_filename=transcript.filename,
        citations=[
            MemorySourceView(
                memory_id=source.memory_id,
                transcript_id=source.transcript_id,
                segment_id=source.segment_id,
                start_offset=source.start_offset,
                end_offset=source.end_offset,
                start_line=_line_number(transcript.normalized_content, source.start_offset),
                end_line=_line_number(
                    transcript.normalized_content,
                    max(source.end_offset - 1, source.start_offset),
                ),
            )
            for source in repository.list_memory_sources(memory.memory_id)
        ],
    )


def _line_number(content: str, offset: int) -> int:
    """Return the one-based line containing a transcript character offset."""

    bounded_offset = min(max(offset, 0), len(content))
    return content.count("\n", 0, bounded_offset) + 1


def _log_ingest_failure(http_request: Request, exception: Exception) -> None:
    """Log only the safe exception type for an upload failure."""

    root_exception = exception
    while root_exception.__cause__ is not None:
        root_exception = root_exception.__cause__
    log_safe_exception(
        event="transcript_ingest_failed",
        request_id=getattr(http_request.state, "request_id", "unknown"),
        exception=root_exception,
    )


def _root_exception(exception: Exception) -> Exception:
    """Return the deepest cause without exposing its message."""

    current = exception
    while isinstance(current.__cause__, Exception):
        current = current.__cause__
    return current


def _ingest_http_error(exception: Exception) -> HTTPException:
    """Translate ingestion failures into actionable, safe API errors."""

    root = _root_exception(exception)
    if isinstance(root, AuthenticationError):
        return HTTPException(
            status_code=401,
            detail={
                "code": "openai_authentication_error",
                "message": "OpenAI API key authentication failed",
            },
        )
    if isinstance(root, RateLimitError):
        return HTTPException(
            status_code=429,
            detail={
                "code": "openai_rate_limit",
                "message": "OpenAI request limit was reached",
            },
        )
    if isinstance(root, BadRequestError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "openai_bad_request",
                "message": "OpenAI rejected the memory extraction request",
            },
        )
    if isinstance(root, APIConnectionError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "openai_connection_error",
                "message": "Could not connect to OpenAI",
            },
        )
    if isinstance(root, APIError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "openai_service_error",
                "message": "OpenAI service request failed",
            },
        )
    if isinstance(exception, MemoryExtractionOutputError) or isinstance(
        root, MemoryExtractionOutputError
    ):
        validation_locations: list[str] = []
        validation_reasons: list[str] = []
        safe_reasons = {
            "event_date and date_precision do not agree": "event_date/date_precision mismatch",
            "event_date does not match date_precision": "event_date/date_precision format mismatch",
            "unknown date precision requires a null event_date": "unknown date requires null event_date",
            "known date precision requires event_date": "known date requires event_date",
            "low-confidence memories require uncertainty_notes": "low confidence requires uncertainty_notes",
            "approximate dates require uncertainty_notes": "approximate date requires uncertainty_notes",
            "evidence_end_offset must follow evidence_start_offset": "evidence offsets are invalid",
        }
        if isinstance(root, ValidationError):
            for error in root.errors():
                location = ".".join(str(part) for part in error.get("loc", ()))
                if location and location not in validation_locations:
                    validation_locations.append(location)
                message = error.get("msg")
                if isinstance(message, str):
                    normalized_message = message.removeprefix("Value error, ")
                    reason = safe_reasons.get(normalized_message)
                else:
                    reason = None
                if reason is not None:
                    if reason not in validation_reasons:
                        validation_reasons.append(reason)
        suffix = (
            f": {', '.join(validation_locations[:5])}"
            if validation_locations
            else ""
        )
        reason_suffix = (
            f" ({', '.join(validation_reasons[:3])})"
            if validation_reasons
            else ""
        )
        known_messages = {
            "Model response could not be parsed as structured output":
                "model response parsing failed",
            "Model returned an invalid output envelope":
                "model response envelope is invalid",
            "Model returned no parsed memories":
                "model did not return structured memories",
            "Memory evidence falls outside the transcript segment":
                "evidence range is outside the transcript chunk",
            "Memory evidence quote was not found in the transcript segment":
                "evidence quote was not found in the transcript chunk",
            "Memory evidence must contain source text":
                "evidence range is empty",
            "Model returned conflicting event dates for the same evidence":
                "conflicting dates were returned for the same evidence",
        }
        message_suffix = ""
        for candidate_message in (str(exception), str(root)):
            if candidate_message in known_messages:
                message_suffix = f" ({known_messages[candidate_message]})"
                break
        return HTTPException(
            status_code=422,
            detail={
                "code": "memory_output_invalid",
                "message": (
                    "Memory extraction returned invalid structured data"
                    f"{suffix}{reason_suffix}{message_suffix}"
                ),
            },
        )
    if isinstance(exception, MemoryExtractionRefusalError) or isinstance(
        root, MemoryExtractionRefusalError
    ):
        return HTTPException(
            status_code=422,
            detail={
                "code": "memory_extraction_refused",
                "message": "Memory extraction was refused for this transcript",
            },
        )
    if isinstance(exception, MemoryExtractionError) or isinstance(
        root, MemoryExtractionError
    ):
        return HTTPException(
            status_code=503,
            detail={
                "code": "memory_extraction_failed",
                "message": "Memory extraction service failed",
            },
        )
    if isinstance(exception, StorageError) or isinstance(root, StorageError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "storage_unavailable",
                "message": "Application storage is unavailable",
            },
        )
    root_type = type(root).__name__.lower()
    return HTTPException(
        status_code=503,
        detail={
            "code": f"transcript_processing_{root_type}",
            "message": "Transcript processing is unavailable",
        },
    )


@lru_cache(maxsize=1)
def get_memory_repository() -> SQLiteRepository:
    """Build the shared SQLite-backed memory reader lazily."""

    database = SQLiteDatabase(get_settings().sqlite_database_path)
    database.initialize()
    return SQLiteRepository(database)


@lru_cache(maxsize=1)
def get_ingestion_service() -> TranscriptIngestionService:
    """Build ingestion dependencies without exposing them to Streamlit."""

    settings = get_settings()
    repository = get_memory_repository()
    return TranscriptIngestionService(
        settings.transcript_upload_directory,
        repository,
        build_openai_memory_model(
            settings.openai_model,
            api_key=settings.openai_api_key,
        ),
        MemoryVectorIndex(
            repository,
            settings.chroma_persist_directory,
            embedding_model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        ),
        localization_model=build_openai_memory_localization_model(
            settings.openai_model,
            api_key=settings.openai_api_key,
        ),
    )


@router.post("/memories/ingest", response_model=IngestionResult)
def ingest_transcript(
    http_request: Request,
    request: IngestTranscriptRequest,
    service: TranscriptIngestionService = Depends(get_ingestion_service),
) -> IngestionResult:
    """Upload and process one immutable UTF-8 TXT transcript."""

    try:
        content = b64decode(request.content_base64, validate=True)
    except (Base64Error, ValueError) as exception:
        _log_ingest_failure(http_request, exception)
        raise HTTPException(
            status_code=422,
            detail="TXT upload encoding is invalid",
        ) from exception
    try:
        return service.ingest(
            filename=request.filename,
            content=content,
            language=request.language,
            recorded_at=request.recorded_at,
        )
    except UploadConflictError as exception:
        _log_ingest_failure(http_request, exception)
        raise HTTPException(
            status_code=409,
            detail="A transcript with this filename already exists",
        ) from exception
    except InvalidUploadError as exception:
        _log_ingest_failure(http_request, exception)
        raise HTTPException(status_code=422, detail=str(exception)) from exception
    except (IngestionError, MemoryExtractionError, StorageError) as exception:
        # Error reporting must never turn a handled upload failure into a 500.
        try:
            _log_ingest_failure(http_request, exception)
        except Exception:
            pass
        try:
            translated = _ingest_http_error(exception)
        except Exception:
            translated = HTTPException(
                status_code=503,
                detail={
                    "code": "transcript_processing_unavailable",
                    "message": "Transcript processing is unavailable",
                },
            )
        raise translated from exception
    except Exception as exception:
        # Keep an unexpected application bug distinguishable from dependency
        # and OpenAI failures while still returning a safe response.
        try:
            _log_ingest_failure(http_request, exception)
        except Exception:
            pass
        root_type = type(_root_exception(exception)).__name__.lower()
        raise HTTPException(
            status_code=500,
            detail={
                "code": f"ingest_unexpected_{root_type}",
                "message": "An unexpected error occurred during transcript upload",
            },
        ) from exception


@lru_cache(maxsize=1)
def get_correction_service() -> MemoryCorrectionService:
    """Build the SQLite-backed correction writer lazily."""

    return MemoryCorrectionService(get_memory_repository())


@router.post(
    "/memories/{memory_id}/corrections",
    response_model=MemoryView,
    status_code=201,
)
def correct_memory(
    memory_id: str,
    correction: MemoryCorrection,
    service: MemoryCorrectionService = Depends(get_correction_service),
    repository: SQLiteRepository = Depends(get_memory_repository),
) -> MemoryView:
    """Append a corrected memory that supersedes the stored one."""

    try:
        memory = service.correct_memory(memory_id, correction)
    except MemoryNotFoundError as exception:
        raise HTTPException(status_code=404, detail="Memory was not found") from exception
    except MemoryAlreadyCorrectedError as exception:
        raise HTTPException(
            status_code=409,
            detail="Memory was already corrected",
        ) from exception
    except UntraceableMemoryError as exception:
        raise HTTPException(
            status_code=422,
            detail="Memory has no transcript source to inherit",
        ) from exception
    except StorageError as exception:
        raise HTTPException(
            status_code=503,
            detail="Memory correction is unavailable",
        ) from exception
    return _memory_view(memory, repository)


@router.get("/memories", response_model=list[MemoryView])
def list_memories(
    transcript_id: str | None = Query(default=None, min_length=1),
    repository: SQLiteRepository = Depends(get_memory_repository),
) -> list[MemoryView]:
    """Return active structured memories with source offsets."""

    return [
        _memory_view(memory, repository)
        for memory in repository.list_memories(transcript_id=transcript_id)
    ]
