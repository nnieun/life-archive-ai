"""Safe UTF-8 transcript loading and conservative whitespace normalization."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from backend.app.models.transcript import LoadedTranscript, TranscriptLoadRequest

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class TranscriptLoadError(ValueError):
    """Base class for privacy-safe transcript loading failures."""


class InvalidTranscriptPathError(TranscriptLoadError):
    """The requested source is not an allowed TXT file."""


class EmptyTranscriptError(TranscriptLoadError):
    """The transcript has no meaningful text after whitespace normalization."""


class InvalidTranscriptEncodingError(TranscriptLoadError):
    """The transcript is not valid UTF-8 or UTF-8 with BOM."""


class DuplicateTranscriptError(TranscriptLoadError):
    """The original file bytes have already been loaded."""


class TranscriptChangedError(TranscriptLoadError):
    """The source changed concurrently while it was being read."""


def normalize_transcript(text: str) -> str:
    """Normalize layout while preserving every spoken token and expression."""
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = [
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        for line in normalized_newlines.split("\n")
    ]
    normalized_text = "\n".join(normalized_lines).strip()
    return _EXCESS_BLANK_LINES.sub("\n\n", normalized_text)


class TranscriptLoader:
    """Load immutable transcripts from one explicitly allowed directory."""

    def __init__(
        self,
        transcript_root: Path,
        known_content_hashes: Iterable[str] = (),
    ) -> None:
        try:
            resolved_root = transcript_root.resolve(strict=True)
        except OSError as exception:
            raise InvalidTranscriptPathError(
                "Transcript input directory is unavailable"
            ) from exception
        if not resolved_root.is_dir():
            raise InvalidTranscriptPathError(
                "Transcript input directory is unavailable"
            )
        self._transcript_root = resolved_root
        self._known_content_hashes = set(known_content_hashes)

    @property
    def known_content_hashes(self) -> frozenset[str]:
        """Expose a read-only snapshot for later persistence integration."""
        return frozenset(self._known_content_hashes)

    def load(self, request: TranscriptLoadRequest) -> LoadedTranscript:
        """Read, validate, normalize, and register one transcript."""
        source_path = request.source_path
        candidate = (
            source_path
            if source_path.is_absolute()
            else self._transcript_root / source_path
        )
        try:
            resolved_path = candidate.resolve(strict=True)
        except OSError as exception:
            raise InvalidTranscriptPathError(
                "Transcript file is unavailable"
            ) from exception

        if (
            not resolved_path.is_file()
            or not resolved_path.is_relative_to(self._transcript_root)
            or resolved_path.suffix.casefold() != ".txt"
        ):
            raise InvalidTranscriptPathError(
                "Expected a TXT file in the transcript input directory"
            )

        try:
            stat_before = resolved_path.stat()
            raw_bytes = resolved_path.read_bytes()
            stat_after = resolved_path.stat()
        except OSError as exception:
            raise InvalidTranscriptPathError(
                "Transcript file could not be read"
            ) from exception

        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise TranscriptChangedError("Transcript changed while being read")

        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        if content_hash in self._known_content_hashes:
            raise DuplicateTranscriptError("Transcript content already loaded")

        encoding = "utf-8-sig" if raw_bytes.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            raw_content = raw_bytes.decode(encoding)
        except UnicodeDecodeError as exception:
            raise InvalidTranscriptEncodingError(
                "Transcript must be valid UTF-8"
            ) from exception

        normalized_content = normalize_transcript(raw_content)
        if not normalized_content:
            raise EmptyTranscriptError("Transcript must contain text")

        transcript = LoadedTranscript(
            transcript_id=f"tr_{content_hash[:24]}",
            filename=resolved_path.name,
            recording_id=request.recording_id,
            language=request.language,
            source_type=request.source_type,
            uploaded_at=request.uploaded_at,
            recorded_at=request.recorded_at,
            content_hash=content_hash,
            raw_content=raw_content,
            normalized_content=normalized_content,
        )
        self._known_content_hashes.add(content_hash)
        return transcript
