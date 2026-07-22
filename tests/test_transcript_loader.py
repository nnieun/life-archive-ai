"""Transcript loading and conservative normalization tests."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.models.transcript import TranscriptLoadRequest
from backend.app.services.transcript_loader import (
    DuplicateTranscriptError,
    EmptyTranscriptError,
    InvalidTranscriptEncodingError,
    InvalidTranscriptPathError,
    TranscriptLoader,
    normalize_transcript,
)


def _request(path: Path) -> TranscriptLoadRequest:
    return TranscriptLoadRequest(
        source_path=path,
        uploaded_at=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
        recorded_at=datetime(2025, 1, 2, 9, 30, tzinfo=UTC),
        recording_id="recording_001",
        language="ko",
        source_type="stt_text",
    )


def test_normalization_preserves_memory_semantics() -> None:
    original = (
        "어...  나는\t나는 민수, 아니 영수와\r\n"
        " 작년쯤  만났어.\r"
        "별명은  곰이야.\n\n\n"
        "같은 기억을 또 말해."
    )

    normalized = normalize_transcript(original)

    assert normalized == (
        "어... 나는 나는 민수, 아니 영수와\n"
        "작년쯤 만났어.\n"
        "별명은 곰이야.\n\n"
        "같은 기억을 또 말해."
    )
    for preserved_expression in (
        "어...",
        "나는 나는",
        "민수, 아니 영수",
        "작년쯤",
        "곰이야",
        "같은 기억을 또 말해",
    ):
        assert preserved_expression in normalized


def test_loader_reads_utf8_and_preserves_original_file(tmp_path: Path) -> None:
    transcript_root = tmp_path / "transcripts"
    transcript_root.mkdir()
    source = transcript_root / "recording_001.txt"
    original_bytes = "첫째 줄  입니다.\r\n둘째 줄입니다.".encode("utf-8")
    source.write_bytes(original_bytes)
    hash_before = hashlib.sha256(source.read_bytes()).hexdigest()

    transcript = TranscriptLoader(transcript_root).load(_request(source.name))

    assert transcript.content_hash == hash_before
    assert transcript.transcript_id == f"tr_{hash_before[:24]}"
    assert transcript.raw_content == original_bytes.decode("utf-8")
    assert transcript.normalized_content == "첫째 줄 입니다.\n둘째 줄입니다."
    assert transcript.uploaded_at == datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    assert transcript.recorded_at == datetime(2025, 1, 2, 9, 30, tzinfo=UTC)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == hash_before


def test_loader_accepts_utf8_bom(tmp_path: Path) -> None:
    transcript_root = tmp_path / "transcripts"
    transcript_root.mkdir()
    source = transcript_root / "bom.txt"
    source.write_bytes(b"\xef\xbb\xbf" + "기억".encode("utf-8"))

    transcript = TranscriptLoader(transcript_root).load(_request(source))

    assert transcript.raw_content == "기억"
    assert transcript.normalized_content == "기억"


@pytest.mark.parametrize("content", [b"", b" \t\r\n\n "])
def test_loader_rejects_empty_or_whitespace_only_files(
    tmp_path: Path,
    content: bytes,
) -> None:
    transcript_root = tmp_path / "transcripts"
    transcript_root.mkdir()
    source = transcript_root / "empty.txt"
    source.write_bytes(content)

    with pytest.raises(EmptyTranscriptError, match="must contain text"):
        TranscriptLoader(transcript_root).load(_request(source))


def test_loader_rejects_invalid_encoding(tmp_path: Path) -> None:
    transcript_root = tmp_path / "transcripts"
    transcript_root.mkdir()
    source = transcript_root / "invalid.txt"
    source.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(InvalidTranscriptEncodingError, match="valid UTF-8"):
        TranscriptLoader(transcript_root).load(_request(source))


def test_loader_detects_duplicate_file_content(tmp_path: Path) -> None:
    transcript_root = tmp_path / "transcripts"
    transcript_root.mkdir()
    first = transcript_root / "first.txt"
    second = transcript_root / "second.txt"
    content = "같은 원본 기억".encode("utf-8")
    first.write_bytes(content)
    second.write_bytes(content)
    loader = TranscriptLoader(transcript_root)

    loaded = loader.load(_request(first))

    with pytest.raises(DuplicateTranscriptError, match="already loaded"):
        loader.load(_request(second))
    assert loader.known_content_hashes == frozenset({loaded.content_hash})


def test_loader_rejects_files_outside_input_directory(tmp_path: Path) -> None:
    transcript_root = tmp_path / "transcripts"
    transcript_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("외부 파일", encoding="utf-8")

    with pytest.raises(InvalidTranscriptPathError, match="input directory"):
        TranscriptLoader(transcript_root).load(_request(outside))


def test_input_model_rejects_naive_timestamps_and_hides_local_path() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        TranscriptLoadRequest(
            source_path=Path("private.txt"),
            uploaded_at=datetime(2026, 7, 22, 12, 0),
        )

    request = _request(Path("private.txt"))
    assert "source_path" not in request.model_dump()
    assert "private.txt" not in repr(request)
