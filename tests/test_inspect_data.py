"""Privacy-safe transcript dataset inspection tests."""

import json
from pathlib import Path

import pytest

from scripts.inspect_data import inspect_dataset


def _write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _manifest_record(filename: str, recording_id: str) -> dict[str, str]:
    return {
        "filename": filename,
        "language": "ko",
        "recorded_at": "2025-01-01T00:00:00+09:00",
        "recording_id": recording_id,
        "source_type": "stt_text",
    }


def test_inspection_reports_healthy_dataset_without_copying_text(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    transcript_dir = raw_dir / "transcripts"
    transcript_dir.mkdir(parents=True)
    private_text = "A private memory that must not appear in reports."
    (transcript_dir / "recording_001.txt").write_text(
        private_text, encoding="utf-8"
    )
    _write_manifest(
        raw_dir / "upload_manifest.jsonl",
        [_manifest_record("recording_001.txt", "recording_001")],
    )
    report_path = tmp_path / "processed" / "dataset_report.json"

    report = inspect_dataset(raw_dir, report_path)

    assert report["raw_unchanged"] is True
    assert report["dataset"] == {
        "txt_file_count": 1,
        "readable_txt_count": 1,
        "empty_txt_count": 0,
        "invalid_encoding_count": 0,
        "duplicate_group_count": 0,
        "manifest_record_count": 1,
        "manifest_parse_error_count": 0,
        "manifest_file_missing_count": 0,
        "txt_not_in_manifest_count": 0,
    }
    assert report["exceptions"] == []
    assert private_text not in report_path.read_text(encoding="utf-8")


def test_inspection_detects_dataset_exceptions(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    transcript_dir = raw_dir / "transcripts"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "empty.txt").write_bytes(b"")
    duplicate_text = b"same private content"
    (transcript_dir / "duplicate_a.txt").write_bytes(duplicate_text)
    (transcript_dir / "duplicate_b.txt").write_bytes(duplicate_text)
    (transcript_dir / "invalid.txt").write_bytes(b"\xff\xfe")
    _write_manifest(
        raw_dir / "upload_manifest.jsonl",
        [
            _manifest_record("missing.txt", "recording_001"),
            _manifest_record("empty.txt", "recording_002"),
        ],
    )

    report = inspect_dataset(raw_dir, tmp_path / "report.json")
    codes = {item["code"] for item in report["exceptions"]}

    assert {
        "duplicate_content",
        "empty_txt",
        "invalid_utf8",
        "manifest_file_missing",
        "txt_not_in_manifest",
    } <= codes
    assert report["dataset"]["duplicate_group_count"] == 1
    assert report["dataset"]["invalid_encoding_count"] == 1


def test_inspection_rejects_report_inside_raw_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    with pytest.raises(ValueError, match="outside data/raw"):
        inspect_dataset(raw_dir, raw_dir / "dataset_report.json")
