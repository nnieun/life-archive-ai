"""Inspect transcript inputs without copying personal text into the report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_MANIFEST_FIELDS = {
    "filename",
    "language",
    "recorded_at",
    "recording_id",
    "source_type",
}


def _private_id(value: str) -> str:
    """Return a stable identifier that does not disclose a file name."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _filename_pattern(filename: str) -> str:
    """Describe name structure without retaining any original characters."""
    categories: list[str] = []
    for character in Path(filename).stem:
        if character.isdigit():
            categories.append("D")
        elif "a" <= character.casefold() <= "z":
            categories.append("A")
        elif "\uac00" <= character <= "\ud7a3":
            categories.append("K")
        elif character.isspace():
            categories.append("S")
        else:
            categories.append("P")
    return re.sub(r"(.)\1+", r"\1+", "".join(categories))


def _raw_snapshot(raw_dir: Path) -> str:
    """Fingerprint raw paths and bytes to prove the inspection is read-only."""
    digest = hashlib.sha256()
    for path in sorted(item for item in raw_dir.rglob("*") if item.is_file()):
        relative_path = path.relative_to(raw_dir).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _read_manifest(
    manifest_path: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    exceptions: list[dict[str, Any]] = []

    if not manifest_path.is_file():
        exceptions.append({"code": "manifest_missing"})
        return records, exceptions

    try:
        lines = manifest_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        exceptions.append({"code": "manifest_invalid_utf8"})
        return records, exceptions

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            exceptions.append(
                {"code": "manifest_invalid_json", "manifest_line": line_number}
            )
            continue
        if not isinstance(value, dict):
            exceptions.append(
                {"code": "manifest_record_not_object", "manifest_line": line_number}
            )
            continue
        records.append((line_number, value))

    return records, exceptions


def _reference_keys(filename: str) -> list[str] | None:
    normalized = filename.replace("\\", "/")
    reference = PurePosixPath(normalized)
    if (
        reference.is_absolute()
        or ".." in reference.parts
        or not reference.parts
        or ":" in reference.parts[0]
    ):
        return None

    relative = reference.as_posix().casefold()
    keys = [relative]
    if not relative.startswith("transcripts/"):
        keys.append(f"transcripts/{relative}")
    if len(reference.parts) == 1:
        keys.append(reference.name.casefold())
    return list(dict.fromkeys(keys))


def _valid_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def inspect_dataset(raw_dir: Path, report_path: Path) -> dict[str, Any]:
    """Inspect TXT and manifest structure while leaving raw data unchanged."""
    raw_dir = raw_dir.resolve()
    report_path = report_path.resolve()
    if report_path.is_relative_to(raw_dir):
        raise ValueError("The report must be written outside data/raw")
    transcript_dir = raw_dir / "transcripts"
    manifest_path = raw_dir / "upload_manifest.jsonl"
    snapshot_before = _raw_snapshot(raw_dir)

    txt_files = sorted(
        path
        for path in transcript_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".txt"
    )
    records, exceptions = _read_manifest(manifest_path)

    path_lookup: dict[str, list[Path]] = defaultdict(list)
    for path in txt_files:
        relative_raw = path.relative_to(raw_dir).as_posix().casefold()
        relative_transcripts = path.relative_to(transcript_dir).as_posix().casefold()
        for key in {relative_raw, relative_transcripts, path.name.casefold()}:
            path_lookup[key].append(path)

    encodings: Counter[str] = Counter()
    text_lengths: list[int] = []
    content_hashes: dict[str, list[Path]] = defaultdict(list)
    readable_paths: set[Path] = set()
    empty_count = 0
    invalid_encoding_count = 0

    for path in txt_files:
        relative_path = path.relative_to(raw_dir).as_posix()
        identifier = _private_id(relative_path)
        raw_bytes = path.read_bytes()
        content_hashes[hashlib.sha256(raw_bytes).hexdigest()].append(path)

        if not raw_bytes:
            empty_count += 1
            exceptions.append({"code": "empty_txt", "file_id": identifier})

        encoding = "utf-8-sig" if raw_bytes.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            text = raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            invalid_encoding_count += 1
            exceptions.append({"code": "invalid_utf8", "file_id": identifier})
            continue

        encodings[encoding] += 1
        text_lengths.append(len(text))
        readable_paths.add(path)

    duplicate_groups: list[dict[str, Any]] = []
    for content_hash, paths in sorted(content_hashes.items()):
        if len(paths) < 2:
            continue
        group = {
            "group_id": _private_id(f"duplicate:{content_hash}"),
            "file_ids": sorted(
                _private_id(path.relative_to(raw_dir).as_posix()) for path in paths
            ),
        }
        duplicate_groups.append(group)
        exceptions.append({"code": "duplicate_content", **group})

    referenced_paths: set[Path] = set()
    manifest_filenames: list[str] = []
    recording_ids: list[str] = []

    for line_number, record in records:
        missing_fields = sorted(REQUIRED_MANIFEST_FIELDS - record.keys())
        if missing_fields:
            exceptions.append(
                {
                    "code": "manifest_missing_fields",
                    "manifest_line": line_number,
                    "fields": missing_fields,
                }
            )

        for field in sorted(REQUIRED_MANIFEST_FIELDS):
            if field in record and not isinstance(record[field], str):
                exceptions.append(
                    {
                        "code": "manifest_invalid_field_type",
                        "manifest_line": line_number,
                        "field": field,
                    }
                )

        filename = record.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            continue
        manifest_filenames.append(filename)
        reference_keys = _reference_keys(filename)
        if reference_keys is None:
            exceptions.append(
                {"code": "unsafe_manifest_path", "manifest_line": line_number}
            )
            continue

        matches = {
            path for key in reference_keys for path in path_lookup.get(key, [])
        }
        if not matches:
            exceptions.append(
                {
                    "code": "manifest_file_missing",
                    "manifest_line": line_number,
                    "file_id": _private_id(filename.casefold()),
                }
            )
        elif len(matches) > 1:
            exceptions.append(
                {
                    "code": "manifest_file_ambiguous",
                    "manifest_line": line_number,
                    "file_id": _private_id(filename.casefold()),
                }
            )
        else:
            referenced_paths.update(matches)

        recording_id = record.get("recording_id")
        if isinstance(recording_id, str) and recording_id:
            recording_ids.append(recording_id)

        recorded_at = record.get("recorded_at")
        if isinstance(recorded_at, str) and recorded_at and not _valid_iso_datetime(recorded_at):
            exceptions.append(
                {"code": "invalid_recorded_at", "manifest_line": line_number}
            )

    for filename, count in Counter(name.casefold() for name in manifest_filenames).items():
        if count > 1:
            exceptions.append(
                {
                    "code": "duplicate_manifest_filename",
                    "file_id": _private_id(filename),
                    "count": count,
                }
            )

    for recording_id, count in Counter(recording_ids).items():
        if count > 1:
            exceptions.append(
                {
                    "code": "duplicate_recording_id",
                    "recording_id_hash": _private_id(recording_id),
                    "count": count,
                }
            )

    for path in sorted(set(txt_files) - referenced_paths):
        exceptions.append(
            {
                "code": "txt_not_in_manifest",
                "file_id": _private_id(path.relative_to(raw_dir).as_posix()),
            }
        )

    manifest_keys = sorted(
        {key for _line_number, record in records for key in record.keys()}
    )
    manifest_types = {
        key: sorted(
            {
                type(record[key]).__name__
                for _line_number, record in records
                if key in record
            }
        )
        for key in manifest_keys
    }
    filename_suffixes = Counter(
        PurePosixPath(name.replace("\\", "/")).suffix.casefold() or "[none]"
        for name in manifest_filenames
    )
    txt_name_patterns = Counter(_filename_pattern(path.name) for path in txt_files)
    manifest_name_patterns = Counter(
        _filename_pattern(name) for name in manifest_filenames
    )
    txt_name_lengths = [len(path.stem) for path in txt_files]
    manifest_name_lengths = [
        len(PurePosixPath(name.replace("\\", "/")).stem)
        for name in manifest_filenames
    ]

    snapshot_after = _raw_snapshot(raw_dir)
    raw_unchanged = snapshot_before == snapshot_after
    report: dict[str, Any] = {
        "schema_version": 1,
        "raw_snapshot_sha256": snapshot_after,
        "raw_unchanged": raw_unchanged,
        "dataset": {
            "txt_file_count": len(txt_files),
            "readable_txt_count": len(readable_paths),
            "empty_txt_count": empty_count,
            "invalid_encoding_count": invalid_encoding_count,
            "duplicate_group_count": len(duplicate_groups),
            "manifest_record_count": len(records),
            "manifest_parse_error_count": sum(
                1
                for item in exceptions
                if item["code"]
                in {"manifest_invalid_json", "manifest_record_not_object"}
            ),
            "manifest_file_missing_count": sum(
                item["code"] == "manifest_file_missing" for item in exceptions
            ),
            "txt_not_in_manifest_count": sum(
                item["code"] == "txt_not_in_manifest" for item in exceptions
            ),
        },
        "text_length_characters": {
            "average": round(sum(text_lengths) / len(text_lengths), 2)
            if text_lengths
            else 0,
            "maximum": max(text_lengths, default=0),
        },
        "encoding_counts": dict(sorted(encodings.items())),
        "filename_summary": {
            "manifest_suffix_counts": dict(sorted(filename_suffixes.items())),
            "txt_stem_length_minimum": min(txt_name_lengths, default=0),
            "txt_stem_length_maximum": max(txt_name_lengths, default=0),
            "manifest_stem_length_minimum": min(manifest_name_lengths, default=0),
            "manifest_stem_length_maximum": max(manifest_name_lengths, default=0),
            "txt_pattern_counts": dict(sorted(txt_name_patterns.items())),
            "manifest_pattern_counts": dict(sorted(manifest_name_patterns.items())),
            "basename_only_count": sum(
                len(PurePosixPath(name.replace("\\", "/")).parts) == 1
                for name in manifest_filenames
            ),
            "unique_filename_count": len(
                {name.casefold() for name in manifest_filenames}
            ),
        },
        "manifest_schema": {
            "required_fields": sorted(REQUIRED_MANIFEST_FIELDS),
            "observed_fields": manifest_keys,
            "observed_types": manifest_types,
        },
        "duplicate_groups": duplicate_groups,
        "exceptions": sorted(
            exceptions,
            key=lambda item: (
                item["code"],
                item.get("manifest_line", 0),
                item.get("file_id", ""),
            ),
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not raw_unchanged:
        raise RuntimeError("Raw data changed during inspection")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/processed/dataset_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = inspect_dataset(args.raw_dir, args.report_path)
    dataset = report["dataset"]
    print(
        "Dataset inspection complete: "
        f"{dataset['txt_file_count']} TXT files, "
        f"{dataset['manifest_record_count']} manifest records, "
        f"{len(report['exceptions'])} exceptions, raw unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
