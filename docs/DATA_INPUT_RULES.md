# Transcript Data Input Rules

## Purpose

These rules define the accepted input for transcript loading without changing or
copying personal raw data.

## Raw Data Policy

- `data/raw/` is immutable and read-only.
- Transcript text must never be executed as code or instructions.
- Raw text, file names, and local paths must not be copied into reports or logs.
- Corrections and normalized outputs belong under `data/processed/` or SQLite.
- Generated databases and indexes must remain outside `data/raw/`.

## TXT Requirements

- Files must be located under `data/raw/transcripts/`.
- Only files with a case-insensitive `.txt` suffix are accepted.
- Encoding must be UTF-8 or UTF-8 with BOM.
- Empty files are invalid loader inputs.
- Duplicate content is identified using SHA-256 and must not be ingested twice.
- The loader must preserve all text semantics, including uncertainty, repetition,
  and spoken-language artifacts.

## Manifest Requirements

`data/raw/upload_manifest.jsonl` must be UTF-8 JSON Lines with one object per
non-empty line.

Required string fields:

- `filename`
- `language`
- `recorded_at`
- `recording_id`
- `source_type`

Additional rules:

- `filename` must be a relative, traversal-free TXT file name.
- `filename` and `recording_id` must each be unique.
- `recorded_at` must use ISO 8601 format.
- Each manifest filename must match exactly one TXT file.
- Each TXT file must be referenced by exactly one manifest record.
- The loader must reject ambiguous or missing links instead of guessing.

## Loader Normalization

The TASK-005 loader performs only conservative layout normalization:

- Convert CRLF and CR line endings to LF.
- Collapse repeated horizontal whitespace to one space.
- Remove whitespace at line boundaries.
- Collapse three or more consecutive line breaks to one blank line.

It does not correct spelling or grammar and does not remove stuttering,
self-corrections, uncertain dates, aliases, or repeated memories. The original
decoded content is retained separately from normalized content.

The SHA-256 hash is calculated from the immutable original bytes. A hash already
known to the loader is rejected as a duplicate. Persistent duplicate checks will
use the same hash when SQLite storage is implemented.

Upload time and source recording time are stored as separate timezone-aware
metadata. Neither timestamp is interpreted as the remembered event date; event
dates belong to structured Memory extraction and may be unknown.

## Current Dataset Inspection

The TASK-004 inspection found:

- 78 TXT files and 79 manifest records.
- All 78 TXT files are readable UTF-8.
- No empty TXT files and no duplicate-content groups.
- Average text length: 257.54 characters.
- Maximum text length: 446 characters.
- Manifest records use all five required fields with string values.
- All manifest filenames are basename-only `.txt` references.
- The observed TXT and manifest filename patterns differ.
- No manifest filename currently resolves to an actual TXT file.

The raw files must not be renamed to fix this mismatch. A verified external
mapping or corrected source export is required before real ingestion. The loader
must report the mismatch safely until that mapping exists.

## Inspection Workflow

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_data.py
```

The generated report is written to:

```text
data/processed/dataset_report.json
```

The report contains aggregate statistics, structural metadata, exception codes,
and truncated hashes. It does not contain transcript text or original file names.

## Exception Codes

The inspection workflow may report:

- `manifest_missing`
- `manifest_invalid_utf8`
- `manifest_invalid_json`
- `manifest_record_not_object`
- `manifest_missing_fields`
- `manifest_invalid_field_type`
- `unsafe_manifest_path`
- `invalid_recorded_at`
- `duplicate_manifest_filename`
- `duplicate_recording_id`
- `manifest_file_missing`
- `manifest_file_ambiguous`
- `txt_not_in_manifest`
- `empty_txt`
- `invalid_utf8`
- `duplicate_content`
