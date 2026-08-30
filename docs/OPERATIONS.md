# Local Operations Runbook

## Environment

1. Create `.env` from `.env.example`.
2. Set `OPENAI_API_KEY` only in the local `.env` file.
3. Do not paste the key into issues, logs, screenshots, or committed files.

The optional real evaluation script loads the project-root `.env` before it
checks the key.

## Start the MVP

Run these commands in separate PowerShell windows:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Verify the backend at `http://127.0.0.1:8000/api/v1/health` and open the UI at
`http://localhost:8501`.

## Data and recovery

- `data/raw/transcripts/` contains immutable uploaded source files.
- `data/db/` contains SQLite, the only source of truth.
- `data/indexes/` contains rebuildable ChromaDB and BM25 indexes.
- Deleting a transcript through the privacy API invalidates its derived data.
- If an index is damaged, rebuild it from SQLite; do not restore data from
  ChromaDB alone.

## Verification before a demo

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

The real embedding benchmark is opt-in and can incur API charges:

```powershell
.\.venv\Scripts\python.exe scripts\run_real_evaluation.py
```

Use the deterministic evaluation for normal regression checks. Use the real
benchmark only when comparing the configured embedding model, and inspect the
manifest and CSV together.

## Release boundary

This MVP is a local single-user application. Authentication, multi-user
authorization, rate limiting, background jobs, cloud deployment, and automatic
raw-file retention deletion remain outside the one-week MVP scope.
