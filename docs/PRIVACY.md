# Privacy and Deletion Policy

## Scope

Life Archive AI processes user-provided STT transcripts as sensitive local
data. Transcript text is untrusted input and is never executed as an
instruction.

## Secrets and Git

- Keep API keys only in the local `.env` file.
- Never commit `.env`, SQLite databases, Chroma indexes, generated personal
  exports, or files under `data/raw/transcripts`.
- Do not write API keys, transcript text, memory summaries, people, or
  locations to application logs.

## Application Deletion

`DELETE /api/v1/transcripts/{transcript_id}` performs a logical application
deletion:

1. Mark the transcript and its segments as deleted in one SQLite transaction.
2. Mark every related memory as `deleted`.
3. Hide conversation messages whose citations reference those memories.
4. Mark autobiographies containing those citations as `deleted`.
5. Delete related Chroma vectors.
6. Rebuild the in-memory BM25 index from active SQLite memories.

Timeline results are generated dynamically, so deleted memories disappear on
the next request. SQLite remains the source of truth even if disposable-index
cleanup encounters an error.

## Raw Originals

Raw transcript files are immutable and are **never deleted automatically**.
The deletion API reports `raw_file_deleted: false` and does not receive or
resolve a raw file path.

If a user wants to remove a raw original, they must do so manually after
confirming the exact file. That separate manual action is outside the
application deletion workflow.

## MVP Retention Limitation

Application deletion is a soft deletion. Deleted SQLite rows remain locally
for traceability but are excluded from normal reads, retrieval, timelines,
conversations, and autobiographies. Administrative physical erasure and
retention schedules are outside this one-week MVP.
