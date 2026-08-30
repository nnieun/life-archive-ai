"""SQLite connection management and schema initialization."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_path TEXT,
    recording_id TEXT UNIQUE,
    language TEXT,
    source_type TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    recorded_at TEXT,
    content_hash TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    segment_id TEXT PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE (transcript_id, chunk_index),
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    people_json TEXT NOT NULL CHECK (json_valid(people_json)),
    location TEXT,
    event_date TEXT,
    date_precision TEXT NOT NULL DEFAULT 'unknown'
        CHECK (
            date_precision IN (
                'exact', 'day', 'month', 'year', 'approximate', 'unknown'
            )
        )
        CHECK (
            (event_date IS NULL AND date_precision = 'unknown')
            OR (event_date IS NOT NULL AND date_precision != 'unknown')
        ),
    emotion TEXT,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    uncertainty_notes TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'corrected', 'deleted')),
    supersedes_memory_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (supersedes_memory_id) REFERENCES memories(memory_id)
        ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_source_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    transcript_id TEXT NOT NULL,
    segment_id TEXT,
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (memory_id, transcript_id, segment_id, start_offset, end_offset),
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (segment_id) REFERENCES transcript_segments(segment_id)
        ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS memory_gaps (
    gap_id TEXT PRIMARY KEY,
    memory_id TEXT,
    gap_type TEXT NOT NULL
        CHECK (
            gap_type IN (
                'MISSING_LOCATION', 'MISSING_PERSON', 'MISSING_DATE',
                'UNCERTAIN_EVENT', 'CONFLICTING_FACT', 'WEAK_PROVENANCE'
            )
        ),
    clue_text TEXT NOT NULL,
    missing_field TEXT,
    period_start TEXT,
    period_end TEXT,
    location TEXT,
    people_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(people_json)),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    importance_score REAL NOT NULL DEFAULT 0.5
        CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    status TEXT NOT NULL DEFAULT 'OPEN'
        CHECK (
            status IN (
                'OPEN', 'SEARCHING', 'CANDIDATE_FOUND', 'WAITING_USER',
                'RESOLVED', 'DISMISSED'
            )
        ),
    source_type TEXT NOT NULL
        CHECK (source_type IN ('MEMORY', 'TRANSCRIPT_SEGMENT', 'EXTERNAL')),
    source_id TEXT NOT NULL,
    web_search_consent INTEGER NOT NULL DEFAULT 0
        CHECK (web_search_consent IN (0, 1)),
    user_clues_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(user_clues_json)),
    resolved_candidate_id TEXT,
    resolved_memory_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (resolved_memory_id) REFERENCES memories(memory_id)
        ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS memory_gap_candidates (
    candidate_id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL,
    value TEXT NOT NULL,
    explanation TEXT NOT NULL,
    deterministic_score REAL NOT NULL
        CHECK (deterministic_score >= 0.0 AND deterministic_score <= 1.0),
    llm_relation TEXT NOT NULL DEFAULT 'UNKNOWN'
        CHECK (llm_relation IN ('SUPPORTS', 'RELATED', 'CONFLICTS', 'UNKNOWN')),
    supporting_source_ids_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(supporting_source_ids_json)),
    external_sources_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(external_sources_json)),
    status TEXT NOT NULL DEFAULT 'PROPOSED'
        CHECK (status IN ('PROPOSED', 'ACCEPTED', 'REJECTED')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (gap_id) REFERENCES memory_gaps(gap_id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(citations_json)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autobiographies (
    autobiography_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content_json TEXT NOT NULL CHECK (json_valid(content_json)),
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'completed', 'deleted')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_segments_transcript
    ON transcript_segments(transcript_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_memories_transcript_status
    ON memories(transcript_id, status);
CREATE INDEX IF NOT EXISTS idx_memories_event_date
    ON memories(event_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_supersedes
    ON memories(supersedes_memory_id)
    WHERE supersedes_memory_id IS NOT NULL AND status != 'deleted';
CREATE INDEX IF NOT EXISTS idx_memory_sources_memory
    ON memory_sources(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_gaps_status_importance
    ON memory_gaps(status, importance_score DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_gaps_memory
    ON memory_gaps(memory_id, status);
CREATE INDEX IF NOT EXISTS idx_memory_gap_candidates_gap
    ON memory_gap_candidates(gap_id, status, deterministic_score DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_active_content_hash
    ON transcripts(content_hash)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON conversation_messages(session_id, created_at);
"""

MEMORY_COLUMN_MIGRATIONS = {
    "title": "TEXT",
    "date_precision": (
        "TEXT NOT NULL DEFAULT 'unknown' "
        "CHECK (date_precision IN "
        "('exact', 'day', 'month', 'year', 'approximate', 'unknown'))"
    ),
    "emotion": "TEXT",
    "uncertainty_notes": "TEXT",
}

TRANSCRIPT_COLUMN_MIGRATIONS = {
    "source_path": "TEXT",
}


class SQLiteDatabase:
    """Own one SQLite connection with enforced foreign keys and transactions."""

    def __init__(self, database_path: Path | str) -> None:
        path_value = str(database_path)
        if path_value != ":memory:":
            path = Path(database_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path_value = str(path)

        self._connection = sqlite3.connect(
            path_value,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        foreign_keys_enabled = self._connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]
        if foreign_keys_enabled != 1:
            self._connection.close()
            raise RuntimeError("SQLite foreign key enforcement is unavailable")

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the managed connection for repository operations and tests."""
        return self._connection

    def initialize(self) -> None:
        """Create every application table and index idempotently."""
        with self._lock:
            self._connection.executescript(SCHEMA_SQL)
            existing_transcript_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(transcripts)"
                ).fetchall()
            }
            for column, definition in TRANSCRIPT_COLUMN_MIGRATIONS.items():
                if column not in existing_transcript_columns:
                    self._connection.execute(
                        f"ALTER TABLE transcripts ADD COLUMN {column} {definition}"
                    )
            self._connection.execute(
                "UPDATE transcripts SET source_path = 'data/raw/transcripts/' || filename "
                "WHERE source_path IS NULL"
            )
            existing_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(memories)"
                ).fetchall()
            }
            for column, definition in MEMORY_COLUMN_MIGRATIONS.items():
                if column not in existing_columns:
                    self._connection.execute(
                        f"ALTER TABLE memories ADD COLUMN {column} {definition}"
                    )
            self._connection.execute(
                "UPDATE memories SET title = summary "
                "WHERE title IS NULL OR trim(title) = ''"
            )
            self._connection.execute(
                "UPDATE memories SET date_precision = 'exact' "
                "WHERE event_date IS NOT NULL AND date_precision = 'unknown'"
            )
            self._connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit one atomic operation or roll it back on any failure."""
        with self._lock:
            try:
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def table_names(self) -> set[str]:
        """Return application table names, excluding SQLite internals."""
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {str(row["name"]) for row in rows}

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteDatabase:
        self.initialize()
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()
