"""Typed CRUD operations for the SQLite source of truth."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from backend.app.models.gap import (
    MemoryGapCandidateCreate,
    MemoryGapCandidateRecord,
    MemoryGapCandidateStatus,
    MemoryGapCreate,
    MemoryGapRecord,
    MemoryGapStatus,
    MemoryGapUpdate,
)
from backend.app.models.transcript import LoadedTranscript
from backend.app.models.privacy import SQLiteTranscriptDeletion
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import (
    AutobiographyContent,
    AutobiographyCreate,
    AutobiographyRecord,
    AutobiographyStatus,
    ConversationMessageCreate,
    ConversationMessageRecord,
    ConversationSessionCreate,
    ConversationSessionRecord,
    MemoryCreate,
    MemoryRecord,
    MemorySourceCreate,
    MemorySourceRecord,
    MemoryStatus,
    MemoryUpdate,
    TranscriptMetadataUpdate,
    TranscriptRecord,
    TranscriptSegmentCreate,
    TranscriptSegmentRecord,
)


class StorageError(RuntimeError):
    """Base class for privacy-safe persistence errors."""


class StorageConflictError(StorageError):
    """A unique identifier or content hash already exists."""


class StorageIntegrityError(StorageError):
    """A foreign key, check, or required-field constraint was rejected."""


class StorageNotFoundError(StorageError):
    """The requested active record does not exist."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _datetime_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_references_memory(
    serialized: str,
    memory_ids: set[str],
) -> bool:
    """Check structured citation fields without searching prose text."""

    if not memory_ids:
        return False
    try:
        root = json.loads(serialized)
    except json.JSONDecodeError as exception:
        raise StorageIntegrityError("Stored citation JSON is invalid") from exception

    pending = [root]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            memory_id = value.get("memory_id")
            if isinstance(memory_id, str) and memory_id in memory_ids:
                return True
            referenced_ids = value.get("memory_ids")
            if (
                isinstance(referenced_ids, list)
                and any(
                    isinstance(item, str) and item in memory_ids
                    for item in referenced_ids
                )
            ):
                return True
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return False


def _translate_integrity(exception: sqlite3.IntegrityError) -> StorageIntegrityError:
    return StorageIntegrityError("SQLite rejected a record constraint")


def _transcript_record(row: sqlite3.Row) -> TranscriptRecord:
    return TranscriptRecord.model_validate(dict(row))


def _segment_record(row: sqlite3.Row) -> TranscriptSegmentRecord:
    return TranscriptSegmentRecord.model_validate(dict(row))


def _memory_record(row: sqlite3.Row) -> MemoryRecord:
    data = dict(row)
    data["people"] = json.loads(data.pop("people_json"))
    return MemoryRecord.model_validate(data)


def _memory_source_record(row: sqlite3.Row) -> MemorySourceRecord:
    return MemorySourceRecord.model_validate(dict(row))


def _memory_gap_record(row: sqlite3.Row) -> MemoryGapRecord:
    data = dict(row)
    data["people"] = json.loads(data.pop("people_json"))
    data["user_clues"] = json.loads(data.pop("user_clues_json"))
    data["web_search_consent"] = bool(data["web_search_consent"])
    return MemoryGapRecord.model_validate(data)


def _memory_gap_candidate_record(row: sqlite3.Row) -> MemoryGapCandidateRecord:
    data = dict(row)
    data["supporting_source_ids"] = json.loads(
        data.pop("supporting_source_ids_json")
    )
    data["external_sources"] = json.loads(data.pop("external_sources_json"))
    return MemoryGapCandidateRecord.model_validate(data)


def _session_record(row: sqlite3.Row) -> ConversationSessionRecord:
    return ConversationSessionRecord.model_validate(dict(row))


def _message_record(row: sqlite3.Row) -> ConversationMessageRecord:
    data = dict(row)
    data["citations"] = json.loads(data.pop("citations_json"))
    return ConversationMessageRecord.model_validate(data)


def _autobiography_record(row: sqlite3.Row) -> AutobiographyRecord:
    data = dict(row)
    data["content"] = json.loads(data.pop("content_json"))
    return AutobiographyRecord.model_validate(data)


class SQLiteRepository:
    """Explicit storage operations; SQLite remains the only source of truth."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def create_transcript(self, transcript: LoadedTranscript) -> TranscriptRecord:
        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO transcripts (
                        transcript_id, filename, source_path, recording_id, language,
                        source_type, uploaded_at, recorded_at, content_hash,
                        raw_content, normalized_content, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transcript.transcript_id,
                        transcript.filename,
                        transcript.source_path,
                        transcript.recording_id,
                        transcript.language,
                        transcript.source_type,
                        transcript.uploaded_at.isoformat(),
                        _datetime_iso(transcript.recorded_at),
                        transcript.content_hash,
                        transcript.raw_content,
                        transcript.normalized_content,
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise StorageConflictError("Transcript already exists") from exception
        record = self.get_transcript(transcript.transcript_id)
        if record is None:
            raise StorageError("Transcript was not persisted")
        return record

    def delete_transcript(self, transcript_id: str) -> bool:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM transcripts WHERE transcript_id = ?",
                (transcript_id,),
            )
        return cursor.rowcount == 1

    def get_transcript(
        self,
        transcript_id: str,
        *,
        include_deleted: bool = False,
    ) -> TranscriptRecord | None:
        where = "transcript_id = ?"
        if not include_deleted:
            where += " AND deleted_at IS NULL"
        with self._database.transaction() as connection:
            row = connection.execute(
                f"SELECT * FROM transcripts WHERE {where}",
                (transcript_id,),
            ).fetchone()
        return _transcript_record(row) if row is not None else None

    def list_transcripts(self, *, include_deleted: bool = False) -> list[TranscriptRecord]:
        where = "" if include_deleted else "WHERE deleted_at IS NULL"
        with self._database.transaction() as connection:
            rows = connection.execute(
                f"SELECT * FROM transcripts {where} ORDER BY created_at, transcript_id"
            ).fetchall()
        return [_transcript_record(row) for row in rows]

    def update_transcript_metadata(
        self,
        transcript_id: str,
        update: TranscriptMetadataUpdate,
    ) -> TranscriptRecord:
        fields = update.model_fields_set
        if not fields:
            record = self.get_transcript(transcript_id)
            if record is None:
                raise StorageNotFoundError("Transcript was not found")
            return record
        if ("event_date" in fields) != ("date_precision" in fields):
            raise StorageIntegrityError(
                "event_date and date_precision must be updated together"
            )

        assignments: list[str] = []
        values: list[Any] = []
        for field in sorted(fields):
            value = getattr(update, field)
            if field == "source_type" and value is None:
                raise StorageIntegrityError("source_type cannot be cleared")
            assignments.append(f"{field} = ?")
            values.append(_datetime_iso(value) if isinstance(value, datetime) else value)
        assignments.append("updated_at = ?")
        values.extend((_now_iso(), transcript_id))

        try:
            with self._database.transaction() as connection:
                cursor = connection.execute(
                    f"UPDATE transcripts SET {', '.join(assignments)} "
                    "WHERE transcript_id = ? AND deleted_at IS NULL",
                    values,
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        if cursor.rowcount != 1:
            raise StorageNotFoundError("Transcript was not found")
        record = self.get_transcript(transcript_id)
        if record is None:
            raise StorageNotFoundError("Transcript was not found")
        return record

    def soft_delete_transcript(self, transcript_id: str) -> bool:
        timestamp = _now_iso()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE transcripts SET deleted_at = ?, updated_at = ? "
                "WHERE transcript_id = ? AND deleted_at IS NULL",
                (timestamp, timestamp, transcript_id),
            )
        return cursor.rowcount == 1

    def soft_delete_transcript_cascade(
        self,
        transcript_id: str,
    ) -> SQLiteTranscriptDeletion:
        """Logically delete one transcript and invalidate cited derivatives."""

        timestamp = _now_iso()
        with self._database.transaction() as connection:
            transcript = connection.execute(
                "SELECT transcript_id FROM transcripts "
                "WHERE transcript_id = ? AND deleted_at IS NULL",
                (transcript_id,),
            ).fetchone()
            if transcript is None:
                raise StorageNotFoundError("Transcript was not found")

            memory_rows = connection.execute(
                "SELECT memory_id, status FROM memories "
                "WHERE transcript_id = ? ORDER BY memory_id",
                (transcript_id,),
            ).fetchall()
            memory_ids = [str(row["memory_id"]) for row in memory_rows]
            memory_id_set = set(memory_ids)

            message_ids = [
                str(row["message_id"])
                for row in connection.execute(
                    "SELECT message_id, citations_json "
                    "FROM conversation_messages WHERE deleted_at IS NULL"
                ).fetchall()
                if _json_references_memory(
                    row["citations_json"],
                    memory_id_set,
                )
            ]
            autobiography_ids = [
                str(row["autobiography_id"])
                for row in connection.execute(
                    "SELECT autobiography_id, content_json "
                    "FROM autobiographies WHERE status != 'deleted'"
                ).fetchall()
                if _json_references_memory(
                    row["content_json"],
                    memory_id_set,
                )
            ]

            segment_cursor = connection.execute(
                "UPDATE transcript_segments "
                "SET deleted_at = ?, updated_at = ? "
                "WHERE transcript_id = ? AND deleted_at IS NULL",
                (timestamp, timestamp, transcript_id),
            )
            gap_cursor = connection.execute(
                "UPDATE memory_gaps SET status = 'DISMISSED', "
                "resolved_at = COALESCE(resolved_at, ?), updated_at = ? "
                "WHERE memory_id IN ("
                "SELECT memory_id FROM memories WHERE transcript_id = ?"
                ") AND status NOT IN ('RESOLVED', 'DISMISSED')",
                (timestamp, timestamp, transcript_id),
            )
            memory_cursor = connection.execute(
                "UPDATE memories SET status = 'deleted', "
                "deleted_at = ?, updated_at = ? "
                "WHERE transcript_id = ? AND status != 'deleted'",
                (timestamp, timestamp, transcript_id),
            )
            if message_ids:
                connection.executemany(
                    "UPDATE conversation_messages "
                    "SET deleted_at = ?, updated_at = ? "
                    "WHERE message_id = ? AND deleted_at IS NULL",
                    [
                        (timestamp, timestamp, message_id)
                        for message_id in message_ids
                    ],
                )
            if autobiography_ids:
                connection.executemany(
                    "UPDATE autobiographies SET status = 'deleted', "
                    "deleted_at = ?, updated_at = ? "
                    "WHERE autobiography_id = ? AND status != 'deleted'",
                    [
                        (timestamp, timestamp, autobiography_id)
                        for autobiography_id in autobiography_ids
                    ],
                )
            connection.execute(
                "UPDATE transcripts SET deleted_at = ?, updated_at = ? "
                "WHERE transcript_id = ? AND deleted_at IS NULL",
                (timestamp, timestamp, transcript_id),
            )

        return SQLiteTranscriptDeletion(
            transcript_id=transcript_id,
            memory_ids=memory_ids,
            deleted_segment_count=segment_cursor.rowcount,
            deleted_memory_count=memory_cursor.rowcount,
            dismissed_gap_count=gap_cursor.rowcount,
            invalidated_conversation_message_count=len(message_ids),
            invalidated_autobiography_count=len(autobiography_ids),
        )

    def create_segment(
        self,
        segment: TranscriptSegmentCreate,
    ) -> TranscriptSegmentRecord:
        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO transcript_segments (
                        segment_id, transcript_id, chunk_index, content,
                        start_offset, end_offset, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        segment.segment_id,
                        segment.transcript_id,
                        segment.chunk_index,
                        segment.content,
                        segment.start_offset,
                        segment.end_offset,
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        rows = self.list_segments(segment.transcript_id)
        return next(row for row in rows if row.segment_id == segment.segment_id)

    def create_segments(
        self,
        segments: list[TranscriptSegmentCreate],
    ) -> list[TranscriptSegmentRecord]:
        """Persist one transcript's chunks atomically in chunk order."""
        if not segments:
            return []

        transcript_ids = {segment.transcript_id for segment in segments}
        if len(transcript_ids) != 1:
            raise StorageIntegrityError(
                "A segment batch must belong to one transcript"
            )
        transcript_id = next(iter(transcript_ids))
        transcript = self.get_transcript(transcript_id)
        if transcript is None:
            raise StorageIntegrityError("Segments require an active transcript")

        expected_indexes = list(range(len(segments)))
        actual_indexes = [segment.chunk_index for segment in segments]
        if actual_indexes != expected_indexes:
            raise StorageIntegrityError(
                "Segment chunk indexes must be contiguous and ordered"
            )
        for segment in segments:
            expected_content = transcript.normalized_content[
                segment.start_offset : segment.end_offset
            ]
            if segment.content != expected_content:
                raise StorageIntegrityError(
                    "Segment content must match its transcript offset range"
                )

        timestamp = _now_iso()
        values = [
            (
                segment.segment_id,
                segment.transcript_id,
                segment.chunk_index,
                segment.content,
                segment.start_offset,
                segment.end_offset,
                timestamp,
                timestamp,
            )
            for segment in segments
        ]
        try:
            with self._database.transaction() as connection:
                connection.executemany(
                    """
                    INSERT INTO transcript_segments (
                        segment_id, transcript_id, chunk_index, content,
                        start_offset, end_offset, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        return self.list_segments(transcript_id)

    def list_segments(
        self,
        transcript_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[TranscriptSegmentRecord]:
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM transcript_segments WHERE transcript_id = ?"
                f"{deleted_clause} ORDER BY chunk_index",
                (transcript_id,),
            ).fetchall()
        return [_segment_record(row) for row in rows]

    def get_segment(
        self,
        segment_id: str,
        *,
        include_deleted: bool = False,
    ) -> TranscriptSegmentRecord | None:
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM transcript_segments WHERE segment_id = ?"
                f"{deleted_clause}",
                (segment_id,),
            ).fetchone()
        return _segment_record(row) if row is not None else None

    def soft_delete_segment(self, segment_id: str) -> bool:
        timestamp = _now_iso()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE transcript_segments SET deleted_at = ?, updated_at = ? "
                "WHERE segment_id = ? AND deleted_at IS NULL",
                (timestamp, timestamp, segment_id),
            )
        return cursor.rowcount == 1

    def create_memory(self, memory: MemoryCreate) -> MemoryRecord:
        if self.get_transcript(memory.transcript_id) is None:
            raise StorageIntegrityError("Memory requires an active transcript")
        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO memories (
                        memory_id, transcript_id, title, summary, people_json,
                        location, event_date, date_precision, emotion, confidence,
                        uncertainty_notes, status, supersedes_memory_id,
                        created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.memory_id,
                        memory.transcript_id,
                        memory.title,
                        memory.summary,
                        _json_dump(memory.people),
                        memory.location,
                        memory.event_date,
                        memory.date_precision.value,
                        memory.emotion,
                        memory.confidence,
                        memory.uncertainty_notes,
                        memory.status.value,
                        memory.supersedes_memory_id,
                        timestamp,
                        timestamp,
                        timestamp if memory.status is MemoryStatus.DELETED else None,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        record = self.get_memory(memory.memory_id, include_deleted=True)
        if record is None:
            raise StorageError("Memory was not persisted")
        return record

    def get_memory(
        self,
        memory_id: str,
        *,
        include_deleted: bool = False,
    ) -> MemoryRecord | None:
        where = "memory_id = ?"
        if not include_deleted:
            where += " AND status != 'deleted'"
        with self._database.transaction() as connection:
            row = connection.execute(
                f"SELECT * FROM memories WHERE {where}",
                (memory_id,),
            ).fetchone()
        return _memory_record(row) if row is not None else None

    def list_memories(
        self,
        transcript_id: str | None = None,
        *,
        include_deleted: bool = False,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]:
        """List visible memories, hiding rows a live correction replaced.

        Every reader of the memory set - timeline, retrieval, both search
        indexes and ``GET /memories`` - goes through here, so filtering
        supersession at this one point is what keeps them from disagreeing.
        """

        clauses: list[str] = []
        values: list[Any] = []
        if transcript_id is not None:
            clauses.append("memories.transcript_id = ?")
            values.append(transcript_id)
        if not include_deleted:
            clauses.append("memories.status != 'deleted'")
        if not include_superseded:
            clauses.append(
                "NOT EXISTS ("
                "SELECT 1 FROM memories AS correction "
                "WHERE correction.supersedes_memory_id = memories.memory_id "
                "AND correction.status != 'deleted')"
            )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._database.transaction() as connection:
            rows = connection.execute(
                f"SELECT memories.* FROM memories {where} "
                "ORDER BY memories.created_at, memories.memory_id",
                values,
            ).fetchall()
        return [_memory_record(row) for row in rows]

    def get_correction_of(self, memory_id: str) -> MemoryRecord | None:
        """Return the live correction that replaced this memory, if any."""

        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE supersedes_memory_id = ? "
                "AND status != 'deleted'",
                (memory_id,),
            ).fetchone()
        return _memory_record(row) if row is not None else None

    def create_memory_correction(
        self,
        correction: MemoryCreate,
        sources: list[MemorySourceCreate],
    ) -> MemoryRecord:
        """Persist one correction and its inherited evidence atomically."""

        if correction.supersedes_memory_id is None:
            raise StorageIntegrityError("A correction must supersede a memory")
        if correction.status is not MemoryStatus.CORRECTED:
            raise StorageIntegrityError("A correction must have corrected status")
        if not sources:
            raise StorageIntegrityError("A correction requires inherited evidence")
        if any(source.memory_id != correction.memory_id for source in sources):
            raise StorageIntegrityError("Memory and source identifiers must match")
        superseded = self.get_memory(correction.supersedes_memory_id)
        if superseded is None:
            raise StorageNotFoundError("Superseded memory was not found")

        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO memories (
                        memory_id, transcript_id, title, summary, people_json,
                        location, event_date, date_precision, emotion, confidence,
                        uncertainty_notes, status, supersedes_memory_id,
                        created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        correction.memory_id,
                        correction.transcript_id,
                        correction.title,
                        correction.summary,
                        _json_dump(correction.people),
                        correction.location,
                        correction.event_date,
                        correction.date_precision.value,
                        correction.emotion,
                        correction.confidence,
                        correction.uncertainty_notes,
                        correction.status.value,
                        correction.supersedes_memory_id,
                        timestamp,
                        timestamp,
                        None,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO memory_sources (
                        memory_source_id, memory_id, transcript_id, segment_id,
                        start_offset, end_offset, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            source.memory_source_id,
                            source.memory_id,
                            source.transcript_id,
                            source.segment_id,
                            source.start_offset,
                            source.end_offset,
                            timestamp,
                            timestamp,
                        )
                        for source in sources
                    ],
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception

        record = self.get_memory(correction.memory_id)
        if record is None:
            raise StorageError("Memory correction was not persisted")
        return record

    def update_memory(self, memory_id: str, update: MemoryUpdate) -> MemoryRecord:
        fields = update.model_fields_set
        if not fields:
            record = self.get_memory(memory_id, include_deleted=True)
            if record is None:
                raise StorageNotFoundError("Memory was not found")
            return record

        assignments: list[str] = []
        values: list[Any] = []
        for field in sorted(fields):
            value = getattr(update, field)
            if (
                field
                in {
                    "title",
                    "summary",
                    "people",
                    "date_precision",
                    "confidence",
                    "status",
                }
                and value is None
            ):
                raise StorageIntegrityError(f"{field} cannot be cleared")
            column = "people_json" if field == "people" else field
            if field == "people":
                value = _json_dump(value)
            elif field in {"date_precision", "status"} and value is not None:
                value = value.value
            assignments.append(f"{column} = ?")
            values.append(value)

        status = update.status if "status" in fields else None
        if status is MemoryStatus.DELETED:
            assignments.append("deleted_at = ?")
            values.append(_now_iso())
        elif status is not None:
            assignments.append("deleted_at = NULL")
        assignments.append("updated_at = ?")
        values.extend((_now_iso(), memory_id))

        try:
            with self._database.transaction() as connection:
                cursor = connection.execute(
                    f"UPDATE memories SET {', '.join(assignments)} WHERE memory_id = ?",
                    values,
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        if cursor.rowcount != 1:
            raise StorageNotFoundError("Memory was not found")
        record = self.get_memory(memory_id, include_deleted=True)
        if record is None:
            raise StorageNotFoundError("Memory was not found")
        return record

    def soft_delete_memory(self, memory_id: str) -> MemoryRecord:
        return self.update_memory(
            memory_id,
            MemoryUpdate(status=MemoryStatus.DELETED),
        )

    def create_memory_source(
        self,
        source: MemorySourceCreate,
    ) -> MemorySourceRecord:
        memory = self.get_memory(source.memory_id, include_deleted=True)
        if memory is None or memory.transcript_id != source.transcript_id:
            raise StorageIntegrityError("Memory source transcript does not match")
        if source.segment_id is not None:
            segments = self.list_segments(source.transcript_id, include_deleted=True)
            if source.segment_id not in {segment.segment_id for segment in segments}:
                raise StorageIntegrityError("Memory source segment does not match")
        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO memory_sources (
                        memory_source_id, memory_id, transcript_id, segment_id,
                        start_offset, end_offset, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source.memory_source_id,
                        source.memory_id,
                        source.transcript_id,
                        source.segment_id,
                        source.start_offset,
                        source.end_offset,
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        record = self.get_memory_source(source.memory_source_id)
        if record is None:
            raise StorageError("Memory source was not persisted")
        return record

    def create_memories_with_sources(
        self,
        items: list[tuple[MemoryCreate, MemorySourceCreate]],
    ) -> list[MemoryRecord]:
        """Atomically persist extracted memories and their transcript evidence."""
        if not items:
            return []

        validated: list[
            tuple[MemoryCreate, MemorySourceCreate, TranscriptSegmentRecord]
        ] = []
        for memory, source in items:
            if (
                memory.memory_id != source.memory_id
                or memory.transcript_id != source.transcript_id
            ):
                raise StorageIntegrityError(
                    "Memory and source identifiers must match"
                )
            if self.get_transcript(memory.transcript_id) is None:
                raise StorageIntegrityError("Memory requires an active transcript")
            if source.segment_id is None:
                raise StorageIntegrityError(
                    "Extracted memory requires a transcript segment"
                )
            segment = self.get_segment(source.segment_id)
            if segment is None or segment.transcript_id != memory.transcript_id:
                raise StorageIntegrityError("Memory source segment does not match")
            if (
                source.start_offset < segment.start_offset
                or source.end_offset > segment.end_offset
                or source.end_offset <= source.start_offset
            ):
                raise StorageIntegrityError(
                    "Memory source offsets must be inside its segment"
                )
            validated.append((memory, source, segment))

        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                for memory, source, _segment in validated:
                    connection.execute(
                        """
                        INSERT INTO memories (
                            memory_id, transcript_id, title, summary, people_json,
                            location, event_date, date_precision, emotion,
                            confidence, uncertainty_notes, status,
                            supersedes_memory_id, created_at, updated_at, deleted_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            memory.memory_id,
                            memory.transcript_id,
                            memory.title,
                            memory.summary,
                            _json_dump(memory.people),
                            memory.location,
                            memory.event_date,
                            memory.date_precision.value,
                            memory.emotion,
                            memory.confidence,
                            memory.uncertainty_notes,
                            memory.status.value,
                            memory.supersedes_memory_id,
                            timestamp,
                            timestamp,
                            timestamp
                            if memory.status is MemoryStatus.DELETED
                            else None,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_sources (
                            memory_source_id, memory_id, transcript_id, segment_id,
                            start_offset, end_offset, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source.memory_source_id,
                            source.memory_id,
                            source.transcript_id,
                            source.segment_id,
                            source.start_offset,
                            source.end_offset,
                            timestamp,
                            timestamp,
                        ),
                    )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception

        records = [
            self.get_memory(memory.memory_id)
            for memory, _source, _segment in validated
        ]
        if any(record is None for record in records):
            raise StorageError("Extracted memories were not persisted")
        return [record for record in records if record is not None]

    def get_memory_source(
        self,
        memory_source_id: str,
    ) -> MemorySourceRecord | None:
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memory_sources WHERE memory_source_id = ?",
                (memory_source_id,),
            ).fetchone()
        return _memory_source_record(row) if row is not None else None

    def list_memory_sources(self, memory_id: str) -> list[MemorySourceRecord]:
        with self._database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_sources WHERE memory_id = ? "
                "ORDER BY start_offset, memory_source_id",
                (memory_id,),
            ).fetchall()
        return [_memory_source_record(row) for row in rows]

    def delete_memory_source(self, memory_source_id: str) -> bool:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_sources WHERE memory_source_id = ?",
                (memory_source_id,),
            )
        return cursor.rowcount == 1

    def create_memory_gap(self, gap: MemoryGapCreate) -> MemoryGapRecord:
        """Persist one detected gap without modifying its source memory."""

        if gap.memory_id is not None and self.get_memory(gap.memory_id) is None:
            raise StorageIntegrityError("Memory gap requires an active memory")
        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO memory_gaps (
                        gap_id, memory_id, gap_type, clue_text, missing_field,
                        period_start, period_end, location, people_json,
                        confidence, importance_score, status, source_type,
                        source_id, web_search_consent, user_clues_json,
                        resolved_candidate_id, resolved_memory_id,
                        created_at, updated_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              NULL, NULL, ?, ?, NULL)
                    """,
                    (
                        gap.gap_id,
                        gap.memory_id,
                        gap.gap_type.value,
                        gap.clue_text,
                        gap.missing_field,
                        gap.period_start,
                        gap.period_end,
                        gap.location,
                        _json_dump(gap.people),
                        gap.confidence,
                        gap.importance_score,
                        gap.status.value,
                        gap.source_type.value,
                        gap.source_id,
                        int(gap.web_search_consent),
                        _json_dump(gap.user_clues),
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        record = self.get_memory_gap(gap.gap_id)
        if record is None:
            raise StorageError("Memory gap was not persisted")
        return record

    def get_memory_gap(self, gap_id: str) -> MemoryGapRecord | None:
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memory_gaps WHERE gap_id = ?",
                (gap_id,),
            ).fetchone()
        return _memory_gap_record(row) if row is not None else None

    def list_memory_gaps(
        self,
        *,
        statuses: set[MemoryGapStatus] | None = None,
        memory_id: str | None = None,
    ) -> list[MemoryGapRecord]:
        clauses: list[str] = []
        values: list[Any] = []
        if statuses is not None:
            if not statuses:
                return []
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            values.extend(status.value for status in sorted(statuses))
        if memory_id is not None:
            clauses.append("memory_id = ?")
            values.append(memory_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._database.transaction() as connection:
            rows = connection.execute(
                f"SELECT * FROM memory_gaps {where} "
                "ORDER BY importance_score DESC, created_at, gap_id",
                values,
            ).fetchall()
        return [_memory_gap_record(row) for row in rows]

    def update_memory_gap(
        self,
        gap_id: str,
        update: MemoryGapUpdate,
    ) -> MemoryGapRecord:
        fields = update.model_fields_set
        if not fields:
            record = self.get_memory_gap(gap_id)
            if record is None:
                raise StorageNotFoundError("Memory gap was not found")
            return record

        assignments: list[str] = []
        values: list[Any] = []
        for field in sorted(fields):
            value = getattr(update, field)
            if field == "status" and value is not None:
                value = value.value
            elif field == "web_search_consent" and value is not None:
                value = int(value)
            elif field == "user_clues" and value is not None:
                value = _json_dump(value)
            elif field == "resolved_at":
                value = _datetime_iso(value)
            column = "user_clues_json" if field == "user_clues" else field
            assignments.append(f"{column} = ?")
            values.append(value)
        assignments.append("updated_at = ?")
        values.extend((_now_iso(), gap_id))
        try:
            with self._database.transaction() as connection:
                cursor = connection.execute(
                    f"UPDATE memory_gaps SET {', '.join(assignments)} "
                    "WHERE gap_id = ?",
                    values,
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        if cursor.rowcount != 1:
            raise StorageNotFoundError("Memory gap was not found")
        record = self.get_memory_gap(gap_id)
        if record is None:
            raise StorageNotFoundError("Memory gap was not found")
        return record

    def create_memory_gap_candidate(
        self,
        candidate: MemoryGapCandidateCreate,
    ) -> MemoryGapCandidateRecord:
        if self.get_memory_gap(candidate.gap_id) is None:
            raise StorageIntegrityError("Candidate requires a memory gap")
        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO memory_gap_candidates (
                        candidate_id, gap_id, value, explanation,
                        deterministic_score, llm_relation,
                        supporting_source_ids_json, external_sources_json,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.candidate_id,
                        candidate.gap_id,
                        candidate.value,
                        candidate.explanation,
                        candidate.deterministic_score,
                        candidate.llm_relation.value,
                        _json_dump(candidate.supporting_source_ids),
                        _json_dump(
                            [
                                source.model_dump(mode="json")
                                for source in candidate.external_sources
                            ]
                        ),
                        candidate.status.value,
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        record = self.get_memory_gap_candidate(candidate.candidate_id)
        if record is None:
            raise StorageError("Memory gap candidate was not persisted")
        return record

    def get_memory_gap_candidate(
        self,
        candidate_id: str,
    ) -> MemoryGapCandidateRecord | None:
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memory_gap_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return _memory_gap_candidate_record(row) if row is not None else None

    def list_memory_gap_candidates(
        self,
        gap_id: str,
        *,
        statuses: set[MemoryGapCandidateStatus] | None = None,
    ) -> list[MemoryGapCandidateRecord]:
        clauses = ["gap_id = ?"]
        values: list[Any] = [gap_id]
        if statuses is not None:
            if not statuses:
                return []
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            values.extend(status.value for status in sorted(statuses))
        with self._database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_gap_candidates WHERE "
                f"{' AND '.join(clauses)} "
                "ORDER BY deterministic_score DESC, created_at, candidate_id",
                values,
            ).fetchall()
        return [_memory_gap_candidate_record(row) for row in rows]

    def update_memory_gap_candidate_status(
        self,
        candidate_id: str,
        status: MemoryGapCandidateStatus,
    ) -> MemoryGapCandidateRecord:
        timestamp = _now_iso()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE memory_gap_candidates SET status = ?, updated_at = ? "
                "WHERE candidate_id = ?",
                (status.value, timestamp, candidate_id),
            )
        if cursor.rowcount != 1:
            raise StorageNotFoundError("Memory gap candidate was not found")
        record = self.get_memory_gap_candidate(candidate_id)
        if record is None:
            raise StorageNotFoundError("Memory gap candidate was not found")
        return record

    def create_conversation_session(
        self,
        session: ConversationSessionCreate,
    ) -> ConversationSessionRecord:
        timestamp = _now_iso()
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    "INSERT INTO conversation_sessions "
                    "(session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session.session_id, session.title, timestamp, timestamp),
                )
        except sqlite3.IntegrityError as exception:
            raise StorageConflictError("Conversation session already exists") from exception
        record = self.get_conversation_session(session.session_id)
        if record is None:
            raise StorageError("Conversation session was not persisted")
        return record

    def get_conversation_session(
        self,
        session_id: str,
        *,
        include_deleted: bool = False,
    ) -> ConversationSessionRecord | None:
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_sessions WHERE session_id = ?"
                f"{deleted_clause}",
                (session_id,),
            ).fetchone()
        return _session_record(row) if row is not None else None

    def update_conversation_session_title(
        self,
        session_id: str,
        title: str | None,
    ) -> ConversationSessionRecord:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET title = ?, updated_at = ? "
                "WHERE session_id = ? AND deleted_at IS NULL",
                (title, _now_iso(), session_id),
            )
        if cursor.rowcount != 1:
            raise StorageNotFoundError("Conversation session was not found")
        record = self.get_conversation_session(session_id)
        if record is None:
            raise StorageNotFoundError("Conversation session was not found")
        return record

    def add_conversation_message(
        self,
        message: ConversationMessageCreate,
    ) -> ConversationMessageRecord:
        timestamp = _now_iso()
        citations_json = _json_dump(
            [citation.model_dump(mode="json") for citation in message.citations]
        )
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO conversation_messages (
                        message_id, session_id, role, content, citations_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.session_id,
                        message.role,
                        message.content,
                        citations_json,
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        record = self.get_conversation_message(message.message_id)
        if record is None:
            raise StorageError("Conversation message was not persisted")
        return record

    def get_conversation_message(
        self,
        message_id: str,
        *,
        include_deleted: bool = False,
    ) -> ConversationMessageRecord | None:
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_messages WHERE message_id = ?"
                f"{deleted_clause}",
                (message_id,),
            ).fetchone()
        return _message_record(row) if row is not None else None

    def list_conversation_messages(
        self,
        session_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[ConversationMessageRecord]:
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM conversation_messages WHERE session_id = ?"
                f"{deleted_clause} ORDER BY created_at, message_id",
                (session_id,),
            ).fetchall()
        return [_message_record(row) for row in rows]

    def soft_delete_conversation_message(self, message_id: str) -> bool:
        timestamp = _now_iso()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE conversation_messages SET deleted_at = ?, updated_at = ? "
                "WHERE message_id = ? AND deleted_at IS NULL",
                (timestamp, timestamp, message_id),
            )
        return cursor.rowcount == 1

    def soft_delete_conversation_session(self, session_id: str) -> bool:
        timestamp = _now_iso()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET deleted_at = ?, updated_at = ? "
                "WHERE session_id = ? AND deleted_at IS NULL",
                (timestamp, timestamp, session_id),
            )
        return cursor.rowcount == 1

    def create_autobiography(
        self,
        autobiography: AutobiographyCreate,
    ) -> AutobiographyRecord:
        timestamp = _now_iso()
        content_json = _json_dump(autobiography.content.model_dump(mode="json"))
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO autobiographies (
                        autobiography_id, title, content_json, status,
                        created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        autobiography.autobiography_id,
                        autobiography.title,
                        content_json,
                        autobiography.status.value,
                        timestamp,
                        timestamp,
                        timestamp
                        if autobiography.status is AutobiographyStatus.DELETED
                        else None,
                    ),
                )
        except sqlite3.IntegrityError as exception:
            raise _translate_integrity(exception) from exception
        record = self.get_autobiography(
            autobiography.autobiography_id,
            include_deleted=True,
        )
        if record is None:
            raise StorageError("Autobiography was not persisted")
        return record

    def get_autobiography(
        self,
        autobiography_id: str,
        *,
        include_deleted: bool = False,
    ) -> AutobiographyRecord | None:
        deleted_clause = "" if include_deleted else " AND status != 'deleted'"
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM autobiographies WHERE autobiography_id = ?"
                f"{deleted_clause}",
                (autobiography_id,),
            ).fetchone()
        return _autobiography_record(row) if row is not None else None

    def update_autobiography(
        self,
        autobiography_id: str,
        *,
        title: str | None = None,
        content: AutobiographyContent | None = None,
        status: AutobiographyStatus | None = None,
    ) -> AutobiographyRecord:
        assignments: list[str] = []
        values: list[Any] = []
        if title is not None:
            assignments.append("title = ?")
            values.append(title)
        if content is not None:
            assignments.append("content_json = ?")
            values.append(_json_dump(content.model_dump(mode="json")))
        if status is not None:
            assignments.append("status = ?")
            values.append(status.value)
            if status is AutobiographyStatus.DELETED:
                assignments.append("deleted_at = ?")
                values.append(_now_iso())
            else:
                assignments.append("deleted_at = NULL")
        if not assignments:
            record = self.get_autobiography(autobiography_id, include_deleted=True)
            if record is None:
                raise StorageNotFoundError("Autobiography was not found")
            return record
        assignments.append("updated_at = ?")
        values.extend((_now_iso(), autobiography_id))
        with self._database.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE autobiographies SET {', '.join(assignments)} "
                "WHERE autobiography_id = ?",
                values,
            )
        if cursor.rowcount != 1:
            raise StorageNotFoundError("Autobiography was not found")
        record = self.get_autobiography(autobiography_id, include_deleted=True)
        if record is None:
            raise StorageNotFoundError("Autobiography was not found")
        return record
