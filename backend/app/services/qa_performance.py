"""Session-scoped verified cache and content-free performance measurements."""

import hashlib
import json
import sqlite3
from uuid import uuid4

from backend.app.models.qa import QAResult
from backend.app.storage.repository import SQLiteRepository


class QAPerformanceStore:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.database = repository._database

    def key(self, session_id: str, question: str, top_k: int, configuration: str) -> str:
        """Invalidate on any authoritative memory, source, or transcript change."""
        with self.database.transaction() as connection:
            return self._key(connection, session_id, question, top_k, configuration)

    @staticmethod
    def _key(connection: sqlite3.Connection, session_id: str, question: str, top_k: int, configuration: str) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps([session_id, question, top_k, configuration], ensure_ascii=False).encode())
        for table, order in (("memories", "memory_id"), ("memory_sources", "memory_source_id"),
                             ("transcript_segments", "segment_id"), ("transcripts", "transcript_id")):
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                digest.update(json.dumps(dict(row), sort_keys=True, ensure_ascii=False).encode())
        return digest.hexdigest()

    def get(self, key: str, session_id: str) -> QAResult | None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT result_json FROM qa_answer_cache c JOIN conversation_sessions s ON s.session_id=c.session_id WHERE c.cache_key=? AND c.session_id=? AND s.deleted_at IS NULL",
                                     (key, session_id)).fetchone()
        return QAResult.model_validate_json(row[0]) if row else None

    def save(self, key: str, result: QAResult, top_k: int, configuration: str) -> None:
        if result.validation_result.passed and result.citations and result.error is None:
            with self.database.transaction() as connection:
                connection.execute('BEGIN IMMEDIATE')
                session = connection.execute('SELECT deleted_at FROM conversation_sessions WHERE session_id=?', (result.session_id,)).fetchone()
                if session is None or session['deleted_at'] is not None:
                    return
                if key != self._key(connection, result.session_id, result.question, top_k, configuration):
                    return
                connection.execute("INSERT OR REPLACE INTO qa_answer_cache(cache_key,session_id,result_json) VALUES (?,?,?)",
                                   (key, result.session_id, result.model_dump_json()))
                connection.execute("DELETE FROM qa_answer_cache WHERE cache_key NOT IN (SELECT cache_key FROM qa_answer_cache ORDER BY rowid DESC LIMIT 512)")

    def measure(self, result: QAResult, configuration: str) -> None:
        payload = {
            "configuration": configuration, "elapsed_ms": result.elapsed_ms,
            "cache_hit": result.cache_hit, "passed": result.validation_result.passed,
            "failure_code": result.validation_result.failure_code, "retry_count": result.retry_count,
            "steps": [{"node": step.node, "elapsed_ms": step.elapsed_ms} for step in result.steps],
        }
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO qa_performance(measurement_id,session_id,payload_json) VALUES (?,?,?)",
                               (uuid4().hex, result.session_id, json.dumps(payload)))
