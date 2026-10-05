"""SQLite job tracking; execution is delegated to FastAPI background tasks."""

from datetime import UTC, datetime
from uuid import uuid4
import json

from backend.app.models.chat_job import ChatJob
from backend.app.models.qa import QAResult
from backend.app.storage.database import SQLiteDatabase


class ChatJobStore:
    def __init__(self, database: SQLiteDatabase):
        self.database = database

    def create(self, session_id: str) -> ChatJob:
        job_id = f"job_{uuid4().hex}"
        created = datetime.now(UTC).isoformat()
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO chat_jobs (job_id,session_id,status,created_at,result_json,error) VALUES (?, ?, 'queued', ?, NULL, NULL)",
                               (job_id, session_id, created))
        return self.get(job_id, session_id)

    def get(self, job_id: str, session_id: str) -> ChatJob | None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM chat_jobs WHERE job_id = ? AND session_id = ?",
                                     (job_id, session_id)).fetchone()
        if row is None:
            return None
        return ChatJob(job_id=row["job_id"], session_id=row["session_id"], status=row["status"],
                       created_at=row["created_at"], error=row["error"],
                       failure_code=row["failure_code"], failure_stage=row["failure_stage"],
                       progress=json.loads(row['progress_json']) if row['progress_json'] else {},
                       result=QAResult.model_validate_json(row["result_json"]) if row["result_json"] else None)

    def start(self, job_id: str) -> bool:
        with self.database.transaction() as connection:
            cursor = connection.execute("UPDATE chat_jobs SET status='running' WHERE job_id=? AND status='queued'", (job_id,))
        return cursor.rowcount == 1

    def progress(self, job_id: str, payload: dict) -> None:
        with self.database.transaction() as connection:
            connection.execute("UPDATE chat_jobs SET progress_json=? WHERE job_id=? AND status='running'",
                               (json.dumps(payload, ensure_ascii=False), job_id))

    def finish(self, job_id: str, result: QAResult | None = None, *, failure_code: str | None = None,
               failure_stage: str | None = None, error_message: str | None = None) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE chat_jobs SET status=?, result_json=?, error=?, failure_code=?, failure_stage=? WHERE job_id=? AND status='running'",
                ("completed" if result else "failed", result.model_dump_json() if result else None,
                 None if result else error_message or "질문 처리에 실패했습니다. 잠시 후 다시 시도해 주세요.",
                 failure_code, failure_stage, job_id),
            )

    def recover_interrupted(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("UPDATE chat_jobs SET status='failed', failure_code='server_restarted', failure_stage='background', error=? WHERE status IN ('queued','running')",
                               ("서버가 재시작되어 질문 처리가 중단됐습니다. 다시 질문해 주세요.",))
