"""Track background uploads in SQLite without persisting duplicate file bodies."""

from datetime import UTC, datetime
import json
from uuid import uuid4

from backend.app.models.ingestion import IngestionResult
from backend.app.models.ingestion_job import IngestionJob
from backend.app.storage.database import SQLiteDatabase


class IngestionJobCancelled(RuntimeError):
    pass


class IngestionJobStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def create(self, session_id: str, filename: str, content_hash: str) -> IngestionJob:
        job_id = 'ingest_' + uuid4().hex
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO ingestion_jobs(job_id,session_id,filename,content_hash,status,created_at) VALUES(?,?,?,?,'queued',?)",
                               (job_id, session_id, filename, content_hash, datetime.now(UTC).isoformat()))
        return self.get(job_id, session_id)

    def get(self, job_id: str, session_id: str) -> IngestionJob | None:
        with self.database.transaction() as connection:
            row = connection.execute('SELECT * FROM ingestion_jobs WHERE job_id=? AND session_id=?', (job_id, session_id)).fetchone()
        if row is None:
            return None
        return IngestionJob(job_id=row['job_id'], session_id=row['session_id'], filename=row['filename'],
            created_at=row['created_at'], status=row['status'], error=row['error'], failure_code=row['failure_code'],
            progress=json.loads(row['progress_json']) if row['progress_json'] else {},
            result=IngestionResult.model_validate_json(row['result_json']) if row['result_json'] else None)

    def start(self, job_id: str) -> bool:
        with self.database.transaction() as connection:
            return connection.execute("UPDATE ingestion_jobs SET status='running' WHERE job_id=? AND status='queued'", (job_id,)).rowcount == 1

    def progress(self, job_id: str, payload: dict) -> None:
        with self.database.transaction() as connection:
            changed = connection.execute("UPDATE ingestion_jobs SET progress_json=?,transcript_id=COALESCE(?,transcript_id) WHERE job_id=? AND status='running'",
                (json.dumps(payload), payload.get('transcript_id'), job_id)).rowcount
        if not changed:
            raise IngestionJobCancelled('Upload processing was cancelled')

    def finish(self, job_id: str, result: IngestionResult | None = None, *, code: str | None = None, error: str | None = None, diagnostic: dict | None = None) -> None:
        with self.database.transaction() as connection:
            connection.execute("UPDATE ingestion_jobs SET status=?,result_json=?,error=?,failure_code=?,diagnostic_json=? WHERE job_id=? AND status='running'",
                ('completed' if result else 'failed', result.model_dump_json() if result else None, error, code,
                 json.dumps(diagnostic, ensure_ascii=False) if diagnostic else None, job_id))

    def recover_interrupted(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("UPDATE ingestion_jobs SET status='failed',failure_code='server_restarted',error=? WHERE status IN ('queued','running')",
                               ('서버 재시작으로 업로드 처리가 중단되었습니다. 기억 등록 상태를 확인한 뒤 다시 시도해 주세요.',))
