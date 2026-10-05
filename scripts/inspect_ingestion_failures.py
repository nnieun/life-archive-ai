"""Show recent failed uploads with the recorded cause chain."""

import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.core.config import get_settings


def main() -> None:
    connection = sqlite3.connect(get_settings().sqlite_database_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT job_id, filename, created_at, failure_code, diagnostic_json FROM ingestion_jobs "
        "WHERE status='failed' ORDER BY rowid DESC LIMIT ?", (int(sys.argv[1]) if len(sys.argv) > 1 else 5,))
    for row in rows:
        diagnostic = json.loads(row['diagnostic_json']) if row['diagnostic_json'] else None
        print(json.dumps({'job_id': row['job_id'], 'filename': row['filename'], 'created_at': row['created_at'],
                          'failure_code': row['failure_code'], 'diagnostic': diagnostic}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
