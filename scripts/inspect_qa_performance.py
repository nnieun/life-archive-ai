"""Print timing/status only, without questions, memories, or answer text."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    path = get_settings().sqlite_database_path.resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'qa_performance' not in tables:
            print('No timing records yet. Restart the backend and submit a question.')
            return
        rows = connection.execute('SELECT created_at,payload_json FROM qa_performance ORDER BY rowid DESC LIMIT ?',
                                  (max(1, min(args.limit, 1000)),)).fetchall()
        if not rows:
            print('No timing records yet. Submit a question to collect a measurement.')
        for row in rows:
            print(json.dumps({'created_at': row[0], **json.loads(row[1])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
