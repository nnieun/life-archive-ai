"""Inspect local failure diagnostics for one conversation session."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    args = parser.parse_args()
    database = SQLiteDatabase(get_settings().sqlite_database_path)
    database.initialize()
    try:
        for failure in SQLiteRepository(database).list_qa_failures(args.session_id):
            print(failure.model_dump_json(indent=2))
    finally:
        database.close()


if __name__ == "__main__":
    main()
