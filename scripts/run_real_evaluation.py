"""Run the opt-in real-embedding retrieval evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.real_runner import run_real_evaluation  # noqa: E402


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for the real-embedding evaluation.")
        return 2
    output = run_real_evaluation(
        PROJECT_ROOT / "evaluation" / "dataset.json",
        PROJECT_ROOT / "reports",
    )
    print(output.relative_to(PROJECT_ROOT))
    print("reports/real_evaluation_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
