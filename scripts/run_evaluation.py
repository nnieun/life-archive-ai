"""Run the privacy-safe TASK-015 evaluation matrix."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import run_evaluation  # noqa: E402


def main() -> int:
    outputs = run_evaluation(
        PROJECT_ROOT / "evaluation" / "dataset.json",
        PROJECT_ROOT / "reports",
    )
    for output in outputs:
        print(output.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
