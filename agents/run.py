"""Run the home-win loop once and print the holdout score."""

from __future__ import annotations

import json
from pathlib import Path

from agents.graph import build_graph, initial_state
from pipeline.cbb import PUBLISHED

DEFAULT_PATH = PUBLISHED / "merged_latest.csv"


def main() -> None:
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the Airflow DAG or python -m pipeline.cbb --skip-db first.")
    final = build_graph().invoke(initial_state(str(path)))
    print(json.dumps({"stop_reason": final.get("stop_reason"), "best": final.get("best"), "holdout": final.get("holdout_metrics")}, indent=2))


if __name__ == "__main__":
    main()
