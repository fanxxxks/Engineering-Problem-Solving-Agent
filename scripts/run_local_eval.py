"""Run the local evaluation loop against the development set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eng_solver_agent.eval.local_eval import LocalEvalPaths, run_local_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local development evaluation.")
    parser.add_argument("--dev-path", default=ROOT / "data" / "dev" / "dev.json")
    parser.add_argument("--predictions-path", default=ROOT / "data" / "exports" / "dev_predictions.json")
    parser.add_argument("--report-path", default=ROOT / "data" / "exports" / "dev_report.json")
    args = parser.parse_args()

    paths = LocalEvalPaths(
        dev_path=Path(args.dev_path),
        predictions_path=Path(args.predictions_path),
        report_path=Path(args.report_path),
    )
    predictions, report = run_local_eval(paths.dev_path, paths.predictions_path, paths.report_path)
    print(f"saved predictions to {paths.predictions_path}")
    print(f"saved report to {paths.report_path}")
    print(f"questions evaluated: {len(predictions)}")
    print(report)


if __name__ == "__main__":
    main()
