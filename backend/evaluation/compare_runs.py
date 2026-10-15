"""Print a side-by-side table of several evaluation runs (for ablations).

    cd backend
    uv run python -m evaluation.compare_runs evaluation/results/*.json
"""
import json
import sys

from evaluation.report import render_comparison


def main() -> None:
    paths = sys.argv[1:]
    if len(paths) < 2:
        raise SystemExit("Usage: python -m evaluation.compare_runs RESULT1.json RESULT2.json [...]")
    results = [json.load(open(p, encoding="utf-8")) for p in paths]
    print(render_comparison(results))


if __name__ == "__main__":
    main()
