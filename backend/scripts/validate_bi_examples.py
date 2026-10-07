"""Run every verified BI example against the real database and report problems.

    cd backend
    uv run python scripts/validate_bi_examples.py

For each example it checks the Python validator, runs the SQL through the same read-only
function the agent uses, and shows the row count and the chart type the agent would pick.
Exits with a non-zero status if any example fails, so it can also run in CI later.
"""
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from src.memory.supabase_client import supabase_admin  # noqa: E402
from src.utils import bi_utils as U  # noqa: E402

EXAMPLES_PATH = os.path.join(BACKEND_DIR, "data", "bi_examples.json")


def main() -> int:
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        examples = json.load(f)

    failures = 0
    print(f"Running {len(examples)} examples against the database...\n")
    for ex in examples:
        question, sql = ex["question"], ex["sql"]

        problem = U.validate_sql(sql)
        if problem:
            failures += 1
            print(f"FAIL  {question}\n      validator: {problem}")
            continue

        try:
            res = supabase_admin.rpc("execute_sql_query", {"sql_query": sql}).execute()
            data = res.data if res.data is not None else []
        except Exception as e:  # network / RPC problems
            failures += 1
            print(f"FAIL  {question}\n      rpc error: {e}")
            continue

        if isinstance(data, list) and data and isinstance(data[0], dict) and set(data[0]) == {"error"}:
            failures += 1
            print(f"FAIL  {question}\n      database: {data[0]['error']}")
            continue

        analysis = U.analyze_result(question, data)
        print(f"ok    {question}  ->  {len(data)} rows, chart: {analysis['chart_type']}")

    print(f"\n{len(examples) - failures}/{len(examples)} examples ran successfully.")
    if failures:
        print("Fix the failing examples (or the views in schema.sql) and re-run.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
