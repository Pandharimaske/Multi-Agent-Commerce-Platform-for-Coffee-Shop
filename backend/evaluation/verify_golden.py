"""Check that the benchmark's answer key is correct, before trusting any score.

    cd backend
    uv run python -m evaluation.verify_golden

It (1) runs every reference query (and accepted alternatives) against the evaluation database, and
(2) for the questions that have an independent ground truth computed in Python straight from the
generated data, compares the SQL result to that truth. A mismatch means a wrong reference query, a
wrong view, or a data-loading problem. Exit status is non-zero on any problem.
"""
import json
import sys
from datetime import datetime

from evaluation.env import BACKEND, load_eval_env


def main() -> int:
    load_eval_env()
    from src.memory.supabase_client import supabase_admin as db  # noqa: E402
    from evaluation import compare as K, seed_data as S  # noqa: E402
    from evaluation.ground_truth import GROUND_TRUTH  # noqa: E402

    golden = json.loads((BACKEND / "evaluation" / "golden_bi.json").read_text(encoding="utf-8"))
    dataset_path = BACKEND / "evaluation" / "data" / "eval_dataset.json"
    if not dataset_path.exists():
        raise SystemExit("evaluation/data/eval_dataset.json not found. Run: uv run python -m evaluation.setup_eval_db")
    ds = S.dataset_from_json(dataset_path.read_text(encoding="utf-8"))

    now = datetime.now(S.IST)
    generated = datetime.fromisoformat(ds["meta"]["now_ist"]).date()
    if generated != now.date():
        print(f"NOTE: the data was generated on {generated}, today is {now.date()}. Time-relative checks (today, "
              "yesterday, last N days) will drift. Re-run setup_eval_db on the day you evaluate.\n")

    def run(sql):
        res = db.rpc("execute_sql_query", {"sql_query": sql}).execute()
        data = res.data if res.data is not None else []
        if data and isinstance(data[0], dict) and set(data[0]) == {"error"}:
            return None, str(data[0]["error"])
        return data, None

    problems = 0
    checked = 0
    for q in golden["questions"]:
        rows_by_sql = []
        for sql in [q["reference_sql"]] + q.get("accept_sql", []):
            rows, err = run(sql)
            if err:
                problems += 1
                print(f"FAIL  {q['id']:<4} reference SQL error: {err}")
                break
            rows_by_sql.append(rows)
        else:
            truth_fn = GROUND_TRUTH.get(q["id"])
            if truth_fn is None:
                print(f"ok    {q['id']:<4} runs ({len(rows_by_sql[0])} rows)")
                continue
            checked += 1
            truth = truth_fn(ds, now)
            if K.results_match(truth, rows_by_sql[0], order_sensitive=bool(q.get("order_sensitive"))):
                print(f"ok    {q['id']:<4} runs AND matches the Python ground truth")
            else:
                problems += 1
                print(f"FAIL  {q['id']:<4} SQL result differs from the Python ground truth: "
                      f"{K.describe_difference(truth, rows_by_sql[0])}")

    print(f"\n{len(golden['questions'])} reference queries ran; {checked} cross-checked against independent Python answers.")
    print("All good: the answer key is verified." if not problems else f"{problems} problem(s) found. Fix them before evaluating.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
