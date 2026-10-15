"""Check that the BI view descriptions match the real views in the database.

    cd backend
    uv run python scripts/validate_schema_docs.py

For every view in data/bi_schema_docs.json it reads one row through the same read-only function the BI
agent uses, and compares the real column names with the documented ones. Run it after any schema change:
a renamed, added or removed column shows up here instead of silently misleading the agent.
Exits with a non-zero status on any mismatch.

(An empty view has no row to read; it is reported as unchecked.)
"""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from src.memory.supabase_client import supabase_admin  # noqa: E402
from src.utils import bi_context as C  # noqa: E402


def main() -> int:
    docs = C.load_table_docs()
    problems = 0
    for doc in docs:
        name = doc["name"]
        try:
            res = supabase_admin.rpc("execute_sql_query", {"sql_query": f"SELECT * FROM {name} LIMIT 1"}).execute()
            data = res.data if res.data is not None else []
        except Exception as e:
            problems += 1
            print(f"FAIL  {name}: rpc error: {e}")
            continue

        if data and isinstance(data[0], dict) and set(data[0]) == {"error"}:
            problems += 1
            print(f"FAIL  {name}: database: {data[0]['error']}")
            continue
        if not data:
            print(f"skip  {name}: the view is empty, columns not checked")
            continue

        real = set(data[0].keys())
        documented = {c["name"] for c in doc["columns"]}
        missing, undocumented = sorted(documented - real), sorted(real - documented)
        if missing or undocumented:
            problems += 1
            print(f"FAIL  {name}: documented but not in the view: {missing or '-'}; in the view but not documented: {undocumented or '-'}")
        else:
            print(f"ok    {name}: {len(real)} columns match")

    print("\nAll view descriptions match the database." if not problems else f"\n{problems} view(s) need attention: update data/bi_schema_docs.json (and re-run scripts/index_metadata.py).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
