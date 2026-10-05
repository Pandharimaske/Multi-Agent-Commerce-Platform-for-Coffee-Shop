"""DEPRECATED: do not use.

This script used to create the BI tables and the `execute_sql_query` /
`match_schema_metadata` functions. Everything now lives in
backend/supabase_db/schema.sql, which creates them with stricter permissions
(service role only, read-only transaction, system schemas blocked).

Running the old version would have re-created `execute_sql_query` and granted
it to every logged-in user, so it has been replaced by this stub.

Setup order:
  1. Run backend/supabase_db/schema.sql in the Supabase SQL editor
     (run reset_schema.sql first only if you want to wipe an existing project).
  2. uv run python scripts/seed_products.py
  3. uv run python scripts/index_metadata.py
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
