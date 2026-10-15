"""One-command setup of the evaluation database.

    cd backend
    uv run python -m evaluation.setup_eval_db

Prerequisite: a SEPARATE Supabase project with backend/supabase_db/schema.sql already run in it, and its
keys in backend/.env.eval. This script then:
  1. seeds the menu (58 products + embeddings) if the project has none
  2. generates deterministic synthetic customers and orders (relative to today) and loads them
  3. indexes the BI view descriptions (schema linking, pgvector)
  4. saves the dataset locally so the ground truth can be re-checked later

Safe to repeat: it replaces only the demo rows (@demo.merrysway.test). Re-run it on the day you evaluate,
because the data is relative to "today".
"""
import argparse
import runpy
from datetime import datetime
from pathlib import Path

from evaluation.env import BACKEND, load_eval_env

DATA_DIR = BACKEND / "evaluation" / "data"
DATASET_PATH = DATA_DIR / "eval_dataset.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-index", action="store_true", help="do not re-embed the BI view descriptions")
    args = parser.parse_args()

    load_eval_env()
    from src.memory.supabase_client import supabase_admin as db  # noqa: E402  (after the env is loaded)
    from evaluation import seed_data as S  # noqa: E402

    print("Evaluation database:", __import__("os").environ["SUPABASE_URL"])

    # 0. schema present?
    try:
        db.table("coffee_shop_products").select("id", count="exact").limit(1).execute()
        db.rpc("execute_sql_query", {"sql_query": "SELECT 1 AS ok"}).execute()
    except Exception as e:
        raise SystemExit(f"The schema is not installed in this project (or its keys are wrong): {e}\n"
                         "Run backend/supabase_db/schema.sql in the evaluation project's SQL editor first.")

    # 1. menu
    count = db.table("coffee_shop_products").select("id", count="exact").limit(1).execute().count or 0
    if count == 0:
        print("Seeding the menu (products + embeddings)...")
        runpy.run_path(str(BACKEND / "scripts" / "seed_products.py"), run_name="__main__")
        count = db.table("coffee_shop_products").select("id", count="exact").limit(1).execute().count or 0
    if count == 0:
        raise SystemExit("The products table is still empty after seeding.")
    print(f"Menu: {count} products")

    # 2. demo data
    others = db.table("coffee_shop_profiles").select("user_email").not_.like("user_email", f"%@{S.DEMO_DOMAIN}").execute()
    if others.data:
        print(f"WARNING: {len(others.data)} non-demo profile(s) exist in this project. Ground-truth checks "
              "assume a clean evaluation project, so counts such as 'total customers' may not match.")

    products = S.load_products(str(BACKEND / "data" / "products_data" / "products.jsonl"))
    product_ids = {r["name"]: r["id"] for r in db.table("coffee_shop_products").select("id,name").execute().data}
    missing = [p["name"] for p in products if p["name"] not in product_ids]
    if missing:
        raise SystemExit(f"These catalog products are not in the database: {missing[:5]} ...")

    now_ist = datetime.now(S.IST)
    ds = S.generate_dataset(products, now_ist, seed=args.seed)
    loaded = S.load_into_db(db, ds, product_ids)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(S.dataset_to_json(ds), encoding="utf-8")
    print(f"Loaded demo data (seed {ds['meta']['seed_used']}): {loaded}")

    # 3. examples
    if not args.skip_index:
        print("Indexing the BI view descriptions...")
        runpy.run_path(str(BACKEND / "scripts" / "index_metadata.py"), run_name="__main__")

    # 4. sanity
    res = db.rpc("execute_sql_query", {"sql_query":
        "SELECT (SELECT COUNT(*) FROM orders WHERE status = 'confirmed') AS confirmed_orders, "
        "(SELECT COUNT(*) FROM orders WHERE status = 'pending') AS pending_carts, "
        "(SELECT COUNT(*) FROM customers) AS customers"}).execute()
    print("BI views report:", res.data)
    print("\nDone. Next: uv run python -m evaluation.verify_golden")


if __name__ == "__main__":
    main()
