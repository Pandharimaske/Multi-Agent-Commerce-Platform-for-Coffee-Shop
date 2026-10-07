"""Index the BI agent's verified question -> SQL examples into pgvector.

The BI agent retrieves the most similar examples for each question and shows them to the
model as guidance (retrieval-augmented Text-to-SQL). Re-run this whenever
data/bi_examples.json changes:

    cd backend
    uv run python scripts/index_metadata.py

It clears coffee_shop_schema_metadata and re-inserts every example, so it is safe to repeat.
"""
import json
import logging
import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json

# Allow running as `python scripts/index_metadata.py` from the backend folder
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from src.utils.util import get_embedding_model  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")
EXAMPLES_PATH = os.path.join(BACKEND_DIR, "data", "bi_examples.json")


def main() -> None:
    if not DB_URL:
        raise SystemExit("SUPABASE_DB_URL is not set in .env")

    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        examples = json.load(f)

    logger.info(f"Embedding {len(examples)} BI examples...")
    embeddings = get_embedding_model().embed_documents([e["question"] for e in examples])

    rows = []
    for example, embedding in zip(examples, embeddings):
        metadata = {
            "type": "example",
            "question": example["question"],
            "sql": example["sql"],
            "tags": example.get("tags", []),
        }
        vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        rows.append((example["question"], Json(metadata), vector_literal))

    conn = psycopg2.connect(DB_URL)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM coffee_shop_schema_metadata")
            cur.executemany(
                "INSERT INTO coffee_shop_schema_metadata (content, metadata, embedding) "
                "VALUES (%s, %s, %s::vector)",
                rows,
            )
        logger.info(f"✅ Indexed {len(rows)} examples into coffee_shop_schema_metadata.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
