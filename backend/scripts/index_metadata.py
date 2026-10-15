"""Index the BI agent's view descriptions (schema linking) into pgvector.

For every BI view, data/bi_schema_docs.json holds a short description and its columns. This script embeds
each description. At question time the BI agent embeds the question, retrieves the views it probably
needs, and shows only those to the model, so the prompt stays small however many views exist.

Re-run this whenever data/bi_schema_docs.json changes:

    cd backend
    uv run python scripts/index_metadata.py

It clears coffee_shop_schema_metadata and re-inserts one row per view, so it is safe to repeat.
"""
import logging
import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json

# Allow running as `python scripts/index_metadata.py` from the backend folder
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from src.utils import bi_context as C  # noqa: E402
from src.utils.util import get_embedding_model  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")


def main() -> None:
    if not DB_URL:
        raise SystemExit("SUPABASE_DB_URL is not set in .env")

    docs = C.load_table_docs()
    texts = [C.table_embedding_text(d) for d in docs]

    logger.info(f"Embedding {len(docs)} view descriptions...")
    embeddings = get_embedding_model().embed_documents(texts)

    rows = []
    for doc, text, embedding in zip(docs, texts, embeddings):
        metadata = {"type": "table", "table": doc["name"], "columns": [c["name"] for c in doc["columns"]]}
        vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        rows.append((text, Json(metadata), vector_literal))

    conn = psycopg2.connect(DB_URL)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM coffee_shop_schema_metadata")
            cur.executemany(
                "INSERT INTO coffee_shop_schema_metadata (content, metadata, embedding) "
                "VALUES (%s, %s, %s::vector)",
                rows,
            )
        logger.info(f"✅ Indexed {len(rows)} view descriptions into coffee_shop_schema_metadata.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
