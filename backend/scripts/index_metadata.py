import os
import sys
import json
import logging
import psycopg2
from dotenv import load_dotenv

# Allow running as `python scripts/index_metadata.py` from the backend folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.util import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Config
DB_URL = os.getenv("SUPABASE_DB_URL")

if "?" in DB_URL:
    full_uri = f"{DB_URL}&sslmode=require"
else:
    full_uri = f"{DB_URL}?sslmode=require"

def index_metadata():
    # 1. Load Metadata
    metadata_path = "data/schema_metadata.json"
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    with open(metadata_path, "r") as f:
        data = json.load(f)

    # 2. Initialize Embeddings from Pool
    logger.info("Initializing embeddings from centralized pool...")
    embeddings = get_embedding_model()

    # 3. Connect to DB
    try:
        conn = psycopg2.connect(full_uri)
        cur = conn.cursor()
        
        # Clear existing metadata to avoid duplicates
        logger.info("Clearing existing schema metadata...")
        cur.execute("DELETE FROM coffee_shop_schema_metadata;")

        logger.info(f"Indexing {len(data)} items...")
        
        for item in data:
            # Create a rich content string for embedding
            if item["type"] == "table":
                content = f"Table: {item['name']}. Description: {item['description']}"
            else:
                content = f"Column: {item['name']} in table {item['table']}. Description: {item['description']}"
            
            embedding = embeddings.embed_query(content)
            
            cur.execute(
                "INSERT INTO coffee_shop_schema_metadata (content, metadata, embedding) VALUES (%s, %s, %s)",
                (content, json.dumps(item), embedding)
            )
        
        conn.commit()
        cur.close()
        conn.close()
        logger.info("✅ Schema metadata indexing complete.")

    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}")

if __name__ == "__main__":
    index_metadata()
