"""REMOVED: do not use.

This module used to index products into Pinecone using Hugging Face embeddings. Both are gone:
  * the vector store is Supabase pgvector (see supabase_db/schema.sql and src/rag/retriever.py)
  * embeddings come from NVIDIA NIM (see src/utils/embeddings.py and src/utils/util.py)

To (re)build the menu index run:
    uv run python scripts/seed_products.py
    uv run python scripts/index_metadata.py

This stub can be deleted from the repository.
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
