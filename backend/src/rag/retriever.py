"""Scalable Retriever helpers for Coffee Shop RAG - Supabase pgvector version

Provides utilities to query Supabase with vector similarity search for:
- Hybrid lookup (Exact match + Semantic fallback)
- Unified product data & pricing source of truth
- Low latency search (<300ms)
- Multi-user / Multi-tenant support
"""

import logging
from typing import List, Dict, Any, Optional

import dotenv
from src.config import Config, RetrieverConfig as ConfigRetrieverConfig
from src.memory.supabase_client import supabase_admin as supabase
from src.utils.util import get_embedding_model

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Use centralized RetrieverConfig from src/config.py
RetrieverConfig = ConfigRetrieverConfig


def search_products(
    query: str,
    top_k: int = 5,
    match_threshold: float = 0.35,   # nemotron-3-embed-1b: correct menu matches score ~0.42-0.76, unrelated ~0.3-0.38
) -> List[Dict[str, Any]]:
    """Search products using Supabase pgvector similarity search."""
    try:
        embeddings = get_embedding_model()
        q_vec = embeddings.embed_query(query)
        
        # Call the Supabase RPC function 'match_coffee_products'
        rpc_params = {
            'query_embedding': q_vec,
            'match_threshold': match_threshold,
            'match_count': top_k
        }
        
        res = supabase.rpc('match_coffee_products', rpc_params).execute()
        
        if not res.data:
            return []
            
        results = []
        for p in res.data:
            results.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "price": p.get("price"),
                "score": p.get("similarity"), # Map similarity to score for backward compatibility
                "category": p.get("category"),
                "description": p.get("description"),
                "image_url": p.get("image_url"),
                "metadata": p
            })
            
        return results
    except Exception as e:
        logger.error(f"Supabase vector search failed: {str(e)}")
        return []


def get_product_by_name(name: str) -> Dict[str, Any]:
    """Fetch product by name with Hybrid Logic using Supabase ONLY:
    1. Try exact lookup in Supabase.
    2. Fallback to Supabase pgvector semantic search to "translate" misspelt names.
    """
    try:
        normalized = name.lower().strip()
        
        # --- Step 1: Direct Supabase Lookup (Exact/ILIKE) ---
        res = supabase.table("coffee_shop_products").select("*").ilike("name", normalized).execute()
        
        if res.data:
            p = res.data[0]
            logger.debug(f"Direct Supabase match found for '{name}'")
            return {
                "found": True,
                "name": p.get("name"),
                "price": p.get("price"),
                "image_url": p.get("image_url"),
                "metadata": p,
                "source": "supabase_direct"
            }

        # --- Step 2: Semantic Translation fallback (Supabase pgvector) ---
        logger.debug(f"No direct match for '{name}', using Supabase semantic fallback...")
        fallback = search_products(name, top_k=1, match_threshold=0.5)
        
        if fallback:
            top_match = fallback[0]
            match_name = top_match.get("name")
            score = top_match.get("score")
            
            # Stricter than browse search on purpose: a wrong name substitution is worse than "not found".
            # With nemotron-3-embed-1b, 'capuchino' -> Cappuccino scores 0.55, while 'cold coffee' -> a cold brew scores 0.47.
            if score and score > 0.5:
                logger.info(f"Supabase translated '{name}' -> '{match_name}' (score: {score:.2f})")
                return {
                    "found": True,
                    "name": top_match.get("name"),
                    "price": top_match.get("price"),
                    "image_url": top_match.get("image_url"),
                    "metadata": top_match.get("metadata"),
                    "score": score,
                    "source": "supabase_semantic"
                }

        logger.warning(f"Product '{name}' not found locally or via pgvector.")
        return {"found": False, "query": name}

    except Exception as e:
        logger.error(f"get_product_by_name (supabase hybrid) failed for '{name}': {str(e)}")
        return {"found": False, "query": name, "error": str(e)}
