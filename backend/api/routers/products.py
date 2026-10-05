import json
import logging
import os
from fastapi import APIRouter, Query
from typing import Optional
from api.schemas import ProductResponse
from src.memory.supabase_client import supabase_admin as supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["products"])

import time
from typing import List, Dict

# Local fallback — used when Supabase is unreachable
_LOCAL_PRODUCTS_PATH = os.path.join(
    os.path.dirname(__file__), "../../data/products_data/products.jsonl"
)

# Cache settings
_PRODUCT_CACHE: List[Dict] = []
_CACHE_TIMESTAMP: float = 0.0
_CACHE_TTL = 600  # 10 minutes in seconds

def _load_local_products() -> list[dict]:
    """Load products from local JSONL file as fallback."""
    products = []
    try:
        with open(_LOCAL_PRODUCTS_PATH, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                products.append({
                    "id": str(i + 1),
                    "name": data["name"],
                    "category": data.get("category"),
                    "description": data.get("description"),
                    "price": float(data["price"]),
                    "rating": float(data["rating"]) if data.get("rating") else None,
                    "ingredients": data.get("ingredients", []),
                    "image_url": f"/images/{data['image_path']}" if data.get("image_path") else None,
                })
        logger.info(f"Loaded {len(products)} products from local fallback")
    except Exception as e:
        logger.error(f"Local product fallback failed: {e}")
    return products


@router.get("", response_model=list[ProductResponse])
async def get_products(
    category: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
):
    global _PRODUCT_CACHE, _CACHE_TIMESTAMP
    
    # 1. Check if cache is valid
    now = time.time()
    if not _PRODUCT_CACHE or (now - _CACHE_TIMESTAMP > _CACHE_TTL):
        # Cache expired or empty — reload from Supabase
        try:
            res = supabase.table("coffee_shop_products").select("*").order("name").execute()
            if res.data:
                _PRODUCT_CACHE = res.data
                _CACHE_TIMESTAMP = now
                logger.info(f"Product cache refreshed from Supabase: {len(_PRODUCT_CACHE)} items")
        except Exception as e:
            logger.warning(f"Supabase products unavailable ({e}) — will use local fallback or existing cache")

    # 2. Use cache (or local fallback if cache still empty)
    products = _PRODUCT_CACHE if _PRODUCT_CACHE else _load_local_products()

    # 3. Apply filters in memory (extremely fast)
    if category:
        products = [p for p in products if p.get("category", "").lower() == category.lower()]
    if search:
        products = [p for p in products if search.lower() in p["name"].lower()]

    return [ProductResponse(**p) for p in sorted(products, key=lambda x: x["name"])]

    if category:
        products = [p for p in products if p.get("category", "").lower() == category.lower()]
    if search:
        products = [p for p in products if search.lower() in p["name"].lower()]

    return [ProductResponse(**p) for p in sorted(products, key=lambda x: x["name"])]
