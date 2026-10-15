"""
Seed Supabase products table from products.jsonl

Run once:
    python scripts/seed_products.py
"""

import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

from src.utils.util import get_embedding_model  # noqa: E402  (NVIDIA NIM embeddings)

# Seed script needs service role key to bypass RLS
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")  # service_role key, not anon
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/products_data/products.jsonl"
)



def seed():
    products = []
    names = []
    
    with open(DATA_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            names.append(data["name"])
            # Construct Supabase Storage URL
            image_path = data.get("image_path")
            supabase_url = os.getenv("SUPABASE_URL")
            bucket_name = "product-images"
            image_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{image_path}" if image_path else None

            products.append({
                "name": data["name"],
                "category": data.get("category"),
                "description": data.get("description"),
                "price": float(data["price"]),
                "rating": float(data["rating"]) if data.get("rating") else None,
                "ingredients": data.get("ingredients", []),
                "image_url": image_url,
            })

    if not products:
        print("No products found in JSONL file.")
        return

    print(f"Generating embeddings for {len(names)} products via NVIDIA NIM...")
    # Generate embeddings in batch
    embeddings = get_embedding_model().embed_documents(names)
    
    for i, product in enumerate(products):
        product["embedding"] = embeddings[i]

    print(f"Seeding {len(products)} products to Supabase...")
    res = supabase.table("coffee_shop_products").upsert(products, on_conflict="name").execute()

    if res.data:
        print(f"✅ Seeded {len(res.data)} products successfully with embeddings.")
    else:
        print("⚠️  Upsert returned no data — check Supabase logs and if 'embedding' column exists.")


if __name__ == "__main__":
    seed()
