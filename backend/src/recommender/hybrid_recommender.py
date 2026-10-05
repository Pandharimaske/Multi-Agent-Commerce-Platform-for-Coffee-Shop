"""
Hybrid Recommender — combines three signals:
  1. Popularity     — fallback for cold start (no cart, no preferences)
  2. Apriori        — market basket ("people who bought X also bought Y")
  3. Content-based  — user memory (likes, dislikes, allergies, last_order)

Usage:
    rec = HybridRecommender.load()          # loads from artifacts/
    results = rec.recommend(
        cart=["Cappuccino"],
        likes=["sweet", "chocolate"],
        dislikes=["bitter"],
        allergies=["nuts"],
        top_k=3,
    )
"""

import json
import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Optional
import joblib

from src.utils.allergens import user_allergy_conflicts

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
_DATA = os.path.join(_BASE, "../../data")
_ARTIFACTS = os.path.join(_BASE, "../../artifacts")

APRIORI_PATH = os.path.join(_DATA, "apriori_recommendations.json")
POPULARITY_PATH = os.path.join(_DATA, "popularity_recommendation.csv")
PRODUCTS_PATH = os.path.join(_DATA, "products_data/products.jsonl")
MODEL_PATH = os.path.join(_ARTIFACTS, "hybrid_recommender.joblib")


# ── Recommender ───────────────────────────────────────────────────────────────
class HybridRecommender:
    """
    Hybrid recommendation model.
    Weights:
        popularity_weight   — how much global popularity matters
        apriori_weight      — how much market basket rules matter
        content_weight      — how much user preferences matter
    """

    def __init__(
        self,
        popularity_weight: float = 0.2,
        apriori_weight: float = 0.5,
        content_weight: float = 0.3,
    ):
        self.popularity_weight = popularity_weight
        self.apriori_weight = apriori_weight
        self.content_weight = content_weight

        self.products: dict = {}          # name → product dict
        self.popularity: dict = {}        # name → normalised score
        self.apriori: dict = {}           # name → [{product, confidence}]
        self.content_matrix: Optional[pd.DataFrame] = None  # product × feature
        self.mlb: Optional[MultiLabelBinarizer] = None
        self._fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self):
        """Load data and build all three scoring components."""
        self._load_products()
        self._build_popularity()
        self._build_apriori()
        self._build_content_matrix()
        self._fitted = True
        logger.info(f"HybridRecommender fitted on {len(self.products)} products")
        return self

    def _load_products(self):
        with open(PRODUCTS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                self.products[p["name"]] = p

    def _build_popularity(self):
        df = pd.read_csv(POPULARITY_PATH)
        # Sum transactions per product (dark chocolate appears twice)
        grouped = df.groupby("product")["number_of_transactions"].sum()
        max_txn = grouped.max()
        self.popularity = (grouped / max_txn).to_dict()

    def _build_apriori(self):
        with open(APRIORI_PATH) as f:
            self.apriori = json.load(f)

    def _build_content_matrix(self):
        """
        Build a product × feature matrix using:
        - ingredient one-hot encoding
        - category one-hot encoding
        - normalised price
        - normalised rating
        """
        records = []
        for name, p in self.products.items():
            records.append({
                "name": name,
                "ingredients": p.get("ingredients", []),
                "category": p.get("category", "Unknown"),
                "price": float(p.get("price", 0)),
                "rating": float(p.get("rating", 0)),
            })
        df = pd.DataFrame(records).set_index("name")

        # Ingredient features
        self.mlb = MultiLabelBinarizer()
        ing_matrix = pd.DataFrame(
            self.mlb.fit_transform(df["ingredients"]),
            index=df.index,
            columns=[f"ing_{i}" for i in self.mlb.classes_],
        )

        # Category dummies
        cat_dummies = pd.get_dummies(df["category"], prefix="cat")

        # Normalised numeric features
        df["price_norm"] = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min() + 1e-9)
        df["rating_norm"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min() + 1e-9)

        self.content_matrix = pd.concat(
            [ing_matrix, cat_dummies, df[["price_norm", "rating_norm"]]],
            axis=1,
        ).fillna(0)

    # ── Score components ──────────────────────────────────────────────────────
    def _popularity_scores(self) -> dict:
        return {name: self.popularity.get(name, 0.0) for name in self.products}

    def _apriori_scores(self, cart: list[str]) -> dict:
        """
        For each item in cart, look up associated products.
        Accumulate confidence scores.
        """
        scores = {name: 0.0 for name in self.products}
        for cart_item in cart:
            rules = self.apriori.get(cart_item, [])
            for rule in rules:
                product = rule["product"]
                if product in scores:
                    scores[product] += rule["confidence"]
        # Normalise
        max_score = max(scores.values()) if any(scores.values()) else 1.0
        return {k: v / max_score for k, v in scores.items()}

    def _content_scores(self, likes: list[str], dislikes: list[str], last_order: str) -> dict:
        """
        Build a user preference vector from likes + last_order.
        Compute cosine similarity with each product's feature vector.
        """
        if self.content_matrix is None or not (likes or last_order):
            return {name: 0.0 for name in self.products}

        # Build user vector by averaging feature vectors of liked products + last order
        anchor_products = []
        for like in likes:
            # Try to find a matching product name
            matches = [p for p in self.products if like.lower() in p.lower()]
            anchor_products.extend(matches[:1])
        if last_order:
            matches = [p for p in self.products if last_order.lower() in p.lower()]
            anchor_products.extend(matches[:1])

        if not anchor_products:
            return {name: 0.0 for name in self.products}

        # Average the feature vectors of anchor products
        valid = [p for p in anchor_products if p in self.content_matrix.index]
        if not valid:
            return {name: 0.0 for name in self.products}

        user_vector = self.content_matrix.loc[valid].mean(axis=0).values.reshape(1, -1)
        product_vectors = self.content_matrix.values
        sims = cosine_similarity(user_vector, product_vectors)[0]

        return dict(zip(self.content_matrix.index, sims.tolist()))

    # ── Recommend ─────────────────────────────────────────────────────────────
    def recommend(
        self,
        cart: list[str] = None,
        likes: list[str] = None,
        dislikes: list[str] = None,
        allergies: list[str] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Returns top_k recommended products as list of dicts:
        [{name, price, category, description, score, reason}]
        """
        assert self._fitted, "Call .fit() before .recommend()"

        cart = cart or []
        likes = likes or []
        dislikes = dislikes or []
        allergies = allergies or []

        # ── Score ─────────────────────────────────────────────────────────────
        pop = self._popularity_scores()
        apr = self._apriori_scores(cart)
        con = self._content_scores(likes, dislikes, "")

        final_scores = {}
        for name in self.products:
            final_scores[name] = (
                self.popularity_weight * pop.get(name, 0)
                + self.apriori_weight * apr.get(name, 0)
                + self.content_weight * con.get(name, 0)
            )

        # ── Hard filters ──────────────────────────────────────────────────────
        def _is_safe(name):
            p = self.products[name]
            ingredients = " ".join(str(i) for i in p.get("ingredients", [])).lower()
            # Remove allergens (word-aware and synonym-aware, see src/utils/allergens.py)
            if user_allergy_conflicts(allergies, name, p.get("ingredients", [])):
                return False
            # Remove cart items (already ordered)
            if name in cart:
                return False
            # Remove disliked items
            for dislike in dislikes:
                if dislike.lower() in name.lower() or dislike.lower() in ingredients:
                    return False
            return True

        filtered = {k: v for k, v in final_scores.items() if _is_safe(k)}

        # ── Rank ──────────────────────────────────────────────────────────────
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # ── Build reason ──────────────────────────────────────────────────────
        results = []
        for name, score in ranked:
            p = self.products[name]
            reason = self._reason(name, score, cart, likes, apr)
            results.append({
                "name": name,
                "price": p.get("price"),
                "category": p.get("category"),
                "description": p.get("description", ""),
                "score": round(score, 4),
                "reason": reason,
            })
        return results

    def _reason(self, name: str, score: float, cart: list, likes: list, apr_scores: dict) -> str:
        """Generate a human-readable reason for the recommendation."""
        p = self.products[name]
        if apr_scores.get(name, 0) > 0.3 and cart:
            return f"Often ordered with {', '.join(cart[:2])}"
        for like in likes:
            if like.lower() in name.lower() or like.lower() in p.get("description", "").lower():
                return f"Matches your preference for {like}"
        pop_score = self.popularity.get(name, 0)
        if pop_score > 0.7:
            return "One of our most popular items"
        return f"Highly rated at {p.get('rating', 'N/A')}★"

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "HybridRecommender":
        if not os.path.exists(path):
            logger.warning(f"No saved model at {path} — fitting fresh model")
            rec = cls().fit()
            rec.save(path)
            return rec
        rec = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return rec

    # ── Metrics ───────────────────────────────────────────────────────────────
    def coverage(self) -> float:
        """% of products that can be recommended (not hard-blocked by default filters)."""
        return len(self.products) / max(len(self.products), 1)

    def catalog_size(self) -> int:
        return len(self.products)
