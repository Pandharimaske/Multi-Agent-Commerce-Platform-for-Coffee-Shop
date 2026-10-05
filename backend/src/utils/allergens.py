"""Deterministic allergen detection for menu items.

Used by:
  * the recommender   - hard filter on the user's saved allergies
  * the order agent   - small warning when a cart item matches a saved allergy

Matching is word-aware and synonym-aware and only looks at the product NAME and
INGREDIENT list (never free-text descriptions), so "nutmeg" or "coconut" do not
trigger a nut warning and "Oat Milk" does not count as dairy.

Limits: this is ingredient based. Cross-contact (shared equipment) and anything
not listed in the ingredients cannot be detected.

Review the derived tags for every product with:
    uv run python -m src.utils.allergens
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Iterable, Optional

_CATALOG_PATH = os.getenv("ALLERGEN_CATALOG_PATH") or os.path.join(
    os.path.dirname(__file__), "../../data/products_data/products.jsonl"
)

# Canonical allergen -> keywords (matched on word boundaries, plural-tolerant).
ALLERGEN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tree nuts": (
        "almond", "hazelnut", "pecan", "walnut", "pistachio", "cashew",
        "macadamia", "brazil nut", "pine nut", "chestnut", "praline",
        "marzipan", "nougat", "nut",
    ),
    "peanuts": ("peanut", "groundnut"),
    "dairy": (
        "milk", "cream", "butter", "cheese", "feta", "custard", "gelato",
        "yogurt", "yoghurt", "whey", "casein", "ghee", "paneer", "mozzarella",
        "cheddar", "parmesan", "ricotta", "cold foam", "white chocolate",
    ),
    "gluten": (
        "flour", "wheat", "sourdough", "granola", "bread", "biscotti",
        "croissant", "scone", "danish", "muffin", "brownie", "cake", "pastry",
        "dough", "barley", "rye", "malt", "semolina", "couscous", "pasta",
        "bagel",
    ),
    "eggs": ("egg", "custard", "mayonnaise", "mayo", "meringue"),
    "soy": ("soy", "soya", "soybean", "tofu", "edamame", "miso", "tempeh"),
    "sesame": ("sesame", "tahini"),
}

# Phrases that contain a dairy keyword but are not dairy.
NON_DAIRY_PHRASES: tuple[str, ...] = (
    "oat milk", "soy milk", "soya milk", "almond milk", "coconut milk",
    "rice milk", "cashew milk", "coconut cream", "cocoa butter", "shea butter",
    "peanut butter", "almond butter", "cashew butter", "nut butter",
)

# Manual additions where the ingredient list under-reports (name -> extra tags).
# Keep each entry justified in a comment.
PRODUCT_OVERRIDES: dict[str, set[str]] = {
    # Description says "brioche dough", which is made with eggs; not in ingredient list.
    "cinnamon roll": {"eggs"},
}

# What users say -> canonical categories. Unknown terms fall back to a literal word match.
USER_TERM_CATEGORIES: dict[str, set[str]] = {
    "nut": {"tree nuts", "peanuts"}, "nuts": {"tree nuts", "peanuts"},
    "tree nut": {"tree nuts"}, "tree nuts": {"tree nuts"},
    "peanut": {"peanuts"}, "peanuts": {"peanuts"},
    "dairy": {"dairy"}, "milk": {"dairy"}, "lactose": {"dairy"},
    "gluten": {"gluten"}, "wheat": {"gluten"}, "celiac": {"gluten"}, "coeliac": {"gluten"},
    "egg": {"eggs"}, "eggs": {"eggs"},
    "soy": {"soy"}, "soya": {"soy"},
    "sesame": {"sesame"},
}

_STRIP_PREFIX = re.compile(r"^(?:allergic to|allergy to|intolerant to|no)\s+")
_STRIP_SUFFIX = re.compile(r"\s+(?:allergy|allergies|allergic|intolerance|intolerant|free)$")


def _compile(words: Iterable[str]) -> re.Pattern:
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(r"\b(?:" + alts + r")(?:es|s)?\b")


_CATEGORY_PATTERNS = {cat: _compile(words) for cat, words in ALLERGEN_KEYWORDS.items()}


def _product_text(name: Optional[str], ingredients) -> str:
    if isinstance(ingredients, str):
        ingr = ingredients
    else:
        ingr = ", ".join(str(i) for i in (ingredients or []))
    return f"{name or ''} {ingr}".lower()


def _normalize_term(term: str) -> str:
    t = re.sub(r"\s+", " ", str(term or "").strip().lower())
    t = _STRIP_PREFIX.sub("", t)
    t = _STRIP_SUFFIX.sub("", t)
    if t.startswith("lactose"):
        return "lactose"
    return t.strip()


def _literal_pattern(term: str) -> re.Pattern:
    singular = term[:-1] if term.endswith("s") and len(term) > 3 else term
    alts = "|".join(re.escape(x) for x in {term, singular})
    return re.compile(r"\b(?:" + alts + r")(?:es|s)?\b")


def product_allergens(name: Optional[str], ingredients) -> set[str]:
    """Canonical allergen categories a product may contain."""
    text = _product_text(name, ingredients)
    dairy_text = text
    for phrase in NON_DAIRY_PHRASES:
        dairy_text = dairy_text.replace(phrase, " ")

    found: set[str] = set()
    for cat, pattern in _CATEGORY_PATTERNS.items():
        scan = dairy_text if cat == "dairy" else text
        if pattern.search(scan):
            found.add(cat)
    found |= PRODUCT_OVERRIDES.get((name or "").strip().lower(), set())
    return found


def user_allergy_conflicts(allergies: Optional[Iterable[str]], name: Optional[str], ingredients) -> list[str]:
    """Return the user's allergy terms (as stored) that this product may conflict with."""
    cats = product_allergens(name, ingredients)
    text = _product_text(name, ingredients)
    hits: list[str] = []
    for raw in allergies or []:
        term = _normalize_term(raw)
        if not term:
            continue
        mapped = USER_TERM_CATEGORIES.get(term)
        matched = bool(mapped & cats) if mapped else bool(_literal_pattern(term).search(text))
        if matched and raw not in hits:
            hits.append(raw)
    return hits


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict]:
    items: dict[str, dict] = {}
    try:
        with open(_CATALOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    items[p["name"].strip().lower()] = p
    except Exception:
        pass
    return items


def catalog_product(name: str) -> Optional[dict]:
    return _catalog().get((name or "").strip().lower())


def allergy_warning(allergies: Optional[Iterable[str]], product_names: Iterable[str]) -> str:
    """Small, deterministic warning for the cart summary. Empty string if nothing matches."""
    if not allergies:
        return ""
    lines, seen = [], set()
    for n in product_names:
        p = catalog_product(n)
        if not p or p["name"] in seen:
            continue
        seen.add(p["name"])
        hits = user_allergy_conflicts(allergies, p["name"], p.get("ingredients", []))
        if hits:
            lines.append(f"  • {p['name']} (matches: {', '.join(hits)})")
    if not lines:
        return ""
    return (
        "⚠️ Allergy heads-up: these items may conflict with the allergies on your profile:\n"
        + "\n".join(lines)
        + "\nIgnore this if the order is for someone else, and please check with our staff if your allergy is severe."
    )


if __name__ == "__main__":
    rows = sorted(_catalog().values(), key=lambda p: p["name"].lower())
    print(f"{len(rows)} products from {os.path.normpath(_CATALOG_PATH)}\n")
    for p in rows:
        tags = ", ".join(sorted(product_allergens(p["name"], p.get("ingredients", [])))) or "-"
        print(f"{p['name']:<32} {tags}")
