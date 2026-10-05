"""Tests for src/utils/allergens.py.

Run from the backend folder:
    uv run --with pytest pytest tests -q
"""
from src.utils.allergens import (
    _catalog,
    allergy_warning,
    catalog_product,
    product_allergens,
    user_allergy_conflicts,
)


def hits(allergies, product_name):
    p = catalog_product(product_name)
    assert p is not None, f"{product_name} not found in catalog"
    return user_allergy_conflicts(allergies, p["name"], p.get("ingredients", []))


# ── The cases the old substring check ("nuts" in ingredients) got wrong ───────

def test_nut_allergy_flags_every_nut_product():
    must_flag = [
        "Pecan Brownie", "Almond Croissant", "Chocolate Chip Biscotti",
        "Hazelnut syrup", "Hazelnut Biscotti", "Banana Nut Bread", "Toffee Nut Syrup",
    ]
    for name in must_flag:
        assert hits(["nuts"], name), f"{name} should be flagged for a nut allergy"


def test_nut_allergy_leaves_safe_items_alone():
    for name in ["Latte", "Americano", "Cappuccino", "Croissant", "Brazil Santos (250g)",
                 "Pumpkin Spice Latte", "Cinnamon Roll"]:
        assert not hits(["nuts"], name), f"{name} should NOT be flagged for a nut allergy"


def test_nut_allergy_phrasings():
    for phrase in ["nuts", "nut", "Nuts", "nut allergy", "tree nuts", "allergic to nuts"]:
        assert hits([phrase], "Pecan Brownie"), phrase


def test_peanut_allergy_is_not_a_tree_nut_allergy():
    assert not hits(["peanuts"], "Pecan Brownie")
    assert hits(["tree nuts"], "Pecan Brownie")


def test_dairy_allergy_phrasings():
    for phrase in ["dairy", "milk", "lactose", "lactose intolerance", "Dairy allergy"]:
        assert hits([phrase], "Latte"), phrase
        assert hits([phrase], "Jumbo Savory Scone"), phrase  # butter + cheese


def test_plant_milk_is_not_dairy():
    assert not hits(["dairy"], "Oat Milk Honey Latte")
    assert not hits(["milk"], "Vegan Blueberry Muffin")


def test_gluten_and_eggs():
    assert hits(["gluten"], "Croissant")
    assert hits(["gluten"], "Pecan Brownie")          # flagged by name
    assert not hits(["gluten"], "Latte")
    assert hits(["eggs"], "Berry Danish")              # custard
    assert hits(["eggs"], "Cinnamon Roll")             # override (brioche)
    assert not hits(["eggs"], "Croissant")


# ── Word-awareness ─────────────────────────────────────────────────────────────

def test_coconut_and_nutmeg_are_not_tree_nuts():
    assert "tree nuts" not in product_allergens("Coconut Latte", ["Espresso", "Coconut Milk"])
    assert "tree nuts" not in product_allergens("Spice Cake", ["Flour", "Nutmeg"])
    assert "tree nuts" not in product_allergens("Squash Soup", ["Butternut Squash"])


def test_cocoa_butter_and_peanut_butter_are_not_dairy():
    assert "dairy" not in product_allergens("Sauce", ["Sugar", "Cocoa Butter"])
    assert "dairy" not in product_allergens("Toast", ["Bread", "Peanut Butter"])
    assert "peanuts" in product_allergens("Toast", ["Bread", "Peanut Butter"])


# ── Free-text allergies the user may have stored ───────────────────────────────

def test_literal_terms_match_whole_words():
    assert hits(["pecans"], "Pecan Brownie")
    assert hits(["pecan"], "Pecan Brownie")
    assert not hits(["pecans"], "Banana Nut Bread")
    assert hits(["walnut"], "Banana Nut Bread")
    assert hits(["cinnamon"], "Cinnamon Syrup")
    assert not hits(["shellfish"], "Latte")


def test_no_allergies_means_no_conflicts():
    assert user_allergy_conflicts([], "Pecan Brownie", ["Pecans"]) == []
    assert user_allergy_conflicts(None, "Pecan Brownie", ["Pecans"]) == []
    assert user_allergy_conflicts([""], "Pecan Brownie", ["Pecans"]) == []


# ── Cart warning ───────────────────────────────────────────────────────────────

def test_warning_lists_only_conflicting_items():
    msg = allergy_warning(["nuts"], ["Latte", "Pecan Brownie"])
    assert "Pecan Brownie" in msg
    assert "Latte" not in msg
    assert "someone else" in msg


def test_warning_empty_when_safe_or_no_allergies():
    assert allergy_warning(["nuts"], ["Latte", "Americano"]) == ""
    assert allergy_warning([], ["Pecan Brownie"]) == ""
    assert allergy_warning(["nuts"], ["Not On The Menu"]) == ""


def test_warning_deduplicates_items():
    msg = allergy_warning(["nuts"], ["Pecan Brownie", "Pecan Brownie"])
    assert msg.count("Pecan Brownie") == 1


# ── Whole-catalog audit ────────────────────────────────────────────────────────

def test_every_product_with_a_nut_ingredient_is_flagged():
    nut_words = ("almond", "hazelnut", "pecan", "walnut", "pistachio", "cashew")
    products = list(_catalog().values())
    assert products, "catalog failed to load"
    for p in products:
        text = (p["name"] + " " + " ".join(p.get("ingredients", []))).lower()
        if any(w in text for w in nut_words):
            assert user_allergy_conflicts(["nuts"], p["name"], p.get("ingredients", [])), p["name"]
