"""Tests for the admin BI helpers (src/utils/bi_utils.py) and the verified examples bank.

Run from the backend folder:
    uv run --with pytest pytest tests -q
"""
import datetime as dt
import json
import os
import re

from src.utils import bi_utils as U

TODAY = dt.date(2026, 10, 5)


# ── SQL cleaning ──────────────────────────────────────────────────────────────

def test_scrub_removes_fences_comments_and_semicolons():
    raw = "```sql\nSELECT 1 AS x -- trailing comment\n;\n```"
    assert U.scrub_sql(raw) == "SELECT 1 AS x"


def test_scrub_keeps_dashes_inside_string_literals():
    assert U.scrub_sql("SELECT '--not a comment' AS a") == "SELECT '--not a comment' AS a"


def test_scrub_removes_block_comments():
    assert "DROP" not in U.scrub_sql("SELECT 1 /* DROP TABLE x */ AS a")


# ── SQL validation ────────────────────────────────────────────────────────────

def test_validate_accepts_select_and_with():
    assert U.validate_sql("SELECT order_total FROM orders WHERE status = 'confirmed'") is None
    assert U.validate_sql("WITH m AS (SELECT 1 AS x) SELECT x FROM m") is None


def test_validate_rejects_writes_and_ddl():
    for bad in [
        "DELETE FROM orders",
        "UPDATE orders SET status = 'confirmed'",
        "INSERT INTO orders VALUES (1)",
        "DROP TABLE orders",
        "TRUNCATE orders",
        "ALTER TABLE orders ADD COLUMN x int",
        "CREATE TABLE x (a int)",
        "SELECT * INTO newtable FROM orders",
        "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
    ]:
        assert U.validate_sql(bad) is not None, bad


def test_validate_rejects_stacked_statements():
    assert U.validate_sql("SELECT 1; DROP TABLE orders") is not None


def test_validate_rejects_system_and_base_tables():
    for bad in [
        "SELECT * FROM auth.users",
        "SELECT * FROM coffee_shop_profiles",
        "SELECT * FROM public.coffee_shop_orders",
        "SELECT * FROM information_schema.tables",
        "SELECT pg_sleep(10)",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM bi.orders",
    ]:
        assert U.validate_sql(bad) is not None, bad


def test_validate_ignores_keywords_inside_string_literals():
    assert U.validate_sql("SELECT product_name FROM products WHERE product_name ILIKE '%update%'") is None
    assert U.validate_sql("SELECT product_name FROM products WHERE description ILIKE '%drop%'") is None


def test_validate_allows_updated_at_column():
    assert U.validate_sql("SELECT COUNT(*) AS n FROM orders WHERE updated_at < now()") is None


def test_validate_rejects_empty_and_unbalanced():
    assert U.validate_sql("") is not None
    assert U.validate_sql("SELECT (1 FROM orders") is not None
    assert U.validate_sql("EXPLAIN SELECT 1") is not None


def test_comment_cannot_hide_a_forbidden_keyword():
    cleaned = U.scrub_sql("SELECT 1 AS a /* harmless */ ; DELETE FROM orders")
    assert U.validate_sql(cleaned) is not None


# ── Formatting ────────────────────────────────────────────────────────────────

def test_format_inr_indian_grouping():
    assert U.format_inr(0) == "₹0"
    assert U.format_inr(380) == "₹380"
    assert U.format_inr(1440) == "₹1,440"
    assert U.format_inr(144000) == "₹1,44,000"
    assert U.format_inr(1234567.5) == "₹12,34,567.50"
    assert U.format_inr(-2500) == "-₹2,500"


def test_column_kind():
    assert U.column_kind("revenue") == "money"
    assert U.column_kind("order_total") == "money"
    assert U.column_kind("total_spent") == "money"
    assert U.column_kind("average_order_value") == "money"
    assert U.column_kind("potential_revenue") == "money"
    assert U.column_kind("growth_percent") == "percent"
    assert U.column_kind("revenue_share_percent") == "percent"
    assert U.column_kind("orders") == "number"
    assert U.column_kind("total_orders") == "number"
    assert U.column_kind("units_sold") == "number"
    assert U.column_kind("rating") == "number"
    assert U.column_kind("value", "show revenue by category") == "money"
    assert U.column_kind("value", "how many orders") == "number"


def test_redact_emails():
    out = U.redact_emails([{"who": "mahi@gmail.com", "n": 3, "note": "hello"}])
    assert out[0]["who"] == "m***@gmail.com" and out[0]["n"] == 3 and out[0]["note"] == "hello"


# ── Chart selection ───────────────────────────────────────────────────────────

def test_empty_result():
    a = U.analyze_result("anything", [])
    assert a["shape"] == "empty" and a["chart_type"] == "none"
    assert U.build_chart_data(a, []) == []


def test_single_metric_has_no_chart():
    a = U.analyze_result("today's sales", [{"revenue_today": 1440}])
    assert a["shape"] == "metric" and a["chart_type"] == "none"


def test_ranking_is_bar_chart():
    rows = [{"product_name": "Cappuccino", "revenue": 760}, {"product_name": "Latte", "revenue": 400}]
    a = U.analyze_result("top products", rows)
    assert a["chart_type"] == "bar" and a["label_col"] == "product_name" and a["value_col"] == "revenue"
    assert U.build_chart_data(a, rows) == [{"name": "Cappuccino", "value": 760}, {"name": "Latte", "value": 400}]


def test_monthly_series_is_line_chart_with_month_labels():
    rows = [{"month": f"2026-{m:02d}-01", "revenue": r} for m, r in [(8, 0), (9, 0), (10, 1440)]]
    a = U.analyze_result("this year's sales", rows)
    assert a["chart_type"] == "line"
    data = U.build_chart_data(a, rows)
    assert [d["name"] for d in data] == ["Aug 2026", "Sep 2026", "Oct 2026"]
    assert data[-1]["value"] == 1440


def test_daily_series_labels_are_short():
    rows = [{"day": f"2026-10-{d:02d}", "revenue": d} for d in (3, 4, 5)]
    data = U.build_chart_data(U.analyze_result("last 3 days", rows), rows)
    assert [d["name"] for d in data] == ["03 Oct", "04 Oct", "05 Oct"]


def test_two_points_in_time_is_bar_not_line():
    rows = [{"month": "2026-09-01", "revenue": 10}, {"month": "2026-10-01", "revenue": 20}]
    assert U.analyze_result("compare", rows)["chart_type"] == "bar"


def test_share_question_gives_pie():
    rows = [{"category": "Coffee", "revenue_share_percent": 60.5}, {"category": "Bakery", "revenue_share_percent": 39.5}]
    assert U.analyze_result("what share of revenue does each category contribute", rows)["chart_type"] == "pie"


def test_hour_of_day_is_labelled_and_charted():
    rows = [{"order_hour": 9, "orders": 4}, {"order_hour": 13, "orders": 7}]
    a = U.analyze_result("busiest hours", rows)
    assert a["chart_type"] == "bar" and a["label_col"] == "order_hour"
    assert [d["name"] for d in U.build_chart_data(a, rows)] == ["09:00", "13:00"]


def test_multi_metric_and_wide_results_are_tables():
    one_row = [{"this_month": 100, "last_month": 80, "growth_percent": 25}]
    assert U.analyze_result("growth", one_row)["chart_type"] == "table"
    three_cols = [{"a": "x", "b": "y", "n": 1}, {"a": "p", "b": "q", "n": 2}]
    assert U.analyze_result("pairs", three_cols)["chart_type"] == "table"


def test_table_data_keeps_column_names_and_caps_rows():
    rows = [{"product_name": f"P{i}", "category": "C"} for i in range(80)]
    a = U.analyze_result("list", rows)
    data = U.build_chart_data(a, rows)
    assert a["chart_type"] == "table" and len(data) == U.MAX_TABLE_ROWS and set(data[0]) == {"product_name", "category"}


def test_too_many_categories_fall_back_to_table():
    rows = [{"product_name": f"P{i}", "revenue": i} for i in range(40)]
    assert U.analyze_result("revenue by product", rows)["chart_type"] == "table"


def test_chart_values_come_from_rows_not_from_text():
    rows = [{"product_name": "A", "revenue": 12.3456}, {"product_name": "B", "revenue": 7}]
    data = U.build_chart_data(U.analyze_result("x", rows), rows)
    assert data == [{"name": "A", "value": 12.35}, {"name": "B", "value": 7}]


# ── Facts and narrative ───────────────────────────────────────────────────────

def _ranking_rows():
    return [
        {"product_name": "Cappuccino", "revenue": 760},
        {"product_name": "Latte", "revenue": 400},
        {"product_name": "Americano", "revenue": 280},
    ]


def test_facts_for_ranking():
    rows = _ranking_rows()
    a = U.analyze_result("top products by revenue", rows)
    f = U.compute_facts(a, rows, "top products by revenue", TODAY)
    assert f["top"] == {"label": "Cappuccino", "value": 760, "share_percent": 52.8}
    assert f["bottom"]["label"] == "Americano" and f["total"] == 1440 and f["value_kind"] == "money"


def test_in_progress_month_is_flagged_and_not_called_a_decline():
    rows = [{"month": "2026-04-01", "revenue": 5350}, {"month": "2026-09-01", "revenue": 0}, {"month": "2026-10-01", "revenue": 1440}]
    a = U.analyze_result("this year's sales", rows)
    f = U.compute_facts(a, rows, "this year's sales", TODAY)
    assert "in progress" in f["in_progress_note"] and "Oct 2026" in f["in_progress_note"]
    text = U.deterministic_narrative("this year's sales", a, rows, f)
    assert "Oct 2026 is the current month and still in progress." in text
    assert "decline" not in text.lower()


def test_deterministic_narrative_ranking_uses_rupees_and_exact_numbers():
    rows = _ranking_rows()
    a = U.analyze_result("top products by revenue", rows)
    f = U.compute_facts(a, rows, "top products by revenue", TODAY)
    text = U.deterministic_narrative("top products by revenue", a, rows, f)
    assert "Cappuccino" in text and "₹760" in text and "₹1,440" in text and "$" not in text
    assert U.narrative_is_faithful(text, rows, f)


def test_deterministic_narrative_metric_empty_and_table():
    a = U.analyze_result("today's sales", [{"revenue_today": 1440}])
    assert U.deterministic_narrative("today's sales", a, [{"revenue_today": 1440}], {}) == "Revenue today: ₹1,440."
    e = U.analyze_result("x", [])
    assert U.deterministic_narrative("x", e, [], {}).startswith("No data found")
    rows = [{"a": "x", "b": "y", "n": 1}]
    t = U.analyze_result("x", rows)
    assert U.deterministic_narrative("x", t, rows, {}) == "a: x; b: y; n: 1."
    many = [{"a": "x", "b": "y", "n": 1}, {"a": "p", "b": "q", "n": 2}]
    t2 = U.analyze_result("x", many)
    assert U.deterministic_narrative("x", t2, many, {}) == "Showing 2 rows."


def test_truncation_is_mentioned():
    rows = [{"product_name": f"P{i}", "category": "C"} for i in range(U.MAX_ROWS)]
    a = U.analyze_result("list", rows)
    assert str(U.MAX_ROWS) in U.deterministic_narrative("list", a, rows, {}, truncated=True)


# ── Narrative faithfulness ────────────────────────────────────────────────────

def test_faithful_narrative_passes():
    rows = _ranking_rows()
    f = U.compute_facts(U.analyze_result("x", rows), rows, "x", TODAY)
    assert U.narrative_is_faithful("Cappuccino leads with ₹760, followed by Latte at ₹400.", rows, f)
    assert U.narrative_is_faithful("Cappuccino brought in ₹760, about 52.8% of the ₹1,440 total.", rows, f)
    assert U.narrative_is_faithful("Cappuccino brought in ₹760, about 53% of the ₹1,440 total.", rows, f)


def test_invented_numbers_are_rejected():
    rows = _ranking_rows()
    f = U.compute_facts(U.analyze_result("x", rows), rows, "x", TODAY)
    assert not U.narrative_is_faithful("Cappuccino brought in ₹900.", rows, f)
    assert not U.narrative_is_faithful("Sales grew 40% versus last month.", rows, f)


def test_dollar_sign_is_rejected():
    rows = _ranking_rows()
    assert not U.narrative_is_faithful("Cappuccino brought in $760.", rows, {})


def test_empty_narrative_is_rejected():
    assert not U.narrative_is_faithful("   ", _ranking_rows(), {})


# ── Verified examples bank ────────────────────────────────────────────────────

EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bi_examples.json")

KNOWN_COLUMNS = {
    # view names that contain an underscore
    "order_items",
    # orders
    "order_id", "customer_id", "status", "order_total", "created_at", "confirmed_at", "updated_at",
    "order_date", "order_month", "order_week", "order_hour", "order_weekday", "order_weekday_num",
    # order_items
    "item_id", "product_id", "product_name", "category", "quantity", "unit_price", "line_total",
    # products
    "description", "price", "rating", "ingredients",
    # customers
    "location", "likes", "dislikes", "allergies", "joined_at", "orders_count", "total_spent",
    "first_order_at", "last_order_at",
}
SQL_FUNCTIONS = {
    "date_trunc", "make_interval", "make_date", "generate_series", "jsonb_array_elements_text", "row_number",
}


def _examples():
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_examples_exist_and_are_unique():
    ex = _examples()
    assert len(ex) >= 30
    questions = [e["question"] for e in ex]
    assert len(set(questions)) == len(questions)
    for e in ex:
        assert e["question"].strip() and e["sql"].strip()


def test_every_example_passes_the_sql_validator():
    for e in _examples():
        assert U.validate_sql(e["sql"]) is None, (e["question"], U.validate_sql(e["sql"]))
        assert U.scrub_sql(e["sql"]) == e["sql"].strip(), e["question"]


def test_every_example_uses_only_known_columns():
    """Catches typos like order_totla: every snake_case word must be a known column, function or alias."""
    for e in _examples():
        sql = U.mask_literals(e["sql"]).lower()
        aliases = set(re.findall(r"\bas\s+([a-z_][a-z0-9_]*)", sql))
        for word in re.findall(r"[a-z_][a-z0-9_]*", sql):
            if "_" not in word:
                continue
            assert word in KNOWN_COLUMNS or word in SQL_FUNCTIONS or word in aliases, (e["question"], word)


def test_examples_never_reference_base_tables_or_schemas():
    for e in _examples():
        sql = U.mask_literals(e["sql"]).lower()
        for forbidden in ("coffee_shop_", "bi.", "public.", "auth."):
            assert forbidden not in sql, (e["question"], forbidden)


# ── Explicitly requested chart types ──────────────────────────────────────────

def test_requested_chart_detection():
    assert U.requested_chart("show revenue by category as a pie chart") == "pie"
    assert U.requested_chart("top 5 products by revenue this month as a bar chart") == "bar"
    assert U.requested_chart("sales trend as a line chart") == "line"
    assert U.requested_chart("give me the products in a table") == "table"
    assert U.requested_chart("show me a table of revenue by category") == "table"
    assert U.requested_chart("top 5 products by revenue this month") is None
    assert U.requested_chart("") is None
    assert U.requested_chart("which coffee bar location sells most") is None


def _cat_rows():
    return [{"category": "Coffee", "revenue": 600}, {"category": "Bakery", "revenue": 300}, {"category": "Beans", "revenue": 100}]


def test_requested_pie_overrides_default_bar():
    assert U.analyze_result("revenue by category", _cat_rows())["chart_type"] == "bar"
    assert U.analyze_result("revenue by category as a pie chart", _cat_rows())["chart_type"] == "pie"


def test_requested_table_and_bar_and_line():
    rows = _cat_rows()
    assert U.analyze_result("revenue by category in a table", rows)["chart_type"] == "table"
    assert U.analyze_result("revenue by category as a bar chart", rows)["chart_type"] == "bar"
    months = [{"month": f"2026-{m:02d}-01", "revenue": m * 10} for m in (7, 8, 9, 10)]
    assert U.analyze_result("monthly revenue as a bar chart", months)["chart_type"] == "bar"
    assert U.analyze_result("monthly revenue", months)["chart_type"] == "line"


def test_impossible_pie_falls_back_gracefully():
    negative = [{"category": "A", "profit": 10}, {"category": "B", "profit": -5}]
    assert U.analyze_result("profit by category as a pie chart", negative)["chart_type"] == "bar"
    single = [{"category": "A", "revenue": 10}]
    assert U.analyze_result("revenue as a pie chart", single)["chart_type"] == "table"


def test_line_chart_on_categories_keeps_the_query_order():
    rows = [{"product_name": "Zebra", "revenue": 5}, {"product_name": "Apple", "revenue": 9}, {"product_name": "Mango", "revenue": 7}]
    a = U.analyze_result("revenue by product as a line chart", rows)
    assert a["chart_type"] == "line"
    assert [d["name"] for d in U.build_chart_data(a, rows)] == ["Zebra", "Apple", "Mango"]
    f = U.compute_facts(a, rows, "x", TODAY)
    assert f["first"]["label"] == "Zebra" and f["last"]["label"] == "Mango"
    assert "items" in U.deterministic_narrative("x", a, rows, f) and "periods" not in U.deterministic_narrative("x", a, rows, f)
