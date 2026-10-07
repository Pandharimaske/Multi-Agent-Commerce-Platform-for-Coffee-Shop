"""Prompt building blocks for the admin BI agent (pure strings, no heavy imports)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.utils.bi_utils import MAX_ROWS

SCHEMA_DOC = """\
## Views you can query (reference them by bare name, no schema prefix)

orders - one row per order or cart (header)
  order_id, customer_id (anonymised, e.g. 'C1a2b3c4d'), status ('confirmed' = paid order, 'pending' = open cart),
  order_total (rupees), created_at, confirmed_at, updated_at (timestamptz),
  order_date (date, IST), order_month (date, first day of the month, IST), order_week (date, Monday of the week, IST),
  order_hour (0-23, IST), order_weekday ('Monday'...), order_weekday_num (1=Monday ... 7=Sunday)
  The order_* columns are NULL for pending carts.

order_items - one row per product per order (line items, joined to order info)
  item_id, order_id, customer_id, product_id, product_name, category, quantity, unit_price, line_total (rupees),
  status, confirmed_at, order_date, order_month, order_week, order_hour, order_weekday, order_weekday_num

products - the menu
  product_id, product_name, category, description, price (rupees), rating, ingredients (jsonb array of text)

customers - one row per customer (anonymised, no names or emails)
  customer_id, location, likes / dislikes / allergies (jsonb arrays of text), joined_at,
  orders_count and total_spent (confirmed orders only), first_order_at, last_order_at
"""

BUSINESS_DEFINITIONS = """\
## Business definitions
- Revenue = SUM(order_total) over confirmed orders. Product or category revenue = SUM(line_total) on order_items.
- Orders = COUNT of confirmed orders. Average order value = revenue / orders. Units sold = SUM(quantity).
- A customer is a distinct customer_id. Repeat customer = orders_count >= 2. New customer in a period = first_order_at in that period.
- Abandoned cart = pending order whose updated_at is older than 24 hours.
- Weeks start on Monday. All dates and hours are India time (IST); the order_* columns are already in IST.
- Growth % = (current - previous) * 100.0 / NULLIF(previous, 0).
- Currency is Indian rupees.
"""

SQL_RULES = """\
## SQL rules
1. Write ONE read-only query. A query may start with SELECT or WITH. No semicolon, no comments, no markdown.
2. Use only the views above (orders, order_items, products, customers), never schema prefixes or other tables.
3. Today (IST) is (now() AT TIME ZONE 'Asia/Kolkata')::date. This month is date_trunc('month', now() AT TIME ZONE 'Asia/Kolkata')::date, compared with order_month. Use the pre-computed order_date / order_month / order_week / order_hour columns instead of converting time zones yourself.
4. Revenue questions use status = 'confirmed' unless the question is about carts or pending orders.
5. Product or category revenue: SUM(line_total) from order_items. Order-level revenue: SUM(order_total) from orders. Never join orders to order_items and then sum order_total.
6. Chart-friendly answers: for a ranking, breakdown or trend return EXACTLY two columns, the label first and the number second, with meaningful aliases (product_name, revenue, units_sold, orders ...). For a single number return one column. For comparisons or several metrics return several columns.
7. Reports or trends over a period (this year, last 7 days, last 8 weeks) must include EVERY period, with 0 where there were no orders (generate_series plus a LEFT JOIN, as in the examples).
8. Top-N: ORDER BY the metric DESC with a LIMIT (10 if the user says "top" without a number). Always use a deterministic ORDER BY.
9. Percentages: ROUND(x * 100.0 / NULLIF(y, 0), 1). Round money to 2 decimals. Guard every division with NULLIF. ROUND(x, n) only works on numeric values: cast the floating-point columns price and rating with ::numeric before rounding.
10. Match product names with ILIKE '%name%', using the product list below to resolve plurals, typos and partial names.
11. No SELECT *. Do not select description or ingredients unless asked.
12. Customers are anonymous ids. You cannot get names or emails.
13. For follow-ups ("compare that to last month", "now by category"), reuse the previous question's filters and change only what the user asked to change.
"""

INTENT_GUIDE = """\
## Decide the intent first
- "query": the question can be answered from the views. Write the SQL.
- "clarify": the question is truly ambiguous and no sensible default exists. Put ONE short question in "message". Prefer making a reasonable assumption over asking.
- "out_of_scope": not about this shop's sales, products, customers or orders (weather, general knowledge, coding, opinions). Put a short, polite reply in "message".
- "write_request": asks to change, delete, add or update data or settings. Explain in "message" that you are read-only.
Always list the assumptions you made (periods, status filters, definitions) in "assumptions" as short phrases, for example "This year = 1 Jan to today (IST)".
"""


def format_examples(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return "(none retrieved)"
    blocks = []
    for ex in examples:
        blocks.append(f"Q: {ex.get('question', '').strip()}\nSQL: {ex.get('sql', '').strip()}")
    return "\n\n".join(blocks)


def format_profile(profile: Optional[Dict[str, Any]]) -> str:
    if not profile:
        return "(data profile unavailable)"
    lines = []
    if profile.get("categories"):
        lines.append("Menu categories: " + ", ".join(profile["categories"]))
    if profile.get("products"):
        lines.append(f"Products ({len(profile['products'])}): " + ", ".join(profile["products"]))
    if profile.get("stats"):
        s = profile["stats"]
        lines.append(
            f"Confirmed orders: {s.get('confirmed_orders', 0)}; pending carts: {s.get('pending_carts', 0)}; "
            f"first order date: {s.get('first_order_date') or 'n/a'}; last order date: {s.get('last_order_date') or 'n/a'}."
        )
    return "\n".join(lines) if lines else "(data profile unavailable)"


def build_planner_prompt(
    now_ist: str,
    profile: Optional[Dict[str, Any]],
    examples: List[Dict[str, Any]],
    prev_question: Optional[str] = None,
    prev_sql: Optional[str] = None,
    error: Optional[str] = None,
    failed_sql: Optional[str] = None,
) -> str:
    context = ""
    if prev_question:
        context = f"\n## Previous turn (for follow-up questions)\nPrevious question: {prev_question}\n"
        if prev_sql:
            context += f"Previous SQL: {prev_sql}\n"

    retry = ""
    if error:
        retry = (
            "\n## Your previous attempt FAILED\n"
            f"SQL: {failed_sql or '(none)'}\nError: {error}\n"
            "Fix the problem. Use only the views and columns listed above.\n"
        )

    return (
        "You are the BI analyst for Merry's Way, a coffee shop in India. "
        "Turn the owner's question into ONE read-only PostgreSQL query over the views below, or decline politely.\n"
        f"Current time: {now_ist}\n\n"
        f"{SCHEMA_DOC}\n{BUSINESS_DEFINITIONS}\n"
        f"## Data currently available\n{format_profile(profile)}\n\n"
        f"{SQL_RULES}\n"
        f"## Similar verified examples\n{format_examples(examples)}\n"
        f"{context}{retry}\n"
        f"{INTENT_GUIDE}"
    )


NARRATIVE_SYSTEM = f"""\
You are a BI analyst writing a short answer for a coffee shop owner in India.
Rules:
1. Write 1 to 3 plain sentences. No markdown, no bullet points.
2. Money is in Indian rupees: write it with the ₹ symbol, never $.
3. Use ONLY numbers that appear in FACTS or ROWS. Do not calculate new numbers, percentages or growth rates yourself.
4. Do not invent causes, seasonality, marketing or operational reasons, and do not give advice unless asked.
5. If FACTS contain "in_progress_note", mention it and do not call the latest period a decline.
6. Describe a trend only with at least 3 complete periods; otherwise just state and compare the figures.
7. If "truncated" is true, say only the first {MAX_ROWS} rows were analysed.
"""


def build_narrative_messages(question: str, analysis: Dict[str, Any], rows: List[Dict[str, Any]],
                             facts: Dict[str, Any], assumptions: List[str], truncated: bool):
    human = (
        f"QUESTION: {question}\n"
        f"ASSUMPTIONS: {json.dumps(assumptions)}\n"
        f"TRUNCATED: {str(truncated).lower()}\n"
        f"FACTS: {json.dumps(facts, default=str)}\n"
        f"ROWS (first 20 of {len(rows)}): {json.dumps(rows[:20], default=str)}\n"
        "Write the answer."
    )
    return NARRATIVE_SYSTEM, human


DEFAULT_MESSAGES = {
    "out_of_scope": "I can only answer questions about your shop's sales, products, customers and orders.",
    "write_request": "I'm read-only: I can analyse your data but I can't change, add or delete anything.",
    "clarify": "Could you tell me a bit more about what you'd like to see?",
}
