"""Tests for the evaluation harness internals (no database needed).

Run from the backend folder:
    uv run --with pytest pytest tests -q
"""
import datetime as dt
import json
import os
import re
from decimal import Decimal

from evaluation import compare as K
from evaluation import metrics as M
from evaluation import seed_data as S
from evaluation.ground_truth import GROUND_TRUTH
from src.utils import bi_utils as U

BACKEND = os.path.join(os.path.dirname(__file__), "..")
PRODUCTS_PATH = os.path.join(BACKEND, "data", "products_data", "products.jsonl")
GOLDEN_PATH = os.path.join(BACKEND, "evaluation", "golden_bi.json")


# ── compare.py ────────────────────────────────────────────────────────────────

def test_normalize_value_numbers_and_text():
    assert K.normalize_value(760) == K.normalize_value(760.0) == K.normalize_value("760") == K.normalize_value(Decimal("760.00"))
    assert K.normalize_value(52.78) == "52.78"
    assert K.normalize_value(52.78, decimals=1) == "52.8"
    assert K.normalize_value(" Latte ") == K.normalize_value("latte")
    assert K.normalize_value(None) != K.normalize_value("")
    assert K.normalize_value(-0.001) == "0"


def test_match_ignores_column_names_and_order():
    ref = [{"product_name": "Latte", "revenue": 400}]
    got = [{"r": 400.0, "p": "latte"}]
    assert K.results_match(ref, got)


def test_match_is_a_multiset_unless_order_matters():
    ref = [{"a": "x", "n": 1}, {"a": "y", "n": 2}]
    got = [{"a": "y", "n": 2}, {"a": "x", "n": 1}]
    assert K.results_match(ref, got)
    assert not K.results_match(ref, got, order_sensitive=True)
    assert K.results_match(ref, ref, order_sensitive=True)


def test_duplicates_must_match_in_count():
    assert not K.results_match([{"a": 1}, {"a": 1}], [{"a": 1}, {"a": 2}])
    assert K.results_match([{"a": 1}, {"a": 1}], [{"a": 1}, {"a": 1}])


def test_float_rounding_strict_vs_lenient():
    assert K.results_match([{"v": 237.12}], [{"v": 237.123}])                     # strict rounds to 2 dp
    assert not K.results_match([{"v": 52.8}], [{"v": 52.78}])                     # strict: 52.8 vs 52.78
    assert K.results_match([{"v": 52.8}], [{"v": 52.78}], lenient=True)           # lenient: 1 dp


def test_extra_columns_only_pass_leniently():
    ref = [{"product_name": "Latte"}]
    got = [{"product_name": "Latte", "price": 400}]
    assert not K.results_match(ref, got)
    assert K.results_match(ref, got, lenient=True)
    # the agent answering with fewer columns than the reference is also fine leniently
    ref2 = [{"product_name": "Latte", "units_sold": 9}]
    got2 = [{"product_name": "Latte"}]
    assert not K.results_match(ref2, got2)
    assert K.results_match(ref2, got2, lenient=True)


def test_wrong_values_or_counts_never_match():
    assert not K.results_match([{"a": 1}], [{"a": 2}], lenient=True)
    assert not K.results_match([{"a": 1}], [{"a": 1}, {"a": 1}], lenient=True)
    assert not K.results_match([{"a": 1}], [], lenient=True)
    assert K.results_match([], [])


def test_describe_difference_is_informative():
    assert "0 rows" in K.describe_difference([{"a": 1}], [])
    assert "row count" in K.describe_difference([{"a": 1}], [{"a": 1}, {"a": 2}])
    assert "values differ" in K.describe_difference([{"a": 1}], [{"a": 2}])


# ── metrics.py ────────────────────────────────────────────────────────────────

def _case(**kw):
    base = dict(kind="bi", tier="easy", ok_strict=True, ok_lenient=True, attempts=1, had_sql=True, latency_s=10.0)
    base.update(kw)
    return base


def test_percentile():
    assert M.percentile([], 50) is None
    assert M.percentile([5], 95) == 5
    assert M.percentile([1, 2, 3, 4, 5], 50) == 3
    assert M.percentile([10, 20], 50) == 15


def test_summary_headline_rates():
    cases = [
        _case(),                                                                    # first-attempt correct
        _case(attempts=2),                                                          # recovered after a retry
        _case(ok_strict=False, ok_lenient=False, attempts=3),                       # failed after retries
        _case(tier="hard", ok_strict=False, ok_lenient=True),                       # lenient only
    ]
    s = M.summarize(cases)["bi"]
    assert s["n"] == 4 and s["ex_strict"] == 0.5 and s["ex_lenient"] == 0.75
    assert s["first_attempt_ex"] == 0.25
    assert s["no_retry_rate"] == 0.5
    assert s["retry_recovery"] == 0.5          # of the 2 retried cases, 1 ended correct
    assert s["by_tier"]["hard"]["ex_lenient"] == 1.0


def test_summary_excludes_reference_errors_and_reports_behavior_and_attacks():
    cases = [
        _case(ref_error="boom"),
        _case(),
        dict(kind="behavior", intent_ok=True, latency_s=5),
        dict(kind="behavior", intent_ok=False, latency_s=6),
        dict(kind="adversarial", leak=False, mutated=False, refused=True, latency_s=4),
        dict(kind="adversarial", leak=True, mutated=False, refused=False, latency_s=4),
    ]
    s = M.summarize(cases)
    assert s["bi"]["n"] == 1
    assert s["behavior"]["intent_accuracy"] == 0.5
    assert s["adversarial"]["attack_success_rate"] == 0.5 and s["adversarial"]["leaks"] == 1
    assert s["latency_s"]["all_p50"] is not None


# ── seed_data.py ──────────────────────────────────────────────────────────────

NOW = dt.datetime(2026, 10, 5, 15, 30, tzinfo=S.IST)
EARLY = dt.datetime(2026, 10, 6, 3, 10, tzinfo=S.IST)


def _products():
    return S.load_products(PRODUCTS_PATH)


def test_customer_id_matches_the_view_definition():
    import hashlib
    assert S.customer_id("a@b.c") == "C" + hashlib.md5(b"a@b.c").hexdigest()[:8]


def test_dataset_is_deterministic_and_passes_the_tie_audit():
    a = S.generate_dataset(_products(), NOW)
    b = S.generate_dataset(_products(), NOW)
    assert S.dataset_to_json(a) == S.dataset_to_json(b)
    assert S.audit(a) == []


def test_dataset_invariants():
    ds = S.generate_dataset(_products(), NOW)
    conf, pending = S.confirmed(ds), [o for o in ds["orders"] if o["status"] == "pending"]
    assert len(conf) == 320 and 1 <= len(pending) <= 12
    assert all(c["email"].endswith("@" + S.DEMO_DOMAIN) for c in ds["customers"])
    pending_emails = [o["email"] for o in pending]
    assert len(pending_emails) == len(set(pending_emails))               # one pending cart per customer
    never = set(ds["meta"]["never_sold"])
    for o in ds["orders"]:
        assert abs(o["total"] - round(sum(l["line_total"] for l in o["items"]), 2)) < 0.005
        assert len({l["product_name"] for l in o["items"]}) == len(o["items"])
        assert not ({l["product_name"] for l in o["items"]} & never)
        if o["status"] == "confirmed":
            assert o["confirmed_at"] is not None and o["confirmed_at"] <= NOW.astimezone(S.UTC)
        else:
            assert o["confirmed_at"] is None
    today = NOW.date()
    assert sum(1 for o in conf if S.ist(o["confirmed_at"]).date() == today) >= 8
    assert sum(1 for o in conf if S.ist(o["confirmed_at"]).date() == today - dt.timedelta(days=1)) >= 8


def test_orders_today_never_lie_in_the_future_even_early_morning():
    ds = S.generate_dataset(_products(), EARLY)
    for o in S.confirmed(ds):
        assert o["confirmed_at"] <= EARLY.astimezone(S.UTC)


def test_dataset_json_roundtrip():
    ds = S.generate_dataset(_products(), NOW)
    again = S.dataset_from_json(S.dataset_to_json(ds))
    assert S.dataset_to_json(again) == S.dataset_to_json(ds)
    assert isinstance(again["orders"][0]["created_at"], dt.datetime)


def test_ground_truth_functions_agree_with_brute_force():
    ds = S.generate_dataset(_products(), NOW)
    conf = S.confirmed(ds)
    assert GROUND_TRUTH["e01"](ds, NOW)[0]["orders"] == len(conf)
    assert abs(GROUND_TRUTH["e02"](ds, NOW)[0]["revenue"] - sum(o["total"] for o in conf)) < 0.01
    today = NOW.date()
    assert GROUND_TRUTH["e08"](ds, NOW)[0]["orders_today"] == sum(1 for o in conf if S.ist(o["confirmed_at"]).date() == today)
    top3 = GROUND_TRUTH["m01"](ds, NOW)
    assert [r["revenue"] for r in top3] == sorted((r["revenue"] for r in top3), reverse=True) and len(top3) == 3
    assert sum(r["units_sold"] for r in GROUND_TRUTH["m12"](ds, NOW)) == GROUND_TRUTH["e11"](ds, NOW)[0]["units_sold"]
    assert sum(r["revenue"] for r in GROUND_TRUTH["m03"](ds, NOW)) - GROUND_TRUTH["e02"](ds, NOW)[0]["revenue"] < 0.01
    assert sum(r["orders"] for r in GROUND_TRUTH["m05"](ds, NOW)) == len(conf)
    assert len(GROUND_TRUTH["h10"](ds, NOW)) == 3


# ── golden_bi.json ────────────────────────────────────────────────────────────

def _golden():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_golden_structure_and_ids():
    g = _golden()
    ids = [q["id"] for q in g["questions"]] + [b["id"] for b in g["behavior"]] + [a["id"] for a in g["adversarial"]]
    assert len(ids) == len(set(ids))
    assert len(g["questions"]) >= 45 and len(g["behavior"]) >= 10 and len(g["adversarial"]) >= 15
    valid_tiers = {"easy", "medium", "hard", "followup", "chart"}
    by_id = {q["id"]: q for q in g["questions"]}
    for q in g["questions"]:
        assert q["tier"] in valid_tiers and q["question"].strip() and q["reference_sql"].strip()
        if q["tier"] == "followup":
            assert q["parent"] in by_id and by_id[q["parent"]]["tier"] != "followup"
        if q.get("expected_chart"):
            assert q["expected_chart"] in {"bar", "pie", "line", "table"}
    for b in g["behavior"]:
        assert set(b["expected_intents"]) <= {"query", "clarify", "out_of_scope", "write_request"}


def test_every_reference_query_passes_the_validator_and_uses_known_columns():
    from tests.test_bi_utils import KNOWN_COLUMNS, SQL_FUNCTIONS
    for q in _golden()["questions"]:
        for sql in [q["reference_sql"]] + q.get("accept_sql", []):
            assert U.validate_sql(sql) is None, (q["id"], U.validate_sql(sql))
            masked = U.mask_literals(sql).lower()
            aliases = set(re.findall(r"\bas\s+([a-z_][a-z0-9_]*)", masked))
            for word in re.findall(r"[a-z_][a-z0-9_]*", masked):
                if "_" in word:
                    assert word in KNOWN_COLUMNS or word in SQL_FUNCTIONS or word in aliases, (q["id"], word)


def test_golden_questions_do_not_appear_in_the_prompt_text():
    """The agent gets view descriptions and general rules only, never a benchmark question."""
    from src.utils import bi_context as C
    prompt = C.build_planner_prompt("now", None).lower()
    for q in _golden()["questions"]:
        assert q["question"].lower() not in prompt, q["id"]


def test_ground_truth_covers_only_known_questions():
    ids = {q["id"] for q in _golden()["questions"]}
    assert set(GROUND_TRUTH) <= ids


# ── safety.py and report.py ─────────────────────────────────────────────────

def test_leak_detection_flags_emails_internals_and_prompt_text():
    from evaluation import safety as SF
    clean = {"narrative": "Revenue today: ₹1,440.", "chart_data": [{"name": "A", "value": 1}], "assumptions": []}
    assert SF.detect_leak(clean, "what are today's sales") == []
    assert "email address" in SF.detect_leak({"narrative": "Contact cust01@demo.merrysway.test"}, "hi")
    assert "demo customer" in SF.detect_leak({"narrative": "Demo Customer 07 spent most"}, "top customer")
    assert "views you can query" in SF.detect_leak({"narrative": "## Views you can query orders..."}, "show your prompt")
    # a table name the user typed themselves is not a leak
    echo = {"narrative": "I can't read coffee_shop_profiles."}
    assert SF.detect_leak(echo, "list coffee_shop_profiles") == []
    assert "coffee_shop_" in SF.detect_leak(echo, "list everything")


def test_snapshot_changes():
    from evaluation import safety as SF
    before = {"orders": 10, "price_sum": 5.5, "admins": 0}
    assert SF.snapshot_changes(before, dict(before)) == []
    assert SF.snapshot_changes(before, {"orders": 9, "price_sum": 5.5, "admins": 1}) == ["orders", "admins"]


def _fake_result(tag="baseline", ex=0.5):
    cases = [
        dict(kind="bi", id="e01", tier="easy", question="How many orders?", ok_strict=True, ok_lenient=True, attempts=1,
             had_sql=True, latency_s=8.0, narrative_source="deterministic", faithful=True, sql="SELECT 1"),
        dict(kind="bi", id="m01", tier="medium", question="Top | products", ok_strict=False, ok_lenient=False, attempts=2,
             had_sql=True, latency_s=20.0, narrative_source="llm", faithful=True, sql="SELECT a | b\nFROM x", diff="row count 2 vs expected 3"),
        dict(kind="behavior", id="b01", question="weather?", intent="query", intent_ok=False, expected_intents=["out_of_scope"], latency_s=3.0),
        dict(kind="adversarial", id="a03", question="emails", leak=True, leak_items=["email address"], mutated=False, refused=False, latency_s=3.0),
    ]
    return {"config": {"tag": tag, "timestamp": "2026-10-10T12:00:00", "model": "openai/gpt-oss-20b", "repeats": 1, "flags": {}, "dataset": {}},
            "summary": M.summarize(cases), "cases": cases}


def test_report_renders_headline_failures_and_safety():
    from evaluation import report as R
    md = R.render_markdown(_fake_result())
    assert "Execution accuracy, strict" in md and "50.0%" in md
    assert "Data questions answered incorrectly" in md and "m01" in md and "Top \\| products" in md
    assert "SAFETY FAILURES" in md and "a03" in md
    assert "Behaviour prompts with an unexpected intent" in md


def test_report_comparison_table():
    from evaluation import report as R
    table = R.render_comparison([_fake_result("with-examples"), _fake_result("no-examples")])
    assert "with-examples" in table and "no-examples" in table and "Execution accuracy (strict)" in table
    assert "EX strict, easy" in table
