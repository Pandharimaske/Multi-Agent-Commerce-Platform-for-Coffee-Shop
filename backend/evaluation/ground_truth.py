"""Independent ground truth for a subset of golden questions, computed in plain Python from the
generated dataset (no SQL). `verify_golden.py` runs each reference SQL against the database and
compares it with these answers, so a wrong reference query, a wrong view, or a seeding bug is
caught before it can distort the benchmark.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, List

from evaluation.seed_data import confirmed, customer_id, ist


def _d(x: Any) -> Decimal:
    return Decimal(str(x))


def _f(x: Decimal) -> float:
    return float(x)


def _revenue(orders: List[Dict[str, Any]]) -> Decimal:
    return sum((_d(o["total"]) for o in orders), Decimal("0"))


def _on(ds, now: datetime, days_ago: int) -> List[Dict[str, Any]]:
    day = now.date() - timedelta(days=days_ago)
    return [o for o in confirmed(ds) if ist(o["confirmed_at"]).date() == day]


def _per_customer(ds) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"spent": Decimal("0"), "orders": 0, "months": set()})
    for o in confirmed(ds):
        d = ist(o["confirmed_at"]).date()
        s = stats[o["email"]]
        s["spent"] += _d(o["total"])
        s["orders"] += 1
        s["months"].add((d.year, d.month))
    return stats


def e01(ds, now): return [{"orders": len(confirmed(ds))}]
def e02(ds, now): return [{"revenue": _f(_revenue(confirmed(ds)))}]
def e05(ds, now): return [{"customers": len({o["email"] for o in confirmed(ds)})}]
def e06(ds, now): return [{"pending_carts": sum(1 for o in ds["orders"] if o["status"] == "pending")}]
def e08(ds, now): return [{"orders_today": len(_on(ds, now, 0))}]
def e09(ds, now): return [{"revenue_yesterday": _f(_revenue(_on(ds, now, 1)))}]
def e10(ds, now): return [{"products_sold": len({l["product_name"] for o in confirmed(ds) for l in o["items"]})}]
def e11(ds, now): return [{"units_sold": sum(l["quantity"] for o in confirmed(ds) for l in o["items"])}]
def e12(ds, now): return [{"customers": len(ds["customers"])}]


def e13(ds, now):
    conf = confirmed(ds)
    avg = (_revenue(conf) / len(conf)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return [{"average_order_value": _f(avg)}]


def _product_totals(ds):
    rev: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    units: Counter = Counter()
    for o in confirmed(ds):
        for l in o["items"]:
            rev[l["product_name"]] += _d(l["line_total"])
            units[l["product_name"]] += l["quantity"]
    return rev, units


def m01(ds, now):
    rev, _ = _product_totals(ds)
    top = sorted(rev.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return [{"product_name": n, "revenue": _f(v)} for n, v in top]


def m03(ds, now):
    cat: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for o in confirmed(ds):
        for l in o["items"]:
            cat[l["category"]] += _d(l["line_total"])
    return [{"category": c, "revenue": _f(v)} for c, v in cat.items()]


def m05(ds, now):
    ctr = Counter(ist(o["confirmed_at"]).strftime("%A") for o in confirmed(ds))
    return [{"order_weekday": d, "orders": n} for d, n in ctr.items()]


def m10(ds, now):
    ctr = Counter(ist(o["confirmed_at"]).hour for o in confirmed(ds))
    hour, n = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return [{"order_hour": hour, "orders": n}]


def m12(ds, now):
    cat: Counter = Counter()
    for o in confirmed(ds):
        for l in o["items"]:
            cat[l["category"]] += l["quantity"]
    return [{"category": c, "units_sold": n} for c, n in cat.items()]


def m13(ds, now):
    ctr = Counter(c["location"] or "Unknown" for c in ds["customers"])
    return [{"location": k, "customers": n} for k, n in ctr.items()]


def m15(ds, now): return [{"revenue": _f(_revenue([o for o in confirmed(ds) if o["total"] > 800]))}]


def m18(ds, now):
    ctr = Counter(ist(o["confirmed_at"]).date() for o in confirmed(ds))
    day, n = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return [{"order_date": day.isoformat(), "orders": n}]


def h04(ds, now):
    return [{"customers": sum(1 for s in _per_customer(ds).values() if len(s["months"]) > 1)}]


def h06(ds, now):
    n = 0
    for o in confirmed(ds):
        cats = {l["category"] for l in o["items"]}
        if "Coffee" in cats and "Bakery" in cats:
            n += 1
    return [{"orders": n}]


def h10(ds, now):
    stats = _per_customer(ds)
    top = sorted(stats.items(), key=lambda kv: (-kv[1]["spent"], customer_id(kv[0])))[:3]
    return [{"customer_id": customer_id(e), "total_spent": _f(s["spent"]), "orders_count": s["orders"]} for e, s in top]


def h12(ds, now):
    nutty = {c["email"] for c in ds["customers"] if any("nut" in a.lower() for a in c["allergies"])}
    return [{"orders": sum(1 for o in confirmed(ds) if o["email"] in nutty)}]


GROUND_TRUTH: Dict[str, Callable[[Dict[str, Any], datetime], List[Dict[str, Any]]]] = {
    "e01": e01, "e02": e02, "e05": e05, "e06": e06, "e08": e08, "e09": e09, "e10": e10, "e11": e11,
    "e12": e12, "e13": e13, "m01": m01, "m03": m03, "m05": m05, "m10": m10, "m12": m12, "m13": m13,
    "m15": m15, "m18": m18, "h04": h04, "h06": h06, "h10": h10, "h12": h12,
}
