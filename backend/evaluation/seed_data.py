"""Deterministic synthetic data for the evaluation database (pure Python generator).

Everything is generated relative to "today" (IST) with a fixed random seed, so the SHAPE of the data
is identical on every run. Customers use @demo.merrysway.test emails, so they can be found (and
removed) with one query, and so any leak of an email in an agent answer is detectable.

A tie audit re-draws the data (seed + 1, seed + 2, ...) until the rankings the golden questions rely
on (top products, busiest hour, ...) have a single clear answer. Without that, "correct" would be
ambiguous and the evaluation would measure luck.
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
DEMO_DOMAIN = "demo.merrysway.test"

LOCATIONS = ["Koregaon Park", "Baner", "Kothrud", "Viman Nagar", "Camp", "Hinjewadi", "Aundh"]
LIKES = ["cold brew", "latte", "chocolate", "croissant", "oat milk", "caramel", "espresso", "matcha"]
DISLIKES = ["bitter", "very sweet", "cinnamon"]
ALLERGIES = ["nuts", "dairy", "gluten", "eggs"]
HOURS = list(range(7, 22))
HOUR_WEIGHTS = [2, 6, 8, 5, 3, 6, 9, 6, 3, 3, 5, 8, 7, 4, 2]


# ── helpers shared with the ground-truth module ───────────────────────────────

def customer_id(email: str) -> str:
    """Same anonymised id the BI views compute: 'C' + first 8 hex chars of md5(email)."""
    return "C" + hashlib.md5(email.encode()).hexdigest()[:8]


def ist(ts: datetime) -> datetime:
    return ts.astimezone(IST)


def confirmed(ds: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [o for o in ds["orders"] if o["status"] == "confirmed"]


def load_products(path: str) -> List[Dict[str, Any]]:
    products = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                products.append({"name": p["name"], "category": p["category"], "price": float(p["price"])})
    return products


# ── generator ─────────────────────────────────────────────────────────────────

def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _time_of_day(rng: random.Random, day_offset: int, now_ist: datetime):
    if day_offset == 0:  # today: only times that have already happened
        now_min = now_ist.hour * 60 + now_ist.minute
        lo = 6 * 60 if now_min > 6 * 60 + 30 else 0
        minute = rng.randint(lo, max(lo, now_min - 1))
        return minute // 60, minute % 60
    return rng.choices(HOURS, weights=HOUR_WEIGHTS)[0], rng.randint(0, 59)


def _generate(products: List[Dict[str, Any]], now_ist: datetime, seed: int, n_customers: int,
              n_confirmed: int, n_pending: int, days: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    today = now_ist.date()

    names = sorted(p["name"] for p in products)
    shuffled = names[:]
    rng.shuffle(shuffled)
    never_sold = set(shuffled[:6])
    sellable = [p for p in products if p["name"] not in never_sold]
    ranks = [p["name"] for p in sellable]
    rng.shuffle(ranks)
    weight = {name: (i + 1) ** 1.6 for i, name in enumerate(ranks)}

    customers = []
    for i in range(1, n_customers + 1):
        joined_offset = rng.randint(1, 119)
        joined_local = datetime.combine(today - timedelta(days=joined_offset),
                                        time(rng.randint(8, 20), rng.randint(0, 59)), tzinfo=IST)
        customers.append({
            "email": f"cust{i:02d}@{DEMO_DOMAIN}",
            "name": f"Demo Customer {i:02d}",
            "location": rng.choice(LOCATIONS + [None, None]),
            "likes": rng.sample(LIKES, rng.randint(0, 3)),
            "dislikes": rng.sample(DISLIKES, rng.randint(0, 1)),
            "allergies": [rng.choice(ALLERGIES)] if rng.random() < 0.25 else [],
            "joined_offset": joined_offset,
            "joined_at": joined_local.astimezone(UTC),
        })
    cweight = {c["email"]: rng.random() ** 2.5 + 0.03 for c in customers}

    def make_lines(fixed_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = fixed_k or rng.choices([1, 2, 3, 4], weights=[45, 35, 15, 5])[0]
        pool, chosen = list(sellable), []
        for _ in range(min(k, len(pool))):
            pick = rng.choices(pool, weights=[weight[p["name"]] for p in pool])[0]
            chosen.append(pick)
            pool.remove(pick)
        lines = []
        for n, p in enumerate(chosen, start=1):
            qty = rng.choices([1, 2, 3], weights=[65, 25, 10])[0]
            lines.append({
                "id": _uuid(rng), "product_name": p["name"], "category": p["category"], "quantity": qty,
                "unit_price": p["price"], "line_total": round(p["price"] * qty, 2), "line_no": n,
            })
        return lines

    orders: List[Dict[str, Any]] = []
    offsets = [0] * 8 + [1] * 8 + [min(days - 1, int(rng.random() ** 1.4 * days)) for _ in range(n_confirmed - 16)]
    for d in offsets:
        eligible = [c for c in customers if c["joined_offset"] > d] or customers
        cust = rng.choices(eligible, weights=[cweight[c["email"]] for c in eligible])[0]
        hour, minute = _time_of_day(rng, d, now_ist)
        local = datetime.combine(today - timedelta(days=d), time(hour, minute, rng.randint(0, 59)), tzinfo=IST)
        confirmed_at = local.astimezone(UTC)
        lines = make_lines()
        orders.append({
            "id": _uuid(rng), "email": cust["email"], "status": "confirmed",
            "total": round(sum(l["line_total"] for l in lines), 2),
            "created_at": confirmed_at - timedelta(minutes=rng.randint(2, 25)),
            "confirmed_at": confirmed_at, "updated_at": confirmed_at, "items": lines,
        })

    pending_pool = [c for c in customers if c["joined_offset"] >= 11] or customers
    for i, cust in enumerate(rng.sample(pending_pool, min(n_pending, len(pending_pool)))):
        hours_ago = rng.randint(30, 24 * 10) if i < n_pending // 2 else rng.randint(1, 12)
        updated = (now_ist - timedelta(hours=hours_ago, minutes=rng.randint(0, 59))).astimezone(UTC)
        lines = make_lines(rng.choice([1, 2, 3]))
        orders.append({
            "id": _uuid(rng), "email": cust["email"], "status": "pending",
            "total": round(sum(l["line_total"] for l in lines), 2),
            "created_at": updated - timedelta(minutes=rng.randint(1, 15)),
            "confirmed_at": None, "updated_at": updated, "items": lines,
        })

    return {
        "meta": {"seed": seed, "now_ist": now_ist.isoformat(), "never_sold": sorted(never_sold),
                 "n_customers": n_customers, "n_confirmed": n_confirmed, "n_pending": n_pending, "days": days},
        "customers": customers,
        "orders": orders,
    }


def audit(ds: Dict[str, Any]) -> List[str]:
    """Reasons the dataset would make some golden answer ambiguous (empty list = good)."""
    problems: List[str] = []
    today = datetime.fromisoformat(ds["meta"]["now_ist"]).date()
    conf = confirmed(ds)

    rev, units = Counter(), Counter()
    cat_rev: Counter = Counter()
    cat_units: Dict[str, Counter] = defaultdict(Counter)
    for o in conf:
        for l in o["items"]:
            rev[l["product_name"]] += l["line_total"]
            units[l["product_name"]] += l["quantity"]
            cat_rev[l["category"]] += l["line_total"]
            cat_units[l["category"]][l["product_name"]] += l["quantity"]

    for label, counter in (("product revenue", rev), ("product units", units)):
        top = [round(v, 2) for _, v in counter.most_common(7)]
        if len(top) < 7 or len(set(top)) < len(top):
            problems.append(f"tie in the top-7 {label}")

    total_rev = sum(cat_rev.values()) or 1
    for cat, v in cat_rev.items():
        if abs(v * 100 / total_rev - 20) < 1.0:
            problems.append(f"category {cat} is within 1 point of the 20% revenue threshold")
    for cat, ctr in cat_units.items():
        vals = sorted(ctr.values(), reverse=True)
        if len(vals) >= 2 and vals[0] == vals[1]:
            problems.append(f"tie for best seller in {cat}")

    spend, count, months, last = Counter(), Counter(), defaultdict(set), {}
    for o in conf:
        d = ist(o["confirmed_at"]).date()
        spend[o["email"]] += o["total"]
        count[o["email"]] += 1
        months[o["email"]].add((d.year, d.month))
        last[o["email"]] = max(last.get(o["email"], d), d)
    top_spend = [round(v, 2) for _, v in spend.most_common(6)]
    if len(top_spend) < 6 or len(set(top_spend)) < len(top_spend):
        problems.append("tie in the top-6 customer spend")
    if sum(1 for c in count.values() if c > 5) < 3:
        problems.append("fewer than 3 customers with more than 5 orders")
    if sum(1 for s in months.values() if len(s) > 1) < 5:
        problems.append("fewer than 5 customers ordering in more than one month")
    if any(d == today - timedelta(days=30) for d in last.values()):
        problems.append("a customer's last order is exactly 30 days ago (boundary ambiguity)")
    if sum(1 for d in last.values() if d < today - timedelta(days=30)) < 2:
        problems.append("fewer than 2 customers inactive for more than 30 days")

    for label, keyfn in (("hour", lambda t: ist(t).hour), ("weekday", lambda t: ist(t).strftime("%A")),
                         ("day", lambda t: ist(t).date())):
        top2 = Counter(keyfn(o["confirmed_at"]) for o in conf).most_common(2)
        if len(top2) < 2 or top2[0][1] == top2[1][1]:
            problems.append(f"tie for the busiest {label}")

    for window in (30, 29):
        ctr: Counter = Counter()
        for o in conf:
            if ist(o["confirmed_at"]).date() >= today - timedelta(days=window):
                for l in o["items"]:
                    ctr[l["product_name"]] += l["quantity"]
        top2 = ctr.most_common(2)
        if len(top2) < 2 or top2[0][1] == top2[1][1]:
            problems.append(f"tie for the best seller of the last {window} days")
    return problems


def generate_dataset(products: List[Dict[str, Any]], now_ist: datetime, seed: int = 42, n_customers: int = 40,
                     n_confirmed: int = 320, n_pending: int = 12, days: int = 90) -> Dict[str, Any]:
    last: List[str] = []
    for attempt in range(400):
        ds = _generate(products, now_ist, seed + attempt, n_customers, n_confirmed, n_pending, days)
        last = audit(ds)
        if not last:
            ds["meta"]["seed_used"] = seed + attempt
            return ds
    raise RuntimeError("Could not generate a tie-free dataset: " + "; ".join(last))


# ── persistence (so ground truth can be checked on a later day) ───────────────

def _enc(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(type(o))


def dataset_to_json(ds: Dict[str, Any]) -> str:
    return json.dumps(ds, default=_enc)


def dataset_from_json(text: str) -> Dict[str, Any]:
    ds = json.loads(text)
    for c in ds["customers"]:
        c["joined_at"] = datetime.fromisoformat(c["joined_at"])
    for o in ds["orders"]:
        for key in ("created_at", "confirmed_at", "updated_at"):
            if o[key] is not None:
                o[key] = datetime.fromisoformat(o[key])
    return ds


# ── database loading ──────────────────────────────────────────────────────────

def _batches(rows: List[Dict[str, Any]], size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def load_into_db(client, ds: Dict[str, Any], product_ids: Dict[str, str], batch: int = 200) -> Dict[str, int]:
    """Replace all demo rows in the database with this dataset. Real (non-demo) rows are untouched."""
    client.table("coffee_shop_profiles").delete().like("user_email", f"%@{DEMO_DOMAIN}").execute()

    profiles = [{
        "user_email": c["email"], "name": c["name"], "likes": c["likes"], "dislikes": c["dislikes"],
        "allergies": c["allergies"], "location": c["location"], "created_at": c["joined_at"].isoformat(),
    } for c in ds["customers"]]
    orders, items = [], []
    for o in ds["orders"]:
        orders.append({
            "id": o["id"], "user_email": o["email"], "status": o["status"], "total": o["total"],
            "created_at": o["created_at"].isoformat(),
            "confirmed_at": o["confirmed_at"].isoformat() if o["confirmed_at"] else None,
            "updated_at": o["updated_at"].isoformat(),
        })
        for l in o["items"]:
            items.append({
                "id": l["id"], "order_id": o["id"], "product_id": product_ids[l["product_name"]],
                "product_name": l["product_name"], "quantity": l["quantity"], "unit_price": l["unit_price"],
                "line_total": l["line_total"], "line_no": l["line_no"],
            })

    for table, rows in (("coffee_shop_profiles", profiles), ("coffee_shop_orders", orders),
                        ("coffee_shop_order_items", items)):
        for chunk in _batches(rows, batch):
            client.table(table).insert(chunk).execute()
    return {"profiles": len(profiles), "orders": len(orders), "items": len(items)}
