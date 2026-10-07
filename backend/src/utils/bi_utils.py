"""Pure helpers for the admin BI agent (standard library only, so they are easy to test).

Everything that can be decided by code is decided here, not by the LLM:
  * SQL cleaning and validation
  * chart type selection and chart data building
  * facts about a result (totals, top/bottom, shares, in-progress periods)
  * a deterministic narrative, and a check that an LLM narrative only uses
    numbers that really exist in the data
  * Indian-rupee formatting and email masking
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Dict, List, Optional, Tuple

MAX_ROWS = 200          # rows returned to the agent (the DB function fetches MAX_ROWS + 1)
MAX_TABLE_ROWS = 50     # rows shown in a table
IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ══════════════════════════════════════════════════════════════════════════════
# SQL cleaning and validation
# ══════════════════════════════════════════════════════════════════════════════

def _segments(sql: str) -> List[Tuple[str, str]]:
    """Split SQL into ('code' | 'string' | 'ident' | 'comment', text) segments."""
    segs: List[Tuple[str, str]] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":   # escaped quote ''
                        j += 2
                        continue
                    break
                j += 1
            segs.append(("string", sql[i:j + 1]))
            i = j + 1
        elif ch == '"':
            j = sql.find('"', i + 1)
            if j == -1:
                j = n - 1
            segs.append(("ident", sql[i:j + 1]))
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            if j == -1:
                j = n
            segs.append(("comment", sql[i:j]))
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            segs.append(("comment", sql[i:j]))
            i = j
        else:
            j = i
            while j < n and sql[j] not in "'\"" and not sql.startswith("--", j) and not sql.startswith("/*", j):
                j += 1
            segs.append(("code", sql[i:j]))
            i = j
    return segs


def strip_sql_comments(sql: str) -> str:
    return "".join(" " if kind == "comment" else text for kind, text in _segments(sql))


def mask_literals(sql: str) -> str:
    """Blank out string literals and quoted identifiers (and drop comments) for keyword scanning."""
    out = []
    for kind, text in _segments(sql):
        if kind == "comment":
            out.append(" ")
        elif kind == "string":
            out.append("''")
        elif kind == "ident":
            out.append('""')
        else:
            out.append(text)
    return "".join(out)


def scrub_sql(sql: str) -> str:
    """Remove markdown fences, comments and trailing semicolons the LLM tends to add."""
    sql = re.sub(r"```(?:sql)?", "", sql or "", flags=re.IGNORECASE)
    sql = strip_sql_comments(sql).strip()
    return re.sub(r"[;\s]+$", "", sql)


FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "truncate", "create", "grant",
    "revoke", "copy", "vacuum", "call", "execute", "merge", "into", "reindex", "notify",
)

_FORBIDDEN_REFERENCES = (
    (r"\bpg_\w*", "System functions and catalogs (pg_*) are not available."),
    (r"\binformation_schema\b", "System schemas are not available."),
    (r"\b(auth|storage|public|bi|extensions|vault)\s*\.", "Do not use schema prefixes. Query the views directly: orders, order_items, products, customers."),
    (r"\bcoffee_shop_\w*", "Query only the BI views: orders, order_items, products, customers."),
)


def validate_sql(sql: str) -> Optional[str]:
    """Return None if the SQL looks safe to run, otherwise a message the LLM can use to fix it."""
    if not sql or not sql.strip():
        return "The query is empty."
    masked = mask_literals(sql).lower()

    if not re.match(r"^\s*(select|with)\b", masked):
        return "Only SELECT queries are allowed (a query may start with WITH)."
    if ";" in masked:
        return "Multiple statements are not allowed. Remove the semicolon."
    if masked.count("(") != masked.count(")"):
        return "Unbalanced parentheses."
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", masked):
            return f"The keyword '{kw}' is not allowed. Only read-only SELECT queries are permitted."
    for pattern, message in _FORBIDDEN_REFERENCES:
        if re.search(pattern, masked):
            return message
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Formatting
# ══════════════════════════════════════════════════════════════════════════════

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _r2(v: Any) -> Any:
    """Round to 2 decimals, and return an int when the value is a whole number."""
    if not _is_number(v):
        return v
    f = float(v)
    return int(f) if f.is_integer() else round(f, 2)


def _indian_group(whole: str) -> str:
    if len(whole) <= 3:
        return whole
    head, tail = whole[:-3], whole[-3:]
    parts: List[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def format_inr(value: float) -> str:
    neg = value < 0
    whole, frac = f"{abs(float(value)):.2f}".split(".")
    text = _indian_group(whole) + ("" if frac == "00" else f".{frac}")
    return ("-" if neg else "") + "₹" + text


def format_number(value: float) -> str:
    neg = value < 0
    whole, frac = f"{abs(float(value)):.2f}".split(".")
    frac = frac.rstrip("0")
    text = _indian_group(whole) + (f".{frac}" if frac else "")
    return ("-" if neg else "") + text


_PERCENT_RE = re.compile(r"(percent|pct|share|growth|rate|ratio)", re.I)
_MONEY_RE = re.compile(
    r"(revenue|sales|spend|spent|amount|price|aov|average_order_value|income|worth|cost|order_total|line_total|(^|_)total$)",
    re.I,
)
_QUESTION_MONEY_RE = re.compile(r"(revenue|sales|spend|spent|income|earn|price|cost|worth|₹|rupee)", re.I)


def column_kind(name: str, question: str = "") -> str:
    """'money' | 'percent' | 'number' for a result column, from its alias (and the question)."""
    if _PERCENT_RE.search(name):
        return "percent"
    if _MONEY_RE.search(name):
        return "money"
    if name.lower() in ("value", "amount_value", "metric") and _QUESTION_MONEY_RE.search(question or ""):
        return "money"
    return "number"


def format_value(v: Any, kind: str) -> str:
    if v is None:
        return "-"
    if not _is_number(v):
        return str(v)
    if kind == "money":
        return format_inr(v)
    if kind == "percent":
        return f"{_r2(v)}%"
    return format_number(v)


def humanize(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).replace("_", " ")).strip()


def redact_emails(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Defence in depth: the BI views contain no emails, but mask any that slip through."""
    email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    out = []
    for row in rows:
        new = {}
        for k, v in row.items():
            if isinstance(v, str) and email.search(v):
                new[k] = email.sub(lambda m: m.group(0)[0] + "***@" + m.group(0).split("@")[1], v)
            else:
                new[k] = v
        out.append(new)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Result analysis: chart type, chart data
# ══════════════════════════════════════════════════════════════════════════════

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_MONTH_START_RE = re.compile(r"^\d{4}-\d{2}-01")
_TIME_NAME_RE = re.compile(r"(date|month|week|day|hour|year|time|period)", re.I)
_PIE_RE = re.compile(r"(share|percent|%|breakdown|distribution|proportion|split|mix|composition)", re.I)

_REQUESTED_CHART_PATTERNS = (
    ("table", re.compile(r"\b(?:as|in|into)\s+(?:a\s+)?(?:table|tabular)\b|\btabular\b|\btable\s+of\b|\b(?:show|give)\s+(?:me\s+)?(?:a\s+)?table\b", re.I)),
    ("pie", re.compile(r"\b(?:pie|donut|doughnut)\b", re.I)),
    ("line", re.compile(r"\bline\s*(?:chart|graph|plot)\b|\b(?:as|in)\s+(?:a\s+)?line\b", re.I)),
    ("bar", re.compile(r"\b(?:bar|column)\s*(?:chart|graph|plot)\b|\b(?:as|in)\s+(?:a\s+)?bar\b|\bbars\b", re.I)),
)


def requested_chart(question: str) -> Optional[str]:
    """The chart type the owner explicitly asked for ('pie', 'bar', 'line', 'table'), if any."""
    for name, pattern in _REQUESTED_CHART_PATTERNS:
        if pattern.search(question or ""):
            return name
    return None


def _numeric_cols(rows: List[Dict[str, Any]], cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if vals and all(_is_number(v) for v in vals):
            out.append(c)
    return out


def _all_dates(values: List[Any]) -> bool:
    vals = [v for v in values if v is not None]
    return bool(vals) and all(isinstance(v, str) and _DATE_RE.match(v) for v in vals)


def analyze_result(question: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide the shape of a result and the chart type. Pure code, no LLM."""
    if not rows:
        return {"shape": "empty", "chart_type": "none", "label_col": None, "value_col": None, "cols": []}

    cols = list(rows[0].keys())
    nums = _numeric_cols(rows, cols)
    labels = [c for c in cols if c not in nums]

    # e.g. (order_hour, orders): both numeric, first column is really a label
    if not labels and len(cols) == 2 and cols[0] in nums and _TIME_NAME_RE.search(cols[0]) and cols[1] in nums:
        labels, nums = [cols[0]], [cols[1]]

    base: Dict[str, Any] = {"cols": cols, "label_col": None, "value_col": None}
    n = len(rows)

    if len(cols) == 1:
        if n == 1 and nums:
            return {**base, "shape": "metric", "chart_type": "none"}
        return {**base, "shape": "table", "chart_type": "table"}

    if len(cols) == 2 and len(labels) == 1 and len(nums) == 1:
        lc, vc = labels[0], nums[0]
        base.update(label_col=lc, value_col=vc)
        if n == 1:
            return {**base, "shape": "row", "chart_type": "table"}
        values = [r[vc] for r in rows if r.get(vc) is not None]
        label_values = [r.get(lc) for r in rows]
        time_like = _all_dates(label_values)
        if time_like and n >= 3:
            chart = "line"
        elif time_like:
            chart = "bar"
        elif _PIE_RE.search(question or "") and 2 <= n <= 6 and values and all(v >= 0 for v in values) and sum(values) > 0:
            chart = "pie"
        elif n <= 30:
            chart = "bar"
        else:
            chart = "table"
        wanted = requested_chart(question)
        if wanted == "table":
            chart = "table"
        elif wanted == "bar" and n <= 60:
            chart = "bar"
        elif wanted == "line" and n >= 2:
            chart = "line"
        elif wanted == "pie" and 2 <= n <= 12 and values and all(v >= 0 for v in values) and sum(values) > 0:
            chart = "pie"
        return {**base, "shape": "pairs", "chart_type": chart, "time_like": time_like}

    if n == 1:
        return {**base, "shape": "row", "chart_type": "table"}
    return {**base, "shape": "table", "chart_type": "table"}


def _label_formatter(labels: List[Any], col: str):
    strs = [l for l in labels if isinstance(l, str)]
    if strs and all(_DATE_RE.match(s) for s in strs):
        if all(_MONTH_START_RE.match(s) for s in strs) and len({s[:7] for s in strs}) == len(strs):
            fn = lambda s: f"{MONTHS[int(s[5:7]) - 1]} {s[:4]}"
        elif len({s[:4] for s in strs}) > 1:
            fn = lambda s: f"{int(s[8:10]):02d} {MONTHS[int(s[5:7]) - 1]} {s[:4]}"
        else:
            fn = lambda s: f"{int(s[8:10]):02d} {MONTHS[int(s[5:7]) - 1]}"
        return lambda v: fn(v) if isinstance(v, str) and _DATE_RE.match(v) else "Unknown"
    nums = [l for l in labels if l is not None]
    if nums and all(_is_number(l) for l in nums) and re.search("hour", col or "", re.I):
        return lambda v: f"{int(v):02d}:00" if _is_number(v) else "Unknown"
    return lambda v: "Unknown" if v is None else str(v)


def build_chart_data(analysis: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chart points built straight from the result rows, so numbers never pass through the LLM."""
    chart = analysis.get("chart_type")
    if chart in ("bar", "pie", "line"):
        lc, vc = analysis["label_col"], analysis["value_col"]
        pairs = [(r.get(lc), r[vc]) for r in rows if _is_number(r.get(vc))]
        if chart == "line" and _all_dates([p[0] for p in pairs]):
            pairs.sort(key=lambda p: str(p[0]))
        fmt = _label_formatter([p[0] for p in pairs], lc)
        return [{"name": fmt(l), "value": _r2(v)} for l, v in pairs]
    if chart == "table":
        return [{k: _r2(v) for k, v in r.items()} for r in rows[:MAX_TABLE_ROWS]]
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Facts and narrative
# ══════════════════════════════════════════════════════════════════════════════

def _in_progress_note(raw_labels: List[Any], formatted_last: str, today: _dt.date) -> Optional[str]:
    strs = [l for l in raw_labels if isinstance(l, str) and _DATE_RE.match(l)]
    if not strs:
        return None
    last = strs[-1]
    try:
        d = _dt.date(int(last[:4]), int(last[5:7]), int(last[8:10]))
    except ValueError:
        return None
    if all(_MONTH_START_RE.match(s) for s in strs) and (d.year, d.month) == (today.year, today.month):
        return f"{formatted_last} is the current month and still in progress."
    if d == today:
        return f"{formatted_last} is today and still in progress."
    if d.weekday() == 0 and 0 <= (today - d).days <= 6 and len(strs) > 1:
        prev = strs[-2]
        try:
            pd = _dt.date(int(prev[:4]), int(prev[5:7]), int(prev[8:10]))
        except ValueError:
            return None
        if (d - pd).days == 7:
            return f"The week of {formatted_last} is the current week and still in progress."
    return None


def compute_facts(analysis: Dict[str, Any], rows: List[Dict[str, Any]], question: str = "",
                  today: Optional[_dt.date] = None) -> Dict[str, Any]:
    today = today or _dt.datetime.now(IST).date()
    facts: Dict[str, Any] = {"row_count": len(rows), "shape": analysis.get("shape"), "today": today.isoformat()}
    lc, vc = analysis.get("label_col"), analysis.get("value_col")
    if not (lc and vc and rows):
        return facts

    kind = column_kind(vc, question)
    pairs = [(r.get(lc), r[vc]) for r in rows if _is_number(r.get(vc))]
    if not pairs:
        return facts
    if analysis.get("chart_type") == "line" and _all_dates([p[0] for p in pairs]):
        pairs.sort(key=lambda p: str(p[0]))

    fmt = _label_formatter([p[0] for p in pairs], lc)
    values = [v for _, v in pairs]
    total = sum(values)
    top = max(pairs, key=lambda p: p[1])
    bottom = min(pairs, key=lambda p: p[1])
    facts.update({
        "value_kind": kind,
        "label_name": humanize(lc),
        "value_name": humanize(vc),
        "top": {"label": fmt(top[0]), "value": _r2(top[1])},
        "bottom": {"label": fmt(bottom[0]), "value": _r2(bottom[1])},
    })
    if kind != "percent":
        facts["total"] = _r2(total)
        if total > 0 and all(v >= 0 for v in values):
            facts["top"]["share_percent"] = round(top[1] * 100 / total, 1)
            facts["shares_percent"] = {fmt(l): round(v * 100 / total, 1) for l, v in pairs}
    if analysis.get("chart_type") == "line":
        facts["first"] = {"label": fmt(pairs[0][0]), "value": _r2(pairs[0][1])}
        facts["last"] = {"label": fmt(pairs[-1][0]), "value": _r2(pairs[-1][1])}
        note = _in_progress_note([p[0] for p in pairs], fmt(pairs[-1][0]), today)
        if note:
            facts["in_progress_note"] = note
    return facts


def deterministic_narrative(question: str, analysis: Dict[str, Any], rows: List[Dict[str, Any]],
                            facts: Dict[str, Any], truncated: bool = False) -> str:
    """A short, always-correct summary built only from the data."""
    shape = analysis.get("shape")
    n = len(rows)
    if shape == "empty":
        return "No data found for that question. There may be no confirmed orders matching it."

    cols = analysis["cols"]
    if shape == "metric":
        c = cols[0]
        return f"{humanize(c).capitalize()}: {format_value(rows[0][c], column_kind(c, question))}."

    if shape == "row" and not analysis.get("value_col"):
        parts = [f"{humanize(c)}: {format_value(rows[0][c], column_kind(c, question))}" for c in cols]
        return "; ".join(parts) + "."

    if shape in ("row", "pairs") and facts.get("top"):
        kind = facts["value_kind"]
        vname = facts["value_name"]
        top, bottom = facts["top"], facts["bottom"]
        if shape == "row":
            return f"{top['label']}: {format_value(top['value'], kind)} ({vname})."
        fmt = lambda v: format_value(v, kind)
        text: List[str] = []
        if analysis.get("chart_type") == "line":
            if kind != "percent":
                text.append(f"{vname.capitalize()} totals {fmt(facts['total'])} across {n} {'periods' if analysis.get('time_like') else 'items'}.")
            text.append(f"Highest: {top['label']} ({fmt(top['value'])}); lowest: {bottom['label']} ({fmt(bottom['value'])}).")
            if facts.get("in_progress_note"):
                text.append(facts["in_progress_note"])
        else:
            share = top.get("share_percent")
            line = f"Highest {vname}: {top['label']} ({fmt(top['value'])}"
            line += f", {share:g}% of the total)." if share is not None and n > 1 else ")."
            text.append(line)
            if n > 1:
                text.append(f"Lowest: {bottom['label']} ({fmt(bottom['value'])}).")
                if kind != "percent":
                    text.append(f"Total across the {n} shown: {fmt(facts['total'])}.")
        if truncated:
            text.append(f"Only the first {MAX_ROWS} rows are shown.")
        return " ".join(text)

    text = f"Showing {n} row{'s' if n != 1 else ''}."
    if truncated:
        text = f"Showing the first {MAX_ROWS} rows."
    return text


# ── Faithfulness of an LLM-written narrative ─────────────────────────────────

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> List[float]:
    out = []
    for token in _NUM_RE.findall(text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def allowed_numbers(rows: List[Dict[str, Any]], facts: Optional[Dict[str, Any]] = None,
                    extra_text: str = "") -> set:
    """Every number the narrative is allowed to mention (with common roundings)."""
    blob = json.dumps(rows, default=str) + " " + json.dumps(facts or {}, default=str) + " " + (extra_text or "")
    allowed: set = {float(len(rows))}
    for x in _numbers_in(blob):
        allowed.update({x, float(round(x)), round(x, 1), round(x, 2)})
    return allowed


def narrative_is_faithful(narrative: str, rows: List[Dict[str, Any]], facts: Optional[Dict[str, Any]] = None,
                          extra_text: str = "") -> bool:
    """True if every number in the narrative appears in the data/facts, and no '$' is used."""
    if not narrative or not narrative.strip():
        return False
    if "$" in narrative:
        return False
    allowed = allowed_numbers(rows, facts, extra_text)
    for x in _numbers_in(narrative):
        if not any(abs(x - a) <= 1e-6 for a in allowed):
            return False
    return True
