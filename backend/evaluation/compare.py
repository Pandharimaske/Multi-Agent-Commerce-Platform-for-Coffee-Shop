"""Result-set comparison for execution accuracy (pure Python, no database needed).

Two result sets are compared by VALUES, never by SQL text:
  * column names and column order are ignored
  * floats are rounded (2 decimals strict, 1 decimal lenient)
  * text is compared case-insensitively
  * rows are compared as a multiset, unless the question asks for a specific order

Strict  : same rows, same number of columns.
Lenient : additionally allows one side to have EXTRA columns (e.g. the agent also returns the
          product price, or only the product name where the reference has name + revenue),
          and rounds to 1 decimal.
"""
from __future__ import annotations

import datetime as dt
import itertools
import re
from collections import Counter
from decimal import Decimal
from typing import Any, Dict, List, Sequence, Tuple

_NUMERIC_TEXT = re.compile(r"^-?\d+(\.\d+)?$")
# A timestamp at exactly midnight (e.g. date_trunc('month', ...) without ::date) is the same value as its date.
_MIDNIGHT_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ]00:00:00(\.0+)?(Z|[+-]00:?00)?$", re.IGNORECASE)
_NULL = "<null>"


def normalize_value(v: Any, decimals: int = 2) -> str:
    if v is None:
        return _NULL
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, str):
        s = v.strip()
        midnight = _MIDNIGHT_TS.match(s)
        if midnight:
            return midnight.group(1)
        if _NUMERIC_TEXT.match(s):
            v = float(s)
        else:
            return s.lower()
    if isinstance(v, dt.datetime) and v.time() == dt.time(0, 0):
        return v.date().isoformat()
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (int, float, Decimal)):
        f = round(float(v), decimals)
        if f == 0:
            f = 0.0  # avoid "-0"
        if f.is_integer():
            return str(int(f))
        return f"{f:.{decimals}f}".rstrip("0").rstrip(".")
    return str(v).strip().lower()


def _row_tuple(row: Dict[str, Any], cols: Sequence[str], decimals: int) -> Tuple[str, ...]:
    return tuple(sorted(normalize_value(row.get(c), decimals) for c in cols))


def _columns(rows: List[Dict[str, Any]]) -> List[str]:
    return list(rows[0].keys()) if rows else []


def normalize_rows(rows: List[Dict[str, Any]], decimals: int = 2, cols: Sequence[str] = None) -> List[Tuple[str, ...]]:
    cols = list(cols) if cols is not None else _columns(rows)
    return [_row_tuple(r, cols, decimals) for r in rows]


def _same(a: List[Tuple[str, ...]], b: List[Tuple[str, ...]], order_sensitive: bool) -> bool:
    if len(a) != len(b):
        return False
    return a == b if order_sensitive else Counter(a) == Counter(b)


def results_match(ref_rows: List[Dict[str, Any]], got_rows: List[Dict[str, Any]],
                  order_sensitive: bool = False, lenient: bool = False) -> bool:
    """True if the two result sets hold the same values (see module docstring)."""
    if not ref_rows and not got_rows:
        return True
    if not ref_rows or not got_rows or len(ref_rows) != len(got_rows):
        return False

    decimals = 1 if lenient else 2
    ref_cols, got_cols = _columns(ref_rows), _columns(got_rows)
    ref_norm = normalize_rows(ref_rows, decimals)

    if len(ref_cols) == len(got_cols):
        return _same(ref_norm, normalize_rows(got_rows, decimals), order_sensitive)
    if not lenient:
        return False

    # Lenient: one side has extra columns. Find a column subset of the wider side that matches.
    wide_rows, wide_cols, narrow_rows, narrow_cols = (
        (got_rows, got_cols, ref_rows, ref_cols) if len(got_cols) > len(ref_cols)
        else (ref_rows, ref_cols, got_rows, got_cols)
    )
    if len(wide_cols) > 10:
        return False
    narrow_norm = normalize_rows(narrow_rows, decimals)
    for subset in itertools.combinations(wide_cols, len(narrow_cols)):
        if _same(normalize_rows(wide_rows, decimals, subset), narrow_norm, order_sensitive):
            return True
    return False


def describe_difference(ref_rows: List[Dict[str, Any]], got_rows: List[Dict[str, Any]], limit: int = 3) -> str:
    """A short human-readable reason for a mismatch, used in the failure report."""
    if not got_rows and ref_rows:
        return f"agent returned 0 rows, expected {len(ref_rows)}"
    if got_rows and not ref_rows:
        return f"agent returned {len(got_rows)} rows, expected 0"
    if len(ref_rows) != len(got_rows):
        return f"row count {len(got_rows)} vs expected {len(ref_rows)}"
    ref_cols, got_cols = _columns(ref_rows), _columns(got_rows)
    if len(ref_cols) != len(got_cols):
        return f"column count {len(got_cols)} ({got_cols}) vs expected {len(ref_cols)} ({ref_cols})"
    missing = Counter(normalize_rows(ref_rows)) - Counter(normalize_rows(got_rows))
    extra = Counter(normalize_rows(got_rows)) - Counter(normalize_rows(ref_rows))
    return (f"values differ. missing e.g. {list(missing)[:limit]}; unexpected e.g. {list(extra)[:limit]}")
