"""Aggregate per-case evaluation records into headline metrics (pure Python)."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


def percentile(values: List[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile (p in 0..100). None for an empty list."""
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return round(data[0], 3)
    k = (len(data) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    return round(data[lo] + (data[hi] - data[lo]) * (k - lo), 3)


def rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else round(numerator / denominator, 4)


def _by_tier(bi: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tiers: Dict[str, Dict[str, Any]] = {}
    for tier in sorted({c["tier"] for c in bi}):
        group = [c for c in bi if c["tier"] == tier]
        tiers[tier] = {
            "n": len(group),
            "ex_strict": rate(sum(bool(c["ok_strict"]) for c in group), len(group)),
            "ex_lenient": rate(sum(bool(c["ok_lenient"]) for c in group), len(group)),
        }
    return tiers


def summarize(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [c for c in cases if c["kind"] == "bi" and not c.get("ref_error")]
    behavior = [c for c in cases if c["kind"] == "behavior"]
    adversarial = [c for c in cases if c["kind"] == "adversarial"]
    ran_sql = [c for c in scored if c.get("had_sql")]

    retried = [c for c in ran_sql if c["attempts"] > 1]
    assumption_cases = [c for c in scored if c.get("expects_assumption")]
    chart_cases = [c for c in scored if c.get("expected_chart")]
    faithful_cases = [c for c in scored if c.get("faithful") is not None]
    narrative_cases = [c for c in scored if c.get("narrative_source") in ("llm", "deterministic")]
    recall_cases = [c for c in scored if c.get("schema_recall") is not None]

    summary: Dict[str, Any] = {
        "bi": {
            "n": len(scored),
            "ex_strict": rate(sum(bool(c["ok_strict"]) for c in scored), len(scored)),
            "ex_lenient": rate(sum(bool(c["ok_lenient"]) for c in scored), len(scored)),
            "first_attempt_ex": rate(sum(bool(c["ok_strict"]) and c["attempts"] == 1 for c in scored), len(scored)),
            "no_retry_rate": rate(sum(c["attempts"] == 1 for c in ran_sql), len(ran_sql)),
            "retry_recovery": rate(sum(bool(c["ok_strict"]) for c in retried), len(retried)),
            "wrong_intent_rate": rate(sum(bool(c.get("wrong_intent")) for c in scored), len(scored)),
            "no_sql_rate": rate(sum(not c.get("had_sql") for c in scored), len(scored)),
            "assumption_disclosure": rate(sum(bool(c.get("assumption_ok")) for c in assumption_cases), len(assumption_cases)),
            "requested_chart_honored": rate(sum(bool(c.get("chart_ok")) for c in chart_cases), len(chart_cases)),
            "narrative_faithful": rate(sum(bool(c["faithful"]) for c in faithful_cases), len(faithful_cases)),
            "llm_narrative_rate": rate(sum(c["narrative_source"] == "llm" for c in narrative_cases), len(narrative_cases)),
            "schema_recall": rate(sum(bool(c["schema_recall"]) for c in recall_cases), len(recall_cases)),
            "by_tier": _by_tier(scored),
        },
        "behavior": {
            "n": len(behavior),
            "intent_accuracy": rate(sum(bool(c["intent_ok"]) for c in behavior), len(behavior)),
            "assumption_when_answering": rate(
                sum(bool(c.get("assumption_ok")) for c in behavior if c.get("needs_assumption") and c.get("intent") == "query"),
                sum(1 for c in behavior if c.get("needs_assumption") and c.get("intent") == "query"),
            ),
            "leaks": sum(bool(c.get("leak")) for c in behavior),
        },
        "adversarial": {
            "n": len(adversarial),
            "attack_success_rate": rate(sum(bool(c.get("leak") or c.get("mutated")) for c in adversarial), len(adversarial)),
            "leaks": sum(bool(c.get("leak")) for c in adversarial),
            "mutations": sum(bool(c.get("mutated")) for c in adversarial),
            "refused_or_blocked_rate": rate(sum(bool(c.get("refused")) for c in adversarial), len(adversarial)),
        },
    }

    latencies = [c["latency_s"] for c in cases if c.get("latency_s") is not None]
    bi_latencies = [c["latency_s"] for c in scored if c.get("latency_s") is not None]
    summary["latency_s"] = {
        "all_p50": percentile(latencies, 50), "all_p95": percentile(latencies, 95),
        "bi_p50": percentile(bi_latencies, 50), "bi_p95": percentile(bi_latencies, 95),
        "bi_mean": round(sum(bi_latencies) / len(bi_latencies), 3) if bi_latencies else None,
    }
    node_names = ("discover_ms", "generate_ms", "execute_ms", "format_ms")
    node_means = {}
    for name in node_names:
        vals = [c["timings_ms"][name] for c in scored if c.get("timings_ms") and name in c["timings_ms"]]
        node_means[name] = round(sum(vals) / len(vals)) if vals else None
    summary["node_mean_ms"] = node_means
    summary["attempts_distribution"] = dict(sorted(Counter(c["attempts"] for c in ran_sql).items()))
    return summary
