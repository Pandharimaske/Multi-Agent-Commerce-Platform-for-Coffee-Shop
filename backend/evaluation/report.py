"""Render evaluation results as Markdown (pure Python)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _secs(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1f}s"


def _cell(text: Any, n: int = 140) -> str:
    s = " ".join(str(text or "").split()).replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def _flags(config: Dict[str, Any]) -> str:
    f = config.get("flags", {})
    return (f"schema retrieval top-{f.get('BI_SCHEMA_TOP_K') or '3'} views (0 = every view), "
            f"max retries {f.get('BI_MAX_RETRIES') or '2'}, narrative {f.get('BI_NARRATIVE_MODE') or 'llm'}, "
            f"timeout {f.get('BI_TIMEOUT_SECONDS') or '100'}s")


def render_markdown(result: Dict[str, Any]) -> str:
    cfg, s, cases = result["config"], result["summary"], result["cases"]
    bi, beh, adv, lat = s["bi"], s["behavior"], s["adversarial"], s["latency_s"]
    n_adv = adv["n"]
    lines = [
        f"# Admin BI evaluation: {cfg.get('tag', 'run')}",
        "",
        f"- Run: {cfg.get('timestamp')} | model: `{cfg.get('model')}` | repeats: {cfg.get('repeats', 1)}",
        f"- Cases: {bi['n']} data questions, {beh['n']} behaviour prompts, {n_adv} adversarial prompts",
        f"- Embeddings: `{cfg.get('embedding_model', 'n/a')}`",
        f"- Settings: {_flags(cfg)}",
        f"- Data: synthetic (seed {cfg.get('dataset', {}).get('seed_used', '?')}), generated {cfg.get('dataset', {}).get('now_ist', '?')}",
        "",
        "## Headline",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Execution accuracy, strict | **{pct(bi['ex_strict'])}** |",
        f"| Execution accuracy, lenient (extra/fewer columns, 1 dp) | {pct(bi['ex_lenient'])} |",
        f"| Correct on the first attempt (no retry) | {pct(bi['first_attempt_ex'])} |",
        f"| SQL ran without error on the first attempt | {pct(bi['no_retry_rate'])} |",
        f"| Of the retried queries, ended correct | {pct(bi['retry_recovery'])} |",
        f"| Wrongly declined a data question | {pct(bi['wrong_intent_rate'])} |",
        f"| Behaviour: chose the right intent | {pct(beh['intent_accuracy'])} |",
        f"| Safety: attack success rate | **{pct(adv['attack_success_rate'])}** ({adv['leaks']} leaks, {adv['mutations']} data changes in {n_adv} prompts) |",
        f"| Safety: refused or blocked | {pct(adv['refused_or_blocked_rate'])} |",
        f"| Narrative uses only numbers from the data | {pct(bi['narrative_faithful'])} |",
        f"| LLM-written narrative accepted (else code summary) | {pct(bi['llm_narrative_rate'])} |",
        f"| Assumptions stated when the question was ambiguous | {pct(bi['assumption_disclosure'])} |",
        f"| Explicitly requested chart type honoured | {pct(bi['requested_chart_honored'])} |",
        f"| Schema retrieval recall (the views the question needs were retrieved) | {pct(bi.get('schema_recall'))} |",
        f"| Latency per data question, p50 / p95 | {_secs(lat['bi_p50'])} / {_secs(lat['bi_p95'])} |",
        "",
        "## By tier (execution accuracy)",
        "",
        "| Tier | Cases | Strict | Lenient |",
        "|---|---|---|---|",
    ]
    for tier, t in bi["by_tier"].items():
        lines.append(f"| {tier} | {t['n']} | {pct(t['ex_strict'])} | {pct(t['ex_lenient'])} |")

    nm = s["node_mean_ms"]
    lines += [
        "",
        "## Where the time goes (mean per data question)",
        "",
        f"discovery {nm.get('discover_ms')} ms | planner {nm.get('generate_ms')} ms | "
        f"execution {nm.get('execute_ms')} ms | formatting {nm.get('format_ms')} ms | "
        f"attempts used: {s['attempts_distribution']}",
    ]

    bi_fail = [c for c in cases if c["kind"] == "bi" and not c.get("ref_error") and not c.get("ok_strict")]
    ref_err = [c for c in cases if c.get("ref_error")]
    if ref_err:
        lines += ["", "## Reference query errors (excluded from scores, fix the golden set)", ""]
        lines += [f"- {c['id']}: {_cell(c['ref_error'])}" for c in ref_err]
    if bi_fail:
        lines += ["", "## Data questions answered incorrectly", "", "| id | question | why | agent SQL |", "|---|---|---|---|"]
        for c in bi_fail[:30]:
            why = ("declined: " + str(c.get("intent"))) if c.get("wrong_intent") else (c.get("diff") or c.get("exec_error") or "no result")
            lines.append(f"| {c['id']} | {_cell(c['question'], 70)} | {_cell(why, 90)} | {_cell(c.get('sql'), 120)} |")
        if len(bi_fail) > 30:
            lines.append(f"\n…and {len(bi_fail) - 30} more (see the JSON file).")

    beh_fail = [c for c in cases if c["kind"] == "behavior" and not c.get("intent_ok")]
    if beh_fail:
        lines += ["", "## Behaviour prompts with an unexpected intent", "", "| id | prompt | got | expected |", "|---|---|---|---|"]
        for c in beh_fail:
            lines.append(f"| {c['id']} | {_cell(c['question'], 70)} | {c.get('intent')} | {', '.join(c.get('expected_intents', []))} |")

    bad = [c for c in cases if c["kind"] in ("adversarial", "behavior") and (c.get("leak") or c.get("mutated"))]
    if bad:
        lines += ["", "## SAFETY FAILURES", "", "| id | prompt | what happened |", "|---|---|---|"]
        for c in bad:
            what = ("data changed; " if c.get("mutated") else "") + ("leaked: " + ", ".join(c.get("leak_items", [])) if c.get("leak") else "")
            lines.append(f"| {c['id']} | {_cell(c['question'], 80)} | {_cell(what, 100)} |")
    else:
        lines += ["", "## Safety", "", "No leaks and no data changes in any adversarial or behaviour prompt."]
    return "\n".join(lines) + "\n"


_COMPARE_ROWS = (
    ("Execution accuracy (strict)", lambda s: pct(s["bi"]["ex_strict"])),
    ("Execution accuracy (lenient)", lambda s: pct(s["bi"]["ex_lenient"])),
    ("Correct on first attempt", lambda s: pct(s["bi"]["first_attempt_ex"])),
    ("SQL error-free on first attempt", lambda s: pct(s["bi"]["no_retry_rate"])),
    ("Behaviour intent accuracy", lambda s: pct(s["behavior"]["intent_accuracy"])),
    ("Attack success rate", lambda s: pct(s["adversarial"]["attack_success_rate"])),
    ("Narrative faithful", lambda s: pct(s["bi"]["narrative_faithful"])),
    ("LLM narrative accepted", lambda s: pct(s["bi"]["llm_narrative_rate"])),
    ("Latency p50 (data questions)", lambda s: _secs(s["latency_s"]["bi_p50"])),
    ("Latency p95 (data questions)", lambda s: _secs(s["latency_s"]["bi_p95"])),
)


def render_comparison(results: List[Dict[str, Any]]) -> str:
    """Side-by-side table of several runs (ablations: examples on/off, retries 0 vs 2, ...)."""
    tags = [r["config"].get("tag", "run") for r in results]
    lines = ["| Metric | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    lines.append("| Settings | " + " | ".join(_cell(_flags(r["config"]), 60) for r in results) + " |")
    for label, fn in _COMPARE_ROWS:
        lines.append(f"| {label} | " + " | ".join(fn(r["summary"]) for r in results) + " |")
    tiers = sorted({t for r in results for t in r["summary"]["bi"]["by_tier"]})
    for tier in tiers:
        cells = []
        for r in results:
            t = r["summary"]["bi"]["by_tier"].get(tier)
            cells.append(pct(t["ex_strict"]) if t else "n/a")
        lines.append(f"| EX strict, {tier} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
