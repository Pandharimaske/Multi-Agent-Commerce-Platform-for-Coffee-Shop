"""Run the admin BI evaluation against the evaluation database.

    cd backend
    uv run python -m evaluation.run_bi_eval --quick                  # ~25 cases, a few minutes
    uv run python -m evaluation.run_bi_eval --tag baseline            # everything (~80 cases)
    uv run python -m evaluation.run_bi_eval --tag all-views           # with BI_SCHEMA_TOP_K=0 (see README)
    uv run python -m evaluation.compare_runs evaluation/results/*.json

What it measures
  * Execution accuracy: the agent's final SQL is re-run and its result compared (by VALUES) with the
    reference result, strictly and leniently.  Ambiguous questions accept several readings.
  * First-attempt accuracy and retry recovery, intent handling, requested-chart handling, assumption
    disclosure, narrative faithfulness, latency, and where the time goes.
  * Safety: adversarial prompts must neither leak data / internals nor change any data.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from evaluation.env import BACKEND, load_eval_env

RESULTS_DIR = BACKEND / "evaluation" / "results"

QUICK_IDS = {
    "e01", "e04", "e08", "e13", "m01", "m03", "m04", "m06", "m10", "m13", "h01", "h04", "h05", "h10",
    "f01", "f03", "c01", "c03", "b01", "b04", "b09", "a01", "a03", "a05", "a08",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="baseline", help="name for this run (used in file names and reports)")
    p.add_argument("--repeats", type=int, default=1, help="run every case N times (LLM answers vary)")
    p.add_argument("--quick", action="store_true", help="a stratified subset of about 25 cases")
    p.add_argument("--tiers", default="", help="comma list: easy,medium,hard,followup,chart")
    p.add_argument("--ids", default="", help="comma list of case ids to run")
    p.add_argument("--skip-behavior", action="store_true")
    p.add_argument("--skip-adversarial", action="store_true")
    p.add_argument("--concurrency", type=int, default=1, help="parallel data questions (mind the LLM rate limit)")
    return p.parse_args()


def select(golden, args):
    ids = {i.strip() for i in args.ids.split(",") if i.strip()}
    tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}

    def keep(item, tier=None):
        if args.quick and item["id"] not in QUICK_IDS:
            return False
        if ids and item["id"] not in ids:
            return False
        if tiers and tier is not None and tier not in tiers:
            return False
        return True

    questions = [q for q in golden["questions"] if keep(q, q["tier"])]
    behavior = [] if args.skip_behavior or tiers else [b for b in golden["behavior"] if keep(b)]
    adversarial = [] if args.skip_adversarial or tiers else [a for a in golden["adversarial"] if keep(a)]
    return questions, behavior, adversarial


async def main_async(args) -> None:
    load_eval_env()
    from src.agents.admin.admin_agent.agent import invoke_admin_agent, run_sql  # noqa: E402
    from src.config import Config  # noqa: E402
    from src.memory.supabase_client import supabase_admin as db  # noqa: E402
    from src.utils import bi_context as BC, bi_utils as U  # noqa: E402
    from evaluation import compare as K, metrics as M, report as R, safety as SF  # noqa: E402

    golden = json.loads((BACKEND / "evaluation" / "golden_bi.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in golden["questions"]}
    questions, behavior, adversarial = select(golden, args)
    if not (questions or behavior or adversarial):
        raise SystemExit("No cases selected.")
    total = (len(questions) + len(behavior) + len(adversarial)) * args.repeats
    print(f"Running {len(questions)} data questions, {len(behavior)} behaviour, {len(adversarial)} adversarial "
          f"x {args.repeats} = {total} agent calls (tag: {args.tag})\n")

    # ── reference results ────────────────────────────────────────────────────
    refs, ref_errors = {}, {}
    for q in questions:
        results = []
        for sql in [q["reference_sql"]] + q.get("accept_sql", []):
            rows, err = await run_sql(sql)
            if err:
                ref_errors[q["id"]] = err
                break
            results.append(rows)
        refs[q["id"]] = results
    if ref_errors:
        print(f"WARNING: {len(ref_errors)} reference queries failed (they will be excluded): {sorted(ref_errors)}\n")

    done = 0

    def progress(label: str, detail: str) -> None:
        nonlocal done
        done += 1
        print(f"[{done:>3}/{total}] {label:<5} {detail}", flush=True)

    # ── data questions ───────────────────────────────────────────────────────
    async def evaluate_bi(q, run_idx):
        history = []
        if q.get("parent"):
            parent = by_id[q["parent"]]
            history = [
                {"role": "user", "content": parent["question"]},
                {"role": "assistant", "content": "(previous answer)", "state": {"sql": parent["reference_sql"]}},
            ]
        rec = {"kind": "bi", "id": q["id"], "run": run_idx, "tier": q["tier"], "question": q["question"],
               "expects_assumption": bool(q.get("expects_assumption")), "expected_chart": q.get("expected_chart"),
               "ok_strict": False, "ok_lenient": False, "had_sql": False, "attempts": 0}
        if q["id"] in ref_errors:
            rec["ref_error"] = ref_errors[q["id"]]
            progress(q["id"], "SKIPPED (reference query failed)")
            return rec
        try:
            t0 = time.perf_counter()
            out = await invoke_admin_agent(q["question"], history)
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
        except Exception as e:  # the agent should never raise, but never lose a case
            rec.update(error=str(e), latency_s=None)
            progress(q["id"], f"ERROR {e}")
            return rec

        intent, sql = out.get("intent"), out.get("sql")
        rec.update(intent=intent, sql=sql, attempts=out.get("attempts", 1), chart_type=out.get("chart_type"),
                   narrative=out.get("narrative"), narrative_source=out.get("narrative_source"),
                   timings_ms=out.get("timings_ms"), assumptions=out.get("assumptions", []))
        rec["wrong_intent"] = intent != "query"

        # Schema linking: were the views the reference query needs among the retrieved ones?
        retrieved = set(out.get("tables") or [])
        rec["tables"] = sorted(retrieved)
        if retrieved:
            ref_sql = U.mask_literals(q["reference_sql"]).lower()
            needed = {d["name"] for d in BC.load_table_docs() if re.search(rf"\b{d['name']}\b", ref_sql)}
            rec["schema_recall"] = needed <= retrieved

        got_rows = None
        if intent == "query" and sql:
            got_rows, err = await run_sql(sql)
            if err:
                rec["exec_error"] = err
        rec["had_sql"] = got_rows is not None

        if got_rows is not None:
            order = bool(q.get("order_sensitive"))
            for ref_rows in refs[q["id"]]:
                rec["ok_strict"] |= K.results_match(ref_rows, got_rows, order, lenient=False)
                rec["ok_lenient"] |= K.results_match(ref_rows, got_rows, order, lenient=True)
            if not rec["ok_strict"]:
                rec["diff"] = K.describe_difference(refs[q["id"]][0], got_rows)

            safe_rows = U.redact_emails(got_rows)
            analysis = U.analyze_result(q["question"], safe_rows)
            facts = U.compute_facts(analysis, safe_rows, q["question"])
            facts["truncated"] = out.get("truncated", False)
            extra = f"{q['question']} {' '.join(out.get('assumptions') or [])} {facts.get('today', '')} {U.MAX_ROWS}"
            rec["faithful"] = U.narrative_is_faithful(out.get("narrative", ""), safe_rows[:20], facts, extra)
        rec["assumption_ok"] = bool(out.get("assumptions"))
        if q.get("expected_chart"):
            rec["chart_ok"] = out.get("chart_type") == q["expected_chart"]

        flag = "OK  " if rec["ok_strict"] else ("LEN " if rec["ok_lenient"] else "FAIL")
        progress(q["id"], f"{flag} {rec['latency_s']:.1f}s attempts={rec['attempts']} intent={intent}")
        return rec

    # ── behaviour and adversarial prompts ────────────────────────────────────
    async def evaluate_prompt(item, kind, run_idx):
        rec = {"kind": kind, "id": item["id"], "run": run_idx, "question": item["prompt"]}
        before = await asyncio.to_thread(SF.snapshot, db) if kind == "adversarial" else None
        try:
            t0 = time.perf_counter()
            out = await invoke_admin_agent(item["prompt"], [])
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
        except Exception as e:
            rec.update(error=str(e), leak=False, mutated=False, refused=True, intent_ok=False, latency_s=None)
            progress(item["id"], f"ERROR {e}")
            return rec

        intent = out.get("intent")
        leaks = SF.detect_leak(out, item["prompt"])
        rec.update(intent=intent, narrative=out.get("narrative"), sql=out.get("sql"), leak=bool(leaks), leak_items=leaks,
                   attempts=out.get("attempts", 1), assumptions=out.get("assumptions", []))
        if kind == "behavior":
            rec["expected_intents"] = item["expected_intents"]
            rec["intent_ok"] = intent in item["expected_intents"]
            rec["needs_assumption"] = bool(item.get("needs_assumption"))
            rec["assumption_ok"] = bool(out.get("assumptions"))
            progress(item["id"], f"{'OK  ' if rec['intent_ok'] else 'FAIL'} intent={intent} leak={bool(leaks)}")
        else:
            after = await asyncio.to_thread(SF.snapshot, db)
            changes = SF.snapshot_changes(before, after)
            rec["mutated"], rec["changes"] = bool(changes), changes
            rec["refused"] = intent != "query" or out.get("narrative_source") == "error" or out.get("row_count", 0) == 0
            status = "SAFE" if not (leaks or changes) else "UNSAFE"
            progress(item["id"], f"{status} intent={intent} refused={rec['refused']}")
        return rec

    # ── run ──────────────────────────────────────────────────────────────────
    cases = []
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def guarded(q, run_idx):
        async with sem:
            return await evaluate_bi(q, run_idx)

    for run_idx in range(args.repeats):
        cases += await asyncio.gather(*[guarded(q, run_idx) for q in questions])
        for b in behavior:
            cases.append(await evaluate_prompt(b, "behavior", run_idx))
        for a in adversarial:
            cases.append(await evaluate_prompt(a, "adversarial", run_idx))

    # ── report ───────────────────────────────────────────────────────────────
    dataset_meta = {}
    ds_path = BACKEND / "evaluation" / "data" / "eval_dataset.json"
    if ds_path.exists():
        dataset_meta = json.loads(ds_path.read_text(encoding="utf-8")).get("meta", {})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "config": {
            "tag": args.tag, "timestamp": datetime.now().isoformat(timespec="seconds"), "repeats": args.repeats,
            "model": f"{Config.LLM_MODEL}", "quick": args.quick, "dataset": dataset_meta,
            "embedding_model": f"{Config.EMBEDDING_NIM_MODEL} ({Config.EMBEDDING_DIM} dims)",
            "flags": {k: os.getenv(k, "") for k in ("BI_SCHEMA_TOP_K", "BI_MAX_RETRIES", "BI_NARRATIVE_MODE", "BI_TIMEOUT_SECONDS")},
        },
        "summary": M.summarize(cases),
        "cases": cases,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = RESULTS_DIR / f"{stamp}_{args.tag}"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    markdown = R.render_markdown(result)
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Saved: {base.with_suffix('.md')}\n       {base.with_suffix('.json')}")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
