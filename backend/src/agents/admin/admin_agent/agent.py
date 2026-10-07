"""Admin BI agent (Text-to-SQL) - v2.

Pipeline (LangGraph):
    discover -> generate -> execute -> format
                   ^           |
                   +-- retry --+   (SQL error: up to 2 retries, error fed back to the planner)

What the LLM does:   decide the intent, write the SQL, write a short narrative.
What code does:      validate the SQL, run it (read-only), choose the chart, build the chart data
                     straight from the rows, and check that the narrative only uses real numbers.

See src/utils/bi_utils.py (logic) and src/utils/bi_context.py (prompts).
"""
import asyncio
import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.memory.supabase_client import supabase_admin as supabase
from src.utils import bi_context as C
from src.utils import bi_utils as U
from src.utils.util import get_embedding_model, get_llm_error_message, llm

logger = logging.getLogger(__name__)

PROFILE_TTL_SECONDS = 300
OVERALL_TIMEOUT_SECONDS = int(os.getenv("BI_TIMEOUT_SECONDS", "100"))


# ── Models ────────────────────────────────────────────────────────────────────

class AdminState(BaseModel):
    """What the dashboard receives. The first five fields are the original contract."""
    query: str = Field(..., description="The original user query")
    narrative: str = Field(..., description="The textual analysis and response to the user")
    chart_type: str = Field("none", description="bar | pie | line | table | none")
    chart_data: List[Dict[str, Any]] = Field(default_factory=list)
    sql: Optional[str] = Field(None, description="The SQL query that produced this answer")
    intent: str = "query"
    assumptions: List[str] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class BIPlan(BaseModel):
    """Structured output of the planning step."""
    intent: Literal["query", "clarify", "out_of_scope", "write_request"] = Field(
        description="query = answer with SQL; clarify = ask one short question; "
                    "out_of_scope = not about this shop's data; write_request = asks to change data")
    sql: Optional[str] = Field(None, description="One read-only PostgreSQL query. Only when intent is 'query'.")
    assumptions: List[str] = Field(default_factory=list, description="Short phrases listing assumptions made")
    message: Optional[str] = Field(None, description="Reply for clarify / out_of_scope / write_request")


class AdminGraphState(BaseModel):
    query: str
    history: List[Dict] = Field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    prev_question: Optional[str] = None
    prev_sql: Optional[str] = None
    intent: str = "query"
    sql: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    truncated: bool = False
    error: Optional[str] = None
    failed_sql: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    timings: Dict[str, int] = Field(default_factory=dict)
    final_output: Optional[Dict[str, Any]] = None


# ── Small helpers ─────────────────────────────────────────────────────────────

def _now_ist_text() -> str:
    return datetime.datetime.now(U.IST).strftime("%Y-%m-%d %H:%M:%S") + " IST"


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _timed(timings: Dict[str, int], key: str, t0: float) -> Dict[str, int]:
    updated = dict(timings)
    updated[key] = updated.get(key, 0) + _elapsed_ms(t0)
    return updated


def _previous_turn(history: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """The last question and SQL, used to resolve follow-ups like 'compare that to last month'."""
    prev_q: Optional[str] = None
    prev_sql: Optional[str] = None
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        if msg.get("role") == "assistant":
            sql = (msg.get("state") or {}).get("sql")
            if sql:
                prev_sql = sql
                for j in range(i - 1, -1, -1):
                    if history[j].get("role") == "user":
                        prev_q = history[j].get("content")
                        break
                break
    if prev_q is None:
        for msg in reversed(history):
            if msg.get("role") == "user":
                prev_q = msg.get("content")
                break
    return prev_q, prev_sql


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")


def _with_assumptions(narrative: str, assumptions: List[str]) -> str:
    items = [a.strip().rstrip(".")[:160] for a in assumptions if a and a.strip()][:3]
    if not items:
        return narrative
    return f"{narrative} (Assumed: {'; '.join(items)}.)"


# ── Database access ───────────────────────────────────────────────────────────

async def run_sql(sql: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Run SQL through the read-only RPC. Returns (rows, None) or (None, error message)."""
    try:
        res = await asyncio.to_thread(
            lambda: supabase.rpc("execute_sql_query", {"sql_query": sql}).execute()
        )
    except Exception as e:
        logger.error(f"BI RPC failed: {e}")
        return None, str(e)

    data = res.data if res.data is not None else []
    if isinstance(data, list) and data and isinstance(data[0], dict) and set(data[0].keys()) == {"error"}:
        return None, str(data[0]["error"])
    if not isinstance(data, list):
        return None, "The database returned an unexpected response."
    return data, None


_PROFILE_CACHE: Dict[str, Any] = {"at": 0.0, "value": None}


async def get_data_profile() -> Optional[Dict[str, Any]]:
    """Menu categories, product names and order stats, cached for a few minutes. Fails open."""
    now = time.time()
    if _PROFILE_CACHE["value"] and now - _PROFILE_CACHE["at"] < PROFILE_TTL_SECONDS:
        return _PROFILE_CACHE["value"]

    (cats, e1), (prods, e2), (stats, e3) = await asyncio.gather(
        run_sql("SELECT category FROM products GROUP BY category ORDER BY category"),
        run_sql("SELECT product_name FROM products ORDER BY product_name"),
        run_sql(
            "SELECT COUNT(*) FILTER (WHERE status = 'confirmed') AS confirmed_orders, "
            "COUNT(*) FILTER (WHERE status = 'pending') AS pending_carts, "
            "MIN(order_date) AS first_order_date, MAX(order_date) AS last_order_date FROM orders"
        ),
    )
    if e1 and e2 and e3:
        logger.warning(f"BI data profile unavailable: {e1}")
        return None

    profile = {
        "categories": [r["category"] for r in (cats or []) if r.get("category")],
        "products": [r["product_name"] for r in (prods or []) if r.get("product_name")],
        "stats": (stats or [{}])[0],
    }
    _PROFILE_CACHE.update(at=now, value=profile)
    return profile


async def retrieve_examples(query: str, prev_question: Optional[str], k: int = 3) -> List[Dict[str, Any]]:
    """Similar verified question->SQL examples (pgvector). Fails open: no examples is fine."""
    text = query if (len(query.split()) >= 6 or not prev_question) else f"{prev_question} {query}"
    try:
        vector = await asyncio.to_thread(get_embedding_model().embed_query, text)
        res = await asyncio.to_thread(
            lambda: supabase.rpc(
                "match_schema_metadata",
                {"query_embedding": vector, "match_threshold": 0.3, "match_count": k},
            ).execute()
        )
    except Exception as e:
        logger.warning(f"BI example retrieval failed (continuing without examples): {e}")
        return []

    examples = []
    for row in res.data or []:
        meta = row.get("metadata") or {}
        if meta.get("type") == "example" and meta.get("sql"):
            examples.append({"question": meta.get("question", ""), "sql": meta["sql"]})
    return examples[:k]


# ── Narrative ─────────────────────────────────────────────────────────────────

async def write_narrative(question: str, analysis: Dict[str, Any], rows: List[Dict[str, Any]],
                          facts: Dict[str, Any], assumptions: List[str], truncated: bool) -> str:
    """LLM wording, accepted only if every number in it exists in the data; otherwise a code-built summary."""
    fallback = U.deterministic_narrative(question, analysis, rows, facts, truncated)
    mode = os.getenv("BI_NARRATIVE_MODE", "llm").lower()
    if mode != "llm" or analysis.get("shape") in ("empty", "metric", "table"):
        return fallback
    try:
        system, human = C.build_narrative_messages(question, analysis, rows, facts, assumptions, truncated)
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = _text_of(response.content).strip().strip('"').replace("**", "")
    except Exception as e:
        logger.warning(f"BI narrative LLM failed, using deterministic summary: {e}")
        return fallback

    extra = f"{question} {' '.join(assumptions)} {facts.get('today', '')}"
    if len(text) <= 700 and U.narrative_is_faithful(text, rows[:20], facts, extra):
        return text
    logger.info("BI narrative rejected (unsupported numbers or format); using deterministic summary")
    return fallback


# ── Graph nodes ───────────────────────────────────────────────────────────────

async def discovery_node(state: AdminGraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    prev_q, prev_sql = _previous_turn(state.history)
    profile, examples = await asyncio.gather(
        get_data_profile(), retrieve_examples(state.query, prev_q)
    )
    logger.info(f"Node: Discovery | examples={len(examples)} profile={'yes' if profile else 'no'}")
    return {
        "profile": profile,
        "examples": examples,
        "prev_question": prev_q,
        "prev_sql": prev_sql,
        "timings": _timed(state.timings, "discover_ms", t0),
    }


async def generation_node(state: AdminGraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    logger.info(f"Node: Generation | retry_count={state.retry_count}")
    prompt = C.build_planner_prompt(
        now_ist=_now_ist_text(),
        profile=state.profile,
        examples=state.examples,
        prev_question=state.prev_question,
        prev_sql=state.prev_sql,
        error=state.error,
        failed_sql=state.failed_sql,
    )
    try:
        planner = llm.with_structured_output(BIPlan)
        plan = await planner.ainvoke([SystemMessage(content=prompt), HumanMessage(content=state.query)])
    except Exception as e:
        logger.error(f"BI planner failed: {e}")
        return {
            "intent": "query", "sql": None, "error": f"The planner failed: {e}",
            "retry_count": state.retry_count + 1,
            "timings": _timed(state.timings, "generate_ms", t0),
        }

    if plan is None:
        return {
            "intent": "query", "sql": None, "error": "The planner returned no result.",
            "retry_count": state.retry_count + 1,
            "timings": _timed(state.timings, "generate_ms", t0),
        }

    update: Dict[str, Any] = {
        "intent": plan.intent,
        "assumptions": plan.assumptions or [],
        "message": plan.message,
        "timings": _timed(state.timings, "generate_ms", t0),
    }
    if plan.intent == "query":
        sql = U.scrub_sql(plan.sql or "")
        if not sql:
            update.update(sql=None, error="The planner produced no SQL.", retry_count=state.retry_count + 1)
        else:
            update.update(sql=sql, error=None)
    else:
        update.update(sql=None, error=None)
    return update


async def execution_node(state: AdminGraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    logger.info(f"Node: Execution | SQL: {state.sql}")

    problem = U.validate_sql(state.sql or "")
    if problem:
        logger.warning(f"BI SQL rejected before execution: {problem}")
        return {"error": problem, "failed_sql": state.sql, "retry_count": state.retry_count + 1,
                "timings": _timed(state.timings, "execute_ms", t0)}

    rows, err = await run_sql(state.sql)
    if err:
        logger.warning(f"BI SQL error: {err}")
        return {"error": err, "failed_sql": state.sql, "retry_count": state.retry_count + 1,
                "timings": _timed(state.timings, "execute_ms", t0)}

    truncated = len(rows) > U.MAX_ROWS
    return {"results": rows[: U.MAX_ROWS], "truncated": truncated, "error": None,
            "timings": _timed(state.timings, "execute_ms", t0)}


async def formatting_node(state: AdminGraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    q = state.query
    row_count = 0
    truncated = False

    if state.intent != "query":
        narrative = (state.message or "").strip() or C.DEFAULT_MESSAGES.get(state.intent, "I can't help with that.")
        out = AdminState(query=q, narrative=narrative, intent=state.intent)

    elif state.error or state.results is None:
        detail = (state.error or "unknown error")[:160]
        narrative = (
            "I couldn't build a working query for that question. "
            "Try rephrasing it or narrowing the time period. "
            f"(Details: {detail})"
        )
        out = AdminState(query=q, narrative=narrative, sql=state.failed_sql or state.sql, intent="query")

    else:
        rows = U.redact_emails(state.results)
        row_count, truncated = len(rows), state.truncated
        analysis = U.analyze_result(q, rows)
        facts = U.compute_facts(analysis, rows, q)
        facts["truncated"] = truncated
        narrative = await write_narrative(q, analysis, rows, facts, state.assumptions, truncated)
        out = AdminState(
            query=q,
            narrative=_with_assumptions(narrative, state.assumptions),
            chart_type=analysis["chart_type"],
            chart_data=U.build_chart_data(analysis, rows),
            sql=state.sql,
            intent="query",
            assumptions=state.assumptions,
            row_count=row_count,
            truncated=truncated,
        )

    timings = _timed(state.timings, "format_ms", t0)
    out.timings_ms = timings
    logger.info("bi_turn " + json.dumps({
        "query": q, "intent": out.intent, "chart": out.chart_type, "rows": out.row_count,
        "retries": state.retry_count, "error": bool(state.error), "timings_ms": timings,
    }))
    return {"final_output": out.model_dump()}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_generate(state: AdminGraphState) -> str:
    if state.intent != "query":
        return "format"
    if state.sql:
        return "execute"
    return "generate" if state.retry_count <= state.max_retries else "format"


def route_after_execute(state: AdminGraphState) -> str:
    if state.error and state.retry_count <= state.max_retries:
        logger.info(f"Routing: retry {state.retry_count} of {state.max_retries}")
        return "generate"
    return "format"


# ── Graph ─────────────────────────────────────────────────────────────────────

def create_admin_graph():
    workflow = StateGraph(AdminGraphState)
    workflow.add_node("discover", discovery_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("format", formatting_node)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "generate")
    workflow.add_conditional_edges("generate", route_after_generate,
                                   {"execute": "execute", "generate": "generate", "format": "format"})
    workflow.add_conditional_edges("execute", route_after_execute,
                                   {"generate": "generate", "format": "format"})
    workflow.add_edge("format", END)
    return workflow.compile()


admin_graph = create_admin_graph()


async def invoke_admin_agent(query: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Entry point used by the /admin/chat route. Always returns an AdminState-shaped dict."""
    initial: Dict[str, Any] = {"query": query, "history": history or []}
    try:
        final_state = await asyncio.wait_for(admin_graph.ainvoke(initial), timeout=OVERALL_TIMEOUT_SECONDS)
        return final_state["final_output"]
    except asyncio.TimeoutError:
        logger.error(f"Admin agent timed out after {OVERALL_TIMEOUT_SECONDS}s for query: {query}")
        message = "That took too long to analyse. Please try a narrower question."
    except Exception as e:
        logger.error(f"Admin graph failed: {e}", exc_info=True)
        message = get_llm_error_message(e) or "Something went wrong while analysing your data. Please try again."
    return AdminState(query=query, narrative=message).model_dump()
