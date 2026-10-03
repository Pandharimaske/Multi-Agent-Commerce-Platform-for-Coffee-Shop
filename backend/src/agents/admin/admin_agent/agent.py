import logging
import os
import json
import re
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from src.utils.util import llm, get_embedding_model
from src.memory.supabase_client import supabase_admin as supabase
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import datetime

logger = logging.getLogger(__name__)

class AdminState(BaseModel):
    query: str = Field(..., description="The original user query")
    narrative: str = Field(..., description="The textual analysis and response to the user")
    chart_type: str = Field("none", description="The type of chart to render (bar, pie, line, table, none)")
    chart_data: List[Dict[str, Any]] = Field(default_factory=list, description="The data points for the chart")
    sql: Optional[str] = Field(None, description="The SQL query generated for this response")

class AdminGraphState(BaseModel):
    """The state maintained throughout the Admin Agent's execution graph."""
    query: str
    history: List[Dict] = Field(default_factory=list)
    schema_context: Optional[str] = None
    sql: Optional[str] = None
    results: Optional[List[Dict]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    final_output: Optional[Dict[str, Any]] = None

def redact_results(results: List[Dict]) -> List[Dict]:
    """
    Mask PII (emails, names) from results selectively.
    Preserves product names, categories, and brands.
    """
    # Keys that definitely contain PII
    blacklist = ["customer_name", "email", "phone", "user_id", "address", "user_name"]
    # Keys that are always safe
    whitelist = ["product_name", "category", "brand", "name", "value", "count", "revenue", "total"]

    redacted = []
    for row in results:
        new_row = {}
        for k, v in row.items():
            k_lower = k.lower()
            if isinstance(v, str):
                # Always mask emails regardless of key
                if "@" in v and "." in v:
                    parts = v.split("@")
                    new_row[k] = f"{parts[0][0]}***@{parts[1]}"
                # Mask blacklist keys
                elif any(pii in k_lower for pii in blacklist):
                     new_row[k] = f"{v[0]}***" if len(v) > 0 else "***"
                # If key is 'name' but NOT in whitelist and NOT explicitly a product
                elif k_lower == "name" and not any(safe in k_lower for safe in whitelist):
                    # Heuristic: If it looks like a person name (2+ words), mask it
                    if " " in v.strip():
                        new_row[k] = f"{v[0]}***"
                    else:
                        new_row[k] = v
                else:
                    new_row[k] = v
            else:
                new_row[k] = v
        redacted.append(new_row)
    return redacted

async def discover_schema(query: str, top_k_tables: int = 2, top_k_columns: int = 5) -> str:
    """Discovery Phase (Uber-style) with Global Context"""
    try:
        embeddings = get_embedding_model()
        q_vec = embeddings.embed_query(query)

        # 1. Fetch semantic matches
        res = supabase.rpc("match_schema_metadata", {
            "query_embedding": q_vec,
            "match_threshold": 0.2,
            "match_count": 20 # Increased to get more context
        }).execute()

        # 2. ALSO Fetch ALL table summaries for global context
        all_tables_res = supabase.table("coffee_shop_schema_metadata").select("metadata").eq("metadata->>type", "table").execute()
        
        global_tables = "### Global Table Overview\n"
        for item in (all_tables_res.data or []):
            meta = item.get("metadata", {})
            global_tables += f"- {meta.get('name')}: {meta.get('description')}\n"

        if not res.data:
            return global_tables + "\nNo specific columns matched. Please use the global overview."

        tables = {}
        columns = {}

        for item in res.data:
            meta = item.get("metadata", {})
            if meta.get("type") == "table":
                t_name = meta.get("name")
                if t_name not in tables:
                    tables[t_name] = meta.get("description")
            elif meta.get("type") == "column":
                t_name = meta.get("table")
                c_name = meta.get("name")
                if t_name not in columns:
                    columns[t_name] = []
                columns[t_name].append(f"{c_name}: {meta.get('description')}")

        schema_context = global_tables + "\n### Relevant Column Details\n\n"
        for t_name, t_desc in tables.items():
            schema_context += f"Table: {t_name}\nDescription: {t_desc}\n"
            if t_name in columns:
                schema_context += "Columns:\n"
                for col_info in columns[t_name]:
                    schema_context += f"- {col_info}\n"
            schema_context += "\n"
        
        for t_name, cols in columns.items():
            if t_name not in tables:
                schema_context += f"Table: {t_name}\nColumns:\n"
                for col_info in cols:
                    schema_context += f"- {col_info}\n"
                schema_context += "\n"

        return schema_context
    except Exception as e:
        logger.error(f"Discovery phase failed: {e}")
        return "Error during schema discovery."
        
def scrub_sql(sql: str) -> str:
    """
    Clean the SQL query generated by LLM:
    1. Remove markdown code blocks (```sql ... ```)
    2. Remove trailing semicolons (they break RPC subqueries)
    3. Strip whitespace
    """
    # Remove markdown code blocks
    sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'```\s*', '', sql)
    
    # Remove trailing semicolons
    sql = sql.strip().rstrip(';')
    
    return sql.strip()

async def generate_sql(query: str, schema_context: str, history: List[Dict[str, str]] = None, error_context: str = None) -> str:
    """SQL Generation Phase with Optional Retry Context"""
    current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S") + " IST"
    history = history or []
    history_str = ""
    for msg in history[-5:]:
        history_str += f"{msg['role'].upper()}: {msg['content']}\n"

    error_section = ""
    if error_context:
        error_section = f"\n### PREVIOUS ERROR:\nYour previous SQL query failed with this error: {error_context}\nPlease analyze the error and the schema carefully to provide a corrected SQL query.\n"

    system_prompt = f"""You are a PostgreSQL expert.
Today's Date: {current_time}
{schema_context}
{error_section}
CONVERSATION HISTORY:
{history_str}
Guidelines:
1. Return ONLY raw SQL. No markdown block. No trailing semicolon.
2. If charts are needed, aliasing columns as 'name' and 'value' is preferred.
3. For JSONB columns (like 'items' in orders), expand them using: CROSS JOIN LATERAL jsonb_array_elements(o.items) AS item
4. IMPORTANT: Always use parentheses when casting JSONB values for math or aggregation. Example: (item->>'total_price')::numeric.
5. Refer to expanded JSONB items as 'item', NOT 'o.item'.
6. DATE PATTERNS (the shop is in India: ALWAYS convert timestamps to IST with AT TIME ZONE 'Asia/Kolkata'):
   - Today: (updated_at AT TIME ZONE 'Asia/Kolkata')::date = (now() AT TIME ZONE 'Asia/Kolkata')::date
   - Yesterday: (updated_at AT TIME ZONE 'Asia/Kolkata')::date = (now() AT TIME ZONE 'Asia/Kolkata')::date - 1
   - This Month: date_trunc('month', updated_at AT TIME ZONE 'Asia/Kolkata') = date_trunc('month', now() AT TIME ZONE 'Asia/Kolkata')
   - This Year: date_trunc('year', updated_at AT TIME ZONE 'Asia/Kolkata') = date_trunc('year', now() AT TIME ZONE 'Asia/Kolkata')
   - Weekwise: date_trunc('week', updated_at AT TIME ZONE 'Asia/Kolkata')
7. TIME SERIES: when the user asks for a report/trend over a period (this year, this month, last N days), include EVERY period in the range, with 0 for periods that have no orders, using generate_series and a LEFT JOIN. Do NOT use WITH/CTEs (only plain SELECT is allowed). Example for sales per month this year:
   SELECT to_char(m, 'Mon YYYY') AS name, COALESCE(SUM(o.total), 0) AS value FROM generate_series(date_trunc('year', now() AT TIME ZONE 'Asia/Kolkata'), date_trunc('month', now() AT TIME ZONE 'Asia/Kolkata'), interval '1 month') AS m LEFT JOIN coffee_shop_orders o ON date_trunc('month', o.updated_at AT TIME ZONE 'Asia/Kolkata') = m AND o.status = 'confirmed' GROUP BY m ORDER BY m
8. Filter for status = 'confirmed' unless otherwise specified.
"""
    response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return response.content.strip()

async def execute_sql(sql: str) -> List[Dict]:
    """Execute SQL via Supabase RPC"""
    try:
        res = supabase.rpc("execute_sql_query", {"sql_query": sql}).execute()
        return res.data if res.data else []
    except Exception as e:
        logger.error(f"SQL Execution failed: {e}")
        return [{"error": str(e)}]

async def format_response(query: str, results: List[Dict], sql: str, history: List[Dict] = None) -> AdminState:
    """Final Phase: Structured State Generation"""
    history_str = ""
    for msg in (history or [])[-5:]:
         history_str += f"{msg['role'].upper()}: {msg['content']}\n"

    system_prompt = f"""You are a BI Analyst. You MUST return a valid JSON object matching this schema:
{{
  "query": "original user request",
  "narrative": "textual summary of findings",
  "chart_type": "bar | pie | line | table | none",
  "chart_data": [{{ "name": "...", "value": 123 }}],
  "sql": "the executed sql"
}}

Guidelines:
1. All money amounts are in Indian Rupees. Always write currency as ₹ (never $).
1a. Base the narrative ONLY on the numbers in Results. Do NOT invent causes, seasonality, marketing, or operational reasons, and do not give advice unless asked.
1b. Today's date is {datetime.date.today().isoformat()}. If the latest period in the results is the current, still-incomplete day or month, say it is partial so far and do NOT describe it as a decline.
1c. Describe a trend only when there are at least 3 complete periods. With fewer, just state the figures and compare them plainly. If periods have zero or no data, say so.
1d. Be concise: 1-3 sentences with the exact figures from Results.
2. If the data is tabular but not a trend/ranking, use chart_type: "table".
3. Use chart_type: "none" if no data is found or a chart doesn't make sense.
4. Keep the JSON structure strict. No extra text.
"""
    human_msg = f"Query: {query}\nSQL: {sql}\nResults: {json.dumps(results[:20])}\nHistory: {history_str}"
    
    response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
    raw_content = response.content.strip()
    
    # Extract JSON if LLM included markdown fences
    if "```json" in raw_content:
        raw_content = raw_content.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_content:
        raw_content = raw_content.split("```")[1].split("```")[0].strip()

    try:
        state_dict = json.loads(raw_content)
        # Ensure SQL is passed from our actual execution
        state_dict["sql"] = sql
        state_dict["query"] = query
        return AdminState(**state_dict)
    except Exception as e:
        logger.error(f"Failed to parse structured response: {e}\nRaw: {raw_content}")
        return AdminState(
            query=query,
            narrative="I analyzed the data but struggled to format the visual response. Here is what I found: " + raw_content[:200],
            chart_type="none",
            chart_data=[],
            sql=sql
        )

# --- Graph Nodes ---

async def discovery_node(state: AdminGraphState) -> Dict[str, Any]:
    """Phase 1: Discover relevant schema metadata."""
    logger.info(f"Node: Discovery | Query: {state.query}")
    
    discovery_query = state.query
    history = state.history
    
    if history and any(kw in discovery_query.lower() for kw in ["it", "those", "that", "filter", "show more"]):
        discovery_query = f"{history[-1]['content']} {discovery_query}"
        
    schema_context = await discover_schema(discovery_query)
    return {"schema_context": schema_context}

async def generation_node(state: AdminGraphState) -> Dict[str, Any]:
    """Phase 2: Generate PostgreSQL query."""
    logger.info(f"Node: Generation | Retry Count: {state.retry_count}")
    
    sql = await generate_sql(
        query=state.query, 
        schema_context=state.schema_context, 
        history=state.history,
        error_context=state.error
    )
    # SCRUB SQL before execution to handle LLM quirks
    cleaned_sql = scrub_sql(sql)
    return {"sql": cleaned_sql, "error": None} # Clear previous error on regeneration

async def execution_node(state: AdminGraphState) -> Dict[str, Any]:
    """Phase 3: Execute SQL and check for errors."""
    logger.info(f"Node: Execution | SQL: {state.sql}")
    
    raw_results = await execute_sql(state.sql)
    
    # Check for errors in results
    if raw_results and isinstance(raw_results, list) and len(raw_results) > 0 and "error" in raw_results[0]:
        error_msg = raw_results[0]["error"]
        logger.warning(f"SQL Execution Error: {error_msg}")
        return {"error": error_msg, "retry_count": state.retry_count + 1}
    
    return {"results": raw_results, "error": None}

async def formatting_node(state: AdminGraphState) -> Dict[str, Any]:
    """Phase 4: Format final response based on results."""
    logger.info("Node: Formatting")
    
    results = redact_results(state.results or [])
    state_obj = await format_response(
        query=state.query, 
        results=results, 
        sql=state.sql, 
        history=state.history
    )
    return {"final_output": state_obj.model_dump()}

# --- Routing Logic ---

def should_retry(state: AdminGraphState) -> str:
    """Determine if we should retry SQL generation or move to formatting."""
    if state.error and state.retry_count <= state.max_retries:
        logger.info(f"Routing: Retry Logic (Attempt {state.retry_count} of {state.max_retries})")
        return "generate"
    return "format"

# --- Graph Assembly ---

def create_admin_graph():
    workflow = StateGraph(AdminGraphState)
    
    # Add Nodes
    workflow.add_node("discover", discovery_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("format", formatting_node)
    
    # Set Entry Point
    workflow.set_entry_point("discover")
    
    # Add Edges
    workflow.add_edge("discover", "generate")
    workflow.add_edge("generate", "execute")
    
    # Add Conditional Edge for execution
    workflow.add_conditional_edges(
        "execute",
        should_retry,
        {
            "generate": "generate",
            "format": "format"
        }
    )
    
    workflow.add_edge("format", END)
    
    return workflow.compile()

admin_graph = create_admin_graph()

async def invoke_admin_agent(query: str, history: List[Dict] = None) -> Dict[str, Any]:
    """Main Agent Loop powered by LangGraph"""
    try:
        initial_state: AdminGraphState = {
            "query": query,
            "history": history or [],
            "schema_context": None,
            "sql": None,
            "results": None,
            "error": None,
            "retry_count": 0,
            "max_retries": 2, # Allow 2 retries (3 attempts total)
            "final_output": None
        }
        
        final_state = await admin_graph.ainvoke(initial_state)
        return final_state["final_output"]
        
    except Exception as e:
        logger.error(f"Admin Graph failed: {e}", exc_info=True)
        return AdminState(
            query=query, 
            narrative=f"System Error: {str(e)}", 
            chart_type="none", 
            chart_data=[],
            sql=None
        ).model_dump()
