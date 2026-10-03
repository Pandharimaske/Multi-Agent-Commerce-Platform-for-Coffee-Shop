from langgraph.graph import StateGraph, START, END
from src.agents.input_processor_agent import input_processor_agent
from src.agents.memory_management_agent import memory_agent
from src.agents.router_agent import router_agent
from src.agents.details_management_agent import details_management_agent
from src.agents.order_management_agent import order_management_agent
from src.agents.general_agent import general_agent
from src.agents.recommendation_management_agent import recommendation_management_agent

from src.graph.state import CoffeeAgentState
import os

_checkpointer = None

def _get_checkpointer():
    """
    Checkpointer strategy:
    - macOS dev  → MemorySaver (in-memory, resets on restart; avoids psycopg issues)
    - Linux/prod → PostgresSaver via connection pool (Supabase Transaction Pooler)
    - Fallback   → MemorySaver (last resort, resets on restart)

    psycopg's C extension segfaults on macOS ARM when forked by uvicorn's
    multiprocessing — that's why we avoid PostgresSaver on Darwin entirely.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    import platform
    import logging
    log = logging.getLogger(__name__)

    is_mac = platform.system() == "Darwin"
    db_uri = os.getenv("SUPABASE_DB_URL")

    if is_mac or not db_uri:
        # Dev / no DB configured: in-memory checkpointer (works with the async graph).
        # NOTE: AsyncSqliteSaver.from_conn_string() returns an async context manager,
        # not a saver, so it cannot be passed to compile(). State resets on restart.
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        log.info("MemorySaver checkpointer initialised (state resets on restart)")
        return _checkpointer

    # Production (Linux): use persistent Postgres checkpointer
    try:
        from psycopg_pool import ConnectionPool, AsyncConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # Supabase Transaction Pooler (Port 6543) — requires prepare_threshold=0
        separator = "&" if "?" in db_uri else "?"
        full_uri = f"{db_uri}{separator}sslmode=require&tcp_user_timeout=10000"

        # 1. Setup tables synchronously using the sync PostgresSaver
        # We do this because _get_checkpointer is called synchronously
        # during FastAPI startup, where we cannot easily `await setup()`.
        with ConnectionPool(
            conninfo=full_uri,
            max_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        ) as sync_pool:
            PostgresSaver(sync_pool).setup()

        # 2. Create the Async checkpointer for the graph runtime
        # open=False is required here because there is no running async loop
        # to `await async_pool.open()` yet. It will open on first use.
        async_pool = AsyncConnectionPool(
            conninfo=full_uri,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        
        _checkpointer = AsyncPostgresSaver(async_pool)
        log.info("AsyncPostgresSaver checkpointer initialised via Supabase Transaction Pooler")
    except Exception as e:
        log.warning(f"PostgresSaver failed, falling back to MemorySaver: {e}")
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()

    return _checkpointer



def build_coffee_shop_graph():
    builder = StateGraph(CoffeeAgentState)

    builder.add_node("input_processor", input_processor_agent)
    builder.add_node("memory", memory_agent)
    builder.add_node("router", router_agent)
    builder.add_node("details_management_agent", details_management_agent)
    builder.add_node("order_management_agent", order_management_agent)
    builder.add_node("recommendation_management_agent", recommendation_management_agent)
    builder.add_node("general_agent", general_agent)

    builder.add_edge(START, "input_processor")
    builder.add_edge("memory", "router")

    return builder.compile(checkpointer=_get_checkpointer())
