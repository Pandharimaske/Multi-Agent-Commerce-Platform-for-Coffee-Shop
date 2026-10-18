import logging
from datetime import datetime
from threading import Lock
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langgraph.graph import END

from src.utils.util import llm, now_ist_text, time_of_day_ist
from src.graph.state import CoffeeAgentState
from src.recommender import HybridRecommender
from src.agents.recommendation_management_agent.prompt import recommendation_prompt

logger = logging.getLogger(__name__)

_chain = recommendation_prompt | llm

# Thread-safe lazy singleton — load once at first request, never re-load
_recommender: HybridRecommender = None
_recommender_lock = Lock()


def _get_recommender() -> HybridRecommender:
    global _recommender
    if _recommender is None:
        with _recommender_lock:           # only one thread fits/loads at a time
            if _recommender is None:      # double-checked — re-test inside lock
                logger.info("Loading HybridRecommender...")
                _recommender = HybridRecommender.load()
    return _recommender


def _time_context() -> str:
    return time_of_day_ist()


def _format_ml_products(recommendations: list[dict]) -> str:
    if not recommendations:
        return "No specific products found."
    lines = []
    for r in recommendations:
        lines.append(
            f"- {r['name']} (₹{r['price']}) [{r['category']}] — {r['reason']}\n"
            f"  {r['description'][:120]}..."
        )
    return "\n".join(lines)


async def recommendation_management_agent(state: CoffeeAgentState) -> Command:
    try:
        memory = state.user_memory
        rec = _get_recommender()

        # Extract cart names from current order
        cart_names = [item.name for item in (state.order or [])]

        # Get ML recommendations
        recommendations = rec.recommend(
            cart=cart_names,
            likes=memory.likes or [],
            dislikes=memory.dislikes or [],
            allergies=memory.allergies or [],
            top_k=4,
        )

        logger.info(
            f"ML recommendations for user — "
            f"cart={cart_names}, likes={memory.likes}, allergies={memory.allergies} "
            f"→ {[r['name'] for r in recommendations]}"
        )

        # LLM formats the ML output into natural language
        response = await _chain.ainvoke({
            "user_input": state.user_input,
            "likes": memory.likes or "no preferences noted",
            "dislikes": memory.dislikes or "none",
            "allergies": memory.allergies or "none",
            "last_order": memory.last_order or "no previous order",
            "time_of_day": _time_context(),
            "current_time": now_ist_text(),
            "products": _format_ml_products(recommendations),
        })

        msg = response.content
        return Command(
            update={"response_message": msg, "messages": [AIMessage(content=msg)]},
            goto=END,
        )

    except Exception as e:
        logger.error(f"recommendation_management_agent failed: {e}", exc_info=True)
        
        # Check for specific LLM API errors
        from src.utils.util import get_llm_error_message
        msg = get_llm_error_message(e) or "I'd love to recommend something! Could you tell me what kind of mood you're in — something sweet, strong, or a snack?"
        
        return Command(
            update={"response_message": msg, "messages": [AIMessage(content=msg)]},
            goto=END
        )
