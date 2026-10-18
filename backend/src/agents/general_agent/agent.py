"""General Agent - Handles greetings, small talk, memory acknowledgements, order status and fallback."""

import logging
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langgraph.graph import END

from src.utils.util import llm, now_ist_text
from src.agents.general_agent.prompt import general_prompt
from src.graph.state import CoffeeAgentState

logger = logging.getLogger(__name__)

_chain = general_prompt | llm


async def general_agent(state: CoffeeAgentState) -> Command:
    try:
        # Format order for prompt — full details
        if state.order:
            order_lines = [
                f"  • {item.name} x{item.quantity} @ ₹{item.per_unit_price:.2f} = ₹{item.total_price:.2f}"
                for item in state.order
            ]
            current_order = "\n" + "\n".join(order_lines)
            order_total = f"{state.final_price:.2f}"
        else:
            current_order = "empty"
            order_total = "0.00"

        # sliding window: only send last 10 messages to LLM to prevent token limits
        recent_messages = state.messages[-10:] if state.messages else []

        response = await _chain.ainvoke({
            "messages": recent_messages,
            "user_memory": state.user_memory.model_dump(),
            "current_order": current_order,
            "order_total": order_total,
            "current_time": now_ist_text(),
        })
        msg = response.content
        return Command(
            update={
                "response_message": msg,
                "messages": [AIMessage(content=msg)],
            },
            goto=END
        )
    except Exception as e:
        logger.error(f"general_agent failed: {e}", exc_info=True)
        
        # Check for specific LLM API errors
        from src.utils.util import get_llm_error_message
        msg = get_llm_error_message(e) or "Hey there! Sorry, I had a small hiccup. How can I help you today?"
        
        return Command(
            update={"response_message": msg, "messages": [AIMessage(content=msg)]},
            goto=END
        )
