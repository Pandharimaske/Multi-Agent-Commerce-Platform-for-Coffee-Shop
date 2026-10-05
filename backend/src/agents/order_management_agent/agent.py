"""Order Management Agent - Handles order creation, updates, confirmation and cancellation."""

import logging
from typing import List
from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt
from langgraph.graph import END
from langgraph.errors import GraphInterrupt

from src.utils.util import llm
from src.graph.state import CoffeeAgentState, ProductItem
from src.agents.order_management_agent.schema import (
    OrderInput, OrderUpdateState, OrderAction, ActionDecision
)
from src.agents.order_management_agent.prompt import (
    parse_new_order_prompt,
    parse_order_update_prompt,
    detect_order_action_prompt,
    order_responder_prompt,
)
from langchain_core.runnables import RunnableConfig
from src.rag.retriever import get_product_by_name
from src.orders import save_order, confirm_order, cancel_order
from src.utils.email_util import send_order_receipt
from src.utils.allergens import allergy_warning

logger = logging.getLogger(__name__)

_action_llm    = llm.with_structured_output(ActionDecision)
_new_order_llm = llm.with_structured_output(OrderInput)
_update_llm    = llm.with_structured_output(OrderUpdateState)
_responder     = order_responder_prompt | llm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lookup_product(name: str) -> dict:
    result = get_product_by_name(name)
    if result.get("found"):
        return {
            "name": result.get("name"),
            "price": float(result.get("price") or result.get("metadata", {}).get("price", 0.0)),
            "image_url": result.get("image_url")
        }
    return {"found": False}


def _format_order_summary(order: List[ProductItem], final_price: float, notes: str = "") -> str:
    lines = [f"  • {item.name} x{item.quantity} @ ₹{item.per_unit_price:.2f} = ₹{item.total_price:.2f}" for item in order]
    notes_block = f"\n\n{notes}" if notes else ""
    return (
        "Here's your order:\n" + "\n".join(lines)
        + f"\n\n🧾 Total: ₹{final_price:.2f}"
        + notes_block
        + "\n\nShall I confirm this order?"
    )


def _mock_receipt(order: List[ProductItem], final_price: float, order_id: str = None) -> str:
    lines = [f"  • {item.name} x{item.quantity} = ₹{item.total_price:.2f}" for item in order]
    receipt = (
        f"✅ Order confirmed! Here's your receipt:\n" + "\n".join(lines) +
        f"\n\n🧾 **Total: ₹{final_price:.2f}**\n\n"
        f"Your order has been placed successfully. "
        f"You can view it anytime under **Order History** on the Orders page."
    )
    return receipt


async def _generate_dynamic_response(
    action_type: str,
    items_impacted: List[str],
    current_order: List[str],
    total_price: float,
    unavailable_items: List[str],
    status_message: str,
    user_input: str,
    messages: List
) -> str:
    """Invokes the responder LLM to get a natural response."""
    try:
        response = await _responder.ainvoke({
            "action_type": action_type,
            "items_impacted": items_impacted,
            "current_order": current_order,
            "total_price": total_price,
            "unavailable_items": unavailable_items,
            "status_message": status_message,
            "user_input": user_input,
            "messages": messages[-6:] if messages else []
        })
        return response.content
    except Exception as e:
        logger.error(f"Dynamic response generation failed: {e}")
        return status_message # Fallback to hardcoded status


# ── Main agent ────────────────────────────────────────────────────────────────

async def order_management_agent(state: CoffeeAgentState, config: RunnableConfig = {}) -> Command:
    user_input = state.user_input
    existing_order = list(state.order)
    messages = list(state.messages)
    user_id: str = config.get("configurable", {}).get("user_id", "anonymous")

    try:
        # ── LLM classifies all intent ─────────────────────────────────────────
        # Get last bot message for context (helps confirm detection)
        last_bot_msg = ""
        for m in reversed(messages):
            if m.__class__.__name__ == "AIMessage":
                last_bot_msg = m.content[-200:]
                break

        action_decision: ActionDecision = await (detect_order_action_prompt | _action_llm).ainvoke({
            "user_input": user_input,
            "has_existing_order": bool(existing_order),
            "existing_order": [f"{i.name} x{i.quantity}" for i in existing_order],
            "messages": messages,
            "last_bot_message": last_bot_msg,
        })
        action = action_decision.action
        logger.info(f"order_management_agent: action={action} for input='{user_input}'")

        # Safety: existing order + CREATE → treat as UPDATE
        if existing_order and action == OrderAction.CREATE:
            logger.warning("LLM picked CREATE with existing order — overriding to UPDATE")
            action = OrderAction.UPDATE

        # ── CONFIRM ───────────────────────────────────────────────────────────
        if action == OrderAction.CONFIRM:
            if not existing_order:
                msg = await _generate_dynamic_response(
                    action_type="confirm",
                    items_impacted=[],
                    current_order=[],
                    total_price=0.0,
                    unavailable_items=[],
                    status_message="You don't have anything in your order yet. Want to start one?",
                    user_input=user_input,
                    messages=messages
                )
                return Command(update={"response_message": msg, "messages": [AIMessage(content=msg)]}, goto=END)

            # Trigger HITL Order Confirmation & Payment
            payment_status = interrupt({"action": "order_confirmation", "total": state.final_price})

            if payment_status == "payment_success":
                order_id = None
                final_items, final_total = existing_order, state.final_price
                if user_id != "anonymous":
                    confirmed = confirm_order(user_id, existing_order, state.final_price)
                    if not confirmed:
                        msg = "Sorry, something went wrong while placing your order. Your cart is still saved, so please try again in a moment."
                        return Command(
                            update={"response_message": msg, "messages": [AIMessage(content=msg)]},
                            goto=END
                        )
                    # The database re-prices every line from the live menu, so use its values
                    order_id = confirmed["order_id"]
                    final_items, final_total = confirmed["items"], confirmed["total"]
                    # Fire-and-forget email receipt — failures logged, never crash order
                    import asyncio
                    async def _safe_send_receipt():
                        try:
                            await send_order_receipt(user_id, final_items, final_total, order_id)
                        except Exception as mail_err:
                            logger.error(f"Email receipt failed for order {order_id}: {mail_err}")
                    asyncio.ensure_future(_safe_send_receipt())
                
                receipt = _mock_receipt(final_items, final_total, order_id)
                msg = await _generate_dynamic_response(
                    action_type="confirm",
                    items_impacted=[i.name for i in final_items],
                    current_order=[f"{i.name} x{i.quantity}" for i in final_items],
                    total_price=final_total,
                    unavailable_items=[],
                    status_message=receipt,
                    user_input=user_input,
                    messages=messages
                )
                return Command(
                    update={"response_message": msg, "messages": [AIMessage(content=msg)], "order": [], "final_price": 0.0},
                    goto=END
                )
            else:
                msg = await _generate_dynamic_response(
                    action_type="checkout_cancelled",
                    items_impacted=[],
                    current_order=[f"{i.name} x{i.quantity}" for i in existing_order],
                    total_price=state.final_price,
                    unavailable_items=[],
                    status_message="Checkout was cancelled. Your order is still saved in your cart.",
                    user_input=user_input,
                    messages=messages
                )
                return Command(
                    update={"response_message": msg, "messages": [AIMessage(content=msg)]},
                    goto=END
                )

        # ── CANCEL ────────────────────────────────────────────────────────────
        elif action == OrderAction.CANCEL:
            if user_id != "anonymous":
                cancel_order(user_id)
            
            msg = await _generate_dynamic_response(
                action_type="cancel",
                items_impacted=[i.name for i in existing_order],
                current_order=[],
                total_price=0.0,
                unavailable_items=[],
                status_message="Your order has been cancelled. Let me know if you'd like to start a new one!",
                user_input=user_input,
                messages=messages
            )
            return Command(
                update={"response_message": msg, "messages": [AIMessage(content=msg)], "order": [], "final_price": 0.0},
                goto=END
            )

        # ── CREATE ────────────────────────────────────────────────────────────
        elif action == OrderAction.CREATE or (action == OrderAction.UPDATE and not existing_order):
            parsed: OrderInput = await (parse_new_order_prompt | _new_order_llm).ainvoke({
                "user_input": user_input,
                "messages": messages,
            })

            new_order, total, unavailable = [], 0.0, []
            for item in parsed.items:
                product = _lookup_product(item.name)
                if product.get("name"):
                    price = product["price"]
                    line_total = round(price * item.quantity, 2)
                    new_order.append(ProductItem(
                        name=product["name"], 
                        quantity=item.quantity, 
                        per_unit_price=price, 
                        total_price=line_total,
                        image_url=product.get("image_url")
                    ))
                    total += line_total
                else:
                    unavailable.append(item.name)

            total = round(total, 2)

            if not new_order:
                from src.rag.retriever import search_products
                recommendations = []
                for u in unavailable:
                    recs = search_products(u, top_k=2)
                    if recs:
                        recs_str = " or ".join([r.get("name") for r in recs])
                        recommendations.append(f"{recs_str} instead of {u}")
                
                if recommendations:
                    msg = f"Sorry, we don't serve {', '.join(unavailable)}. However, we highly recommend trying our {', '.join(recommendations)}! Would you like me to add that to your order?"
                else:
                    msg = f"Sorry, I couldn't find any of those items on our menu: {', '.join(unavailable)}."
                
                msg = await _generate_dynamic_response(
                    action_type="create_error",
                    items_impacted=[],
                    current_order=[],
                    total_price=0.0,
                    unavailable_items=unavailable,
                    status_message=msg,
                    user_input=user_input,
                    messages=messages
                )
                return Command(update={"response_message": msg, "messages": [AIMessage(content=msg)]}, goto=END)

            notes = []
            if unavailable:
                notes.append(f"⚠️ Skipped (not on menu): {', '.join(unavailable)}")
            warn = allergy_warning(getattr(state.user_memory, "allergies", None) or [], [i.name for i in new_order])
            if warn:
                notes.append(warn)
            summary = _format_order_summary(new_order, total, "\n\n".join(notes))
            if user_id != "anonymous":
                save_order(user_id, new_order, total)

            # Deterministic text: never let the LLM claim the order is confirmed here.
            msg = summary

            return Command(
                update={"order": new_order, "final_price": total, "response_message": msg, "messages": [AIMessage(content=msg)]},
                goto=END
            )

        # ── UPDATE ────────────────────────────────────────────────────────────
        elif action == OrderAction.UPDATE:
            parsed: OrderUpdateState = await (parse_order_update_prompt | _update_llm).ainvoke({
                "user_input": user_input,
                "existing_order": [f"{i.name} x{i.quantity}" for i in existing_order],
                "messages": messages,
            })

            order_dict = {item.name.lower(): item for item in existing_order}

            for update in parsed.updates:
                key = update.name.lower()
                product = _lookup_product(update.name)
                price = product.get("price", 0.0)
                image_url = product.get("image_url")

                if update.set_quantity is not None:
                    if update.set_quantity <= 0:
                        order_dict.pop(key, None)
                    elif product.get("name"):
                        order_dict[key] = ProductItem(
                            name=product["name"], quantity=update.set_quantity,
                            per_unit_price=price, total_price=round(price * update.set_quantity, 2),
                            image_url=image_url
                        )
                elif update.delta_quantity is not None:
                    if key in order_dict:
                        new_qty = order_dict[key].quantity + update.delta_quantity
                        if new_qty <= 0:
                            order_dict.pop(key)
                        else:
                            order_dict[key] = ProductItem(
                                name=order_dict[key].name, quantity=new_qty,
                                per_unit_price=price, total_price=round(price * new_qty, 2),
                                image_url=order_dict[key].image_url or image_url
                            )
                    elif update.delta_quantity > 0 and product.get("name"):
                        order_dict[key] = ProductItem(
                            name=product["name"], quantity=update.delta_quantity,
                            per_unit_price=price, total_price=round(price * update.delta_quantity, 2),
                            image_url=image_url
                        )

            updated_order = list(order_dict.values())
            total = round(sum(i.total_price or 0 for i in updated_order), 2)

            if not updated_order:
                if user_id != "anonymous":
                    cancel_order(user_id)
                
                msg = await _generate_dynamic_response(
                    action_type="update_empty",
                    items_impacted=[],
                    current_order=[],
                    total_price=0.0,
                    unavailable_items=[],
                    status_message="Your order is now empty. Let me know if you'd like to order something!",
                    user_input=user_input,
                    messages=messages
                )
                return Command(
                    update={"order": [], "final_price": 0.0, "response_message": msg, "messages": [AIMessage(content=msg)]},
                    goto=END
                )

            before_names = {i.name.lower() for i in existing_order}
            added_names = [i.name for i in updated_order if i.name.lower() not in before_names]
            warn = allergy_warning(getattr(state.user_memory, "allergies", None) or [], added_names)
            summary = _format_order_summary(updated_order, total, warn)
            if user_id != "anonymous":
                save_order(user_id, updated_order, total)

            # Deterministic text: never let the LLM claim the order is confirmed here.
            msg = summary

            return Command(
                update={"order": updated_order, "final_price": total, "response_message": msg, "messages": [AIMessage(content=msg)]},
                goto=END
            )

    except GraphInterrupt:
        # LangGraph HITL needs this bubble up
        raise
    except Exception as e:
        logger.error(f"order_management_agent failed: {e}", exc_info=True)
        
        # Check for specific LLM API errors
        from src.utils.util import get_llm_error_message
        msg = get_llm_error_message(e) or "Sorry, I ran into an issue processing your order. Please try again."
        
        return Command(update={"response_message": msg, "messages": [AIMessage(content=msg)]}, goto=END)
