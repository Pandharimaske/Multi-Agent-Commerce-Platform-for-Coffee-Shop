import logging
from typing import Any, Dict, List, Optional

from src.graph.state import ProductItem
from src.memory.supabase_client import supabase_admin as supabase

logger = logging.getLogger(__name__)

ORDERS = "coffee_shop_orders"
ITEMS = "coffee_shop_order_items"

_ITEM_COLUMNS = "product_name, quantity, unit_price, line_total, image_url, line_no"
_ORDER_SELECT = f"id, total, status, created_at, confirmed_at, updated_at, {ITEMS}({_ITEM_COLUMNS})"


def _items_from_rows(rows: Optional[List[Dict[str, Any]]]) -> List[ProductItem]:
    """Convert coffee_shop_order_items rows into ProductItem objects (cart order preserved)."""
    ordered = sorted(rows or [], key=lambda r: r.get("line_no") or 0)
    return [
        ProductItem(
            name=r["product_name"],
            quantity=int(r["quantity"]),
            per_unit_price=float(r["unit_price"]),
            total_price=float(r["line_total"]),
            image_url=r.get("image_url"),
        )
        for r in ordered
    ]


def _cart_payload(items: List[ProductItem]) -> List[Dict[str, Any]]:
    """Only name + quantity go to the database. Prices are looked up there, never trusted from callers."""
    return [{"name": i.name, "quantity": i.quantity} for i in items]


def get_active_order(user_email: str) -> tuple[List[ProductItem], float]:
    """Load the user's active pending order. Returns ([], 0.0) if none."""
    try:
        res = (
            supabase.table(ORDERS)
            .select(_ORDER_SELECT)
            .eq("user_email", user_email)
            .eq("status", "pending")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return [], 0.0

        row = res.data[0]
        items = _items_from_rows(row.get(ITEMS))
        total = float(row.get("total") or 0.0)
        return items, total

    except Exception as e:
        logger.error(f"get_active_order failed for {user_email}: {e}")
        return [], 0.0


def save_order(user_email: str, items: List[ProductItem], total: float) -> None:
    """
    Replace the user's pending cart atomically (one database transaction).

    Prices are re-read from the live product table by the database function, so the
    `per_unit_price` / `total` values passed in are advisory only.
    """
    try:
        if not items:
            supabase.table(ORDERS).delete().eq("user_email", user_email).eq("status", "pending").execute()
            return

        supabase.rpc(
            "replace_pending_order",
            {"p_user_email": user_email, "p_items": _cart_payload(items)},
        ).execute()
        logger.info(f"Order saved for {user_email} - {len(items)} items")

    except Exception as e:
        logger.error(f"save_order failed for {user_email}: {e}")


def confirm_order(user_email: str, items: List[ProductItem], total: float) -> Optional[Dict[str, Any]]:
    """
    Confirm the order in one database transaction.

    The cart is re-synced from `items`, every line is re-priced from the live menu, and the
    order is marked confirmed. Returns {"order_id", "total", "items": [ProductItem]} using the
    authoritative database values, or None if the order could not be confirmed.
    """
    try:
        res = supabase.rpc(
            "confirm_pending_order",
            {"p_user_email": user_email, "p_items": _cart_payload(items)},
        ).execute()
        data = res.data
        if not data:
            logger.error(f"confirm_order: nothing to confirm for {user_email}")
            return None

        confirmed_items = [ProductItem(**i) for i in (data.get("items") or [])]
        confirmed_total = float(data["total"])
        if abs(confirmed_total - float(total)) > 0.01:
            logger.warning(
                f"Price changed between cart and checkout for {user_email}: "
                f"cart={total} confirmed={confirmed_total}"
            )

        logger.info(f"Order confirmed for {user_email}, order_id={data['order_id']}")
        return {"order_id": data["order_id"], "total": confirmed_total, "items": confirmed_items}

    except Exception as e:
        logger.error(f"confirm_order failed for {user_email}: {e}")
        return None


def cancel_order(user_email: str) -> None:
    """Hard delete the pending order for this user (line items are removed by cascade)."""
    try:
        supabase.table(ORDERS).delete().eq("user_email", user_email).eq("status", "pending").execute()
        logger.info(f"Pending order cancelled for {user_email}")
    except Exception as e:
        logger.error(f"cancel_order failed for {user_email}: {e}")


def get_order_history(user_email: str) -> List[Dict[str, Any]]:
    """
    Confirmed orders, newest first, in the shape the API returns:
    {id, items: [{name, quantity, per_unit_price, total_price, image_url}], total, status, updated_at}
    """
    res = (
        supabase.table(ORDERS)
        .select(_ORDER_SELECT)
        .eq("user_email", user_email)
        .eq("status", "confirmed")
        .order("confirmed_at", desc=True)
        .execute()
    )
    history: List[Dict[str, Any]] = []
    for row in res.data or []:
        history.append(
            {
                "id": row["id"],
                "items": [i.model_dump() for i in _items_from_rows(row.get(ITEMS))],
                "total": float(row.get("total") or 0.0),
                "status": row["status"],
                "updated_at": row.get("confirmed_at") or row["updated_at"],
            }
        )
    return history
