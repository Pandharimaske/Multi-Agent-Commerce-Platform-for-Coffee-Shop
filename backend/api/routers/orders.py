from fastapi import APIRouter
from api.auth import CurrentUser
from api.schemas import ActiveOrderResponse, OrderHistoryResponse, UpdateOrderRequest
from src.orders import get_active_order, save_order, cancel_order, get_order_history
from src.graph.state import ProductItem

router = APIRouter(prefix="/orders", tags=["orders"])


@router.get("/active", response_model=ActiveOrderResponse)
async def get_active(current_user: CurrentUser):
    items, total = get_active_order(current_user.email)
    return ActiveOrderResponse(
        items=[i.model_dump() for i in items],
        total=total,
    )


@router.put("/active", response_model=ActiveOrderResponse)
async def update_active(body: UpdateOrderRequest, current_user: CurrentUser):
    """
    Frontend calls this whenever cart changes.
    Completely replaces the active pending order with the items the frontend sends.

    Only item names and quantities are used: prices are always looked up from the live
    menu in the database, so the response reflects the authoritative cart.
    """
    items = [
        ProductItem(
            name=item.name,
            quantity=item.quantity,
            per_unit_price=item.per_unit_price,
            total_price=round(item.per_unit_price * item.quantity, 2),
        )
        for item in body.items
    ]
    total = round(sum(i.total_price for i in items), 2)
    save_order(current_user.email, items, total)

    saved_items, saved_total = get_active_order(current_user.email)
    return ActiveOrderResponse(items=[i.model_dump() for i in saved_items], total=saved_total)


@router.delete("/active", status_code=204)
async def clear_active(current_user: CurrentUser):
    """Cancel/clear the active pending order."""
    cancel_order(current_user.email)


@router.get("/history", response_model=list[OrderHistoryResponse])
async def get_history(current_user: CurrentUser):
    try:
        rows = get_order_history(current_user.email)
    except Exception:
        return []
    return [OrderHistoryResponse(**row) for row in rows]
