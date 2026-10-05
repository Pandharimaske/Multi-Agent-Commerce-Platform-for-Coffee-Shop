from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import logging
from api.auth import CurrentUser
from src.memory.supabase_client import supabase, supabase_admin
from src.agents.admin.admin_agent.agent import invoke_admin_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

class AdminChatRequest(BaseModel):
    query: str
    session_id: str | None = None

def verify_admin(current_user: CurrentUser):
    try:
        user_email = current_user.email.lower().strip()
        res = supabase_admin.table("coffee_shop_profiles").select("is_admin").eq("user_email", user_email).execute()
        if not res.data or not res.data[0].get("is_admin"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking admin status: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Error verifying admin status.")
    return current_user

@router.get("/history")
async def get_admin_history(session_id: str, current_user: CurrentUser):
    """Return the full chat history for a given admin session."""
    verify_admin(current_user)
    try:
        res = supabase_admin.table("coffee_shop_admin_sessions").select("history").eq("session_id", session_id).execute()
        if res.data:
            return {"history": res.data[0].get("history", [])}
    except Exception as e:
        logger.error(f"Error fetching admin history: {e}")
    return {"history": []}


@router.post("/chat")
async def chat_with_admin_agent(req: AdminChatRequest, current_user: CurrentUser):
    verify_admin(current_user)
    
    session_id = req.session_id
    history = []
    
    # 1. Load History if session_id provided using supabase_admin
    if session_id:
        try:
            res = supabase_admin.table("coffee_shop_admin_sessions").select("history").eq("session_id", session_id).execute()
            if res.data:
                history = res.data[0].get("history", [])
        except Exception as e:
            logger.error(f"Error loading admin history: {e}")

    # 2. Invoke Agent with History
    # invoke_admin_agent now returns a dict representing AdminState
    state = await invoke_admin_agent(req.query, history=history)
    
    # 3. Save History
    if session_id:
        try:
            new_history = history + [
                {"role": "user", "content": req.query},
                {"role": "assistant", "state": state, "content": state.get("narrative", "")}
            ]
            # Keep only last 10 messages for context (5 turns)
            new_history = new_history[-10:]
            
            supabase_admin.table("coffee_shop_admin_sessions").upsert({
                "session_id": session_id,
                "user_email": current_user.email.lower().strip(),
                "history": new_history,
            }).execute()
        except Exception as e:
            logger.error(f"Error saving admin history: {e}")

    return {"reply": state.get("narrative", ""), "state": state}
