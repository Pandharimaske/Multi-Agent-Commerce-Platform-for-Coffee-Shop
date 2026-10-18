import uuid
import json
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from api.auth import CurrentUser
from api.schemas import ChatRequest, ChatResponse, MessageHistoryResponse, ResumeRequest
from langgraph.types import Command
from src.graph.graph import build_coffee_shop_graph
from src.graph.state import CoffeeAgentState
from src.memory.memory_manager import get_user_memory
from src.memory.schemas import UserMemory
from src.orders import get_active_order
from src.sessions import get_or_create_session, load_messages, save_messages, append_message, append_messages
from src.memory.supabase_client import supabase_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Error messages that should NOT be persisted to chat history
_ERROR_FALLBACKS = {
    "I'm having a little trouble. Can we stick to coffee orders for now?",
    "Sorry, I'm having trouble understanding your request right now. Could you please try again?",
    "I'm having trouble retrieving that information right now. Please try again.",
    "Hey there! Sorry, I had a small hiccup. How can I help you today?",
    "Sorry, I had a little trouble with that. Could you try again?",
    "I'm having trouble updating your order right now. Please try again.",
}

_graph = build_coffee_shop_graph()


@router.post("/upload")
async def upload_image(file: UploadFile = File(...), current_user: CurrentUser = None):
    """Upload an image to Supabase Storage and return the public URL."""
    try:
        file_ext = file.filename.split(".")[-1]
        file_name = f"{uuid.uuid4()}.{file_ext}"
        content = await file.read()
        
        # Upload to 'chat_images' bucket
        res = supabase_admin.storage.from_("chat_images").upload(
            path=file_name,
            file=content,
            file_options={"content-type": file.content_type}
        )
        
        # Get Public URL
        url_res = supabase_admin.storage.from_("chat_images").get_public_url(file_name)
        return {"url": url_res}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/history", response_model=list[MessageHistoryResponse])
async def get_history(session_id: str, current_user: CurrentUser):
    """Load message history for a session — called on page load/refresh."""
    messages = load_messages(session_id)
    result = []
    for m in messages:
        role = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
        content = m.content
        msg_type = "text"
        
        if role == "assistant" and content.startswith("[Approval Required]"):
            msg_type = "interrupt"
            # Extract JSON payload only if it's there
            content = content.replace("[Approval Required] ", "").strip()
                
        result.append(MessageHistoryResponse(
            role=role,
            content=content,
            msg_type=msg_type
        ))
    return result


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, current_user: CurrentUser):
    import asyncio
    import time
    
    start_time = time.time()
    user_email = current_user.email
    session_id = body.session_id or str(uuid.uuid4())

    # Parallelize pre-graph loads
    # Using to_thread because supabase-py is synchronous
    tasks = [
        asyncio.to_thread(get_or_create_session, session_id, user_email),
        asyncio.to_thread(get_user_memory, user_email),
        asyncio.to_thread(get_active_order, user_email),
        asyncio.to_thread(load_messages, session_id),
    ]
    
    # Run all 4 queries simultaneously
    _, user_memory, (order, final_price), messages = await asyncio.gather(*tasks)
    
    # Defaults and safety handling
    if not isinstance(user_memory, UserMemory): user_memory = None
    if order is None: order, final_price = [], 0.0

    state = CoffeeAgentState(
        user_input=body.user_input,
        user_memory=user_memory or CoffeeAgentState().user_memory,
        order=order,
        final_price=final_price,
        messages=messages,
    )
    
    logger.info(f"Pre-graph data loaded in parallel: {(time.time() - start_time)*1000:.2f}ms")

    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_email,
        }
    }

    try:
        final_state = await _graph.ainvoke(state, config=config)
        
        current_state = await _graph.aget_state(config)
        if current_state.tasks and current_state.tasks[0].interrupts:
            payload = current_state.tasks[0].interrupts[0].value
            return ChatResponse(session_id=session_id, response=f"__INTERRUPT__:{json.dumps(payload)}")
        
        response = (
            (final_state.get("response_message") if isinstance(final_state, dict) else final_state.response_message)
            or "Sorry, I had a little trouble with that. Could you try again?"
        )

        if response not in _ERROR_FALLBACKS:
            save_messages(session_id, user_email, body.user_input, response)

        return ChatResponse(session_id=session_id, response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


@router.post("/stream")
async def stream_chat(body: ChatRequest, current_user: CurrentUser):
    user_email = current_user.email
    session_id = body.session_id or str(uuid.uuid4())

    get_or_create_session(session_id, user_email)

    try:
        user_memory = get_user_memory(user_email)
    except Exception:
        user_memory = None

    try:
        order, final_price = get_active_order(user_email)
    except Exception:
        order, final_price = [], 0.0

    messages = load_messages(session_id)
    
    # ATOMIC TURNOUT FOR NEW SESSIONS
    if not messages:
        welcome_msg = f"Hey {current_user.username or user_email.split('@')[0]}! Welcome to Merry's Way ☕ What can I get you today?"
        # Save Welcome + Initial User Input in one go to preserve history flow
        append_messages(session_id, user_email, [
            {"role": "bot", "content": welcome_msg},
            {"role": "user", "content": body.user_input}
        ])
    else:
        # Just save the user message normally
        append_message(session_id, user_email, "user", body.user_input)
    
    # Reload for graph context
    messages = load_messages(session_id)

    state = CoffeeAgentState(
        user_input=body.user_input,
        image_url=body.image_url,
        user_memory=user_memory or CoffeeAgentState().user_memory,
        order=order,
        final_price=final_price,
        messages=messages,
    )

    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_email,
        }
    }

    async def event_generator():
        full_response = ""
        final_state_msg = None
        answering_agents = {
            "general_agent"
        }
        
        try:
            async for event in _graph.astream_events(state, config=config, version="v2"):
                kind = event["event"]
                node_name = event["metadata"].get("langgraph_node")

                # Graph compilation names might vary (e.g. "LangGraph") so we check root chain output
                if kind == "on_chain_end" and not node_name:
                    output = event["data"].get("output", {})
                    if isinstance(output, dict) and "response_message" in output:
                        final_state_msg = output["response_message"]
                    elif hasattr(output, "response_message"):
                        final_state_msg = output.response_message
                
                if kind in ("on_chat_model_start", "on_chain_start") and node_name and not node_name.startswith("_") and node_name not in ("graph", "builder"):
                    yield f"data: {json.dumps({'type': 'status', 'node': node_name})}\n\n"
                    
                elif kind == "on_chat_model_stream" and node_name in answering_agents:
                    chunk = event["data"]["chunk"]
                    text_chunk = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text_chunk and isinstance(text_chunk, str):
                        full_response += text_chunk
                        yield f"data: {json.dumps({'type': 'token', 'content': text_chunk})}\n\n"

            current_state = await _graph.aget_state(config)
            if current_state.tasks and current_state.tasks[0].interrupts:
                payload = current_state.tasks[0].interrupts[0].value
                # SAVE STRUCTURED BUBBLE TO HISTORY
                # We save with a prefix so /history can identify it as an interrupt type
                bubble_text = f"[Approval Required] {json.dumps(payload)}"
                append_message(session_id, user_email, "bot", bubble_text)
                yield f"data: {json.dumps({'type': 'interrupt', 'payload': payload})}\n\n"
                return  # Skip saving to history and returning text since we are paused
            
            if not full_response:
                full_response = final_state_msg or "Sorry, I had a little trouble with that. Could you try again?"
                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                
            if full_response not in _ERROR_FALLBACKS and full_response.strip():
                # SAVE FINAL BOT RESPONSE
                append_message(session_id, user_email, "bot", full_response)
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/resume", response_model=ChatResponse)
async def resume_chat(body: ResumeRequest, current_user: CurrentUser):
    config = {
        "configurable": {
            "thread_id": body.session_id,
            "user_id": current_user.email,
        }
    }
    try:
        final_state = await _graph.ainvoke(
            Command(resume=body.payment_status),
            config=config
        )
        user_label = body.user_content or "Action Confirmed"
        response = (
            (final_state.get("response_message") if isinstance(final_state, dict) else final_state.response_message)
            or "Action processed successfully!"
        )
        
        # USE ATOMIC SAVE FOR THE TURN (Approval + Response)
        try:
            append_messages(body.session_id, current_user.email, [
                {"role": "user", "content": user_label},
                {"role": "bot", "content": response}
            ])
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to save atomic turnout: {e}")
            
        return ChatResponse(session_id=body.session_id, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checkout resume error: {str(e)}")

