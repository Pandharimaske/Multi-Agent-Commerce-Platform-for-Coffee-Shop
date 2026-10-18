from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from src.memory.schemas import UserMemory


class ProductItem(BaseModel):
    name: str
    quantity: int
    per_unit_price: Optional[float] = None
    total_price: Optional[float] = None
    image_url: Optional[str] = None


class CoffeeAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    user_memory: UserMemory = Field(default_factory=UserMemory)
    chat_summary: str = ""
    user_input: str = ""
    image_url: Optional[str] = None
    response_message: Optional[str] = ""
    order: List[ProductItem] = Field(default_factory=list)
    final_price: float = 0.0
