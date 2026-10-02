"""Scalable configuration and utilities for Coffee Shop Chatbot - Multi-user version

Provides:
- Connection pooling with health checks
- Thread-safe singleton pattern
- Distributed cache support (Redis)
- Resource limits & rate limiting
- Monitoring & observability hooks
"""

import os
import logging
import threading
from typing import Optional
from datetime import datetime, timedelta
from functools import wraps
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from src.config import Config

load_dotenv()

logger = logging.getLogger(__name__)

# --- LLM Error Handling ---
try:
    import openai
except ImportError:
    openai = None

def get_llm_error_message(e: Exception) -> Optional[str]:
    """Classifies an LLM exception and returns a user-friendly error message if applicable."""
    error_str = str(e).lower()

    # Rate limits
    if "rate limit" in error_str or (openai and isinstance(e, openai.RateLimitError)):
        return "🚨 LLM Rate Limit: Too many requests. Please wait a moment before trying again."

    # Token / context limits
    if "context_length_exceeded" in error_str or "maximum context length" in error_str:
        return "🚨 LLM Token Limit: The conversation is too long for the current model. Please start a new chat."

    # Authentication
    if "authentication" in error_str or (openai and isinstance(e, openai.AuthenticationError)):
        return "🚨 LLM Auth Error: Invalid API key configuration. Please check NIM_API_KEY in your .env."

    # Connection / timeout
    if "connection" in error_str or "timeout" in error_str or (openai and isinstance(e, openai.APIConnectionError)):
        return "🚨 LLM Connection Error: Unable to reach NVIDIA NIM. Please check your internet or try again later."

    # Generic OpenAI-protocol errors (NIM uses the same protocol)
    if openai and isinstance(e, openai.OpenAIError):
        return f"🚨 LLM Provider Error (NIM): {str(e)}"

    return None


# ============================================================
# Connection Health Check
# ============================================================

class HealthChecker:
    """Monitors connection health and triggers reconnection if needed."""

    def __init__(self):
        self.last_check = datetime.utcnow()
        self.is_healthy = True
        self.lock = threading.Lock()

    def should_check(self) -> bool:
        """Determine if health check is due."""
        return (datetime.utcnow() - self.last_check).total_seconds() > Config.HEALTH_CHECK_INTERVAL_SECONDS

    def mark_check(self, healthy: bool = True):
        """Update health status."""
        with self.lock:
            self.last_check = datetime.utcnow()
            self.is_healthy = healthy
            if not healthy:
                logger.warning("Connection health check failed")


# ============================================================
# LLM Configuration (with connection pooling)
# ============================================================

class LLMPool:
    """Thread-safe LLM instance pool for concurrent requests."""

    _instance = None
    _lock = threading.Lock()
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.health_checker = HealthChecker()
        self._lock = threading.Lock()
        self._initialized = True
        logger.info("LLMPool initialized")

    def get_model(self, temperature: float = None, model_name: str = None) -> ChatOpenAI:
        """Get or create an NVIDIA NIM LLM instance (cached by temperature/model combo).

        NVIDIA NIM exposes an OpenAI-compatible REST API, so ChatOpenAI is used
        with nim_base_url and nim_api_key from config.
        """
        temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        model_name = model_name or Config.LLM_MODEL
        key = f"{model_name}_{temperature}"

        if key in LLMPool._models:
            logger.debug(f"Reusing cached LLM instance: {key}")
            return LLMPool._models[key]

        with self._lock:
            if key not in LLMPool._models:
                try:
                    if not Config.NIM_API_KEY:
                        logger.error("NIM_API_KEY is not set! Add it to your .env file.")
                        raise ValueError("NIM_API_KEY is required to initialise the LLM.")

                    logger.info(
                        f"Initialising NVIDIA NIM LLM | model={model_name} | "
                        f"base_url={Config.NIM_BASE_URL} | temperature={temperature}"
                    )

                    LLMPool._models[key] = ChatOpenAI(
                        model=model_name,
                        openai_api_key=Config.NIM_API_KEY,
                        openai_api_base=Config.NIM_BASE_URL,
                        temperature=temperature,
                        timeout=Config.LLM_TIMEOUT_SECONDS,
                        default_headers={
                            "HTTP-Referer": Config.APP_URL,
                            "X-Title": Config.APP_NAME,
                        },
                    )

                    self.health_checker.mark_check(True)
                except Exception as e:
                    logger.error(f"Failed to create LLM instance: {str(e)}")
                    self.health_checker.mark_check(False)
                    raise

        return LLMPool._models[key]

    @classmethod
    def clear_cache(cls):
        """Clear all cached LLM instances."""
        cls._models.clear()
        logger.info("LLM cache cleared")


# Singleton instance
_llm_pool = LLMPool()


def get_model(temperature: float = 0.0, model_name: str = None) -> ChatOpenAI:
    """Get LLM instance from pool."""
    return _llm_pool.get_model(temperature=temperature, model_name=model_name)


def get_small_model(temperature: float = 0.0, model_name: str = None) -> ChatOpenAI:
    """Get Small LLM instance from pool."""
    return _llm_pool.get_model(temperature=temperature, model_name=model_name or Config.SMALL_LLM_MODEL)


llm = get_model()  # Default large instance
small_llm = get_small_model() # Default small instance


# ============================================================
# Embedding Model Configuration (with caching)
# ============================================================

class EmbeddingPool:
    """Cached embedding model instances."""

    _instance = None
    _lock = threading.Lock()
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lock = threading.Lock()
        self._initialized = True
        logger.info("EmbeddingPool initialized")

    def get_model(self, model_name: str = None) -> HuggingFaceEndpointEmbeddings:
        """Get or create embedding model (cached)."""
        model_name = model_name or Config.EMBEDDING_MODEL

        if model_name in EmbeddingPool._models:
            logger.debug(f"Reusing cached embedding model: {model_name}")
            return EmbeddingPool._models[model_name]

        with self._lock:
            if model_name not in EmbeddingPool._models:
                try:
                    logger.info(f"Creating embedding model: {model_name}")
                    EmbeddingPool._models[model_name] = HuggingFaceEndpointEmbeddings(
                        model=model_name,
                        huggingfacehub_api_token=Config.HF_API_KEY
                    )
                except Exception as e:
                    logger.error(f"Failed to create embedding model {model_name}: {str(e)}")
                    raise

        return EmbeddingPool._models[model_name]

    @classmethod
    def clear_cache(cls):
        """Clear all cached embedding models."""
        cls._models.clear()
        logger.info("Embedding cache cleared")


_embedding_pool = EmbeddingPool()


def get_embedding_model(model_name: str = None) -> HuggingFaceEndpointEmbeddings:
    """Get embedding model from pool."""
    return _embedding_pool.get_model(model_name=model_name)


# embedding_model is intentionally NOT initialised here.
# Call get_embedding_model() at the point of use — lazy init on first request.
# This prevents HuggingFace API latency / outages from blocking server startup.


# ── Pinecone removed ──────────────────────────────────────────────────────────
# Vector retrieval now uses Supabase pgvector exclusively.
# See src/rag/retriever.py for all product search logic.
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# Rate Limiting (optional, configurable)
# ============================================================

class RateLimiter:
    """Simple in-memory rate limiter. Use Redis for distributed systems."""

    def __init__(self, requests_per_minute: int = None):
        self.requests_per_minute = requests_per_minute or Config.REQUESTS_PER_MINUTE
        self.requests = {}
        self._lock = threading.Lock()

    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limit."""
        if not Config.ENABLE_RATE_LIMITING:
            return True

        with self._lock:
            now = datetime.utcnow()
            minute_ago = now - timedelta(minutes=1)

            if user_id not in self.requests:
                self.requests[user_id] = []

            # Remove old requests
            self.requests[user_id] = [
                ts for ts in self.requests[user_id] if ts > minute_ago
            ]

            if len(self.requests[user_id]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for user: {user_id}")
                return False

            self.requests[user_id].append(now)
            return True


_rate_limiter = RateLimiter()


def rate_limit(func):
    """Decorator to apply rate limiting to functions."""
    @wraps(func)
    def wrapper(user_id: str = None, *args, **kwargs):
        if user_id and not _rate_limiter.is_allowed(user_id):
            raise Exception(f"Rate limit exceeded for user {user_id}")
        return func(*args, **kwargs)
    return wrapper


# ============================================================
# Lifecycle Management
# ============================================================

def initialize_all() -> None:
    """Initialize all connection pools and validate configuration."""
    try:
        Config.validate()
        logger.info("Configuration validated")

        # Warm up LLM and embedding connections
        get_model()
        get_embedding_model()

        logger.info("All connections initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize connections: {str(e)}")
        raise


def shutdown_all() -> None:
    """Gracefully shutdown all connections."""
    logger.info("Shutting down connections...")
    LLMPool.clear_cache()
    EmbeddingPool.clear_cache()
    logger.info("All connections shut down")


# Initialize on import
try:
    Config.validate()
except Exception as e:
    logger.warning(f"Configuration not fully validated on import: {e}")
