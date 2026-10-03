"""
Single source of truth for all configuration.
Uses pydantic-settings — reads from .env automatically.
"""

from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Environment ───────────────────────────────────────────────────────────
    env: str = "development"
    debug: bool = False
    app_name: str = "Coffee Shop Chatbot"
    app_url: str = "http://localhost:3000"
    api_port: int = 8000

    # ── Auth ──────────────────────────────────────────────────────────────────
    auth_provider: str = "supabase"           # "supabase" | "aws"
    secret_key: SecretStr = SecretStr("dev")  # set real value in prod
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_key: str = ""                    # anon key
    supabase_service_key: str = ""            # service role key (scripts only)

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Primary LLM model for generation
    llm_model: str = "openai/gpt-oss-20b"
    # Smaller fallback model for cheaper calls
    small_llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0
    # Switching from Groq to NVIDIA NIM – use the NIM endpoint/key
    nim_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    # Deprecated Groq fields left empty for backward compatibility
    groq_api_key: str = ""
    groq_model: str = ""
    llm_timeout_seconds: int = 60

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # must match pgvector index dimension (768)
    hf_api_key: str = ""

    # ── Pinecone (REMOVED: Supabase pgvector is the vector store) ─────────────
    # Fields retained as Optional so old .env files with PINECONE_* don't crash.
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "coffee-products"  # ignored at runtime

    # ── Retriever ─────────────────────────────────────────────────────────────
    retriever_default_top_k: int = 5
    retriever_max_top_k: int = 50
    retriever_timeout_seconds: int = 30
    retriever_retry_attempts: int = 3

    # ── Rate limiting ─────────────────────────────────────────────────────────
    enable_rate_limiting: bool = True
    requests_per_minute: int = 100

    # ── Monitoring ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    enable_request_logging: bool = True

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    langchain_api_key: str = ""
    langsmith_tracing_v2: str = "false"
    langsmith_project: str = "coffee-shop-chatbot"

    # ── External ──────────────────────────────────────────────────────────────
    telegram_token: str = ""

    # ── Email (SMTP) ──────────────────────────────────────────────────────────
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_email: str = "Coffee Shop <noreply@coffeeshop.com>"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_production(self) -> bool:
        return self.env.lower() in ("production", "prod")

    def is_development(self) -> bool:
        return self.env.lower() in ("development", "dev", "local")

    def validate_required(self) -> None:
        """Raise if critical keys are missing."""
        missing = [
            name for name, val in {
                "HF_API_KEY": self.hf_api_key,
                "SUPABASE_URL": self.supabase_url,
                "SUPABASE_KEY": self.supabase_key,
            }.items() if not val
        ]
        # Primary LLM provider is now NVIDIA NIM
        if not self.nim_api_key:
            missing.append("NIM_API_KEY")
        if missing:
            raise ValueError(f"Missing required env vars: {', '.join(missing)}")


# ── Singleton ─────────────────────────────────────────────────────────────────

settings = Settings()


# ── Backward-compatible shim for old src/config.py class-style access ─────────
# Anything that does `from src.config import Config` keeps working.

class Config:
    ENV = settings.env
    DEBUG = settings.debug
    APP_NAME = settings.app_name
    APP_URL = settings.app_url
    API_PORT = settings.api_port

    LLM_MODEL = settings.llm_model
    SMALL_LLM_MODEL = settings.small_llm_model
    LLM_TEMPERATURE = settings.llm_temperature
    # ── NVIDIA NIM (active LLM provider) ──────────────────────────────────────
    NIM_API_KEY = settings.nim_api_key
    NIM_BASE_URL = settings.nim_base_url
    LLM_TIMEOUT_SECONDS = settings.llm_timeout_seconds
    # ── Deprecated provider stubs (kept so old imports don't break) ───────────
    GROQ_API_KEY = settings.groq_api_key   # empty; Groq replaced by NIM
    GROQ_MODEL = settings.groq_model        # empty
    OPENROUTER_API_KEY = ""                 # removed from Settings
    OPENROUTER_BASE_URL = ""                # removed from Settings

    EMBEDDING_MODEL = settings.embedding_model
    HF_API_KEY = settings.hf_api_key

    # Pinecone removed; pgvector is the active vector store
    PINECONE_API_KEY = settings.pinecone_api_key   # None
    PINECONE_INDEX_NAME = settings.pinecone_index_name  # ignored

    RETRIEVER_DEFAULT_TOP_K = settings.retriever_default_top_k
    RETRIEVER_MAX_TOP_K = settings.retriever_max_top_k
    RETRIEVER_TIMEOUT_SECONDS = settings.retriever_timeout_seconds
    RETRIEVER_RETRY_ATTEMPTS = settings.retriever_retry_attempts

    ENABLE_RATE_LIMITING = settings.enable_rate_limiting
    REQUESTS_PER_MINUTE = settings.requests_per_minute
    RATE_LIMIT_ENABLED = settings.enable_rate_limiting

    SUPABASE_URL = settings.supabase_url
    SUPABASE_KEY = settings.supabase_key

    SMTP_HOST = settings.smtp_host
    SMTP_PORT = settings.smtp_port
    SMTP_USER = settings.smtp_user
    SMTP_PASSWORD = settings.smtp_password
    SMTP_FROM_EMAIL = settings.smtp_from_email

    HEALTH_CHECK_INTERVAL_SECONDS = 300
    CONNECTION_POOL_SIZE = 10
    USE_REDIS_CACHE = False
    REDIS_URL = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS = 3600

    @classmethod
    def validate(cls) -> None:
        settings.validate_required()


# Convenience sub-configs (used by retriever)
class RetrieverConfig:
    default_top_k = settings.retriever_default_top_k
    max_top_k = settings.retriever_max_top_k
    timeout_seconds = settings.retriever_timeout_seconds
    retry_attempts = settings.retriever_retry_attempts
