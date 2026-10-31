"""Application configuration settings."""

from functools import lru_cache
from typing import Any, List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Application
    app_name: str = Field(default="Adaptive LLM Orchestrator", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Security & CORS Configuration
    cors_allowed_origins: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("CORS_ALLOWED_ORIGINS", "ALLOWED_ORIGINS", "cors_allowed_origins"),
        description="Allowed CORS origins list or comma-separated string. Empty list by default.",
    )
    cors_allow_credentials: bool = Field(
        default=False,
        validation_alias=AliasChoices("CORS_ALLOW_CREDENTIALS", "cors_allow_credentials"),
        description="Whether cross-origin requests can include credentials (never combined with wildcard).",
    )

    # Redis Cache & Context Management
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    use_redis: bool = Field(
        default=False,
        validation_alias=AliasChoices("USE_REDIS", "use_redis"),
        description="Whether to enable Redis for conversation context persistence.",
    )
    context_ttl_seconds: int = Field(
        default=86400,
        validation_alias=AliasChoices("CONTEXT_TTL_SECONDS", "context_ttl_seconds"),
        description="TTL in seconds for stored conversation context.",
    )

    # Observability & Metrics
    metrics_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("METRICS_ENABLED", "metrics_enabled"),
        description="Whether to expose Prometheus /metrics and /api/v1/metrics.",
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
        description="Application logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    # Provider API Keys (injected via environment; never hardcoded)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")

    # Orchestrator Policies
    default_timeout_seconds: int = Field(default=30, alias="DEFAULT_TIMEOUT_SECONDS")
    gemini_timeout_seconds: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_TIMEOUT_SECONDS", "gemini_timeout_seconds"),
        description="Timeout in seconds for Gemini API requests (falls back to DEFAULT_TIMEOUT_SECONDS if unset).",
    )
    gemini_max_retries: int = Field(
        default=1,
        validation_alias=AliasChoices("GEMINI_MAX_RETRIES", "gemini_max_retries"),
        description="Maximum transient retries for Gemini requests (bounded).",
    )
    max_escalation_retries: int = Field(default=2, alias="MAX_ESCALATION_RETRIES")
    confidence_threshold: float = Field(default=0.75, alias="CONFIDENCE_THRESHOLD")

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> List[str]:
        """Normalize allowed origins from a list, tuple, comma-separated string, or JSON list."""
        if value is None:
            return []
        if isinstance(value, str):
            val = value.strip()
            if not val:
                return []
            if val.startswith("[") and val.endswith("]"):
                import json
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [origin.strip() for origin in val.split(",") if origin.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        return []


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()

