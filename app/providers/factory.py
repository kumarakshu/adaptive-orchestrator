"""Provider factory and registration utilities for Adaptive LLM Orchestrator."""

from typing import Dict, List, Optional
import httpx

from app.core.config import Settings, get_settings
from app.orchestrator.executor import ModelExecutor, ModelProvider
from app.providers.anthropic import AnthropicProvider
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider


def create_configured_providers(
    settings: Optional[Settings] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, ModelProvider]:
    """Create provider instances for all providers that have API keys configured.

    Returns a dict mapping registry provider names ('openai', 'anthropic', 'google')
    to instantiated provider adapters.
    """
    cfg = settings or get_settings()
    providers: Dict[str, ModelProvider] = {}
    timeout = float(cfg.default_timeout_seconds)

    if cfg.openai_api_key and cfg.openai_api_key.strip():
        providers["openai"] = OpenAIProvider(
            api_key=cfg.openai_api_key,
            client=client,
            timeout_seconds=timeout,
        )

    if cfg.anthropic_api_key and cfg.anthropic_api_key.strip():
        providers["anthropic"] = AnthropicProvider(
            api_key=cfg.anthropic_api_key,
            client=client,
            timeout_seconds=timeout,
        )

    if cfg.gemini_api_key and cfg.gemini_api_key.strip():
        providers["google"] = GeminiProvider(
            api_key=cfg.gemini_api_key,
            client=client,
            timeout_seconds=timeout,
        )

    return providers


def register_configured_providers(
    executor: ModelExecutor,
    settings: Optional[Settings] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> List[str]:
    """Register all configured providers into a ModelExecutor instance.

    Only providers with valid non-empty API keys are registered.

    Returns:
        List of registered provider keys (e.g. ['openai', 'anthropic', 'google']).
    """
    configured = create_configured_providers(settings=settings, client=client)
    for provider_name, provider in configured.items():
        executor.register_provider(provider_name, provider)

    return list(configured.keys())

