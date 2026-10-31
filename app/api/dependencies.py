"""Dependency injection providers for FastAPI endpoints."""

from typing import Optional
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.context import BaseContextStore, create_context_store
from app.models.registry import ModelRegistry
from app.orchestrator.analyzer import PromptAnalyzer
from app.orchestrator.engine import OrchestrationEngine
from app.orchestrator.executor import ModelExecutor
from app.orchestrator.router import IntelligentRouter
from app.providers.factory import register_configured_providers

# Shared singleton references (lazy loaded)
_registry_instance: Optional[ModelRegistry] = None
_executor_instance: Optional[ModelExecutor] = None
_engine_instance: Optional[OrchestrationEngine] = None
_context_store_instance: Optional[BaseContextStore] = None


def reset_dependencies() -> None:
    """Reset shared singleton instances (primarily for testing and isolation)."""
    global _registry_instance, _executor_instance, _engine_instance, _context_store_instance
    _registry_instance = None
    _executor_instance = None
    _engine_instance = None
    _context_store_instance = None


async def get_model_registry() -> ModelRegistry:
    """Return the shared ModelRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry.with_default_models()
    return _registry_instance


async def get_model_executor() -> ModelExecutor:
    """Return the shared ModelExecutor instance with configured providers registered."""
    global _executor_instance
    if _executor_instance is None:
        executor = ModelExecutor()
        settings = get_settings()
        register_configured_providers(executor=executor, settings=settings)
        _executor_instance = executor
    return _executor_instance


async def get_prompt_analyzer() -> PromptAnalyzer:
    """Return a PromptAnalyzer instance."""
    return PromptAnalyzer()


async def get_intelligent_router(
    registry: ModelRegistry = Depends(get_model_registry),
    analyzer: PromptAnalyzer = Depends(get_prompt_analyzer),
) -> IntelligentRouter:
    """Return an IntelligentRouter instance with current registry."""
    return IntelligentRouter(registry=registry, analyzer=analyzer)


async def get_context_store() -> BaseContextStore:
    """Return the shared BaseContextStore instance configured with current settings."""
    global _context_store_instance
    if _context_store_instance is None:
        settings = get_settings()
        _context_store_instance = create_context_store(
            use_redis=settings.use_redis,
            redis_url=settings.redis_url,
            default_ttl_seconds=settings.context_ttl_seconds,
        )
    return _context_store_instance


async def get_orchestration_engine(
    registry: ModelRegistry = Depends(get_model_registry),
    executor: ModelExecutor = Depends(get_model_executor),
    analyzer: PromptAnalyzer = Depends(get_prompt_analyzer),
    router: IntelligentRouter = Depends(get_intelligent_router),
    context_store: BaseContextStore = Depends(get_context_store),
) -> OrchestrationEngine:
    """Return an OrchestrationEngine instance configured with current dependencies."""
    return OrchestrationEngine(
        registry=registry,
        executor=executor,
        analyzer=analyzer,
        router=router,
        context_store=context_store,
    )
