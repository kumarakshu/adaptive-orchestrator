"""Unit tests for the intelligent model router."""

import pytest
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.analyzer import (
    AnalysisResult,
    ComplexityLevel,
    PromptAnalyzer,
    TaskType,
)
from app.orchestrator.router import (
    IntelligentRouter,
    NoSuitableModelError,
    RoutingDecision,
)


@pytest.fixture
def configured_registry() -> ModelRegistry:
    """Fixture providing an explicitly configured test registry without real dependencies."""
    registry = ModelRegistry()

    # Lightweight tier
    registry.register(
        ModelDescriptor(
            model_id="mock-lightweight",
            provider="mock-provider",
            tier=ModelTier.LIGHTWEIGHT,
            context_window=4096,
            supported_capabilities=["chat", "general"],
            enabled=True,
        )
    )

    # Balanced tier (general and coder)
    registry.register(
        ModelDescriptor(
            model_id="mock-balanced-general",
            provider="mock-provider",
            tier=ModelTier.BALANCED,
            context_window=8192,
            supported_capabilities=["chat", "summarization"],
            enabled=True,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="mock-balanced-coder",
            provider="mock-provider",
            tier=ModelTier.BALANCED,
            context_window=8192,
            supported_capabilities=["chat", "code", "tools"],
            enabled=True,
        )
    )

    # Advanced tier
    registry.register(
        ModelDescriptor(
            model_id="mock-advanced-reasoner",
            provider="mock-provider",
            tier=ModelTier.ADVANCED,
            context_window=16384,
            supported_capabilities=["chat", "code", "reasoning"],
            enabled=True,
        )
    )

    # Explicitly disabled model for exclusion testing
    registry.register(
        ModelDescriptor(
            model_id="mock-disabled-tier1",
            provider="mock-provider",
            tier=ModelTier.LIGHTWEIGHT,
            context_window=32768,
            supported_capabilities=["chat", "code"],
            enabled=False,
        )
    )

    return registry


# 1. Existing baseline tests preserved
def test_router_select_tier(configured_registry: ModelRegistry):
    """Verify tier selection logic for simple vs complex queries."""
    router = IntelligentRouter(registry=configured_registry)

    simple_analysis = AnalysisResult(requires_reasoning=False)
    assert router.select_tier(simple_analysis) == ModelTier.LIGHTWEIGHT

    complex_analysis = AnalysisResult(requires_reasoning=True)
    assert router.select_tier(complex_analysis) == ModelTier.ADVANCED


def test_router_route_model(configured_registry: ModelRegistry):
    """Verify routing decision contains correct candidate identifier."""
    router = IntelligentRouter(registry=configured_registry)

    analysis = AnalysisResult(requires_reasoning=False)
    selected = router.route(analysis)
    assert selected is not None
    assert selected.model_id == "mock-lightweight"


# 2. Simple query selects lightweight model
def test_simple_query_routes_to_lightweight(configured_registry: ModelRegistry):
    """Verify a simple factual inquiry routes to lightweight tier without fallback."""
    router = IntelligentRouter(registry=configured_registry)
    decision = router.route("What is the capital of France?")

    assert isinstance(decision, RoutingDecision)
    assert decision.selected_tier == ModelTier.LIGHTWEIGHT
    assert decision.preferred_tier == ModelTier.LIGHTWEIGHT
    assert decision.fallback_used is False
    assert decision.model_id == "mock-lightweight"
    assert "Routed to preferred tier 'lightweight'" in decision.routing_reason


# 3. Medium query selects balanced model
def test_medium_query_routes_to_balanced(configured_registry: ModelRegistry):
    """Verify medium difficulty query routes to balanced tier."""
    router = IntelligentRouter(registry=configured_registry)
    query = "Write a Python script with def calculate_average(numbers): return sum(numbers) / len(numbers)"
    decision = router.route(query)

    assert decision.selected_tier == ModelTier.BALANCED
    assert decision.preferred_tier == ModelTier.BALANCED
    assert decision.fallback_used is False
    assert decision.model_id in ["mock-balanced-general", "mock-balanced-coder"]


# 4. Complex query selects advanced model
def test_complex_query_routes_to_advanced(configured_registry: ModelRegistry):
    """Verify complex multi-step prompt routes to advanced tier."""
    router = IntelligentRouter(registry=configured_registry)
    query = """
    Step 1: Refactor the auth middleware for async concurrency.
    Step 2: Optimize algorithm performance.
    Constraints:
    - Must not use external packages.
    - Strictly keep memory footprint below 10MB.
    """
    decision = router.route(query)

    assert decision.selected_tier == ModelTier.ADVANCED
    assert decision.preferred_tier == ModelTier.ADVANCED
    assert decision.fallback_used is False
    assert decision.model_id == "mock-advanced-reasoner"


# 5. Coding tasks prioritize code-capable models
def test_coding_task_capability_matching(configured_registry: ModelRegistry):
    """Verify coding requests prioritize models with declared 'code' capability."""
    router = IntelligentRouter(registry=configured_registry)
    # Use medium coding prompt to target BALANCED tier, where we have general vs coder
    coding_query = "Write a Python function with def parse_data(csv_str):"
    decision = router.route(coding_query)

    assert decision.selected_tier == ModelTier.BALANCED
    assert decision.model_id == "mock-balanced-coder"
    assert "code" in decision.matched_capabilities


# 6. Context window limit exclusion
def test_context_window_limit_exclusion():
    """Verify models that cannot accommodate the prompt token estimate are excluded."""
    small_context_registry = ModelRegistry()
    small_context_registry.register(
        ModelDescriptor(
            model_id="tiny-context-model",
            provider="mock-provider",
            tier=ModelTier.LIGHTWEIGHT,
            context_window=10,  # Far too small for a long prompt
            enabled=True,
        )
    )
    small_context_registry.register(
        ModelDescriptor(
            model_id="large-context-model",
            provider="mock-provider",
            tier=ModelTier.BALANCED,
            context_window=2000,
            enabled=True,
        )
    )

    router = IntelligentRouter(registry=small_context_registry)
    # Long prompt exceeding 10 tokens
    prompt = "This is a prompt with significantly more words than ten tokens can handle for context limits."
    decision = router.route(prompt)

    # The tiny model is excluded; router falls back to balanced
    assert decision.model_id == "large-context-model"
    assert decision.selected_tier == ModelTier.BALANCED


# 7. Disabled models are never selected
def test_disabled_models_never_selected(configured_registry: ModelRegistry):
    """Verify models with enabled=False are strictly excluded from routing."""
    router = IntelligentRouter(registry=configured_registry)
    for _ in range(5):
        decision = router.route("Hello world")
        assert decision.model_id != "mock-disabled-tier1"
        assert decision.model.enabled is True


# 8. Fallback mechanism when preferred tier is unavailable
def test_fallback_when_preferred_tier_has_no_models():
    """Verify fallback ladder selects closest appropriate tier when preferred tier is empty."""
    tier2_only_registry = ModelRegistry()
    tier2_only_registry.register(
        ModelDescriptor(
            model_id="only-balanced-model",
            provider="mock-provider",
            tier=ModelTier.BALANCED,
            context_window=4096,
            enabled=True,
        )
    )

    router = IntelligentRouter(registry=tier2_only_registry)
    # Simple query would prefer LIGHTWEIGHT, but none exist
    decision = router.route("What time is it?")

    assert decision.preferred_tier == ModelTier.LIGHTWEIGHT
    assert decision.selected_tier == ModelTier.BALANCED
    assert decision.fallback_used is True
    assert decision.model_id == "only-balanced-model"
    assert "Fallback applied" in decision.routing_reason


# 9. No suitable model raises NoSuitableModelError
def test_no_suitable_model_raises_exception():
    """Verify NoSuitableModelError is raised when no enabled candidate satisfies constraints."""
    empty_registry = ModelRegistry()
    router = IntelligentRouter(registry=empty_registry)

    with pytest.raises(NoSuitableModelError, match="No enabled model found"):
        router.route("Any prompt")


# 10. Supplied AnalysisResult is reused without re-analysis
def test_supplied_analysis_reused(configured_registry: ModelRegistry):
    """Verify pre-computed AnalysisResult is reused directly."""
    router = IntelligentRouter(registry=configured_registry)
    custom_analysis = AnalysisResult(
        complexity=ComplexityLevel.COMPLEX,
        task_type=TaskType.CODING,
        estimated_input_tokens=100,
        requires_reasoning=True,
    )

    decision = router.route(query="Ignored dummy text", analysis=custom_analysis)

    assert decision.original_complexity == ComplexityLevel.COMPLEX
    assert decision.selected_tier == ModelTier.ADVANCED
    assert decision.model_id == "mock-advanced-reasoner"


# 11. Async routing interface
@pytest.mark.asyncio
async def test_route_async_interface(configured_registry: ModelRegistry):
    """Verify route_async returns valid RoutingDecision when awaited."""
    router = IntelligentRouter(registry=configured_registry)
    decision = await router.route_async("What is the capital of Spain?")

    assert isinstance(decision, RoutingDecision)
    assert decision.selected_tier == ModelTier.LIGHTWEIGHT
    assert decision.model_id == "mock-lightweight"
