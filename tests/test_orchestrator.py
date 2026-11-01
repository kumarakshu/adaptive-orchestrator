"""Unit tests for the central orchestration engine."""

import pytest
from app.core.context import InMemoryContextStore
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.analyzer import (
    AnalysisResult,
    ComplexityLevel,
    PromptAnalyzer,
    TaskType,
)
from app.orchestrator.engine import OrchestrationEngine, OrchestrationResult
from app.orchestrator.executor import ModelExecutor
from app.orchestrator.router import IntelligentRouter


@pytest.fixture
def mock_registry() -> ModelRegistry:
    """Fixture providing a populated test model registry."""
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="mock-fast-model",
            provider="mock-provider",
            tier=ModelTier.LIGHTWEIGHT,
            context_window=4096,
            supported_capabilities=["chat"],
            enabled=True,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="mock-advanced-model",
            provider="mock-provider",
            tier=ModelTier.ADVANCED,
            context_window=16384,
            supported_capabilities=["chat", "code", "reasoning"],
            enabled=True,
        )
    )
    return registry


@pytest.fixture
def mock_executor() -> ModelExecutor:
    """Fixture providing a configured ModelExecutor with a test provider."""
    def fake_provider(prompt, model, context=None):
        return f"Response to '{prompt}' from {model.model_id}"

    executor = ModelExecutor()
    executor.register_provider("mock-provider", fake_provider)
    return executor


# 1. Engine initialization
def test_engine_initialization(mock_registry: ModelRegistry):
    """Verify OrchestrationEngine instantiates with default subcomponents."""
    engine = OrchestrationEngine(registry=mock_registry)
    assert engine.registry == mock_registry
    assert engine.analyzer is not None
    assert engine.router is not None
    assert engine.validator is not None
    assert engine.escalation is not None
    assert engine.executor is None


# 2. Successful end-to-end flow: analyze -> route -> execute
@pytest.mark.asyncio
async def test_engine_successful_flow(mock_registry: ModelRegistry, mock_executor: ModelExecutor):
    """Verify end-to-end execution flow: analyze, route, and execute."""
    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=mock_executor,
    )

    prompt = "What is the speed of sound?"
    result = await engine.orchestrate(prompt)

    assert isinstance(result, OrchestrationResult)
    assert result.success is True
    assert result.error is None
    assert result.analysis is not None
    assert result.analysis.complexity == ComplexityLevel.SIMPLE
    assert result.routing is not None
    assert result.routing.selected_tier == ModelTier.LIGHTWEIGHT
    assert result.routing.model_id == "mock-fast-model"
    assert result.execution is not None
    assert result.execution.success is True
    assert "Response to 'What is the speed of sound?' from mock-fast-model" in result.content
    assert result.model_id == "mock-fast-model"
    assert result.selected_tier == ModelTier.LIGHTWEIGHT


# 3. Supplied analysis is reused
@pytest.mark.asyncio
async def test_engine_supplied_analysis_reused(mock_registry: ModelRegistry, mock_executor: ModelExecutor):
    """Verify precomputed analysis is reused without re-analysis."""
    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=mock_executor,
    )

    # Supply an explicit complex analysis
    precomputed_analysis = AnalysisResult(
        complexity=ComplexityLevel.COMPLEX,
        task_type=TaskType.CODING,
        estimated_input_tokens=50,
        requires_reasoning=True,
    )

    result = await engine.orchestrate(
        prompt="Sample prompt",
        analysis=precomputed_analysis,
    )

    assert result.success is True
    assert result.analysis == precomputed_analysis
    assert result.routing.selected_tier == ModelTier.ADVANCED
    assert result.model_id == "mock-advanced-model"


# 4. Empty or invalid prompt produces clear error
@pytest.mark.asyncio
async def test_engine_empty_prompt_raises_value_error(mock_registry: ModelRegistry):
    """Verify empty prompt raises ValueError."""
    engine = OrchestrationEngine(registry=mock_registry)

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        await engine.orchestrate("")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        await engine.orchestrate("   \t\n  ")


# 5. Missing executor produces structured failure
@pytest.mark.asyncio
async def test_engine_missing_executor_returns_failure(mock_registry: ModelRegistry):
    """Verify engine without an executor returns failure cleanly."""
    engine = OrchestrationEngine(registry=mock_registry, executor=None)

    result = await engine.orchestrate("Hello")

    assert result.success is False
    assert "No executor configured" in result.error
    assert result.routing is not None


# 6. Missing provider returns clear failure
@pytest.mark.asyncio
async def test_engine_missing_provider_returns_failure(mock_registry: ModelRegistry):
    """Verify missing provider in executor surfaces as structured failure."""
    empty_executor = ModelExecutor()  # Has no provider registered for 'mock-provider'
    engine = OrchestrationEngine(registry=mock_registry, executor=empty_executor)

    result = await engine.orchestrate("Hello")

    assert result.success is False
    assert "No provider configured" in result.error
    assert result.execution is not None
    assert result.execution.success is False


# 7. Provider exception is surfaced as structured failure
@pytest.mark.asyncio
async def test_engine_provider_exception_surfaced(mock_registry: ModelRegistry):
    """Verify underlying provider exception surfaces in result without unhandled crash."""
    def crashing_provider(prompt, model):
        raise RuntimeError("External provider rate limit exceeded (HTTP 429)")

    failing_executor = ModelExecutor()
    failing_executor.register_provider("mock-provider", crashing_provider)

    engine = OrchestrationEngine(registry=mock_registry, executor=failing_executor)
    result = await engine.orchestrate("Explain relativity")

    assert result.success is False
    assert "External provider rate limit exceeded" in result.error
    assert result.execution is not None
    assert result.execution.success is False


# 8. Run method alias compatibility
@pytest.mark.asyncio
async def test_engine_run_alias(mock_registry: ModelRegistry, mock_executor: ModelExecutor):
    """Verify engine.run() acts as standard entry point."""
    engine = OrchestrationEngine(registry=mock_registry, executor=mock_executor)

    result = await engine.run("Ping test")

    assert result.success is True
    assert result.content == "Response to 'Ping test' from mock-fast-model"


# 9. Multi-tier Escalation Integration Tests
@pytest.fixture
def multi_tier_registry() -> ModelRegistry:
    """Fixture with models across all 3 tiers."""
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="fast-model",
            provider="fast-provider",
            tier=ModelTier.LIGHTWEIGHT,
            context_window=4096,
            supported_capabilities=["chat"],
            enabled=True,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="balanced-model",
            provider="balanced-provider",
            tier=ModelTier.BALANCED,
            context_window=8192,
            supported_capabilities=["chat", "code"],
            enabled=True,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="advanced-model",
            provider="advanced-provider",
            tier=ModelTier.ADVANCED,
            context_window=16384,
            supported_capabilities=["chat", "code", "reasoning"],
            enabled=True,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_orchestration_first_tier_success(multi_tier_registry: ModelRegistry):
    """Verify first-tier success does not trigger escalation and records metadata."""
    executor = ModelExecutor()
    executor.register_provider("fast-provider", lambda prompt, model: "Fast response")

    engine = OrchestrationEngine(registry=multi_tier_registry, executor=executor)
    result = await engine.orchestrate("What is 2+2?")

    assert result.success is True
    assert result.model_id == "fast-model"
    assert result.selected_tier == ModelTier.LIGHTWEIGHT
    assert result.content == "Fast response"
    assert result.metadata["attempt_count"] == 0
    assert len(result.metadata["escalation_history"]) == 0
    assert result.metadata["validation"]["is_valid"] is True


@pytest.mark.asyncio
async def test_orchestration_escalates_on_validation_failure(multi_tier_registry: ModelRegistry):
    """Verify engine escalates to next tier when response validation fails (e.g. empty content)."""
    executor = ModelExecutor()
    # Fast model returns empty content (triggering validation failure)
    executor.register_provider("fast-provider", lambda prompt, model: "   ")
    # Balanced model returns valid content
    executor.register_provider("balanced-provider", lambda prompt, model: "Balanced response")

    engine = OrchestrationEngine(registry=multi_tier_registry, executor=executor)
    result = await engine.orchestrate("What is 2+2?")

    assert result.success is True
    assert result.model_id == "balanced-model"
    assert result.selected_tier == ModelTier.BALANCED
    assert result.content == "Balanced response"
    assert result.metadata["attempt_count"] == 1
    assert len(result.metadata["escalation_history"]) == 1
    escalation_entry = result.metadata["escalation_history"][0]
    assert escalation_entry["from_model"] == "fast-model"
    assert escalation_entry["to_model"] == "balanced-model"
    assert escalation_entry["from_tier"] == "lightweight"
    assert escalation_entry["to_tier"] == "balanced"


@pytest.mark.asyncio
async def test_orchestration_escalates_on_provider_error(multi_tier_registry: ModelRegistry):
    """Verify engine escalates to next tier when underlying provider crashes."""
    def crashing_provider(prompt, model):
        raise ConnectionResetError("Fast provider network dropped")

    executor = ModelExecutor()
    executor.register_provider("fast-provider", crashing_provider)
    executor.register_provider("balanced-provider", lambda prompt, model: "Balanced recovery")

    engine = OrchestrationEngine(registry=multi_tier_registry, executor=executor)
    result = await engine.orchestrate("Explain photosynthesis")

    assert result.success is True
    assert result.model_id == "balanced-model"
    assert result.selected_tier == ModelTier.BALANCED
    assert result.content == "Balanced recovery"
    assert result.metadata["attempt_count"] == 1
    assert len(result.metadata["escalation_history"]) == 1


@pytest.mark.asyncio
async def test_orchestration_enforces_max_retries(multi_tier_registry: ModelRegistry):
    """Verify engine terminates after reaching max_retries without looping infinitely."""
    executor = ModelExecutor()
    # Both fast and balanced return empty content
    executor.register_provider("fast-provider", lambda prompt, model: "")
    executor.register_provider("balanced-provider", lambda prompt, model: "")
    executor.register_provider("advanced-provider", lambda prompt, model: "Advanced response")

    from app.orchestrator.escalation import EscalationManager

    # Limit escalation retries to 1
    escalation = EscalationManager(registry=multi_tier_registry, max_retries=1)
    engine = OrchestrationEngine(
        registry=multi_tier_registry,
        executor=executor,
        escalation=escalation,
    )

    result = await engine.orchestrate("What is 2+2?")

    # After 1 retry (fast-model -> balanced-model), attempt_count reaches 1, halting escalation
    assert result.success is False
    assert result.model_id == "balanced-model"
    assert result.metadata["attempt_count"] == 1
    assert "Empty response content" in result.error


@pytest.mark.asyncio
async def test_orchestration_stops_at_highest_tier(multi_tier_registry: ModelRegistry):
    """Verify engine does not escalate if initial route is already at highest tier."""
    executor = ModelExecutor()
    executor.register_provider("advanced-provider", lambda prompt, model: "")

    engine = OrchestrationEngine(registry=multi_tier_registry, executor=executor)

    # Supply an explicit complex analysis that routes to ADVANCED tier
    complex_analysis = AnalysisResult(
        complexity=ComplexityLevel.COMPLEX,
        task_type=TaskType.CODING,
        estimated_input_tokens=100,
        requires_reasoning=True,
    )

    result = await engine.orchestrate(
        prompt="Write a distributed consensus algorithm",
        analysis=complex_analysis,
    )

    assert result.success is False
    assert result.model_id == "advanced-model"
    assert result.selected_tier == ModelTier.ADVANCED
    assert result.metadata["attempt_count"] == 0
    assert len(result.metadata["escalation_history"]) == 0


@pytest.mark.asyncio
async def test_orchestration_deduplication_prevents_duplicate_attempts():
    """Verify engine deduplicates model attempts within the escalation loop."""
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="fast-1",
            provider="test-p",
            tier=ModelTier.LIGHTWEIGHT,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="balanced-primary",
            provider="test-p",
            tier=ModelTier.BALANCED,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="balanced-secondary",
            provider="test-p",
            tier=ModelTier.BALANCED,
        )
    )

    call_counts = {"fast-1": 0, "balanced-primary": 0, "balanced-secondary": 0}

    def tracking_provider(prompt, model):
        call_counts[model.model_id] += 1
        if model.model_id == "fast-1":
            return ""  # Fails validation
        if model.model_id == "balanced-primary":
            return "Balanced primary success"
        return "Balanced secondary success"

    executor = ModelExecutor()
    executor.register_provider("test-p", tracking_provider)

    engine = OrchestrationEngine(registry=registry, executor=executor)
    result = await engine.orchestrate("Test query")

    assert result.success is True
    assert result.model_id == "balanced-primary"
    assert call_counts["fast-1"] == 1
    assert call_counts["balanced-primary"] == 1
    assert call_counts["balanced-secondary"] == 0


@pytest.mark.asyncio
async def test_orchestrator_checkpoint_lifecycle_transitions(mock_registry: ModelRegistry, mock_executor: ModelExecutor):
    """Verify engine transitions through started, executing, and completed checkpoints."""
    store = InMemoryContextStore()
    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=mock_executor,
        context_store=store,
    )

    result = await engine.orchestrate(
        prompt="What is photosynthesis?",
        task_id="task-orch-life-1",
    )

    assert result.success is True
    assert result.metadata["task_id"] == "task-orch-life-1"
    assert result.metadata["checkpoint_status"] == "completed"

    cp = await store.get_checkpoint("task-orch-life-1")
    assert cp is not None
    assert cp["status"] == "completed"
    assert cp["prompt"] == "What is photosynthesis?"
    assert cp["last_model_id"] == "mock-fast-model"
    assert "Response to 'What is photosynthesis?'" in cp["result"]["content"]


@pytest.mark.asyncio
async def test_orchestrator_resume_after_interruption(mock_registry: ModelRegistry, mock_executor: ModelExecutor):
    """Verify engine safely resumes an interrupted task from its last checkpoint state."""
    store = InMemoryContextStore()
    # Simulate a task interrupted after escalating to advanced model
    await store.save_checkpoint("task-res-1", {
        "task_id": "task-res-1",
        "status": "escalated",
        "prompt": "Explain quantum computing",
        "attempt_count": 1,
        "attempted_model_ids": ["mock-fast-model"],
        "last_model_id": "mock-advanced-model",
        "escalation_history": [
            {
                "from_model": "mock-fast-model",
                "to_model": "mock-advanced-model",
                "reason": "Escalation triggered",
            }
        ],
        "checkpoint_version": 1,
        "updated_at": 100.0,
    })

    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=mock_executor,
        context_store=store,
    )

    result = await engine.orchestrate(
        prompt="Explain quantum computing",
        task_id="task-res-1",
    )

    assert result.success is True
    assert result.model_id == "mock-advanced-model"
    assert result.metadata.get("resumed_from_checkpoint") is True
    assert result.metadata.get("task_id") == "task-res-1"
    # Ensure completed checkpoint is saved
    cp = await store.get_checkpoint("task-res-1")
    assert cp["status"] == "completed"


@pytest.mark.asyncio
async def test_completed_task_checkpoint_cannot_be_rerun(mock_registry: ModelRegistry):
    """Verify completed tasks return cached results deterministically without rerunning provider."""
    store = InMemoryContextStore()
    await store.save_checkpoint("task-done-1", {
        "task_id": "task-done-1",
        "status": "completed",
        "prompt": "Cached question",
        "attempt_count": 0,
        "attempted_model_ids": ["mock-fast-model"],
        "checkpoint_version": 1,
        "updated_at": 100.0,
        "result": {
            "content": "Deterministic cached answer",
            "model_id": "mock-fast-model",
            "selected_tier": "lightweight",
        },
    })

    # Executor that would raise if called
    def failing_provider(prompt, model, context=None):
        raise AssertionError("Executor should never be called for already-completed task!")

    executor = ModelExecutor()
    executor.register_provider("mock-provider", failing_provider)

    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=executor,
        context_store=store,
    )

    result = await engine.orchestrate(
        prompt="Cached question",
        task_id="task-done-1",
    )

    assert result.success is True
    assert result.content == "Deterministic cached answer"
    assert result.metadata.get("resumed_from_checkpoint") is True


@pytest.mark.asyncio
async def test_malformed_and_unsupported_checkpoint_version_handled_safely(mock_registry: ModelRegistry, mock_executor: ModelExecutor):
    """Verify unsupported checkpoint versions do not crash the engine and proceed safely."""
    store = InMemoryContextStore()
    # Corrupt version
    await store.save_checkpoint("task-bad-ver", {
        "task_id": "task-bad-ver",
        "status": "executing",
        "checkpoint_version": 9999,
        "prompt": "Hello",
    })

    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=mock_executor,
        context_store=store,
    )

    result = await engine.orchestrate(
        prompt="Hello",
        task_id="task-bad-ver",
    )
    assert result.success is True
    assert "Response to 'Hello'" in result.content


@pytest.mark.asyncio
async def test_backward_compatible_when_task_id_omitted(mock_registry: ModelRegistry, mock_executor: ModelExecutor):
    """Verify that when task_id is omitted, execution is normal and no checkpoint is created."""
    store = InMemoryContextStore()
    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=mock_executor,
        context_store=store,
    )

    result = await engine.orchestrate(prompt="Stateless query")
    assert result.success is True
    assert len(store._checkpoints) == 0


@pytest.mark.asyncio
async def test_orchestration_gemini_only_fails_gracefully_without_escalating_to_unconfigured():
    """Verify engine with only Google/Gemini configured fails gracefully without attempting unconfigured OpenAI/Anthropic."""
    # Use real ModelRegistry with all standard models
    registry = ModelRegistry.with_default_models()
    executor = ModelExecutor()

    call_records = []

    async def mock_failing_gemini(prompt: str, model, context=None):
        call_records.append((model.provider, model.model_id))
        raise RuntimeError("Gemini request timed out after 30s")

    executor.register_provider("google", mock_failing_gemini)
    store = InMemoryContextStore()

    engine = OrchestrationEngine(
        registry=registry,
        executor=executor,
        context_store=store,
    )

    result = await engine.orchestrate(
        prompt="Explain quantum physics",
        task_id="task-gemini-fail-1",
    )

    assert result.success is False
    # Ensure ONLY google was called, never openai or anthropic
    for provider, model_id in call_records:
        assert provider == "google", f"Unexpected call to unconfigured provider {provider} ({model_id})"

    # Verify no unconfigured model was escalated to
    assert "No configured fallback models available in higher tiers" in result.error

    # Verify checkpoint is marked failed
    cp = await store.get_checkpoint("task-gemini-fail-1")
    assert cp["status"] == "failed"


@pytest.mark.asyncio
async def test_orchestration_complex_query_falls_back_gracefully_when_only_gemini_configured():
    """Verify complex query preferring advanced tier falls back to Gemini and does not falsely claim escalation."""
    registry = ModelRegistry.with_default_models()
    executor = ModelExecutor()

    call_records = []

    async def mock_gemini(prompt: str, model, context=None):
        call_records.append((model.provider, model.model_id))
        return {
            "content": "Fintech architecture design with event-driven ledger",
            "model_id": model.model_id,
            "prompt_tokens": 50,
            "completion_tokens": 100,
        }

    executor.register_provider("google", mock_gemini)

    engine = OrchestrationEngine(
        registry=registry,
        executor=executor,
    )

    result = await engine.orchestrate(
        prompt="Design a high-availability fintech architecture for real-time transaction processing with microservices and ACID compliance",
    )

    assert result.success is True
    assert result.analysis.task_type.value == "coding"
    assert result.analysis.complexity.value == "complex"
    assert result.routing.preferred_tier.value == "advanced"
    assert result.routing.fallback_used is True
    assert result.model_id == "gemini-3.8-flash"
    assert result.selected_tier.value == "lightweight"
    assert result.metadata.get("escalated") is False
    assert len(result.metadata.get("escalation_history", [])) == 0
    assert len(call_records) == 1
    assert call_records[0] == ("google", "gemini-3.8-flash")


@pytest.mark.asyncio
async def test_failed_task_checkpoint_cannot_be_rerun_requires_fresh_task_id(mock_registry: ModelRegistry):
    """Verify terminally failed tasks cannot be rerun with the same task_id."""
    store = InMemoryContextStore()
    await store.save_checkpoint("task-failed-prior", {
        "task_id": "task-failed-prior",
        "status": "failed",
        "prompt": "Broken query",
        "attempt_count": 2,
        "attempted_model_ids": ["mock-fast-model"],
        "checkpoint_version": 1,
        "updated_at": 100.0,
        "error": "Original fatal failure",
    })

    def failing_provider(prompt, model, context=None):
        raise AssertionError("Executor should never be called for already-failed task!")

    executor = ModelExecutor()
    executor.register_provider("mock-provider", failing_provider)

    engine = OrchestrationEngine(
        registry=mock_registry,
        executor=executor,
        context_store=store,
    )

    result = await engine.orchestrate(
        prompt="Broken query",
        task_id="task-failed-prior",
    )

    assert result.success is False
    assert "A fresh task_id is required" in result.error
    assert result.metadata.get("resumed_from_checkpoint") is True



