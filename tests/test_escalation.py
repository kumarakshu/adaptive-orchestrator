"""Tests for escalation manager."""

import pytest
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.escalation import EscalationManager
from app.orchestrator.validator import ValidationResult


@pytest.fixture
def tiered_registry() -> ModelRegistry:
    """Fixture with models across all tiers."""
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="tier-1",
            provider="test-provider",
            tier=ModelTier.LIGHTWEIGHT,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="tier-2",
            provider="test-provider",
            tier=ModelTier.BALANCED,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="tier-3",
            provider="test-provider",
            tier=ModelTier.ADVANCED,
        )
    )
    return registry


def test_escalation_tier_ladder(tiered_registry: ModelRegistry):
    """Verify tier ladder progression."""
    manager = EscalationManager(registry=tiered_registry)
    assert manager.get_next_tier(ModelTier.LIGHTWEIGHT) == ModelTier.BALANCED
    assert manager.get_next_tier(ModelTier.BALANCED) == ModelTier.ADVANCED
    assert manager.get_next_tier(ModelTier.ADVANCED) is None


def test_escalation_decision(tiered_registry: ModelRegistry):
    """Verify escalation decision when validation demands it."""
    manager = EscalationManager(registry=tiered_registry)
    current_model = tiered_registry.get("tier-1")
    assert current_model is not None

    validation_failure = ValidationResult(
        is_valid=False,
        confidence_score=0.2,
        requires_escalation=True,
        rejection_reasons=["Low confidence"],
    )

    decision = manager.decide_escalation(
        current_model=current_model,
        validation=validation_failure,
        attempt_count=0,
    )

    assert decision.should_escalate is True
    assert decision.next_model is not None
    assert decision.next_model.model_id == "tier-2"


def test_escalation_skips_unavailable_providers():
    """Verify escalation never selects a model whose provider is unconfigured."""
    from app.orchestrator.executor import ModelExecutor

    registry = ModelRegistry()
    registry.register(ModelDescriptor(model_id="gemini-flash", provider="google", tier=ModelTier.LIGHTWEIGHT))
    registry.register(ModelDescriptor(model_id="gpt-4o", provider="openai", tier=ModelTier.BALANCED))
    registry.register(ModelDescriptor(model_id="o1", provider="openai", tier=ModelTier.ADVANCED))

    # Only Google is configured in the executor
    executor = ModelExecutor()
    executor.register_provider("google", lambda prompt, model: "OK")

    manager = EscalationManager(registry=registry, executor=executor)
    current_model = registry.get("gemini-flash")

    validation_failure = ValidationResult(
        is_valid=False,
        confidence_score=0.1,
        requires_escalation=True,
        rejection_reasons=["Upstream timeout"],
    )

    decision = manager.decide_escalation(
        current_model=current_model,
        validation=validation_failure,
        attempt_count=0,
    )

    # Must NOT escalate to gpt-4o or o1
    assert decision.should_escalate is False
    assert decision.next_model is None
    assert "No configured fallback" in decision.reason


def test_escalation_skips_intermediate_tier_if_unconfigured():
    """Verify escalation skips an unconfigured tier if a higher tier has an available provider."""
    from app.orchestrator.executor import ModelExecutor

    registry = ModelRegistry()
    registry.register(ModelDescriptor(model_id="gemini-flash", provider="google", tier=ModelTier.LIGHTWEIGHT))
    registry.register(ModelDescriptor(model_id="gpt-4o", provider="openai", tier=ModelTier.BALANCED))
    registry.register(ModelDescriptor(model_id="gemini-pro-adv", provider="google", tier=ModelTier.ADVANCED))

    executor = ModelExecutor()
    executor.register_provider("google", lambda prompt, model: "OK")

    manager = EscalationManager(registry=registry, executor=executor)
    current_model = registry.get("gemini-flash")

    validation_failure = ValidationResult(
        is_valid=False,
        confidence_score=0.1,
        requires_escalation=True,
    )

    decision = manager.decide_escalation(
        current_model=current_model,
        validation=validation_failure,
        attempt_count=0,
    )

    # Skips unconfigured BALANCED tier (gpt-4o) and selects ADVANCED tier (gemini-pro-adv)
    assert decision.should_escalate is True
    assert decision.next_model is not None
    assert decision.next_model.model_id == "gemini-pro-adv"

