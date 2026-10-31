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

