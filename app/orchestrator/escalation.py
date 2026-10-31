"""Model escalation management module.

Determines fallback strategies and next-tier model upgrades when
validation fails or confidence is below threshold.
"""

from typing import List, Optional
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.validator import ValidationResult


class EscalationDecision(BaseModel if False else object):
    """Container for escalation decisions."""

    def __init__(
        self,
        should_escalate: bool,
        next_model: Optional[ModelDescriptor] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.should_escalate = should_escalate
        self.next_model = next_model
        self.reason = reason


class EscalationManager:
    """Manages escalation ladders across model tiers."""

    TIER_LADDER = [
        ModelTier.LIGHTWEIGHT,
        ModelTier.BALANCED,
        ModelTier.ADVANCED,
    ]

    def __init__(self, registry: ModelRegistry, max_retries: int = 2) -> None:
        self.registry = registry
        self.max_retries = max_retries

    def get_next_tier(self, current_tier: ModelTier) -> Optional[ModelTier]:
        """Get the next tier in the escalation hierarchy."""
        try:
            current_index = self.TIER_LADDER.index(current_tier)
            if current_index + 1 < len(self.TIER_LADDER):
                return self.TIER_LADDER[current_index + 1]
        except ValueError:
            pass
        return None

    def decide_escalation(
        self,
        current_model: ModelDescriptor,
        validation: ValidationResult,
        attempt_count: int,
    ) -> EscalationDecision:
        """Determine whether to escalate and find the appropriate target model."""
        if attempt_count >= self.max_retries or not validation.requires_escalation:
            return EscalationDecision(should_escalate=False)

        next_tier = self.get_next_tier(current_model.tier)
        if not next_tier:
            return EscalationDecision(
                should_escalate=False,
                reason="Already at highest model tier",
            )

        candidates = self.registry.list_by_tier(next_tier)
        if not candidates:
            return EscalationDecision(
                should_escalate=False,
                reason=f"No models registered under escalated tier {next_tier}",
            )

        return EscalationDecision(
            should_escalate=True,
            next_model=candidates[0],
            reason="; ".join(validation.rejection_reasons) or "Validation failed threshold",
        )

