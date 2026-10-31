"""Model escalation management module.

Determines fallback strategies and next-tier model upgrades when
validation fails or confidence is below threshold.
"""

from typing import Any, Iterable, List, Optional, Set
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.validator import ValidationResult


class EscalationDecision:
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

    def __init__(
        self,
        registry: ModelRegistry,
        max_retries: int = 2,
        executor: Optional[Any] = None,
        available_providers: Optional[Iterable[str]] = None,
    ) -> None:
        self.registry = registry
        self.max_retries = max_retries
        self.executor = executor
        self._available_providers = (
            {p.lower() for p in available_providers}
            if available_providers is not None
            else None
        )

    def is_provider_available(self, provider_name: str) -> bool:
        """Check whether a provider is configured and available for escalation."""
        if self._available_providers is not None:
            return provider_name.lower() in self._available_providers
        if self.executor is not None and hasattr(self.executor, "list_providers"):
            configured = self.executor.list_providers()
            if not configured:
                return False
            return self.executor.has_provider(provider_name.lower())
        return True

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

        # Traverse higher tiers in order until an eligible configured model is found
        check_tier = self.get_next_tier(current_model.tier)
        while check_tier is not None:
            tier_candidates = self.registry.list_by_tier(check_tier, only_enabled=True)
            eligible_candidates = [
                m for m in tier_candidates
                if m.enabled and self.is_provider_available(m.provider)
            ]
            if eligible_candidates:
                return EscalationDecision(
                    should_escalate=True,
                    next_model=eligible_candidates[0],
                    reason="; ".join(validation.rejection_reasons) or "Validation failed threshold",
                )
            check_tier = self.get_next_tier(check_tier)

        return EscalationDecision(
            should_escalate=False,
            reason="No configured fallback models available in higher tiers",
        )

