"""Model registry and capability definitions for LLM Orchestration.

Manages catalog metadata, capability declarations, tier assignments, and
availability status for candidate language models without invoking provider APIs.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ModelTier(str, Enum):
    """Classification tiers for routing and escalation."""

    LIGHTWEIGHT = "lightweight"  # Fast, cost-efficient (e.g. gpt-4o-mini, haiku, gemini-flash)
    BALANCED = "balanced"        # Standard reasoning and instruction following (e.g. gpt-4o, sonnet)
    ADVANCED = "advanced"        # High-complexity reasoning and deep analysis (e.g. o1, opus)


class ModelDescriptor(BaseModel):
    """Metadata, capabilities, and pricing specifications for a model candidate.

    Pricing and availability are user-configured and are not guaranteed real-time metrics.
    """

    model_id: str = Field(..., description="Unique model identifier")
    provider: str = Field(..., description="Provider name (e.g., openai, anthropic, google)")
    tier: ModelTier = Field(..., description="Assigned routing tier (lightweight, balanced, advanced)")
    supported_capabilities: List[str] = Field(
        default_factory=list,
        description="Declared capabilities (e.g. ['chat', 'code', 'tools', 'vision', 'reasoning'])",
    )
    context_window: int = Field(default=8192, description="Maximum token context window length")
    input_token_price: Optional[float] = Field(
        default=None,
        description="Optional configured price per 1k input tokens (placeholder/unspecified if None)",
    )
    output_token_price: Optional[float] = Field(
        default=None,
        description="Optional configured price per 1k output tokens (placeholder/unspecified if None)",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the model is enabled and available for routing",
    )

    # Backward compatibility fields
    supports_tools: bool = Field(default=True, description="Whether tool calling is supported")
    tags: List[str] = Field(default_factory=list, description="Tag list alias for capabilities")
    cost_per_1k_input: Optional[float] = Field(default=None, description="Alias for input_token_price")
    cost_per_1k_output: Optional[float] = Field(default=None, description="Alias for output_token_price")

    @model_validator(mode="after")
    def _sync_compatibility_fields(self) -> "ModelDescriptor":
        """Synchronize aliases between modern and legacy schema fields."""
        # Sync tags and supported_capabilities
        if not self.supported_capabilities and self.tags:
            self.supported_capabilities = list(self.tags)
        elif not self.tags and self.supported_capabilities:
            self.tags = list(self.supported_capabilities)

        # Sync tool support capability
        if "tools" in self.supported_capabilities and not self.supports_tools:
            self.supports_tools = True
        elif self.supports_tools and "tools" not in self.supported_capabilities:
            self.supported_capabilities.append("tools")

        # Sync pricing fields
        if self.input_token_price is None and self.cost_per_1k_input is not None:
            self.input_token_price = self.cost_per_1k_input
        elif self.cost_per_1k_input is None and self.input_token_price is not None:
            self.cost_per_1k_input = self.input_token_price

        if self.output_token_price is None and self.cost_per_1k_output is not None:
            self.output_token_price = self.cost_per_1k_output
        elif self.cost_per_1k_output is None and self.output_token_price is not None:
            self.cost_per_1k_output = self.output_token_price

        return self

    @property
    def is_available(self) -> bool:
        """Helper indicating whether model is available for active routing."""
        return self.enabled


class ModelRegistry:
    """In-memory model registry managing model profiles for routing.

    Does not make external network requests or call provider APIs.
    """

    def __init__(self, models: Optional[List[ModelDescriptor]] = None) -> None:
        """Initialize registry with an optional collection of model descriptors."""
        self._models: Dict[str, ModelDescriptor] = {}
        if models:
            for model in models:
                self.register(model)

    def register(self, model: ModelDescriptor) -> None:
        """Register or update a model profile in the registry."""
        self._models[model.model_id] = model

    def unregister(self, model_id: str) -> Optional[ModelDescriptor]:
        """Remove a model from the registry by its ID."""
        return self._models.pop(model_id, None)

    def get(self, model_id: str) -> Optional[ModelDescriptor]:
        """Retrieve a model descriptor by ID. Returns None if not found."""
        return self._models.get(model_id)

    def get_model(self, model_id: str) -> Optional[ModelDescriptor]:
        """Explicit alias to retrieve a model descriptor by ID."""
        return self.get(model_id)

    def get_or_raise(self, model_id: str) -> ModelDescriptor:
        """Retrieve a model descriptor by ID, raising KeyError if unknown."""
        model = self.get(model_id)
        if model is None:
            raise KeyError(f"Model '{model_id}' is not registered in the model registry.")
        return model

    def list_models(self, only_enabled: bool = True) -> List[ModelDescriptor]:
        """List registered models, optionally filtering to only enabled models."""
        if only_enabled:
            return [m for m in self._models.values() if m.enabled]
        return list(self._models.values())

    def list_all(self) -> List[ModelDescriptor]:
        """List all registered models regardless of enabled status (backward-compatible)."""
        return list(self._models.values())

    def get_models_by_tier(
        self,
        tier: Union[ModelTier, str],
        only_enabled: bool = True,
    ) -> List[ModelDescriptor]:
        """Filter registered models by tier and optionally by enabled status."""
        tier_enum = tier if isinstance(tier, ModelTier) else ModelTier(str(tier).lower())
        models = [m for m in self._models.values() if m.tier == tier_enum]
        if only_enabled:
            return [m for m in models if m.enabled]
        return models

    def list_by_tier(
        self,
        tier: Union[ModelTier, str],
        only_enabled: bool = True,
    ) -> List[ModelDescriptor]:
        """Alias for get_models_by_tier for compatibility with router."""
        return self.get_models_by_tier(tier=tier, only_enabled=only_enabled)

    def clear(self) -> None:
        """Clear all models from the registry."""
        self._models.clear()

    @classmethod
    def with_default_models(cls) -> "ModelRegistry":
        """Factory creating a registry pre-populated with standard reference configurations.

        Note: Pricing is set to None (unknown/placeholder) unless configured by the user.
        API keys are never stored here.
        """
        registry = cls()

        # Optional override of the Gemini model id via GEMINI_MODEL env var
        try:
            from app.core.config import get_settings
            gemini_model_id = (get_settings().gemini_model_id or "").strip() or "gemini-3.8-flash"
        except Exception:
            gemini_model_id = "gemini-3.8-flash"

        example_models = [
            # Lightweight tier
            ModelDescriptor(
                model_id="gpt-4o-mini",
                provider="openai",
                tier=ModelTier.LIGHTWEIGHT,
                supported_capabilities=["chat", "code", "tools", "json", "vision"],
                context_window=128000,
                enabled=True,
            ),
            ModelDescriptor(
                model_id=gemini_model_id,
                provider="google",
                tier=ModelTier.LIGHTWEIGHT,
                supported_capabilities=["chat", "code", "tools", "json", "vision", "reasoning"],
                context_window=1048576,
                enabled=True,
            ),
            ModelDescriptor(
                model_id="claude-3-haiku",
                provider="anthropic",
                tier=ModelTier.LIGHTWEIGHT,
                supported_capabilities=["chat", "code", "tools"],
                context_window=200000,
                enabled=True,
            ),
            # Balanced tier
            ModelDescriptor(
                model_id="gpt-4o",
                provider="openai",
                tier=ModelTier.BALANCED,
                supported_capabilities=["chat", "code", "tools", "json", "vision"],
                context_window=128000,
                enabled=True,
            ),
            ModelDescriptor(
                model_id="claude-3-5-sonnet",
                provider="anthropic",
                tier=ModelTier.BALANCED,
                supported_capabilities=["chat", "code", "tools", "vision", "reasoning"],
                context_window=200000,
                enabled=True,
            ),
            # Advanced tier
            ModelDescriptor(
                model_id="o1",
                provider="openai",
                tier=ModelTier.ADVANCED,
                supported_capabilities=["chat", "code", "reasoning"],
                context_window=200000,
                enabled=True,
            ),
            ModelDescriptor(
                model_id="claude-3-opus",
                provider="anthropic",
                tier=ModelTier.ADVANCED,
                supported_capabilities=["chat", "code", "reasoning", "vision"],
                context_window=200000,
                enabled=True,
            ),
            # Explicitly disabled reference model for testing filter exclusions
            ModelDescriptor(
                model_id="legacy-disabled-model",
                provider="custom",
                tier=ModelTier.LIGHTWEIGHT,
                supported_capabilities=["chat"],
                context_window=4096,
                enabled=False,
            ),
        ]

        for model in example_models:
            registry.register(model)

        return registry
