"""Intelligent model router for Adaptive LLM Orchestration.

Selects an appropriate LLM model candidate based on prompt complexity,
task category, capability requirements, context window limits, and availability.
Does not invoke external LLM provider APIs.
"""

from typing import Any, Iterable, List, Optional, Set, Union
from pydantic import BaseModel, Field

from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.analyzer import (
    AnalysisResult,
    ComplexityLevel,
    PromptAnalyzer,
    TaskType,
)


class RoutingError(Exception):
    """Base exception for model routing errors."""
    pass


class NoSuitableModelError(RoutingError):
    """Raised when no enabled model in the registry meets the query constraints."""
    pass


class RoutingDecision(BaseModel):
    """Structured decision returned by the intelligent router."""

    model: ModelDescriptor = Field(..., description="Selected model candidate")
    selected_tier: ModelTier = Field(..., description="Final tier of the selected model")
    preferred_tier: ModelTier = Field(..., description="Initial preferred tier based on complexity")
    routing_reason: str = Field(..., description="Deterministic explanation for the routing choice")
    original_complexity: ComplexityLevel = Field(..., description="Task complexity from query analysis")
    fallback_used: bool = Field(default=False, description="Whether a fallback tier was selected")
    estimated_input_tokens: int = Field(default=0, description="Estimated token count from analyzer")
    matched_capabilities: List[str] = Field(
        default_factory=list,
        description="Model capabilities aligned with query requirements",
    )

    # Convenience helper properties
    @property
    def model_id(self) -> str:
        """Convenience property for accessing the model identifier."""
        return self.model.model_id

    @property
    def selected_model(self) -> ModelDescriptor:
        """Alias for the selected model descriptor."""
        return self.model

    @property
    def selected_model_id(self) -> str:
        """Alias for the selected model ID."""
        return self.model.model_id

    @property
    def provider(self) -> str:
        """Convenience property for accessing the provider name."""
        return self.model.provider

    @property
    def context_window(self) -> int:
        """Convenience property for accessing the model context window."""
        return self.model.context_window


class IntelligentRouter:
    """Intelligently matches user queries with optimal candidate models.

    Uses query complexity, task type, capability matching, context limits,
    and fallback ladders without invoking remote LLM providers.
    """

    # Tier ladders defining search order when the preferred tier has no candidate
    _FALLBACK_LADDERS = {
        ModelTier.LIGHTWEIGHT: [
            ModelTier.LIGHTWEIGHT,
            ModelTier.BALANCED,
            ModelTier.ADVANCED,
        ],
        ModelTier.BALANCED: [
            ModelTier.BALANCED,
            ModelTier.ADVANCED,
            ModelTier.LIGHTWEIGHT,
        ],
        ModelTier.ADVANCED: [
            ModelTier.ADVANCED,
            ModelTier.BALANCED,
            ModelTier.LIGHTWEIGHT,
        ],
    }

    def __init__(
        self,
        registry: ModelRegistry,
        analyzer: Optional[PromptAnalyzer] = None,
        executor: Optional[Any] = None,
        available_providers: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize router with a model registry, optional prompt analyzer, and provider filter."""
        self.registry = registry
        self.analyzer = analyzer or PromptAnalyzer()
        self.executor = executor
        self._available_providers = (
            {p.lower() for p in available_providers}
            if available_providers is not None
            else None
        )

    def is_provider_available(self, provider_name: str) -> bool:
        """Check whether a provider is configured and available for routing."""
        if self._available_providers is not None:
            return provider_name.lower() in self._available_providers
        if self.executor is not None and hasattr(self.executor, "list_providers"):
            configured = self.executor.list_providers()
            if configured:
                return self.executor.has_provider(provider_name.lower())
        return True

    def select_tier(self, analysis: AnalysisResult) -> ModelTier:
        """Determine initial preferred ModelTier from query analysis signals.

        - Simple queries -> LIGHTWEIGHT
        - Medium queries -> BALANCED
        - Complex queries or queries requiring deep reasoning -> ADVANCED
        """
        if analysis.suggested_tier:
            try:
                return ModelTier(analysis.suggested_tier.lower())
            except ValueError:
                pass

        if analysis.requires_reasoning or analysis.complexity == ComplexityLevel.COMPLEX:
            return ModelTier.ADVANCED

        if analysis.complexity == ComplexityLevel.MEDIUM:
            return ModelTier.BALANCED

        return ModelTier.LIGHTWEIGHT

    def _score_model_candidate(
        self,
        model: ModelDescriptor,
        analysis: AnalysisResult,
    ) -> int:
        """Compute alignment score for a model given query requirements.

        Higher score indicates better capability alignment.
        """
        score = 0
        caps = set(model.supported_capabilities)

        # Coding task capability matching
        if analysis.task_type == TaskType.CODING:
            if "code" in caps:
                score += 15

        # Reasoning task capability matching
        if analysis.requires_reasoning or analysis.task_type == TaskType.REASONING:
            if "reasoning" in caps:
                score += 15

        # Structured output capability matching
        if analysis.task_type == TaskType.STRUCTURED_OUTPUT:
            if "json" in caps:
                score += 10
            if "tools" in caps:
                score += 5

        # General tool calling support
        if model.supports_tools or "tools" in caps:
            score += 1

        return score

    def _extract_matched_capabilities(
        self,
        model: ModelDescriptor,
        analysis: AnalysisResult,
    ) -> List[str]:
        """Identify which model capabilities align with the current query."""
        caps = set(model.supported_capabilities)
        matched = []

        if analysis.task_type == TaskType.CODING and "code" in caps:
            matched.append("code")
        if (analysis.requires_reasoning or analysis.task_type == TaskType.REASONING) and "reasoning" in caps:
            matched.append("reasoning")
        if analysis.task_type == TaskType.STRUCTURED_OUTPUT and "json" in caps:
            matched.append("json")
        if model.supports_tools or "tools" in caps:
            matched.append("tools")

        return matched

    def route(
        self,
        query: Union[str, AnalysisResult],
        analysis: Optional[AnalysisResult] = None,
        required_capabilities: Optional[List[str]] = None,
        preferred_tier: Optional[ModelTier] = None,
    ) -> RoutingDecision:
        """Deterministically route a query to the most appropriate candidate model.

        Args:
            query: User query string OR an already computed AnalysisResult.
            analysis: Optional pre-computed AnalysisResult if query is a string.
            required_capabilities: Optional list of mandatory model capabilities.
            preferred_tier: Optional override for the starting target tier.

        Returns:
            RoutingDecision containing selected model, tier, reason, and fallback flag.

        Raises:
            NoSuitableModelError: If no enabled model satisfies context and capability constraints.
        """
        # Resolve AnalysisResult without redundant computation
        if isinstance(query, AnalysisResult):
            analysis_result = query
        elif analysis is not None:
            analysis_result = analysis
        elif isinstance(query, str):
            analysis_result = self.analyzer.analyze(query)
        else:
            raise ValueError("query must be a non-empty string or an AnalysisResult instance.")

        target_tier = preferred_tier or self.select_tier(analysis_result)
        tier_search_ladder = self._FALLBACK_LADDERS.get(
            target_tier,
            [target_tier, ModelTier.BALANCED, ModelTier.ADVANCED, ModelTier.LIGHTWEIGHT],
        )

        required_caps = set(required_capabilities or [])

        # Evaluate candidate models across preferred and fallback tiers
        for candidate_tier in tier_search_ladder:
            # Query registry for enabled models in this tier
            tier_models = self.registry.get_models_by_tier(candidate_tier, only_enabled=True)

            # Filter candidates that have configured providers and can accommodate context window
            eligible_models = [
                m for m in tier_models
                if m.enabled
                and self.is_provider_available(m.provider)
                and m.context_window >= analysis_result.estimated_input_tokens
            ]

            # Filter candidates that satisfy any mandatory capabilities
            if required_caps:
                eligible_models = [
                    m for m in eligible_models
                    if required_caps.issubset(set(m.supported_capabilities))
                ]

            if not eligible_models:
                continue

            # Deterministic ranking: score (descending), context headroom (descending), model_id (alphabetical)
            eligible_models.sort(
                key=lambda m: (
                    -self._score_model_candidate(m, analysis_result),
                    -m.context_window,
                    m.model_id,
                )
            )

            chosen_model = eligible_models[0]
            fallback_used = (candidate_tier != target_tier)
            matched_caps = self._extract_matched_capabilities(chosen_model, analysis_result)

            if fallback_used:
                reason = (
                    f"Fallback applied: preferred tier '{target_tier.value}' had no suitable models; "
                    f"routed to '{candidate_tier.value}' tier candidate '{chosen_model.model_id}' "
                    f"(task: {analysis_result.task_type.value}, complexity: {analysis_result.complexity.value})."
                )
            else:
                reason = (
                    f"Routed to preferred tier '{target_tier.value}' candidate '{chosen_model.model_id}' "
                    f"matching task '{analysis_result.task_type.value}' "
                    f"(complexity: {analysis_result.complexity.value}, context: {chosen_model.context_window})."
                )

            return RoutingDecision(
                model=chosen_model,
                selected_tier=candidate_tier,
                preferred_tier=target_tier,
                routing_reason=reason,
                original_complexity=analysis_result.complexity,
                fallback_used=fallback_used,
                estimated_input_tokens=analysis_result.estimated_input_tokens,
                matched_capabilities=matched_caps,
            )

        # If loop finishes without returning, no model could satisfy constraints
        raise NoSuitableModelError(
            f"No enabled model found in registry capable of handling task '{analysis_result.task_type.value}' "
            f"(complexity: {analysis_result.complexity.value}, estimated tokens: {analysis_result.estimated_input_tokens}, "
            f"required capabilities: {list(required_caps)})."
        )

    async def route_async(
        self,
        query: Union[str, AnalysisResult],
        analysis: Optional[AnalysisResult] = None,
        required_capabilities: Optional[List[str]] = None,
        preferred_tier: Optional[ModelTier] = None,
    ) -> RoutingDecision:
        """Asynchronously route a query, utilizing async analyzer wrapper if analysis is needed."""
        if analysis is None and isinstance(query, str):
            analysis = await self.analyzer.analyze_async(query)

        return self.route(
            query=query,
            analysis=analysis,
            required_capabilities=required_capabilities,
            preferred_tier=preferred_tier,
        )
