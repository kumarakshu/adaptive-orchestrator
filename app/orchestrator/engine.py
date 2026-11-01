"""Core orchestration engine.

Coordinates prompt validation, query analysis, intelligent routing, and
model execution into a unified pipeline.
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.context import BaseContextStore, CHECKPOINT_VERSION, is_safe_id
from app.core.logging import get_logger
from app.models.registry import ModelRegistry, ModelTier
from app.orchestrator.analyzer import AnalysisResult, PromptAnalyzer
from app.orchestrator.escalation import EscalationManager
from app.orchestrator.executor import BaseExecutor, ExecutionResult
from app.orchestrator.router import IntelligentRouter, RoutingDecision
from app.orchestrator.validator import ResponseValidator, ValidationResult

logger = get_logger("orchestrator.engine")


class OrchestrationResult(BaseModel):
    """Structured result returned by the orchestration engine."""

    success: bool = Field(..., description="Whether the orchestration pipeline executed successfully")
    prompt: str = Field(..., description="Original user prompt")
    analysis: Optional[AnalysisResult] = Field(default=None, description="Analysis result for the prompt")
    routing: Optional[RoutingDecision] = Field(default=None, description="Routing decision")
    execution: Optional[ExecutionResult] = Field(default=None, description="Execution outcome from the model provider")
    error: Optional[str] = Field(default=None, description="Error message if orchestration failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata container for context and telemetry")

    @property
    def content(self) -> str:
        """Convenience property for accessing generated content."""
        return self.execution.content if self.execution else ""

    @property
    def model_id(self) -> Optional[str]:
        """Convenience property for accessing the model that executed the prompt."""
        if self.execution:
            return self.execution.model_id
        if self.routing:
            return self.routing.model_id
        return None

    @property
    def selected_tier(self) -> Optional[ModelTier]:
        """Convenience property for accessing the selected model tier."""
        return self.routing.selected_tier if self.routing else None


class OrchestrationEngine:
    """Central orchestrator wiring pipeline stages together."""

    def __init__(
        self,
        registry: ModelRegistry,
        executor: Optional[BaseExecutor] = None,
        analyzer: Optional[PromptAnalyzer] = None,
        router: Optional[IntelligentRouter] = None,
        validator: Optional[ResponseValidator] = None,
        escalation: Optional[EscalationManager] = None,
        context_store: Optional[BaseContextStore] = None,
    ) -> None:
        """Initialize orchestrator with components."""
        self.registry = registry
        self.executor = executor
        self.analyzer = analyzer or PromptAnalyzer()
        self.router = router or IntelligentRouter(
            registry=registry,
            analyzer=self.analyzer,
            executor=self.executor,
        )
        if self.executor is not None and hasattr(self.router, "executor") and self.router.executor is None:
            self.router.executor = self.executor

        self.validator = validator or ResponseValidator()
        self.context_store = context_store
        if escalation is not None:
            self.escalation = escalation
            if self.executor is not None and hasattr(self.escalation, "executor") and self.escalation.executor is None:
                self.escalation.executor = self.executor
        else:
            try:
                settings = get_settings()
                self.escalation = EscalationManager(
                    registry=registry,
                    max_retries=settings.max_escalation_retries,
                    executor=self.executor,
                )
            except Exception:
                self.escalation = EscalationManager(registry=registry, executor=self.executor)

    async def orchestrate(
        self,
        prompt: str,
        analysis: Optional[AnalysisResult] = None,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        context_store: Optional[BaseContextStore] = None,
    ) -> OrchestrationResult:
        """Execute the end-to-end orchestration pipeline.

        1. Validates the input prompt.
        2. Analyzes prompt characteristics (reusing supplied analysis if given).
        3. Routes the prompt to a candidate model.
        4. Executes using the injected executor with validation, multi-tier escalation,
           and optional task checkpointing/recovery.
        5. Returns a structured OrchestrationResult.

        Raises:
            ValueError: If prompt is empty, whitespace-only, or task_id is invalid.
        """
        if prompt is None or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        cleaned_prompt = prompt.strip()
        opts = options or {}
        store = context_store or self.context_store

        valid_task_id: Optional[str] = None
        if task_id:
            if not is_safe_id(task_id):
                raise ValueError("Invalid task_id format. Must be alphanumeric with dashes, underscores, or dots (max 128 chars).")
            valid_task_id = task_id.strip()

        # Step 1: Check for existing task checkpoint
        existing_cp: Optional[Dict[str, Any]] = None
        if valid_task_id and store:
            try:
                existing_cp = await store.get_checkpoint(valid_task_id)
            except Exception as cp_err:
                logger.warning("Failed to retrieve checkpoint for task %s: %s", valid_task_id, type(cp_err).__name__)

        if existing_cp:
            if not isinstance(existing_cp, dict) or existing_cp.get("checkpoint_version") != CHECKPOINT_VERSION:
                logger.warning(
                    "Malformed or unsupported checkpoint version for task %s; proceeding as fresh task.",
                    valid_task_id,
                )
                existing_cp = None

        # Deterministic check for completed task: return cached result without rerun
        if existing_cp and existing_cp.get("status") == "completed":
            saved_result = existing_cp.get("result") or {}
            logger.info("Task %s already completed; returning existing result without rerun.", valid_task_id)
            return OrchestrationResult(
                success=True,
                prompt=existing_cp.get("prompt", cleaned_prompt),
                analysis=analysis,
                routing=None,
                execution=ExecutionResult(
                    content=saved_result.get("content", ""),
                    model_id=saved_result.get("model_id", ""),
                    success=True,
                    prompt_tokens=saved_result.get("prompt_tokens", 0),
                    completion_tokens=saved_result.get("completion_tokens", 0),
                    latency_ms=saved_result.get("latency_ms", 0.0),
                ),
                error=None,
                metadata={
                    **opts.get("metadata", {}),
                    **saved_result.get("metadata", {}),
                    "task_id": valid_task_id,
                    "checkpoint_status": "completed",
                    "resumed_from_checkpoint": True,
                },
            )

        # Deterministic check for terminal failed task: return cached failure without rerun
        if existing_cp and existing_cp.get("status") == "failed":
            logger.info("Task %s previously reached terminal failure; returning recorded failure.", valid_task_id)
            return OrchestrationResult(
                success=False,
                prompt=existing_cp.get("prompt", cleaned_prompt),
                analysis=analysis,
                routing=None,
                execution=None,
                error=existing_cp.get(
                    "last_error",
                    "Task previously reached terminal failure. A fresh task_id is required for a new execution attempt.",
                ),
                metadata={
                    **opts.get("metadata", {}),
                    "task_id": valid_task_id,
                    "checkpoint_status": "failed",
                    "attempt_count": existing_cp.get("attempt_count", 0),
                    "escalation_history": existing_cp.get("escalation_history", []),
                    "resumed_from_checkpoint": True,
                },
            )

        async def _save_cp(
            status: str,
            attempt_count: int,
            attempted_model_ids: List[str],
            escalation_history: List[Dict[str, Any]],
            last_model_id: Optional[str] = None,
            last_error: Optional[str] = None,
            result: Optional[Dict[str, Any]] = None,
        ) -> None:
            if not valid_task_id or not store:
                return
            try:
                cp = {
                    "task_id": valid_task_id,
                    "status": status,
                    "prompt": cleaned_prompt,
                    "attempt_count": attempt_count,
                    "attempted_model_ids": list(attempted_model_ids),
                    "escalation_history": list(escalation_history),
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "updated_at": round(time.time(), 3),
                    "last_model_id": last_model_id,
                    "last_error": last_error,
                    "result": result,
                }
                await store.save_checkpoint(valid_task_id, cp)
            except Exception as exc:
                logger.warning("Failed to save checkpoint for task %s: %s", valid_task_id, type(exc).__name__)

        is_resuming = existing_cp is not None
        if is_resuming:
            attempt_count = existing_cp.get("attempt_count", 0)
            attempted_model_ids = set(existing_cp.get("attempted_model_ids", []))
            escalation_history = list(existing_cp.get("escalation_history", []))
            resumed_model_id = existing_cp.get("last_model_id")
        else:
            attempt_count = 0
            attempted_model_ids = set()
            escalation_history = []
            resumed_model_id = None

        # Step 2: Analyze prompt (reuse if supplied)
        if analysis is not None:
            analysis_result = analysis
        else:
            analysis_result = await self.analyzer.analyze_async(cleaned_prompt)

        # Step 3: Route prompt
        try:
            routing_decision = await self.router.route_async(
                query=cleaned_prompt,
                analysis=analysis_result,
                required_capabilities=opts.get("required_capabilities"),
                preferred_tier=opts.get("preferred_tier"),
            )
        except Exception as route_exc:
            await _save_cp(
                status="failed",
                attempt_count=attempt_count,
                attempted_model_ids=list(attempted_model_ids),
                escalation_history=escalation_history,
                last_error=f"Routing failed: {route_exc}",
            )
            return OrchestrationResult(
                success=False,
                prompt=cleaned_prompt,
                analysis=analysis_result,
                routing=None,
                execution=None,
                error=f"Routing failed: {route_exc}",
                metadata={
                    **opts.get("metadata", {}),
                    **({"task_id": valid_task_id, "checkpoint_status": "failed"} if valid_task_id else {}),
                },
            )

        # Step 4: Execute with validation and escalation loop
        if self.executor is None:
            await _save_cp(
                status="failed",
                attempt_count=attempt_count,
                attempted_model_ids=list(attempted_model_ids),
                escalation_history=escalation_history,
                last_error="No executor configured on the orchestration engine.",
            )
            return OrchestrationResult(
                success=False,
                prompt=cleaned_prompt,
                analysis=analysis_result,
                routing=routing_decision,
                execution=None,
                error="No executor configured on the orchestration engine.",
                metadata={
                    **opts.get("metadata", {}),
                    **({"task_id": valid_task_id, "checkpoint_status": "failed"} if valid_task_id else {}),
                },
            )

        if is_resuming and resumed_model_id:
            resumed_model = self.registry.get(resumed_model_id)
            if resumed_model and resumed_model.enabled:
                current_model = resumed_model
                current_routing = RoutingDecision(
                    model=resumed_model,
                    selected_tier=resumed_model.tier,
                    preferred_tier=routing_decision.preferred_tier,
                    routing_reason=f"Resumed from checkpoint at {resumed_model.tier.value} tier ({resumed_model.model_id})",
                    original_complexity=routing_decision.original_complexity,
                    fallback_used=True,
                    estimated_input_tokens=routing_decision.estimated_input_tokens,
                    matched_capabilities=resumed_model.supported_capabilities,
                )
            else:
                current_routing = routing_decision
                current_model = routing_decision.model
        else:
            current_routing = routing_decision
            current_model = routing_decision.model
            # Save checkpoint: Task started
            await _save_cp(
                status="started",
                attempt_count=0,
                attempted_model_ids=list(attempted_model_ids),
                escalation_history=escalation_history,
                last_model_id=current_model.model_id,
            )

        attempted_model_ids.add(current_model.model_id)
        last_execution_result: Optional[ExecutionResult] = None
        last_validation_result: Optional[ValidationResult] = None

        while True:
            # Save checkpoint transition: Executing model attempt
            await _save_cp(
                status="executing",
                attempt_count=attempt_count,
                attempted_model_ids=list(attempted_model_ids),
                escalation_history=escalation_history,
                last_model_id=current_model.model_id,
            )

            try:
                execution_result = await self.executor.execute(
                    prompt=cleaned_prompt,
                    target=current_routing,
                    context=context,
                )
            except Exception as exec_exc:
                execution_result = ExecutionResult(
                    content="",
                    model_id=current_model.model_id,
                    success=False,
                    error=f"Executor raised unexpected exception: {exec_exc}",
                )

            last_execution_result = execution_result

            # Record model execution and token metrics
            try:
                from app.core.metrics import collector as metrics_collector
                metrics_collector.record_model_execution(
                    provider=current_model.provider,
                    model_id=current_model.model_id,
                    success=execution_result.success,
                )
                if execution_result.prompt_tokens or execution_result.completion_tokens:
                    metrics_collector.record_tokens(
                        model_id=current_model.model_id,
                        prompt_tokens=execution_result.prompt_tokens,
                        completion_tokens=execution_result.completion_tokens,
                    )
            except Exception:
                pass

            # Validate execution result
            if not execution_result.success:
                validation_result = ValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    rejection_reasons=[execution_result.error or "Model execution failed"],
                    requires_escalation=True,
                )
            else:
                try:
                    validation_result = await self.validator.validate(
                        result=execution_result,
                        expected_format=opts.get("expected_format"),
                    )
                except Exception as val_exc:
                    validation_result = ValidationResult(
                        is_valid=False,
                        confidence_score=0.0,
                        rejection_reasons=[f"Validation raised exception: {val_exc}"],
                        requires_escalation=True,
                    )

            last_validation_result = validation_result

            # Success condition: execution succeeded, validation passed, and no escalation demanded
            if execution_result.success and validation_result.is_valid and not validation_result.requires_escalation:
                result_snapshot = {
                    "content": execution_result.content,
                    "model_id": current_model.model_id,
                    "selected_tier": current_model.tier.value,
                    "prompt_tokens": execution_result.prompt_tokens,
                    "completion_tokens": execution_result.completion_tokens,
                    "latency_ms": execution_result.latency_ms,
                }
                await _save_cp(
                    status="completed",
                    attempt_count=attempt_count,
                    attempted_model_ids=list(attempted_model_ids),
                    escalation_history=escalation_history,
                    last_model_id=current_model.model_id,
                    result=result_snapshot,
                )
                return OrchestrationResult(
                    success=True,
                    prompt=cleaned_prompt,
                    analysis=analysis_result,
                    routing=current_routing,
                    execution=execution_result,
                    error=None,
                    metadata={
                        **opts.get("metadata", {}),
                        "attempt_count": attempt_count,
                        "escalation_history": escalation_history,
                        "escalated": len(escalation_history) > 0,
                        "validation": validation_result.model_dump(),
                        **({"task_id": valid_task_id, "checkpoint_status": "completed"} if valid_task_id else {}),
                        **({"resumed_from_checkpoint": True} if is_resuming else {}),
                    },
                )

            # Determine whether escalation is warranted and possible
            decision = self.escalation.decide_escalation(
                current_model=current_model,
                validation=validation_result,
                attempt_count=attempt_count,
            )

            if not decision.should_escalate or not decision.next_model:
                break

            # Deduplication: ensure candidate has not already been attempted in this request
            next_model = decision.next_model
            if next_model.model_id in attempted_model_ids:
                tier_candidates = self.registry.list_by_tier(next_model.tier, only_enabled=True)
                untried = [
                    c for c in tier_candidates
                    if c.model_id not in attempted_model_ids
                    and c.enabled
                    and (not hasattr(self.escalation, "is_provider_available") or self.escalation.is_provider_available(c.provider))
                ]
                if untried:
                    next_model = untried[0]
                else:
                    # All candidates in target tier already attempted; prevent duplicate attempt
                    break

            attempted_model_ids.add(next_model.model_id)
            escalation_history.append({
                "from_model": current_model.model_id,
                "to_model": next_model.model_id,
                "from_tier": current_model.tier.value,
                "to_tier": next_model.tier.value,
                "reason": decision.reason,
                "attempt": attempt_count,
            })
            attempt_count += 1

            # Save checkpoint transition: Escalation
            await _save_cp(
                status="escalated",
                attempt_count=attempt_count,
                attempted_model_ids=list(attempted_model_ids),
                escalation_history=escalation_history,
                last_model_id=next_model.model_id,
            )

            try:
                from app.core.metrics import collector as metrics_collector
                metrics_collector.record_escalation(
                    from_tier=current_model.tier.value,
                    to_tier=next_model.tier.value,
                )
            except Exception:
                pass

            # Advance current routing decision to escalated candidate
            current_routing = RoutingDecision(
                model=next_model,
                selected_tier=next_model.tier,
                preferred_tier=routing_decision.preferred_tier,
                routing_reason=f"Escalated to {next_model.tier.value} tier ({next_model.model_id}): {decision.reason or 'Validation triggered escalation'}",
                original_complexity=routing_decision.original_complexity,
                fallback_used=True,
                estimated_input_tokens=routing_decision.estimated_input_tokens,
                matched_capabilities=next_model.supported_capabilities,
            )
            current_model = next_model

        # Failure outcome when all attempts / tiers are exhausted
        error_msg = None
        if last_execution_result and last_execution_result.error:
            if decision.reason and ("No configured" in decision.reason or "No models" in decision.reason or "highest" in decision.reason):
                error_msg = f"{last_execution_result.error} ({decision.reason})"
            else:
                error_msg = last_execution_result.error
        elif decision.reason:
            error_msg = f"Escalation halted: {decision.reason}"
        elif last_validation_result and last_validation_result.rejection_reasons:
            error_msg = "; ".join(last_validation_result.rejection_reasons)
        else:
            error_msg = "Model execution failed and could not be escalated further."

        await _save_cp(
            status="failed",
            attempt_count=attempt_count,
            attempted_model_ids=list(attempted_model_ids),
            escalation_history=escalation_history,
            last_model_id=current_model.model_id,
            last_error=error_msg,
        )

        return OrchestrationResult(
            success=False,
            prompt=cleaned_prompt,
            analysis=analysis_result,
            routing=current_routing,
            execution=last_execution_result,
            error=error_msg,
            metadata={
                **opts.get("metadata", {}),
                "attempt_count": attempt_count,
                "escalation_history": escalation_history,
                "escalated": len(escalation_history) > 0,
                "validation": last_validation_result.model_dump() if last_validation_result else None,
                **({"task_id": valid_task_id, "checkpoint_status": "failed"} if valid_task_id else {}),
                **({"resumed_from_checkpoint": True} if is_resuming else {}),
            },
        )

    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        analysis: Optional[AnalysisResult] = None,
        task_id: Optional[str] = None,
        context_store: Optional[BaseContextStore] = None,
    ) -> OrchestrationResult:
        """Alias for orchestrate providing standard execution invocation."""
        return await self.orchestrate(
            prompt=prompt,
            analysis=analysis,
            context=context,
            options=options,
            task_id=task_id,
            context_store=context_store,
        )
