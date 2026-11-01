"""API routes definition for LLM Orchestrator."""

from typing import Any, List
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_context_store,
    get_model_executor,
    get_model_registry,
    get_orchestration_engine,
)
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ModelInfoResponse,
    OrchestrationRequest,
    TaskStatusResponse,
)
from app.core.config import Settings, get_settings
from app.core.context import BaseContextStore, is_safe_id
from app.core.logging import get_logger
from app.core.metrics import collector as metrics_collector
from app.models.registry import ModelRegistry
from app.orchestrator.engine import OrchestrationEngine
from app.orchestrator.executor import ModelExecutor

router = APIRouter()
settings = get_settings()
logger = get_logger("api.routes")


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Check the health status of the orchestrator API process.

    Returns basic process status without falsely asserting external provider connectivity.
    """
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version="0.1.0",
    )


@router.get(
    "/models",
    response_model=List[ModelInfoResponse],
    status_code=status.HTTP_200_OK,
    summary="List Registered Models",
    tags=["Models"],
)
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
    executor: ModelExecutor = Depends(get_model_executor),
) -> List[ModelInfoResponse]:
    """Retrieve all models registered in the orchestration registry without exposing credentials."""
    models = registry.list_all()
    active_providers = set(executor.list_providers())
    return [
        ModelInfoResponse(
            model_id=m.model_id,
            provider=m.provider,
            tier=m.tier.value,
            supported_capabilities=m.supported_capabilities,
            context_window=m.context_window,
            enabled=bool(m.enabled and (m.provider.lower() in active_providers if active_providers else m.enabled)),
        )
        for m in models
    ]


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Orchestrate and Execute Chat Prompt",
    tags=["Chat"],
    responses={
        200: {"description": "Prompt successfully routed and executed", "model": ChatResponse},
        422: {"description": "Prompt validation failure (e.g. empty or whitespace-only)"},
        502: {"description": "Routing or provider execution failure", "model": ChatResponse},
    },
)
async def chat_endpoint(
    request: ChatRequest,
    engine: OrchestrationEngine = Depends(get_orchestration_engine),
    context_store: BaseContextStore = Depends(get_context_store),
) -> ChatResponse:
    """Analyze, route, and execute a chat prompt through the orchestration pipeline.

    Supports optional session context persistence via context_id. When context_id is
    omitted, execution remains strictly stateless.
    """
    effective_context = dict(request.context or {})

    # Load and combine session context if context_id is provided
    if request.context_id:
        try:
            stored_data = await context_store.get_context(request.context_id)
            stored_turns = stored_data.get("turns", [])
            variables = stored_data.get("variables", {})
            effective_context = {**variables, **effective_context}
            effective_context["history"] = list(stored_turns)
            effective_context["messages"] = list(stored_turns)
            effective_context["turns"] = list(stored_turns)
        except Exception as exc:
            logger.warning("Failed to load context for session: %s", type(exc).__name__)

    active_backend = await context_store.get_active_backend()
    turns_used = len(
        effective_context.get("history")
        or effective_context.get("messages")
        or effective_context.get("turns")
        or []
    )

    result = await engine.orchestrate(
        prompt=request.prompt,
        context=effective_context,
        options=request.options,
        task_id=request.task_id,
        context_store=context_store,
    )

    # Persist turns if context_id is provided and orchestration succeeded
    if request.context_id and result.success:
        try:
            await context_store.append_turn(
                context_id=request.context_id,
                role="user",
                content=request.prompt,
            )
            await context_store.append_turn(
                context_id=request.context_id,
                role="assistant",
                content=result.content,
                metadata={
                    "model_id": result.model_id,
                    "tier": result.selected_tier.value if result.selected_tier else None,
                },
            )
            result.metadata["context_id"] = request.context_id
        except Exception as exc:
            logger.warning("Failed to persist session context: %s", type(exc).__name__)

    response_payload = ChatResponse(
        success=result.success,
        content=result.content,
        selected_model=result.model_id,
        selected_tier=result.selected_tier.value if result.selected_tier else None,
        routing_reason=result.routing.routing_reason if result.routing else None,
        complexity=result.analysis.complexity.value if result.analysis else None,
        task_type=result.analysis.task_type.value if result.analysis else None,
        latency_ms=result.execution.latency_ms if result.execution else 0.0,
        error=result.error,
        task_id=request.task_id,
        checkpoint_status=result.metadata.get("checkpoint_status"),
        cached=bool(result.metadata.get("cached", False)),
        cache_status=result.metadata.get("cache_status", "MISS"),
        context_backend=active_backend,
        context_turns_used=turns_used,
        cache_saved_tokens=result.metadata.get("cache_saved_tokens"),
        metadata=result.metadata,
    )

    if not result.success:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=response_payload.model_dump(),
        )

    return response_payload


@router.get(
    "/metrics",
    summary="JSON System Metrics",
    tags=["System"],
)
async def api_metrics(request: Request) -> Any:
    """Retrieve cumulative system metrics in structured JSON format."""
    app_settings = getattr(request.app.state, "settings", None) or get_settings()
    if not app_settings.metrics_enabled:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Metrics collection is disabled."},
        )
    return metrics_collector.get_snapshot()


# Backward compatibility route for /orchestrate
@router.post(
    "/orchestrate",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Legacy Orchestration Endpoint",
    tags=["Chat"],
    include_in_schema=False,
)
async def orchestrate_alias(
    request: OrchestrationRequest,
    engine: OrchestrationEngine = Depends(get_orchestration_engine),
    context_store: BaseContextStore = Depends(get_context_store),
) -> ChatResponse:
    """Backward compatible alias endpoint forwarding to the chat orchestration pipeline."""
    chat_req = ChatRequest(
        prompt=request.prompt,
        context_id=request.context_id,
        task_id=request.task_id,
        options=request.options.model_dump() if request.options else {},
        context=request.metadata or {},
    )
    return await chat_endpoint(request=chat_req, engine=engine, context_store=context_store)


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Task Checkpoint Status",
    tags=["Tasks"],
)
async def get_task_status(
    task_id: str,
    context_store: BaseContextStore = Depends(get_context_store),
) -> Any:
    """Retrieve task checkpoint status by task ID without exposing credentials."""
    if not is_safe_id(task_id):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid task_id format. Must be alphanumeric with dashes, underscores, or dots (max 128 chars)."},
        )
    checkpoint = await context_store.get_checkpoint(task_id)
    if not checkpoint:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Task checkpoint '{task_id}' not found."},
        )
    return TaskStatusResponse(
        task_id=checkpoint.get("task_id", task_id),
        status=checkpoint.get("status", "unknown"),
        attempt_count=checkpoint.get("attempt_count", 0),
        attempted_model_ids=checkpoint.get("attempted_model_ids", []),
        escalation_history=checkpoint.get("escalation_history", []),
        checkpoint_version=checkpoint.get("checkpoint_version", 1),
        updated_at=checkpoint.get("updated_at", 0.0),
        last_model_id=checkpoint.get("last_model_id"),
        last_error=checkpoint.get("last_error"),
    )


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear Task Checkpoint",
    tags=["Tasks"],
)
async def clear_task_checkpoint(
    task_id: str,
    context_store: BaseContextStore = Depends(get_context_store),
) -> Any:
    """Clear task checkpoint by task ID."""
    if not is_safe_id(task_id):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid task_id format."},
        )
    await context_store.clear_checkpoint(task_id)
    return {"status": "ok", "message": f"Task checkpoint '{task_id}' cleared."}
