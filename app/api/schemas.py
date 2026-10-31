"""Pydantic schemas for API endpoints."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

SAFE_TASK_ID_REGEX = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")


class HealthResponse(BaseModel):
    """Health check status schema."""

    status: str = Field(default="ok", description="Current health status of the service")
    app: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")


class ModelInfoResponse(BaseModel):
    """Safe model metadata descriptor for API clients without sensitive credentials."""

    model_id: str = Field(..., description="Unique model identifier")
    provider: str = Field(..., description="Model provider (e.g. openai, anthropic, google)")
    tier: str = Field(..., description="Assigned model capability tier")
    supported_capabilities: List[str] = Field(default_factory=list, description="Capabilities supported")
    context_window: int = Field(..., description="Maximum context window tokens")
    enabled: bool = Field(..., description="Whether model is currently enabled for routing")


class ChatRequest(BaseModel):
    """Incoming prompt execution payload for /api/v1/chat."""

    prompt: str = Field(..., description="User input prompt to orchestrate")
    context_id: Optional[str] = Field(default=None, description="Optional session or conversation ID")
    task_id: Optional[str] = Field(default=None, description="Optional task identifier for checkpointing and recovery")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Conversational context dictionary")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Routing or execution override options")

    @field_validator("prompt")
    @classmethod
    def validate_non_empty_prompt(cls, value: str) -> str:
        """Validate that prompt is not empty or whitespace-only."""
        if value is None or not value.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")
        return value.strip()

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if not SAFE_TASK_ID_REGEX.match(cleaned):
            raise ValueError("task_id must contain only alphanumeric characters, dashes, underscores, or dots (max 128 chars).")
        return cleaned


class ChatResponse(BaseModel):
    """Response returned by the chat orchestration endpoint."""

    success: bool = Field(..., description="Whether the request was processed successfully")
    content: str = Field(default="", description="Generated response content from the model")
    selected_model: Optional[str] = Field(default=None, description="Model ID chosen for generation")
    selected_tier: Optional[str] = Field(default=None, description="Model tier selected")
    routing_reason: Optional[str] = Field(default=None, description="Explanation for model selection or fallback")
    complexity: Optional[str] = Field(default=None, description="Assessed query complexity")
    task_type: Optional[str] = Field(default=None, description="Categorized query task type")
    latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")
    task_id: Optional[str] = Field(default=None, description="Task identifier if checkpointing was requested")
    checkpoint_status: Optional[str] = Field(default=None, description="Task checkpoint status if checkpointing was used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or telemetry")


class TaskStatusResponse(BaseModel):
    """Safe task execution checkpoint status schema without exposing credentials."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Current checkpoint status (e.g. started, executing, escalated, completed, failed)")
    attempt_count: int = Field(default=0, description="Number of execution attempts")
    attempted_model_ids: List[str] = Field(default_factory=list, description="Model IDs attempted so far")
    escalation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation history records")
    checkpoint_version: int = Field(default=1, description="Checkpoint schema version")
    updated_at: float = Field(..., description="Timestamp of last checkpoint update")
    last_model_id: Optional[str] = Field(default=None, description="Most recently attempted model ID")
    last_error: Optional[str] = Field(default=None, description="Last sanitized error message, if failed")


# Backward compatibility schemas
class OrchestrationOptions(BaseModel):
    """User-specified routing and escalation override options."""

    prefer_speed: bool = Field(default=False, description="Prioritize response latency")
    prefer_cost: bool = Field(default=False, description="Prioritize lowest token cost")
    max_escalations: Optional[int] = Field(default=None, description="Max allowed model escalations")
    forced_model_tier: Optional[str] = Field(default=None, description="Force a specific tier")


class OrchestrationRequest(BaseModel):
    """Incoming prompt orchestration request payload."""

    prompt: str = Field(..., min_length=1, description="Input user prompt to process")
    context_id: Optional[str] = Field(default=None, description="Session or context identifier")
    task_id: Optional[str] = Field(default=None, description="Optional task identifier for checkpointing and recovery")
    options: Optional[OrchestrationOptions] = Field(default_factory=OrchestrationOptions)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")
        return value.strip()

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if not SAFE_TASK_ID_REGEX.match(cleaned):
            raise ValueError("task_id must contain only alphanumeric characters, dashes, underscores, or dots (max 128 chars).")
        return cleaned


class EscalationRecord(BaseModel):
    """Details of any escalation step during orchestration."""

    from_model: str = Field(..., description="Model before escalation")
    to_model: str = Field(..., description="Model escalated to")
    reason: str = Field(..., description="Reason for escalation")


class OrchestrationResponse(BaseModel):
    """Orchestration completion response payload."""

    content: str = Field(..., description="Generated response content")
    selected_tier: str = Field(..., description="Model tier selected for execution")
    selected_model: str = Field(..., description="Final model that produced the response")
    escalated: bool = Field(default=False, description="Whether escalation occurred")
    escalation_history: List[EscalationRecord] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(default=None, description="Validator confidence score")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Latency and token metrics")
