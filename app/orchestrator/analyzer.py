"""Deterministic query analysis module for LLM orchestration.

Analyzes input user queries to estimate complexity, categorize task types,
approximate input tokens, and detect complexity signals to assist model routing.
"""

from enum import Enum
import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    """Estimated complexity classification of a prompt."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    # Backward compatibility alias
    MODERATE = "medium"


class TaskType(str, Enum):
    """Categorization of query task types."""

    GENERAL_QUESTION = "general_question"
    CODING = "coding"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    STRUCTURED_OUTPUT = "structured_output"
    CREATIVE_WRITING = "creative_writing"
    OTHER = "other"


class AnalyzerConfig(BaseModel):
    """Configurable heuristics and thresholds for prompt analysis."""

    # Token estimation: average characters per token in English / code
    chars_per_token: float = Field(
        default=4.0,
        description="Approximate character count per token for lightweight estimation",
    )
    words_per_token: float = Field(
        default=1.33,
        description="Approximate multiplier from whitespace-delimited words to tokens",
    )

    # Prompt length thresholds
    long_prompt_tokens: int = Field(
        default=150,
        description="Threshold above which a prompt is considered long",
    )
    very_long_prompt_tokens: int = Field(
        default=350,
        description="Threshold above which a prompt is considered very long (automatically complex)",
    )

    # Signal count thresholds
    multi_step_threshold: int = Field(
        default=2,
        description="Number of sequential step markers needed to trigger multi-step signal",
    )
    constraint_threshold: int = Field(
        default=2,
        description="Number of restrictive phrases needed to trigger constraint signal",
    )
    requirement_threshold: int = Field(
        default=3,
        description="Number of numbered/bullet items needed to trigger explicit requirements signal",
    )
    complex_signals_threshold: int = Field(
        default=2,
        description="Number of distinct complex signals required to escalate complexity to COMPLEX",
    )


class AnalysisResult(BaseModel):
    """Structured result produced by prompt analysis."""

    complexity: ComplexityLevel = Field(
        default=ComplexityLevel.SIMPLE,
        description="Classified complexity level: simple, medium, or complex",
    )
    task_type: TaskType = Field(
        default=TaskType.GENERAL_QUESTION,
        description="Identified task category",
    )
    estimated_input_tokens: int = Field(
        default=0,
        description="Lightweight heuristic approximation of input tokens (not an exact tokenizer count)",
    )
    requires_reasoning: bool = Field(
        default=False,
        description="Flag indicating if the query demands logical or multi-step reasoning",
    )
    detected_signals: List[str] = Field(
        default_factory=list,
        description="Specific signals detected during analysis explaining complexity",
    )

    # Optional routing and compatibility fields
    suggested_tier: Optional[str] = Field(
        default=None,
        description="Suggested target model tier (lightweight, balanced, advanced)",
    )
    domain_tags: List[str] = Field(
        default_factory=list,
        description="Associated domain tags",
    )

    @property
    def estimated_tokens(self) -> int:
        """Backward compatibility property for existing callers."""
        return self.estimated_input_tokens


class PromptAnalyzer:
    """Lightweight, deterministic query analyzer.

    Extracts complexity signals, task categories, and token estimates
    without external API calls, networks, or model dependencies.
    """

    # Precompiled regex patterns for task and signal detection
    _CODE_PATTERN = re.compile(
        r"(```|\b(def|class|async def|return|import|lambda|function|const|let|var|SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN|SQL|HTML|CSS|Docker|FastAPI|Pydantic|pytest|git|API endpoint)\b|->|\b(refactor|debug|compile|traceback|syntax error)\b)",
        re.IGNORECASE,
    )

    _ARCHITECTURE_PATTERN = re.compile(
        r"\b(architecture|architect|system design|microservices?|distributed system|high availability|fault tolerance|scalability|fintech|payment gateway|transaction processing|acid compliance|event[- ]driven|event sourcing|cqrs|data pipeline|infrastructure|load balanc(ing|er)|caching layer|database replication)\b",
        re.IGNORECASE,
    )

    _REASONING_PATTERN = re.compile(
        r"\b(why does|solve this|riddle|puzzle|deduce|deduction|logical|prove that|mathematical proof|solve for|calculate probability|trade-off|tradeoff|causal|counterfactual|step-by-step reasoning|think carefully)\b",
        re.IGNORECASE,
    )

    _SUMMARIZATION_PATTERN = re.compile(
        r"\b(summarize|summary|tldr|tl;dr|key takeaways|condense|abstract of|executive summary|shorten|briefly explain)\b",
        re.IGNORECASE,
    )

    _STRUCTURED_PATTERN = re.compile(
        r"\b(json|xml|yaml|csv|table format|markdown table|pydantic model|json schema|key-value pairs|json object)\b",
        re.IGNORECASE,
    )

    _CREATIVE_PATTERN = re.compile(
        r"\b(poem|poetry|haiku|write a story|screenplay|fiction|creative writing|dialogue between|narrative|character background)\b",
        re.IGNORECASE,
    )

    _STEP_MARKERS = re.compile(
        r"(\b(step\s*\d+|firstly|secondly|finally|after that|then|next|follow these steps)\b|^\s*\d+[\.\)]\s+)",
        re.IGNORECASE | re.MULTILINE,
    )

    _CONSTRAINT_MARKERS = re.compile(
        r"\b(must not|do not|without using|no external|strictly|constraint|constraints|limit to|cannot|forbidden|only use|under \d+ words)\b",
        re.IGNORECASE,
    )

    _LIST_ITEMS = re.compile(
        r"^\s*(?:[-*•]|\d+[\.\)])\s+",
        re.MULTILINE,
    )

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        """Initialize the analyzer with configurable thresholds."""
        self.config = config or AnalyzerConfig()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using lightweight word and character heuristics.

        NOTE: This is a fast, deterministic estimate and NOT an exact tokenizer count.
        Actual token counts vary depending on model vocabularies (e.g. cl100k_base).
        """
        stripped = text.strip()
        if not stripped:
            return 0

        words = stripped.split()
        word_estimate = len(words) * self.config.words_per_token
        char_estimate = len(stripped) / self.config.chars_per_token

        # Blended average of word and character heuristics
        blended = (word_estimate + char_estimate) / 2.0
        return max(1, int(round(blended)))

    def detect_task_type(self, text: str) -> TaskType:
        """Determine task category using pattern heuristics."""
        # Check structured output first if explicit
        if self._STRUCTURED_PATTERN.search(text):
            # If code block with json, could be structured or code
            if "```" not in text or "json" in text.lower():
                return TaskType.STRUCTURED_OUTPUT

        # Check coding and software architecture
        if self._CODE_PATTERN.search(text) or self._ARCHITECTURE_PATTERN.search(text):
            return TaskType.CODING

        # Check reasoning
        if self._REASONING_PATTERN.search(text):
            return TaskType.REASONING

        # Check summarization
        if self._SUMMARIZATION_PATTERN.search(text):
            return TaskType.SUMMARIZATION

        # Check creative writing
        if self._CREATIVE_PATTERN.search(text):
            return TaskType.CREATIVE_WRITING

        # Check general inquiry keywords
        if re.search(r"\b(what is|who is|when was|where is|how does|can you explain|define)\b", text, re.IGNORECASE):
            return TaskType.GENERAL_QUESTION

        return TaskType.OTHER

    def detect_signals(self, text: str, estimated_tokens: int, task_type: TaskType) -> List[str]:
        """Detect complexity signals from query structure and vocabulary."""
        signals: List[str] = []

        # 1. Long prompt signal
        if estimated_tokens >= self.config.long_prompt_tokens:
            signals.append(f"long_prompt (estimated {estimated_tokens} tokens)")

        # 2. Multi-step instructions
        step_matches = self._STEP_MARKERS.findall(text)
        if len(step_matches) >= self.config.multi_step_threshold:
            signals.append(f"multi_step_instructions ({len(step_matches)} steps detected)")

        # 3. Explicit requirements (bullet or numbered lists)
        list_matches = self._LIST_ITEMS.findall(text)
        if len(list_matches) >= self.config.requirement_threshold:
            signals.append(f"multiple_explicit_requirements ({len(list_matches)} items detected)")

        # 4. Restrictive constraints
        constraint_matches = self._CONSTRAINT_MARKERS.findall(text)
        if len(constraint_matches) >= self.config.constraint_threshold:
            signals.append(f"multiple_constraints ({len(constraint_matches)} constraints detected)")

        # 5. Complex coding and software architecture requests
        if task_type == TaskType.CODING or self._ARCHITECTURE_PATTERN.search(text):
            if re.search(
                r"\b(refactor|architect|architecture|system design|optimize|concurrency|async|full implementation|unit test|microservices?|distributed system|high availability|fault tolerance|scalability|fintech|transaction processing|acid)\b",
                text,
                re.IGNORECASE,
            ):
                signals.append("complex_coding_request")

        # 6. Deep reasoning requested
        if task_type == TaskType.REASONING:
            signals.append("reasoning_task")

        return signals

    def classify_complexity(
        self,
        estimated_tokens: int,
        task_type: TaskType,
        signals: List[str],
    ) -> ComplexityLevel:
        """Classify prompt into simple, medium, or complex based on signals and size."""
        # Very long prompts are automatically complex
        if estimated_tokens >= self.config.very_long_prompt_tokens:
            return ComplexityLevel.COMPLEX

        # Explicit complex coding or architecture request signal indicates high complexity
        if "complex_coding_request" in signals:
            return ComplexityLevel.COMPLEX

        # Multiple complex signals indicate high complexity
        if len(signals) >= self.config.complex_signals_threshold:
            return ComplexityLevel.COMPLEX

        # Complex task with at least one signal
        if task_type in (TaskType.CODING, TaskType.REASONING) and len(signals) >= 1:
            return ComplexityLevel.COMPLEX

        # Moderate criteria: single signal or moderately demanding tasks
        if len(signals) == 1:
            return ComplexityLevel.MEDIUM

        if task_type in (TaskType.CODING, TaskType.REASONING, TaskType.STRUCTURED_OUTPUT):
            return ComplexityLevel.MEDIUM

        if estimated_tokens >= 60:
            return ComplexityLevel.MEDIUM

        return ComplexityLevel.SIMPLE

    def analyze(self, query: str) -> AnalysisResult:
        """Analyze a user query synchronously or asynchronously.

        Returns structured analysis result containing complexity, task type,
        token approximation, reasoning requirements, and detected signals.

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        if query is None or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        cleaned_query = query.strip()
        estimated_tokens = self.estimate_tokens(cleaned_query)
        task_type = self.detect_task_type(cleaned_query)
        signals = self.detect_signals(cleaned_query, estimated_tokens, task_type)
        complexity = self.classify_complexity(estimated_tokens, task_type, signals)

        # Determine reasoning requirements
        requires_reasoning = (
            task_type == TaskType.REASONING
            or complexity == ComplexityLevel.COMPLEX
            or any("reasoning" in s for s in signals)
        )

        # Map to suggested model tier
        if complexity == ComplexityLevel.COMPLEX or requires_reasoning:
            suggested_tier = "advanced"
        elif complexity == ComplexityLevel.MEDIUM:
            suggested_tier = "balanced"
        else:
            suggested_tier = "lightweight"

        domain_tags = [task_type.value]
        if "complex_coding_request" in signals:
            domain_tags.append("software_engineering")

        return AnalysisResult(
            complexity=complexity,
            task_type=task_type,
            estimated_input_tokens=estimated_tokens,
            requires_reasoning=requires_reasoning,
            detected_signals=signals,
            suggested_tier=suggested_tier,
            domain_tags=domain_tags,
        )

    async def analyze_async(self, query: str) -> AnalysisResult:
        """Asynchronous wrapper for prompt analysis."""
        return self.analyze(query)
