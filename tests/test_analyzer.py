"""Comprehensive unit tests for the deterministic prompt analyzer."""

import pytest
from app.orchestrator.analyzer import (
    AnalysisResult,
    AnalyzerConfig,
    ComplexityLevel,
    PromptAnalyzer,
    TaskType,
)


@pytest.fixture
def analyzer() -> PromptAnalyzer:
    """Fixture providing a standard PromptAnalyzer instance."""
    return PromptAnalyzer()


# 1. Empty and whitespace input handling
def test_analyzer_empty_input_raises_value_error(analyzer: PromptAnalyzer):
    """Verify empty string raises ValueError."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        analyzer.analyze("")


def test_analyzer_whitespace_only_raises_value_error(analyzer: PromptAnalyzer):
    """Verify whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        analyzer.analyze("    \t\n  ")


# 2. Simple factual questions
def test_analyzer_simple_factual_question(analyzer: PromptAnalyzer):
    """Verify a simple factual inquiry is classified as simple and general_question."""
    prompt = "What is the capital of France?"
    result = analyzer.analyze(prompt)

    assert result.complexity == ComplexityLevel.SIMPLE
    assert result.task_type == TaskType.GENERAL_QUESTION
    assert result.requires_reasoning is False
    assert result.suggested_tier == "lightweight"
    assert len(result.detected_signals) == 0


# 3. Coding requests
def test_analyzer_coding_request(analyzer: PromptAnalyzer):
    """Verify coding prompts are categorized as CODING with appropriate complexity."""
    prompt = "Write a Python function to sort a list using quicksort with def quicksort(arr):"
    result = analyzer.analyze(prompt)

    assert result.task_type == TaskType.CODING
    assert result.complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]
    assert "coding" in result.domain_tags


def test_analyzer_complex_coding_with_backticks(analyzer: PromptAnalyzer):
    """Verify coding requests with refactoring or architectural keywords detect complex signals."""
    prompt = """
    Please refactor the following Python class for async concurrency and optimize performance:
    ```python
    class DataWorker:
        def process(self):
            pass
    ```
    Ensure thread safety and write unit test cases.
    """
    result = analyzer.analyze(prompt)

    assert result.task_type == TaskType.CODING
    assert result.complexity == ComplexityLevel.COMPLEX
    assert "complex_coding_request" in result.detected_signals
    assert result.requires_reasoning is True


# 4. Multi-step and complex prompts with constraints
def test_analyzer_multi_step_and_constraints(analyzer: PromptAnalyzer):
    """Verify prompts with multiple sequential steps and constraints trigger complex classification."""
    prompt = """
    Step 1: Parse the incoming CSV telemetry records.
    Step 2: Calculate the standard deviation and rolling average.
    Step 3: Format the output as JSON schema.
    Constraints:
    - Must not use third-party libraries.
    - Strictly limit memory usage to 50MB.
    - Do not perform disk writes.
    """
    result = analyzer.analyze(prompt)

    assert result.complexity == ComplexityLevel.COMPLEX
    assert result.requires_reasoning is True
    assert result.suggested_tier == "advanced"

    signal_text = " ".join(result.detected_signals)
    assert "multi_step_instructions" in signal_text
    assert "multiple_constraints" in signal_text or "multiple_explicit_requirements" in signal_text


# 5. Token estimate output
def test_analyzer_token_estimate_properties(analyzer: PromptAnalyzer):
    """Verify token estimation is positive, scales with length, and is documented as an estimate."""
    short_prompt = "Hello world"
    long_prompt = "This is a much longer prompt designed to verify that the heuristic token count increases proportionally with word and character count."

    short_tokens = analyzer.estimate_tokens(short_prompt)
    long_tokens = analyzer.estimate_tokens(long_prompt)

    assert short_tokens >= 1
    assert long_tokens > short_tokens

    # Verify AnalysisResult fields and compatibility property
    result = analyzer.analyze(short_prompt)
    assert result.estimated_input_tokens == short_tokens
    assert result.estimated_tokens == short_tokens


# 6. Consistent structured results
def test_analyzer_consistent_structured_result(analyzer: PromptAnalyzer):
    """Verify all required fields exist and conform to types on AnalysisResult."""
    result = analyzer.analyze("Explain how photosynthesis converts sunlight into energy.")

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.complexity, ComplexityLevel)
    assert isinstance(result.task_type, TaskType)
    assert isinstance(result.estimated_input_tokens, int)
    assert isinstance(result.requires_reasoning, bool)
    assert isinstance(result.detected_signals, list)
    assert isinstance(result.suggested_tier, str)


# 7. Additional task categories
def test_analyzer_reasoning_task(analyzer: PromptAnalyzer):
    """Verify reasoning riddles/logic puzzles trigger reasoning task type and flag."""
    prompt = "Solve this logic puzzle step-by-step reasoning: Why does water boil at a lower temperature at high altitude?"
    result = analyzer.analyze(prompt)

    assert result.task_type == TaskType.REASONING
    assert result.requires_reasoning is True


def test_analyzer_summarization_task(analyzer: PromptAnalyzer):
    """Verify summarization keywords trigger summarization task type."""
    prompt = "TL;DR summarize the key takeaways of the following quarterly earnings report."
    result = analyzer.analyze(prompt)

    assert result.task_type == TaskType.SUMMARIZATION


def test_analyzer_structured_output_task(analyzer: PromptAnalyzer):
    """Verify structured output requests trigger structured_output category."""
    prompt = "Extract customer name and phone number and format the result as a JSON object."
    result = analyzer.analyze(prompt)

    assert result.task_type == TaskType.STRUCTURED_OUTPUT


def test_analyzer_creative_writing_task(analyzer: PromptAnalyzer):
    """Verify poetry and story requests trigger creative_writing category."""
    prompt = "Write a short haiku about a robot learning to paint."
    result = analyzer.analyze(prompt)

    assert result.task_type == TaskType.CREATIVE_WRITING


# 8. Async compatibility (await analyzer.analyze_async)
@pytest.mark.asyncio
async def test_analyzer_async_wrapper_compatibility(analyzer: PromptAnalyzer):
    """Verify analyze_async returns valid result when awaited in async pipelines."""
    result = await analyzer.analyze_async("What is the speed of light?")
    assert result.complexity == ComplexityLevel.SIMPLE
    assert result.estimated_input_tokens > 0


# 9. Configurable thresholds
def test_analyzer_custom_configuration():
    """Verify custom AnalyzerConfig influences signal detection thresholds."""
    custom_config = AnalyzerConfig(
        long_prompt_tokens=10,  # Unusually low threshold to trigger long_prompt easily
    )
    custom_analyzer = PromptAnalyzer(config=custom_config)

    prompt = "This sentence contains enough words to exceed the ten token custom limit easily."
    result = custom_analyzer.analyze(prompt)

    assert any("long_prompt" in s for s in result.detected_signals)
