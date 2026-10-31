"""Unit tests for the model registry."""

import pytest
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier


@pytest.fixture
def empty_registry() -> ModelRegistry:
    """Fixture providing a fresh empty registry."""
    return ModelRegistry()


@pytest.fixture
def default_registry() -> ModelRegistry:
    """Fixture providing a registry populated with default reference models."""
    return ModelRegistry.with_default_models()


# 1. Listing models
def test_list_models_empty(empty_registry: ModelRegistry):
    """Verify empty registry lists zero models."""
    assert len(empty_registry.list_models()) == 0
    assert len(empty_registry.list_all()) == 0


def test_list_models_default(default_registry: ModelRegistry):
    """Verify default registry contains models."""
    all_models = default_registry.list_all()
    enabled_models = default_registry.list_models(only_enabled=True)

    assert len(all_models) > 0
    assert len(enabled_models) > 0
    # There is 1 disabled reference model in defaults
    assert len(all_models) > len(enabled_models)


# 2. Retrieving a known model
def test_retrieve_known_model(default_registry: ModelRegistry):
    """Verify known model retrieval by model_id."""
    model = default_registry.get("gpt-4o-mini")
    assert model is not None
    assert model.model_id == "gpt-4o-mini"
    assert model.provider == "openai"
    assert model.tier == ModelTier.LIGHTWEIGHT
    assert "tools" in model.supported_capabilities


def test_get_model_alias(default_registry: ModelRegistry):
    """Verify get_model alias behaves identically to get."""
    model = default_registry.get_model("claude-3-5-sonnet")
    assert model is not None
    assert model.model_id == "claude-3-5-sonnet"
    assert model.tier == ModelTier.BALANCED


# 3. Handling unknown model IDs
def test_retrieve_unknown_model_returns_none(default_registry: ModelRegistry):
    """Verify get returns None for unregistered models."""
    assert default_registry.get("non-existent-model-id") is None
    assert default_registry.get_model("non-existent-model-id") is None


def test_get_or_raise_unknown_model_raises_key_error(default_registry: ModelRegistry):
    """Verify get_or_raise raises KeyError with informative message."""
    with pytest.raises(KeyError, match="not registered"):
        default_registry.get_or_raise("non-existent-model-id")


# 4. Filtering by tier
def test_filter_by_tier(default_registry: ModelRegistry):
    """Verify filtering models by tier returns matching candidates."""
    lightweight_models = default_registry.get_models_by_tier(ModelTier.LIGHTWEIGHT)
    assert len(lightweight_models) > 0
    for model in lightweight_models:
        assert model.tier == ModelTier.LIGHTWEIGHT

    advanced_models = default_registry.get_models_by_tier(ModelTier.ADVANCED)
    assert len(advanced_models) > 0
    for model in advanced_models:
        assert model.tier == ModelTier.ADVANCED


def test_filter_by_tier_string(default_registry: ModelRegistry):
    """Verify filtering by string tier representation."""
    balanced_models = default_registry.get_models_by_tier("balanced")
    assert len(balanced_models) > 0
    for model in balanced_models:
        assert model.tier == ModelTier.BALANCED


# 5. Excluding disabled models
def test_excluding_disabled_models(empty_registry: ModelRegistry):
    """Verify disabled models are omitted when only_enabled=True and included when False."""
    active_model = ModelDescriptor(
        model_id="active-model",
        provider="test",
        tier=ModelTier.LIGHTWEIGHT,
        enabled=True,
    )
    disabled_model = ModelDescriptor(
        model_id="disabled-model",
        provider="test",
        tier=ModelTier.LIGHTWEIGHT,
        enabled=False,
    )

    empty_registry.register(active_model)
    empty_registry.register(disabled_model)

    # When only_enabled is True (default)
    enabled_list = empty_registry.list_models(only_enabled=True)
    assert len(enabled_list) == 1
    assert enabled_list[0].model_id == "active-model"

    # When only_enabled is False
    all_list = empty_registry.list_models(only_enabled=False)
    assert len(all_list) == 2

    # Tier filtering exclusion
    tier_enabled = empty_registry.get_models_by_tier(ModelTier.LIGHTWEIGHT, only_enabled=True)
    assert len(tier_enabled) == 1
    assert tier_enabled[0].model_id == "active-model"

    tier_all = empty_registry.get_models_by_tier(ModelTier.LIGHTWEIGHT, only_enabled=False)
    assert len(tier_all) == 2


# 6. Missing or unknown pricing
def test_missing_or_unknown_pricing(empty_registry: ModelRegistry):
    """Verify model entries support optional/None pricing by default without failure."""
    model_unpriced = ModelDescriptor(
        model_id="unpriced-model",
        provider="test-provider",
        tier=ModelTier.LIGHTWEIGHT,
    )
    assert model_unpriced.input_token_price is None
    assert model_unpriced.output_token_price is None

    # Model with explicitly configured pricing
    model_priced = ModelDescriptor(
        model_id="priced-model",
        provider="test-provider",
        tier=ModelTier.BALANCED,
        input_token_price=0.0015,
        output_token_price=0.006,
    )
    assert model_priced.input_token_price == 0.0015
    assert model_priced.output_token_price == 0.006


# 7. Dynamic registration and unregistration
def test_register_and_unregister(empty_registry: ModelRegistry):
    """Verify dynamic adding and removing of models."""
    custom_model = ModelDescriptor(
        model_id="custom-model-x",
        provider="local",
        tier=ModelTier.BALANCED,
    )

    empty_registry.register(custom_model)
    assert empty_registry.get("custom-model-x") is not None

    removed = empty_registry.unregister("custom-model-x")
    assert removed is not None
    assert removed.model_id == "custom-model-x"
    assert empty_registry.get("custom-model-x") is None
    assert empty_registry.unregister("custom-model-x") is None

