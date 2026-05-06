"""Tests for configuration validation, model config dataclasses, and per-model defaults.

Covers:
- ConfigValidator (validate_base_config, validate_sequential_config, validate_implicit_config)
- validate_config top-level function
- BaseModelConfig, BaseTrainingConfig, BaseDataConfig dataclasses
- Per-model default config classes
- Property-based tests for correctness properties 4–8, 39
"""

import pytest
from dataclasses import dataclass
from hypothesis import given, settings
from hypothesis import strategies as st

from rec_arena.configs.validation import ConfigValidator, validate_config
from rec_arena.configs.base import BaseModelConfig, BaseTrainingConfig, BaseDataConfig
from rec_arena.configs import (
    SASRecConfig,
    GRU4RecConfig,
    BERT4RecConfig,
    RecMConfig,
    NCFConfig,
    BPRMFConfig,
    HSTUConfig,
    EASEConfig,
    SLIMConfig,
    ItemKNNConfig,
)

# Imports for config defaults coverage tests (from test_config_defaults_coverage.py)
from rec_arena.configs.defaults.llada4rec import LLaDA4RecConfig
from rec_arena.configs.defaults.lightgcn import LightGCNConfig
from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig
from rec_arena.configs.defaults.caser import CaserConfig
from rec_arena.configs.defaults.fmlprec import FMLPRecConfig

# Imports for config validation branches tests (from test_config_validation_branches.py)
from rec_arena.configs.defaults.twotower import TwoTowerConfig
from rec_arena.configs.defaults.simplex import SimpleXConfig
from rec_arena.configs.defaults.fuxi import FuXiConfig

# Imports for search spaces tests (from test_search_spaces.py)
from unittest.mock import MagicMock
from rec_arena.configs.search_spaces import (
    SASRecSearchSpace,
    GRU4RecSearchSpace,
    BERT4RecSearchSpace,
    NCFSearchSpace,
    BPRMFSearchSpace,
)

# Note: shared hypothesis strategies (valid_base_config, etc.) are available
# via conftest.py auto-loading but are not directly imported here since
# property tests below define their strategies inline for clarity.


# ============================================================================
# 3.1 — Config validation unit tests (Requirements 3.1–3.9)
# ============================================================================


class TestValidateBaseConfig:
    """Unit tests for ConfigValidator.validate_base_config."""

    def test_valid_config_passes(self, base_config_dict):
        """Req 3.1: valid base config passes without error."""
        ConfigValidator.validate_base_config(base_config_dict)

    def test_missing_vocab_size_raises(self):
        """Req 3.2: missing vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size"):
            ConfigValidator.validate_base_config({"embedding_dim": 64})

    def test_missing_embedding_dim_raises(self):
        """Req 3.2: missing embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim"):
            ConfigValidator.validate_base_config({"vocab_size": 100})

    def test_non_positive_vocab_size_raises(self):
        """Req 3.3: non-positive vocab_size raises ValueError."""
        with pytest.raises(ValueError):
            ConfigValidator.validate_base_config({"vocab_size": 0, "embedding_dim": 64})

    def test_negative_embedding_dim_raises(self):
        """Req 3.3: negative embedding_dim raises ValueError."""
        with pytest.raises(ValueError):
            ConfigValidator.validate_base_config({"vocab_size": 100, "embedding_dim": -1})


class TestValidateSequentialConfig:
    """Unit tests for ConfigValidator.validate_sequential_config."""

    def test_valid_config_passes(self, sequential_config_dict):
        """Req 3.4: valid sequential config passes without error."""
        ConfigValidator.validate_sequential_config(sequential_config_dict)

    def test_invalid_loss_type_raises(self, sequential_config_dict):
        """Req 3.5: invalid loss_type raises ValueError."""
        sequential_config_dict["loss_type"] = "invalid_loss"
        with pytest.raises(ValueError, match="Invalid loss_type"):
            ConfigValidator.validate_sequential_config(sequential_config_dict)

    def test_valid_loss_types_accepted(self, sequential_config_dict):
        """Req 3.4: all valid sequential loss types pass."""
        for loss in ["cross_entropy", "bce", "bpr", "sampled_softmax", "gbce"]:
            sequential_config_dict["loss_type"] = loss
            ConfigValidator.validate_sequential_config(sequential_config_dict)

    def test_non_positive_max_seq_length_raises(self, sequential_config_dict):
        """Req 3.3: non-positive max_seq_length raises ValueError."""
        sequential_config_dict["max_seq_length"] = 0
        with pytest.raises(ValueError, match="max_seq_length"):
            ConfigValidator.validate_sequential_config(sequential_config_dict)


class TestValidateImplicitConfig:
    """Unit tests for ConfigValidator.validate_implicit_config."""

    def test_valid_config_passes(self, implicit_config_dict):
        """Req 3.6: valid implicit config passes without error."""
        ConfigValidator.validate_implicit_config(implicit_config_dict)

    def test_invalid_loss_type_raises(self, implicit_config_dict):
        """Req 3.7: invalid loss_type raises ValueError."""
        implicit_config_dict["loss_type"] = "cross_entropy"
        with pytest.raises(ValueError, match="Invalid loss_type"):
            ConfigValidator.validate_implicit_config(implicit_config_dict)

    def test_valid_loss_types_accepted(self, implicit_config_dict):
        """Req 3.6: all valid implicit loss types pass."""
        for loss in ["bce", "bpr"]:
            implicit_config_dict["loss_type"] = loss
            ConfigValidator.validate_implicit_config(implicit_config_dict)

    def test_missing_num_users_raises(self):
        """Req 3.2: missing num_users raises ValueError."""
        with pytest.raises(ValueError, match="num_users"):
            ConfigValidator.validate_implicit_config(
                {"num_items": 100, "embedding_dim": 64}
            )

    def test_non_positive_num_items_raises(self):
        """Req 3.3: non-positive num_items raises ValueError."""
        with pytest.raises(ValueError):
            ConfigValidator.validate_implicit_config(
                {"num_users": 50, "num_items": -1, "embedding_dim": 64}
            )


class TestValidateConfig:
    """Unit tests for the top-level validate_config function."""

    def test_sequential_dict(self, sequential_config_dict):
        """Req 3.8: dict config with model_type='sequential' validates."""
        validate_config(sequential_config_dict, model_type="sequential")

    def test_implicit_dict(self, implicit_config_dict):
        """Req 3.8: dict config with model_type='implicit' validates."""
        validate_config(implicit_config_dict, model_type="implicit")

    def test_dataclass_input(self):
        """Req 3.8: dataclass config is converted to dict and validated."""

        @dataclass
        class FakeConfig:
            vocab_size: int = 100
            embedding_dim: int = 64

        validate_config(FakeConfig(), model_type="sequential")

    def test_unknown_model_type_raises(self, base_config_dict):
        """Req 3.9: unknown model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            validate_config(base_config_dict, model_type="graph")


# ============================================================================
# 3.2 — Property 4: Valid configs pass validation (Requirements 3.1, 3.4, 3.6)
# ============================================================================


# Feature: comprehensive-test-suite, Property 4: Valid configs pass validation
@given(
    vocab_size=st.integers(min_value=1, max_value=10_000),
    embedding_dim=st.integers(min_value=1, max_value=512),
)
@settings(max_examples=100)
def test_property_valid_base_configs_pass(vocab_size, embedding_dim):
    """Property 4: Any config with positive vocab_size and embedding_dim passes validate_base_config."""
    config = {"vocab_size": vocab_size, "embedding_dim": embedding_dim}
    ConfigValidator.validate_base_config(config)


@given(
    vocab_size=st.integers(min_value=1, max_value=10_000),
    embedding_dim=st.integers(min_value=1, max_value=512),
    loss_type=st.sampled_from(["cross_entropy", "bce", "bpr", "sampled_softmax", "gbce"]),
)
@settings(max_examples=100)
def test_property_valid_sequential_configs_pass(vocab_size, embedding_dim, loss_type):
    """Property 4: Valid sequential configs pass validation."""
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "loss_type": loss_type,
    }
    ConfigValidator.validate_sequential_config(config)


@given(
    num_users=st.integers(min_value=1, max_value=10_000),
    num_items=st.integers(min_value=1, max_value=10_000),
    embedding_dim=st.integers(min_value=1, max_value=512),
    loss_type=st.sampled_from(["bce", "bpr"]),
)
@settings(max_examples=100)
def test_property_valid_implicit_configs_pass(num_users, num_items, embedding_dim, loss_type):
    """Property 4: Valid implicit configs pass validation."""
    config = {
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "loss_type": loss_type,
    }
    ConfigValidator.validate_implicit_config(config)


# ============================================================================
# 3.3 — Property 5: Non-positive config values rejected (Requirement 3.3)
# ============================================================================


# Feature: comprehensive-test-suite, Property 5: Non-positive config values are rejected
@given(
    bad_value=st.integers(max_value=0),
    field=st.sampled_from(["vocab_size", "embedding_dim"]),
)
@settings(max_examples=100)
def test_property_non_positive_base_config_rejected(bad_value, field):
    """Property 5: Non-positive vocab_size or embedding_dim raises ValueError."""
    config = {"vocab_size": 100, "embedding_dim": 64}
    config[field] = bad_value
    with pytest.raises(ValueError):
        ConfigValidator.validate_base_config(config)


# ============================================================================
# 3.4 — Property 6: Invalid loss types rejected (Requirements 3.5, 3.7)
# ============================================================================

VALID_SEQUENTIAL_LOSSES = {"cross_entropy", "bce", "bpr", "sampled_softmax", "gbce"}
VALID_IMPLICIT_LOSSES = {"bce", "bpr"}


# Feature: comprehensive-test-suite, Property 6: Invalid loss types are rejected by config validation
@given(
    loss_type=st.text(min_size=1, max_size=30).filter(
        lambda s: s not in VALID_SEQUENTIAL_LOSSES
    ),
)
@settings(max_examples=100)
def test_property_invalid_sequential_loss_rejected(loss_type):
    """Property 6: Invalid loss_type rejected by validate_sequential_config."""
    config = {"vocab_size": 100, "embedding_dim": 64, "loss_type": loss_type}
    with pytest.raises(ValueError, match="Invalid loss_type"):
        ConfigValidator.validate_sequential_config(config)


@given(
    loss_type=st.text(min_size=1, max_size=30).filter(
        lambda s: s not in VALID_IMPLICIT_LOSSES
    ),
)
@settings(max_examples=100)
def test_property_invalid_implicit_loss_rejected(loss_type):
    """Property 6: Invalid loss_type rejected by validate_implicit_config."""
    config = {
        "num_users": 50,
        "num_items": 100,
        "embedding_dim": 64,
        "loss_type": loss_type,
    }
    with pytest.raises(ValueError, match="Invalid loss_type"):
        ConfigValidator.validate_implicit_config(config)


# ============================================================================
# 3.5 — Property 7: Unknown model_type rejected (Requirement 3.9)
# ============================================================================


# Feature: comprehensive-test-suite, Property 7: Unknown model_type is rejected
@given(
    model_type=st.text(min_size=1, max_size=30).filter(
        lambda s: s not in {"sequential", "implicit"}
    ),
)
@settings(max_examples=100)
def test_property_unknown_model_type_rejected(model_type):
    """Property 7: Unknown model_type raises ValueError in validate_config."""
    config = {"vocab_size": 100, "embedding_dim": 64}
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_config(config, model_type=model_type)


# ============================================================================
# 3.6 — Model config dataclass unit tests (Requirements 4.1–4.5)
# ============================================================================


class TestBaseModelConfig:
    """Unit tests for BaseModelConfig defaults and get accessor."""

    def test_defaults(self):
        """Req 4.1: BaseModelConfig defaults match documented values."""
        cfg = BaseModelConfig()
        assert cfg.embedding_dim == 64
        assert cfg.lr == 0.001

    def test_get_existing_attribute(self):
        """Req 4.2: get returns correct value for existing attribute."""
        cfg = BaseModelConfig()
        assert cfg.get("embedding_dim") == 64
        assert cfg.get("lr") == 0.001

    def test_get_nonexistent_attribute_returns_default(self):
        """Req 4.3: get returns default for non-existent attribute."""
        cfg = BaseModelConfig()
        assert cfg.get("nonexistent", 42) == 42
        assert cfg.get("missing") is None


class TestBaseTrainingConfig:
    """Unit tests for BaseTrainingConfig defaults."""

    def test_defaults(self):
        """Req 4.4: BaseTrainingConfig defaults match documented values."""
        cfg = BaseTrainingConfig()
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 256


class TestBaseDataConfig:
    """Unit tests for BaseDataConfig defaults."""

    def test_defaults(self):
        """Req 4.5: BaseDataConfig defaults match documented values."""
        cfg = BaseDataConfig()
        assert cfg.name == "ml100k"
        assert cfg.min_interactions == 5


# ============================================================================
# 3.7 — Property 8: Config get accessor (Requirements 4.2, 4.3)
# ============================================================================


# Feature: comprehensive-test-suite, Property 8: Config get accessor returns correct values
@given(default_val=st.integers() | st.text(max_size=20) | st.none())
@settings(max_examples=100)
def test_property_config_get_existing_attribute(default_val):
    """Property 8: get(attr) returns same as getattr for existing attributes."""
    cfg = BaseModelConfig()
    for attr in ("embedding_dim", "lr", "weight_decay"):
        assert cfg.get(attr) == getattr(cfg, attr)


@given(
    attr_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_").filter(
        lambda s: not hasattr(BaseModelConfig(), s)
    ),
    default_val=st.integers(min_value=-1000, max_value=1000),
)
@settings(max_examples=100)
def test_property_config_get_nonexistent_returns_default(attr_name, default_val):
    """Property 8: get(non_existent, default) returns the default."""
    cfg = BaseModelConfig()
    assert cfg.get(attr_name, default_val) == default_val


# ============================================================================
# 3.8 — Per-model default config tests (Requirements 16.1, 16.2, 16.3)
# ============================================================================

# Configs that require num_users/num_items to instantiate
_CONFIGS_NEEDING_USERS_ITEMS = {
    EASEConfig: {"num_users": 100, "num_items": 200},
    SLIMConfig: {"num_users": 100, "num_items": 200},
    ItemKNNConfig: {"num_users": 100, "num_items": 200},
}

# All model config classes listed in the task
ALL_MODEL_CONFIGS = [
    SASRecConfig,
    GRU4RecConfig,
    BERT4RecConfig,
    RecMConfig,
    NCFConfig,
    BPRMFConfig,
    HSTUConfig,
    EASEConfig,
    SLIMConfig,
    ItemKNNConfig,
]


def _instantiate_config(cls):
    """Instantiate a config class, providing required kwargs if needed."""
    kwargs = _CONFIGS_NEEDING_USERS_ITEMS.get(cls, {})
    return cls(**kwargs)


@pytest.mark.parametrize("config_cls", ALL_MODEL_CONFIGS, ids=lambda c: c.__name__)
class TestPerModelDefaults:
    """Req 16.1–16.3: each model config instantiates and has valid defaults."""

    def test_instantiates_with_no_extra_args(self, config_cls):
        """Req 16.1: config class instantiates (with required kwargs only)."""
        cfg = _instantiate_config(config_cls)
        assert cfg is not None

    def test_embedding_dim_positive_int(self, config_cls):
        """Req 16.2: embedding_dim is a positive integer."""
        cfg = _instantiate_config(config_cls)
        assert isinstance(cfg.embedding_dim, int)
        assert cfg.embedding_dim > 0

    def test_lr_positive_float(self, config_cls):
        """Req 16.3: lr is a positive float."""
        cfg = _instantiate_config(config_cls)
        assert isinstance(cfg.lr, float)
        assert cfg.lr > 0


# ============================================================================
# 3.9 — Property 39: Model config defaults have valid types (Req 16.2, 16.3)
# ============================================================================


# Feature: comprehensive-test-suite, Property 39: Model config defaults have valid types
@given(data=st.data())
@settings(max_examples=100)
def test_property_model_config_defaults_valid_types(data):
    """Property 39: For any model config class, embedding_dim is a positive int and lr is a positive float."""
    config_cls = data.draw(st.sampled_from(ALL_MODEL_CONFIGS))
    cfg = _instantiate_config(config_cls)
    assert isinstance(cfg.embedding_dim, int) and cfg.embedding_dim > 0
    assert isinstance(cfg.lr, float) and cfg.lr > 0


# ============================================================================
# SECTION: Config Defaults Coverage Tests
# (from tests/test_config_defaults_coverage.py)
# Covers: LLaDA4RecConfig, LightGCNConfig, MLP4RecConfig, FuXiGammaConfig,
# CaserConfig, FMLPRecConfig, RecMConfig
# ============================================================================


class TestLLaDA4RecConfig:
    def test_default_instantiation(self):
        config = LLaDA4RecConfig(vocab_size=100)
        assert config.eps == 0.01
        assert config.diffusion_steps == 50

    def test_invalid_eps_zero(self):
        with pytest.raises(ValueError, match="eps"):
            LLaDA4RecConfig(vocab_size=100, eps=0.0)

    def test_invalid_eps_one(self):
        with pytest.raises(ValueError, match="eps"):
            LLaDA4RecConfig(vocab_size=100, eps=1.0)

    def test_invalid_eps_negative(self):
        with pytest.raises(ValueError, match="eps"):
            LLaDA4RecConfig(vocab_size=100, eps=-0.1)

    def test_invalid_diffusion_steps(self):
        with pytest.raises(ValueError, match="diffusion_steps"):
            LLaDA4RecConfig(vocab_size=100, diffusion_steps=0)

    def test_invalid_remasking_strategy(self):
        with pytest.raises(ValueError, match="remasking_strategy"):
            LLaDA4RecConfig(vocab_size=100, remasking_strategy="invalid")

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            LLaDA4RecConfig(vocab_size=100, temperature=-1.0)

    def test_valid_custom_params(self):
        config = LLaDA4RecConfig(
            vocab_size=100, eps=0.5, diffusion_steps=10,
            remasking_strategy="random", temperature=0.5
        )
        assert config.eps == 0.5
        assert config.remasking_strategy == "random"


class TestLightGCNConfig:
    def test_default_instantiation(self):
        config = LightGCNConfig()
        assert config.num_layers == 3
        assert config.val_k_values == [10, 20]

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            LightGCNConfig(num_layers=0)

    def test_custom_val_k_values(self):
        config = LightGCNConfig(val_k_values=[5, 10, 50])
        assert config.val_k_values == [5, 10, 50]


class TestMLP4RecConfig:
    def test_default_instantiation(self):
        config = MLP4RecConfig()
        assert config.embedding_dim == 64
        assert config.hidden_dims is not None
        assert len(config.hidden_dims) == config.num_layers

    def test_auto_hidden_dims(self):
        config = MLP4RecConfig(embedding_dim=32, hidden_multiplier=2, num_layers=3)
        assert config.hidden_dims == [64, 64, 64]

    def test_custom_hidden_dims(self):
        config = MLP4RecConfig(hidden_dims=[128, 64])
        assert config.hidden_dims == [128, 64]

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            MLP4RecConfig(dropout_rate=1.5)

    def test_invalid_dropout_negative(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            MLP4RecConfig(dropout_rate=-0.1)

    def test_invalid_pooling(self):
        with pytest.raises(ValueError, match="pooling"):
            MLP4RecConfig(pooling="invalid")

    def test_valid_pooling_options(self):
        for p in ["mean", "max", "last", "attention", "multi"]:
            config = MLP4RecConfig(pooling=p)
            assert config.pooling == p

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size"):
            MLP4RecConfig(vocab_size=1)

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="lr"):
            MLP4RecConfig(lr=-0.001)


class TestFuXiGammaConfig:
    def test_default_instantiation(self):
        config = FuXiGammaConfig()
        assert config.embedding_dim == 64


class TestCaserConfigDefaults:
    def test_default_instantiation(self):
        config = CaserConfig()
        assert config.embedding_dim == 64

    def test_custom_params(self):
        config = CaserConfig(embedding_dim=128, num_horizontal_filters=8)
        assert config.embedding_dim == 128


class TestFMLPRecConfig:
    def test_default_instantiation(self):
        config = FMLPRecConfig()
        assert config.embedding_dim == 64
        assert config.mlp_hidden_dim == 256  # 4 * 64

    def test_custom_params(self):
        config = FMLPRecConfig(embedding_dim=128, num_blocks=4)
        assert config.num_blocks == 4
        assert config.mlp_hidden_dim == 512  # 4 * 128

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            FMLPRecConfig(dropout_rate=1.5)

    def test_invalid_gradient_clip(self):
        with pytest.raises(ValueError, match="gradient_clip_val"):
            FMLPRecConfig(gradient_clip_val=0)


class TestRecMConfigDefaults:
    def test_default_instantiation(self):
        config = RecMConfig()
        assert config.embedding_dim == 64

    def test_custom_params(self):
        config = RecMConfig(embedding_dim=128)
        assert config.embedding_dim == 128



# ============================================================================
# SECTION: Config Validation Branches Tests
# (from tests/test_config_validation_branches.py)
# Covers: EASE, SLIM, ItemKNN, RecM, SASRec, GRU4Rec, NCF, TwoTower,
# BPRMF, SimpleX, BERT4Rec, Caser, HSTU, FuXi validation branches
# ============================================================================


class TestEASEConfigBranches:
    def test_custom_reg_lambda(self):
        config = EASEConfig(num_users=10, num_items=20, reg_lambda=100.0)
        assert config.reg_lambda == 100.0

    def test_negative_reg_lambda(self):
        with pytest.raises(ValueError):
            EASEConfig(num_users=10, num_items=20, reg_lambda=-1.0)


class TestSLIMConfigBranches:
    def test_custom_alpha(self):
        config = SLIMConfig(num_users=10, num_items=20, alpha=0.5)
        assert config.alpha == 0.5

    def test_negative_alpha(self):
        with pytest.raises(ValueError):
            SLIMConfig(num_users=10, num_items=20, alpha=-1.0)

    def test_negative_l1_ratio(self):
        with pytest.raises(ValueError):
            SLIMConfig(num_users=10, num_items=20, l1_ratio=-0.1)

    def test_l1_ratio_above_one(self):
        with pytest.raises(ValueError):
            SLIMConfig(num_users=10, num_items=20, l1_ratio=1.5)


class TestItemKNNConfigBranches:
    def test_custom_k(self):
        config = ItemKNNConfig(num_users=10, num_items=20, k=50)
        assert config.k == 50

    def test_zero_k(self):
        with pytest.raises(ValueError):
            ItemKNNConfig(num_users=10, num_items=20, k=0)

    def test_invalid_similarity(self):
        with pytest.raises(ValueError):
            ItemKNNConfig(num_users=10, num_items=20, similarity="invalid")

    def test_valid_similarities(self):
        for sim in ["cosine", "jaccard"]:
            config = ItemKNNConfig(num_users=10, num_items=20, similarity=sim)
            assert config.similarity == sim

    def test_negative_shrinkage(self):
        with pytest.raises(ValueError):
            ItemKNNConfig(num_users=10, num_items=20, shrinkage=-1)


class TestRecMConfigBranches:
    def test_invalid_ensemble_size(self):
        with pytest.raises(ValueError, match="ensemble_size"):
            RecMConfig(ensemble_size=0)

    def test_invalid_embedding_dim_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            RecMConfig(embedding_dim=65, num_heads=2)

    def test_invalid_ensemble_loss_count(self):
        with pytest.raises(ValueError, match="ensemble_loss_functions"):
            RecMConfig(ensemble_size=4, ensemble_loss_functions=["bpr", "bce", "gbce"])

    def test_valid_ensemble_loss_single(self):
        config = RecMConfig(ensemble_size=4, ensemble_loss_functions=["bpr"])
        assert config.ensemble_loss_functions == ["bpr"]

    def test_valid_ensemble_loss_per_member(self):
        config = RecMConfig(ensemble_size=2, ensemble_loss_functions=["bpr", "bce"])
        assert len(config.ensemble_loss_functions) == 2

    def test_valid_ensemble_loss_divisible(self):
        config = RecMConfig(ensemble_size=4, ensemble_loss_functions=["bpr", "bce"])
        assert len(config.ensemble_loss_functions) == 2


class TestSASRecConfigBranches:
    def test_invalid_embedding_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            SASRecConfig(vocab_size=100, embedding_dim=65, num_heads=2)

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            SASRecConfig(vocab_size=100, num_layers=0)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            SASRecConfig(vocab_size=100, dropout_rate=1.5)

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size"):
            SASRecConfig(vocab_size=3)

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="lr"):
            SASRecConfig(vocab_size=100, lr=-0.001)


class TestGRU4RecConfigBranches:
    def test_custom_params(self):
        config = GRU4RecConfig(vocab_size=100, embedding_dim=32, num_layers=3)
        assert config.num_layers == 3

    def test_invalid_dropout(self):
        with pytest.raises(ValueError):
            GRU4RecConfig(vocab_size=100, dropout_rate=1.5)

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError):
            GRU4RecConfig(vocab_size=100, num_layers=0)


class TestNCFConfigBranches:
    def test_custom_params(self):
        config = NCFConfig(num_users=100, num_items=200)
        assert config.num_users == 100

    def test_invalid_dropout(self):
        with pytest.raises(ValueError):
            NCFConfig(dropout_rate=1.5)


class TestTwoTowerConfigBranches:
    def test_custom_params(self):
        config = TwoTowerConfig(num_users=100, num_items=200)
        assert config.num_users == 100

    def test_invalid_dropout(self):
        with pytest.raises(ValueError):
            TwoTowerConfig(dropout_rate=1.5)


class TestBPRMFConfigBranches:
    def test_custom_params(self):
        config = BPRMFConfig(num_users=100, num_items=200)
        assert config.num_users == 100

    def test_custom_embedding_dim(self):
        config = BPRMFConfig(num_users=100, num_items=200, embedding_dim=128)
        assert config.embedding_dim == 128


class TestSimpleXConfigBranches:
    def test_custom_params(self):
        config = SimpleXConfig(num_users=100, num_items=200)
        assert config.num_users == 100

    def test_custom_embedding_dim(self):
        config = SimpleXConfig(num_users=100, num_items=200, embedding_dim=128)
        assert config.embedding_dim == 128


class TestBERT4RecConfigBranches:
    def test_custom_params(self):
        config = BERT4RecConfig(vocab_size=100, embedding_dim=32, num_heads=2)
        assert config.embedding_dim == 32

    def test_invalid_embedding_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            BERT4RecConfig(vocab_size=100, embedding_dim=65, num_heads=2)

    def test_invalid_mask_prob(self):
        with pytest.raises(ValueError):
            BERT4RecConfig(vocab_size=100, mask_prob=1.5)


class TestCaserConfigBranches:
    def test_custom_params(self):
        config = CaserConfig(vocab_size=100, embedding_dim=32)
        assert config.embedding_dim == 32

    def test_invalid_dropout(self):
        with pytest.raises(ValueError):
            CaserConfig(dropout_rate=1.5)


class TestHSTUConfigBranches:
    def test_custom_params(self):
        config = HSTUConfig(vocab_size=100, embedding_dim=32, num_heads=2)
        assert config.embedding_dim == 32


class TestFuXiConfigBranches:
    def test_custom_params(self):
        config = FuXiConfig(vocab_size=100, embedding_dim=32, num_heads=2)
        assert config.embedding_dim == 32



# ============================================================================
# SECTION: Search Space Tests
# (from tests/test_search_spaces.py)
# Covers: SASRecSearchSpace, GRU4RecSearchSpace, BERT4RecSearchSpace,
# NCFSearchSpace, BPRMFSearchSpace — get_search_space and suggest_hyperparameters
# ============================================================================

ALL_SPACES = [
    SASRecSearchSpace,
    GRU4RecSearchSpace,
    BERT4RecSearchSpace,
    NCFSearchSpace,
    BPRMFSearchSpace,
]


class TestGetSearchSpace:
    @pytest.mark.parametrize("cls", ALL_SPACES)
    def test_returns_non_empty_dict(self, cls):
        space = cls.get_search_space()
        assert isinstance(space, dict)
        assert len(space) > 0

    @pytest.mark.parametrize("cls", ALL_SPACES)
    def test_contains_embedding_dim(self, cls):
        space = cls.get_search_space()
        assert "embedding_dim" in space

    @pytest.mark.parametrize("cls", ALL_SPACES)
    def test_contains_lr(self, cls):
        space = cls.get_search_space()
        assert "lr" in space


class TestSuggestHyperparameters:
    def _mock_trial(self):
        """Create a mock Optuna trial that returns sensible defaults."""
        trial = MagicMock()
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        trial.suggest_float.side_effect = lambda name, low, high, **kw: (low + high) / 2
        trial.suggest_int.side_effect = lambda name, low, high, **kw: (low + high) // 2
        return trial

    @pytest.mark.parametrize("cls", ALL_SPACES)
    def test_returns_dict(self, cls):
        trial = self._mock_trial()
        params = cls.suggest_hyperparameters(trial)
        assert isinstance(params, dict)
        assert len(params) > 0

    @pytest.mark.parametrize("cls", ALL_SPACES)
    def test_contains_embedding_dim(self, cls):
        trial = self._mock_trial()
        params = cls.suggest_hyperparameters(trial)
        assert "embedding_dim" in params

    @pytest.mark.parametrize("cls", ALL_SPACES)
    def test_contains_lr(self, cls):
        trial = self._mock_trial()
        params = cls.suggest_hyperparameters(trial)
        assert "lr" in params
