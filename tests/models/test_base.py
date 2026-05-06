"""Consolidated tests for all model base classes.

Covers:
  - BaseModel: init, get_config (copy semantics), repr, abstract interface
  - SequentialModel: init, vocab_size validation, max_seq_length defaults,
    predict/recommend NotImplementedError
  - DeepSequentialModel: config wiring, configure_optimizers, training_step,
    validation_step (LOO + fallback), test_step (target + fallback),
    compute_loss (neg_items variants), save/load, set_loss_fn,
    _to_model_indices, _get_activation, recommend_next, get_item_embedding,
    get_output_embeddings, get_sequence_embedding
  - DeepModel: config wiring, configure_optimizers, training_step,
    validation_step, test_step, save/load
  - TraditionalModel: save/load, predict, recommend
  - Property-based tests (Hypothesis): get_config copy, repr class name

Sources merged:
  tests/test_models.py
  tests/test_model_base_classes.py
  tests/test_sequential_base_coverage.py
"""

import os
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from rec_arena.losses import get_loss_function
from rec_arena.models.base import BaseModel
from rec_arena.models.deep import DeepModel
from rec_arena.models.sequential import DeepSequentialModel, SequentialModel
from rec_arena.models.traditional import TraditionalModel


# ===================================================================
# Concrete stubs for abstract classes
# ===================================================================


class ConcreteBaseModel(BaseModel):
    """Minimal concrete subclass of BaseModel for testing."""

    def fit(self, train_data, val_data=None) -> None:
        pass

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(user_ids.shape[0])

    def recommend(
        self, user_ids: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = user_ids.shape[0]
        return torch.zeros(n, k, dtype=torch.long), torch.zeros(n, k)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


class ConcreteSequentialModel(SequentialModel):
    """Minimal concrete subclass of SequentialModel for testing."""

    def predict_next(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        batch = sequences.shape[0]
        return torch.zeros(batch, self.vocab_size)

    def recommend_next(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = sequences.shape[0]
        return torch.zeros(batch, k, dtype=torch.long), torch.zeros(batch, k)

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*item_ids.shape, 64)

    def get_sequence_embedding(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(sequences.shape[0], 64)

    def fit(self, train_data, val_data=None) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


class ConcreteDeepSequentialModel(DeepSequentialModel):
    """Minimal concrete subclass of DeepSequentialModel for config-wiring tests."""

    def get_hidden_states(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len = sequences.shape
        return torch.zeros(batch, seq_len, self.embedding_dim)

    def predict_next(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        batch = sequences.shape[0]
        return torch.zeros(batch, self.vocab_size)

    def get_targets_and_mask(self, batch):
        sequences = batch["sequence"]
        mask = (sequences != 0).float()
        return sequences, mask


class StubDeepSequentialModel(DeepSequentialModel):
    """Full-featured concrete DeepSequentialModel for lifecycle / method tests."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)

    def get_hidden_states(self, sequences, sequence_lengths):
        return self.item_embedding(sequences)

    def predict_next(self, sequences, sequence_lengths):
        hidden = self.get_hidden_states(sequences, sequence_lengths)
        batch_idx = torch.arange(sequences.size(0))
        last_idx = torch.clamp(sequence_lengths - 1, min=0)
        last_hidden = hidden[batch_idx, last_idx]
        logits = last_hidden @ self.item_embedding.weight.T
        return torch.softmax(logits, dim=-1)

    def get_targets_and_mask(self, batch):
        targets = batch["sequence"].clone()
        mask = (targets != 0).float()
        return targets, mask


class StubDeepModel(DeepModel):
    """Minimal concrete DeepModel for testing base class methods."""

    def __init__(self, config):
        super().__init__(config)
        self.user_emb = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_emb = nn.Embedding(self.num_items, self.embedding_dim)

    def predict(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return (u * i).sum(dim=-1)

    def recommend(self, user_ids, k=10):
        u = self.user_emb(user_ids)
        scores = u @ self.item_emb.weight.T
        return torch.topk(scores, k, dim=-1)

    def get_user_embedding(self, user_ids):
        return self.user_emb(user_ids)

    def get_item_embedding(self, item_ids):
        return self.item_emb(item_ids)

    def compute_loss(self, batch):
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        preds = self.predict(user_ids, item_ids)
        return torch.nn.functional.mse_loss(preds, torch.ones_like(preds))


class StubTraditionalModel(TraditionalModel):
    """Minimal concrete TraditionalModel for testing base class methods."""

    def fit(self, train_data, val_data=None):
        import numpy as np

        self.model = np.eye(self.num_items)

    def _predict_numpy(self, user_ids, item_ids):
        import numpy as np

        return np.ones(len(user_ids), dtype=np.float32)

    def _recommend_numpy(self, user_ids, k):
        import numpy as np

        n = len(user_ids)
        items = np.tile(np.arange(k), (n, 1))
        scores = np.ones((n, k), dtype=np.float32)
        return items, scores


# ===================================================================
# Config helpers
# ===================================================================

VALID_BASE_CONFIG = {"vocab_size": 100, "embedding_dim": 64}

VALID_SEQUENTIAL_CONFIG = {
    "vocab_size": 100,
    "embedding_dim": 64,
    "max_seq_length": 50,
    "loss_type": "cross_entropy",
}

VALID_IMPLICIT_CONFIG = {
    "num_users": 50,
    "num_items": 100,
    "embedding_dim": 64,
    "loss_type": "bce",
}

SEQ_CONFIG = {
    "vocab_size": 23,
    "embedding_dim": 16,
    "max_seq_length": 10,
    "loss_type": "cross_entropy",
}

IMPLICIT_CONFIG = {
    "num_users": 5,
    "num_items": 20,
    "embedding_dim": 16,
    "loss_type": "bce",
}


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def seq_model():
    """StubDeepSequentialModel with loss function pre-configured."""
    m = StubDeepSequentialModel(SEQ_CONFIG)
    m.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
    return m


# ===================================================================
# BaseModel tests
# ===================================================================


class TestBaseModelInit:
    def test_config_none_raises_value_error(self):
        with pytest.raises(ValueError, match="[Cc]onfig"):
            ConcreteBaseModel(config=None)

    def test_valid_config_initializes(self):
        model = ConcreteBaseModel(config=VALID_BASE_CONFIG)
        assert model.config == VALID_BASE_CONFIG

    def test_name_is_class_name(self):
        model = ConcreteBaseModel(config=VALID_BASE_CONFIG)
        assert model.name == "ConcreteBaseModel"


class TestBaseModelGetConfig:
    def test_returns_dict_equal_to_config(self):
        model = ConcreteBaseModel(config=VALID_BASE_CONFIG.copy())
        result = model.get_config()
        assert result == VALID_BASE_CONFIG

    def test_returns_copy_not_same_object(self):
        model = ConcreteBaseModel(config=VALID_BASE_CONFIG.copy())
        result = model.get_config()
        assert result is not model.config

    def test_modifying_returned_dict_does_not_affect_model(self):
        config = {"vocab_size": 100, "embedding_dim": 64}
        model = ConcreteBaseModel(config=config)
        returned = model.get_config()
        returned["vocab_size"] = 9999
        assert model.config["vocab_size"] == 100

    def test_modifying_model_config_does_not_affect_returned_copy(self):
        config = {"vocab_size": 100, "embedding_dim": 64}
        model = ConcreteBaseModel(config=config)
        returned = model.get_config()
        model.config["vocab_size"] = 9999
        assert returned["vocab_size"] == 100


class TestBaseModelRepr:
    def test_repr_contains_class_name(self):
        model = ConcreteBaseModel(config=VALID_BASE_CONFIG)
        assert "ConcreteBaseModel" in repr(model)

    def test_repr_is_string(self):
        model = ConcreteBaseModel(config=VALID_BASE_CONFIG)
        assert isinstance(repr(model), str)

    def test_repr_contains_config_info(self):
        model = ConcreteBaseModel(config=VALID_BASE_CONFIG)
        r = repr(model)
        assert "vocab_size" in r or "100" in r or "64" in r


# ===================================================================
# SequentialModel tests
# ===================================================================


class TestSequentialModelInit:
    def test_valid_config_initializes(self):
        model = ConcreteSequentialModel(config=VALID_BASE_CONFIG)
        assert model.vocab_size == 100

    def test_invalid_vocab_size_zero_raises(self):
        config = {"vocab_size": 0, "embedding_dim": 64}
        with pytest.raises(ValueError, match="vocab_size"):
            ConcreteSequentialModel(config=config)

    def test_invalid_vocab_size_negative_raises(self):
        config = {"vocab_size": -5, "embedding_dim": 64}
        with pytest.raises(ValueError, match="vocab_size"):
            ConcreteSequentialModel(config=config)

    def test_vocab_size_none_raises(self):
        config = {"vocab_size": None, "embedding_dim": 64}
        with pytest.raises(ValueError, match="vocab_size"):
            ConcreteSequentialModel(config=config)

    def test_config_none_raises(self):
        with pytest.raises(ValueError):
            ConcreteSequentialModel(config=None)

    def test_max_seq_length_defaults_to_50(self):
        model = ConcreteSequentialModel(config=VALID_BASE_CONFIG)
        assert model.max_seq_length == 50

    def test_max_seq_length_from_config(self):
        config = {"vocab_size": 100, "embedding_dim": 64, "max_seq_length": 30}
        model = ConcreteSequentialModel(config=config)
        assert model.max_seq_length == 30


class TestSequentialModelAbstract:
    def test_predict_raises(self, seq_model):
        with pytest.raises(NotImplementedError, match="predict_next"):
            SequentialModel.predict(seq_model, torch.tensor([0]), torch.tensor([1]))

    def test_recommend_raises(self, seq_model):
        with pytest.raises(NotImplementedError, match="recommend_next"):
            SequentialModel.recommend(seq_model, torch.tensor([0]))


# ===================================================================
# DeepSequentialModel tests — config wiring
# ===================================================================


class TestDeepSequentialModelConfigWiring:
    def test_embedding_dim_wired_from_config(self):
        config = {**VALID_SEQUENTIAL_CONFIG, "embedding_dim": 128}
        model = ConcreteDeepSequentialModel(config=config)
        assert model.embedding_dim == 128

    def test_lr_wired_from_config(self):
        config = {**VALID_SEQUENTIAL_CONFIG, "lr": 0.01}
        model = ConcreteDeepSequentialModel(config=config)
        assert model.lr == 0.01

    def test_vocab_size_wired_from_config(self):
        config = {**VALID_SEQUENTIAL_CONFIG, "vocab_size": 200}
        model = ConcreteDeepSequentialModel(config=config)
        assert model.vocab_size == 200

    def test_item_embedding_created(self):
        model = ConcreteDeepSequentialModel(config=VALID_SEQUENTIAL_CONFIG)
        assert hasattr(model, "item_embedding")
        assert isinstance(model.item_embedding, nn.Embedding)

    def test_item_embedding_size_matches_vocab(self):
        model = ConcreteDeepSequentialModel(config=VALID_SEQUENTIAL_CONFIG)
        assert model.item_embedding.num_embeddings == VALID_SEQUENTIAL_CONFIG["vocab_size"]

    def test_invalid_vocab_size_raises(self):
        config = {**VALID_SEQUENTIAL_CONFIG, "vocab_size": 0}
        with pytest.raises(ValueError):
            ConcreteDeepSequentialModel(config=config)

    def test_config_none_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            ConcreteDeepSequentialModel(config=None)


# ===================================================================
# DeepSequentialModel tests — configure_optimizers
# ===================================================================


class TestDeepSequentialModelConfigureOptimizers:
    def test_returns_optimizer_no_scheduler(self):
        model = StubDeepSequentialModel(SEQ_CONFIG)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.Optimizer)

    def test_cosine_scheduler(self):
        config = {**SEQ_CONFIG, "scheduler": {"type": "cosine"}}
        model = StubDeepSequentialModel(config)
        result = model.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_cosine_with_warmup(self):
        config = {**SEQ_CONFIG, "scheduler": {"type": "cosine", "warmup_steps": 10}}
        model = StubDeepSequentialModel(config)
        result = model.configure_optimizers()
        assert result["lr_scheduler"]["interval"] == "step"

    def test_reduce_on_plateau(self):
        config = {**SEQ_CONFIG, "scheduler": {"type": "reduce_on_plateau"}}
        model = StubDeepSequentialModel(config)
        result = model.configure_optimizers()
        assert result["lr_scheduler"]["monitor"] is not None

    def test_unknown_scheduler_returns_optimizer(self):
        config = {**SEQ_CONFIG, "scheduler": {"type": "unknown_scheduler"}}
        model = StubDeepSequentialModel(config)
        result = model.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)


# ===================================================================
# DeepSequentialModel tests — training_step
# ===================================================================


class TestDeepSequentialModelTrainingStep:
    def test_returns_loss_tensor(self):
        model = StubDeepSequentialModel(SEQ_CONFIG)
        model.set_loss_fn(get_loss_function("cross_entropy", "sequential"))

        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
        }
        loss = model.training_step(batch, 0)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ===================================================================
# DeepSequentialModel tests — validation_step (LOO + fallback)
# ===================================================================


class TestValidationStepLOO:
    def test_loo_validation_returns_loss(self, seq_model):
        seq_model.log = MagicMock()
        seq_model._trainer = MagicMock()
        seq_model._trainer.callback_metrics = {}
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "target": torch.randint(3, 20, (2,)),
        }
        loss = seq_model.validation_step(batch, 0)
        assert torch.isfinite(loss)

    def test_loo_validation_with_neg_items(self, seq_model):
        seq_model.log = MagicMock()
        seq_model._trainer = MagicMock()
        seq_model._trainer.callback_metrics = {}
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "target": torch.randint(3, 20, (2,)),
            "neg_items": torch.randint(3, 20, (2, 10, 4)),
        }
        loss = seq_model.validation_step(batch, 0)
        assert torch.isfinite(loss)

    def test_loo_validation_computes_metrics_at_interval(self, seq_model):
        seq_model.log = MagicMock()
        seq_model._trainer = MagicMock()
        seq_model._trainer.callback_metrics = {"val_ndcg@10": torch.tensor(0.5)}
        seq_model.metric_compute_interval = 1
        seq_model._current_epoch = 0
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "target": torch.randint(3, 20, (2,)),
        }
        loss = seq_model.validation_step(batch, 0)
        assert torch.isfinite(loss)

    def test_fallback_validation_without_target(self, seq_model):
        seq_model.log = MagicMock()
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "negatives": torch.randint(3, 20, (2, 4)),
        }
        loss = seq_model.validation_step(batch, 0)
        assert torch.isfinite(loss)


# ===================================================================
# DeepSequentialModel tests — test_step
# ===================================================================


class TestDeepSeqTestStep:
    def test_with_target_returns_predictions(self, seq_model):
        seq_model.log = MagicMock()
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "target": torch.randint(3, 20, (2,)),
        }
        result = seq_model.test_step(batch, 0)
        assert "predictions" in result
        assert "targets" in result
        assert result["predictions"].shape[0] == 2

    def test_without_target_fallback(self, seq_model):
        seq_model.log = MagicMock()
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
        }
        result = seq_model.test_step(batch, 0)
        assert "predictions" in result
        assert "targets" in result


# ===================================================================
# DeepSequentialModel tests — compute_loss
# ===================================================================


class TestComputeLoss:
    def test_no_loss_fn_raises(self):
        m = StubDeepSequentialModel(SEQ_CONFIG)
        m.loss_fn = None
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
        }
        with pytest.raises(RuntimeError, match="No loss function"):
            m.compute_loss(batch)

    def test_with_neg_items_3d(self, seq_model):
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "neg_items": torch.randint(3, 20, (2, 10, 4)),
        }
        loss = seq_model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_with_neg_items_2d(self, seq_model):
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "neg_items": torch.randint(3, 20, (2, 10)),
        }
        loss = seq_model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_with_batch_shared_neg_items(self, seq_model):
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "neg_items": torch.randint(3, 20, (2, 4)),
        }
        loss = seq_model.compute_loss(batch)
        assert torch.isfinite(loss)


# ===================================================================
# DeepSequentialModel tests — _to_model_indices, set_loss_fn, _get_activation
# ===================================================================


class TestDeepSequentialModelToModelIndices:
    def test_decrements_by_one(self):
        model = StubDeepSequentialModel(SEQ_CONFIG)
        targets = torch.tensor([1, 5, 10])
        result = model._to_model_indices(targets)
        assert torch.equal(result, torch.tensor([0, 4, 9]))


class TestDeepSequentialModelSetLossFn:
    def test_sets_loss_fn(self):
        model = StubDeepSequentialModel(SEQ_CONFIG)
        dummy_fn = MagicMock()
        model.set_loss_fn(dummy_fn)
        assert model.loss_fn is dummy_fn


class TestDeepSequentialModelGetActivation:
    def test_relu(self):
        from rec_arena.configs.defaults.sasrec import SASRecConfig

        config = SASRecConfig(vocab_size=23, embedding_dim=16, activation="relu")
        model = StubDeepSequentialModel(config)
        act = model._get_activation()
        assert isinstance(act, nn.ReLU)

    def test_gelu(self):
        from rec_arena.configs.defaults.sasrec import SASRecConfig

        config = SASRecConfig(vocab_size=23, embedding_dim=16, activation="gelu")
        model = StubDeepSequentialModel(config)
        act = model._get_activation()
        assert isinstance(act, nn.GELU)

    def test_unknown_defaults_to_relu(self):
        from rec_arena.configs.defaults.sasrec import SASRecConfig

        config = SASRecConfig(vocab_size=23, embedding_dim=16, activation="unknown")
        model = StubDeepSequentialModel(config)
        act = model._get_activation()
        assert isinstance(act, nn.ReLU)


# ===================================================================
# DeepSequentialModel tests — save/load
# ===================================================================


class TestDeepSequentialModelSaveLoad:
    def test_save_and_load_roundtrip(self):
        model = StubDeepSequentialModel(SEQ_CONFIG)
        path = "test_model_save_tmp.pt"
        try:
            model.save(path)
            assert os.path.exists(path)
            model2 = StubDeepSequentialModel(SEQ_CONFIG)
            model2.load(path)
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.equal(p1, p2)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_invalid_extension_raises(self):
        model = StubDeepSequentialModel(SEQ_CONFIG)
        with pytest.raises(RuntimeError):
            model.save("model.txt")

    def test_load_nonexistent_raises(self):
        model = StubDeepSequentialModel(SEQ_CONFIG)
        with pytest.raises(RuntimeError):
            model.load("nonexistent.pt")


# ===================================================================
# DeepSequentialModel tests — recommend_next, embeddings
# ===================================================================


class TestRecommendNext:
    def test_returns_items_and_scores(self, seq_model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        items, scores = seq_model.recommend_next(seq, lengths, k=5)
        assert items.shape == (2, 5)
        assert scores.shape == (2, 5)


class TestGetItemEmbedding:
    def test_returns_correct_shape(self, seq_model):
        ids = torch.tensor([1, 5, 10])
        emb = seq_model.get_item_embedding(ids)
        assert emb.shape == (3, 16)

    def test_clamps_out_of_range(self, seq_model):
        ids = torch.tensor([999])
        emb = seq_model.get_item_embedding(ids)
        assert emb.shape == (1, 16)


class TestGetOutputEmbeddings:
    def test_returns_weight_matrix(self, seq_model):
        w = seq_model.get_output_embeddings()
        assert w.shape == (23, 16)


class TestGetSequenceEmbedding:
    def test_returns_last_hidden(self, seq_model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        emb = seq_model.get_sequence_embedding(seq, lengths)
        assert emb.shape == (2, 16)


# ===================================================================
# DeepModel tests — config wiring
# ===================================================================


class TestDeepModelConfigWiring:
    """Test DeepModel config wiring using a minimal concrete subclass."""

    def _make_deep_model(self, config: Dict[str, Any]):
        """Import and create a minimal DeepModel subclass."""

        class ConcreteDeepModel(DeepModel):
            def __init__(self, config):
                super().__init__(config)
                self.dummy = nn.Linear(config.get("embedding_dim", 64), 1)

            def compute_loss(self, batch):
                return torch.tensor(0.0)

            def predict(self, user_ids, item_ids):
                return torch.zeros(user_ids.shape[0])

            def recommend(self, user_ids, k=10):
                n = user_ids.shape[0]
                return torch.zeros(n, k, dtype=torch.long), torch.zeros(n, k)

        return ConcreteDeepModel(config)

    def test_embedding_dim_wired(self):
        config = {**VALID_IMPLICIT_CONFIG, "embedding_dim": 32}
        model = self._make_deep_model(config)
        assert model.embedding_dim == 32

    def test_lr_wired(self):
        config = {**VALID_IMPLICIT_CONFIG, "lr": 0.005}
        model = self._make_deep_model(config)
        assert model.lr == 0.005

    def test_num_users_wired(self):
        config = {**VALID_IMPLICIT_CONFIG, "num_users": 200}
        model = self._make_deep_model(config)
        assert model.num_users == 200

    def test_num_items_wired(self):
        config = {**VALID_IMPLICIT_CONFIG, "num_items": 500}
        model = self._make_deep_model(config)
        assert model.num_items == 500

    def test_config_none_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            self._make_deep_model(None)

    def test_invalid_implicit_config_raises(self):
        with pytest.raises(ValueError):
            self._make_deep_model({"embedding_dim": 64})


# ===================================================================
# DeepModel tests — configure_optimizers
# ===================================================================


class TestDeepModelConfigureOptimizers:
    def test_returns_optimizer_no_scheduler(self):
        model = StubDeepModel(IMPLICIT_CONFIG)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.Optimizer)

    def test_cosine_scheduler(self):
        config = {**IMPLICIT_CONFIG, "scheduler": {"type": "cosine"}}
        model = StubDeepModel(config)
        result = model.configure_optimizers()
        assert "optimizer" in result

    def test_cosine_with_warmup(self):
        config = {**IMPLICIT_CONFIG, "scheduler": {"type": "cosine", "warmup_steps": 5}}
        model = StubDeepModel(config)
        result = model.configure_optimizers()
        assert result["lr_scheduler"]["interval"] == "step"


# ===================================================================
# DeepModel tests — training_step, validation_step, test_step
# ===================================================================


class TestDeepModelTrainingStep:
    def test_returns_loss(self):
        model = StubDeepModel(IMPLICIT_CONFIG)
        batch = {"user_id": torch.tensor([0, 1]), "item_id": torch.tensor([3, 5])}
        loss = model.training_step(batch, 0)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestDeepModelValidationStep:
    def test_returns_loss(self):
        model = StubDeepModel(IMPLICIT_CONFIG)
        batch = {"user_id": torch.tensor([0, 1]), "item_id": torch.tensor([3, 5])}
        model.log = MagicMock()
        loss = model.validation_step(batch, 0)
        assert torch.isfinite(loss)

    def test_with_val_metrics(self):
        config = {**IMPLICIT_CONFIG, "compute_val_metrics": True, "val_k_values": [5]}
        model = StubDeepModel(config)
        model.log = MagicMock()
        batch = {"user_id": torch.tensor([0, 1]), "item_id": torch.tensor([3, 5])}
        loss = model.validation_step(batch, 0)
        assert torch.isfinite(loss)


class TestDeepModelTestStep:
    def test_returns_predictions_and_targets(self):
        model = StubDeepModel(IMPLICIT_CONFIG)
        model.log = MagicMock()
        batch = {"user_id": torch.tensor([0, 1]), "item_id": torch.tensor([3, 5])}
        result = model.test_step(batch, 0)
        assert "predictions" in result
        assert "targets" in result


# ===================================================================
# DeepModel tests — save/load
# ===================================================================


class TestDeepModelSaveLoad:
    def test_save_and_load(self):
        model = StubDeepModel(IMPLICIT_CONFIG)
        path = "test_deep_model_save_tmp.pt"
        try:
            model.save(path)
            assert os.path.exists(path)
            model2 = StubDeepModel(IMPLICIT_CONFIG)
            model2.load(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_invalid_extension_raises(self):
        model = StubDeepModel(IMPLICIT_CONFIG)
        with pytest.raises(RuntimeError):
            model.save("model.txt")


# ===================================================================
# TraditionalModel tests
# ===================================================================


class TestTraditionalModelSaveLoad:
    def test_save_and_load(self):
        model = StubTraditionalModel({"num_users": 5, "num_items": 10})
        model.fit(None)
        path = "test_trad_model_save_tmp.joblib"
        try:
            model.save(path)
            assert os.path.exists(path)
            model2 = StubTraditionalModel({"num_users": 5, "num_items": 10})
            model2.load(path)
            assert model2.model is not None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_invalid_extension_raises(self):
        model = StubTraditionalModel({"num_users": 5, "num_items": 10})
        model.fit(None)
        with pytest.raises(RuntimeError):
            model.save("model.txt")

    def test_load_nonexistent_raises(self):
        model = StubTraditionalModel({"num_users": 5, "num_items": 10})
        with pytest.raises(RuntimeError):
            model.load("nonexistent.joblib")


class TestTraditionalModelPredict:
    def test_returns_tensor(self):
        model = StubTraditionalModel({"num_users": 5, "num_items": 10})
        model.fit(None)
        preds = model.predict(torch.tensor([0, 1]), torch.tensor([3, 5]))
        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (2,)


class TestTraditionalModelRecommend:
    def test_returns_items_and_scores(self):
        model = StubTraditionalModel({"num_users": 5, "num_items": 10})
        model.fit(None)
        items, scores = model.recommend(torch.tensor([0, 1]), k=5)
        assert items.shape == (2, 5)
        assert scores.shape == (2, 5)


# ===================================================================
# Property-based tests (Hypothesis)
# ===================================================================

# Feature: comprehensive-test-suite, Property 33: BaseModel.get_config returns a copy


@given(
    config=st.fixed_dictionaries(
        {
            "vocab_size": st.integers(min_value=1, max_value=10_000),
            "embedding_dim": st.integers(min_value=1, max_value=512),
        }
    ).map(lambda d: {**d, "extra_key": "extra_value"})
)
@settings(max_examples=100)
def test_property_33_get_config_returns_copy(config):
    """Property 33: BaseModel.get_config returns a copy.

    Validates: Requirements 13.2
    """
    model = ConcreteBaseModel(config=config)
    returned = model.get_config()

    # Must be equal in value
    assert returned == config

    # Must not be the same object
    assert returned is not model.config

    # Mutating the returned dict must not affect the model's internal config
    original_vocab = model.config["vocab_size"]
    returned["vocab_size"] = -9999
    assert model.config["vocab_size"] == original_vocab


# Feature: comprehensive-test-suite, Property 34: BaseModel repr contains class name


@given(
    vocab_size=st.integers(min_value=1, max_value=10_000),
    embedding_dim=st.integers(min_value=1, max_value=512),
)
@settings(max_examples=100)
def test_property_34_repr_contains_class_name(vocab_size, embedding_dim):
    """Property 34: BaseModel repr contains class name.

    Validates: Requirements 13.3
    """
    config = {"vocab_size": vocab_size, "embedding_dim": embedding_dim}
    model = ConcreteBaseModel(config=config)
    r = repr(model)
    assert isinstance(r, str)
    assert "ConcreteBaseModel" in r
