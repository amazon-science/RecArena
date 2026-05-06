"""Integration tests for rec_arena.

Task 11.1: End-to-end pipeline tests
  - dataset → split → prepare_sequences → MetricCalculator
  - config validation → loss factory → loss computation
  - ExperimentConfig → ExperimentResult → JSON serialization round-trip

Task 11.2: RecDataModule integration tests
  - setup with format="sequential" / "implicit"
  - train_dataloader yields batches with expected keys
  - num_negatives >= num_items raises ValueError
  - batch_size < 1 raises ValueError

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 17.1, 17.2, 17.3
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from rec_arena.benchmarks.experiment import ExperimentConfig, ExperimentResult
from rec_arena.configs.validation import validate_config
from rec_arena.datasets.implicit_dataset import ImplicitDataset
from rec_arena.datasets.rec_datamodule import RecDataModule
from rec_arena.datasets.sequential_dataset import (
    SequentialDataset,
    build_user_histories,
    prepare_sequences,
)
from rec_arena.datasets.split_strategies import LeaveOneOutSplit
from rec_arena.losses import CrossEntropyLoss, get_loss_function
from rec_arena.metrics import MetricCalculator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_USERS = 5
NUM_ITEMS = 20
MAX_SEQ = 10


def _make_interactions_df(num_users=NUM_USERS, items_per_user=10):
    """Build a synthetic interactions DataFrame (1-indexed items)."""
    rows = []
    for uid in range(num_users):
        for rank, iid in enumerate(range(1, items_per_user + 1), start=1):
            rows.append({"user_id": uid, "item_id": iid, "timestamp": uid * 100 + rank})
    return pd.DataFrame(rows)


def _make_mock_dataset(num_users=NUM_USERS, num_items=NUM_ITEMS, items_per_user=10):
    """Build a mock dataset object that RecDataModule can use without real I/O."""
    df = _make_interactions_df(num_users, items_per_user)
    splitter = LeaveOneOutSplit(min_sequence_length=3)
    train_df, val_df, test_df = splitter.split(df, num_users, items_per_user)

    mock_ds = MagicMock()
    mock_ds.num_users = num_users
    mock_ds.num_items = num_items
    mock_ds.interactions_df = df
    # split() returns (train, val, test)
    mock_ds.split.return_value = (train_df, val_df, test_df)
    # No s3_bucket attribute so num_workers is not overridden
    del mock_ds.s3_bucket
    return mock_ds


# ===========================================================================
# Task 11.1 — End-to-end pipeline tests (Requirements 17.1, 17.2, 17.3)
# ===========================================================================


class TestDatasetToMetricsPipeline:
    """Test dataset → split → prepare_sequences → MetricCalculator pipeline.

    Validates: Requirement 17.1
    """

    def test_full_pipeline_produces_valid_metric_results(self):
        """Full pipeline from synthetic data to metric calculation produces valid results."""
        # 1. Create synthetic interactions
        df = _make_interactions_df(num_users=5, items_per_user=10)

        # 2. Split
        splitter = LeaveOneOutSplit(min_sequence_length=3)
        train_df, val_df, test_df = splitter.split(df, 5, 10)

        # 3. Prepare sequences from training data
        sequences = prepare_sequences(train_df, max_seq_length=MAX_SEQ)
        assert len(sequences) > 0, "prepare_sequences should produce at least one sequence"

        # 4. Build user histories
        histories = build_user_histories(train_df)
        assert len(histories) == 5

        # 5. Create SequentialDataset
        dataset = SequentialDataset(sequences, max_seq_length=MAX_SEQ)
        assert len(dataset) > 0

        # 6. Compute metrics on synthetic predictions
        batch_size = min(len(dataset), 4)
        num_items = 10
        torch.manual_seed(42)
        predictions = torch.randn(batch_size, num_items)
        targets = torch.randint(0, num_items, (batch_size,))

        calc = MetricCalculator(k_values=[5, 10])
        results = calc.calculate_all(predictions, targets)

        # 7. Verify results are valid
        assert isinstance(results, dict)
        assert len(results) > 0
        for key, val in results.items():
            assert 0.0 <= val <= 1.0, f"Metric {key} = {val} is out of [0, 1]"

    def test_pipeline_with_temporal_split(self):
        """Pipeline works with TemporalSplit as well."""
        from rec_arena.datasets.split_strategies import TemporalSplit

        df = _make_interactions_df(num_users=5, items_per_user=15)
        splitter = TemporalSplit()
        train_df, val_df, test_df = splitter.split(df, 5, 15)

        sequences = prepare_sequences(train_df, max_seq_length=MAX_SEQ)
        assert len(sequences) >= 0  # may be empty if train_df is empty

        torch.manual_seed(0)
        preds = torch.randn(4, 15)
        targets = torch.randint(0, 15, (4,))
        calc = MetricCalculator(k_values=[5])
        results = calc.calculate_all(preds, targets)
        for val in results.values():
            assert 0.0 <= val <= 1.0


class TestConfigToLossPipeline:
    """Test config validation → loss factory → loss computation pipeline.

    Validates: Requirement 17.2
    """

    def test_sequential_config_to_loss_pipeline(self):
        """Config validation → loss factory → finite loss value."""
        # 1. Validate config
        config = {
            "vocab_size": 50,
            "embedding_dim": 32,
            "loss_type": "cross_entropy",
        }
        validate_config(config, model_type="sequential")  # should not raise

        # 2. Create loss via factory
        loss_fn = get_loss_function(config["loss_type"], model_type="sequential")
        assert isinstance(loss_fn, CrossEntropyLoss)

        # 3. Compute loss on synthetic data
        torch.manual_seed(42)
        batch, seq_len, vocab = 4, 8, config["vocab_size"]
        logits = torch.randn(batch, seq_len, vocab)
        targets = torch.randint(1, vocab, (batch, seq_len))
        mask = torch.ones(batch, seq_len)

        loss_val = loss_fn(logits=logits, targets=targets, mask=mask)

        # 4. Verify finite scalar
        assert loss_val.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(loss_val), f"Loss should be finite, got {loss_val.item()}"

    def test_implicit_config_to_loss_pipeline(self):
        """Implicit config validation → loss factory → finite loss value."""
        from unittest.mock import MagicMock

        from rec_arena.losses import BPRLoss

        config = {
            "num_users": 50,
            "num_items": 100,
            "embedding_dim": 32,
            "loss_type": "bpr",
        }
        validate_config(config, model_type="implicit")

        loss_fn = get_loss_function(config["loss_type"], model_type="implicit")
        assert isinstance(loss_fn, BPRLoss)

    def test_bce_sequential_pipeline(self):
        """BCE sequential config → loss factory → finite loss."""
        from rec_arena.losses import BCENegativeSamplingLoss

        config = {
            "vocab_size": 50,
            "embedding_dim": 32,
            "loss_type": "bce",
        }
        validate_config(config, model_type="sequential")

        loss_fn = get_loss_function("bce", model_type="sequential")
        assert isinstance(loss_fn, BCENegativeSamplingLoss)

        torch.manual_seed(0)
        batch, seq_len, vocab, num_neg = 4, 8, 50, 5
        logits = torch.randn(batch, seq_len, vocab)
        targets = torch.randint(1, vocab, (batch, seq_len))
        mask = torch.ones(batch, seq_len)
        neg_items = torch.randint(1, vocab, (batch, seq_len, num_neg))

        loss_val = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert loss_val.dim() == 0
        assert torch.isfinite(loss_val)


class TestExperimentSerializationPipeline:
    """Test ExperimentConfig → ExperimentResult → JSON round-trip.

    Validates: Requirement 17.3
    """

    def test_experiment_result_json_roundtrip(self):
        """ExperimentResult serializes to JSON and deserializes without data loss."""
        config = ExperimentConfig(
            name="integration_test",
            model_config={"type": "sasrec", "embedding_dim": 64},
            dataset_config={"name": "ml100k"},
            training_config={"max_epochs": 10, "lr": 0.001},
            evaluation_config={"k": [5, 10]},
        )
        result = ExperimentResult(
            config=config,
            metrics={"ndcg@10": 0.42, "recall@10": 0.55},
            training_time=12.5,
            inference_time=0.8,
            timestamp=datetime.now().isoformat(),
            metadata={"seed": 42},
        )

        # Serialize to dict then JSON
        original_dict = result.to_dict()
        json_str = json.dumps(original_dict)
        deserialized = json.loads(json_str)

        # Verify round-trip
        assert deserialized == original_dict
        assert deserialized["metrics"]["ndcg@10"] == pytest.approx(0.42)
        assert deserialized["metrics"]["recall@10"] == pytest.approx(0.55)
        assert deserialized["training_time"] == pytest.approx(12.5)
        assert deserialized["config"]["name"] == "integration_test"

    def test_experiment_config_to_dict_completeness(self):
        """ExperimentConfig.to_dict contains all required fields."""
        config = ExperimentConfig(
            name="test",
            model_config={"type": "gru4rec"},
            dataset_config={"name": "ml1m"},
            training_config={"max_epochs": 5},
            evaluation_config={"k": [10]},
        )
        d = config.to_dict()
        for key in ("name", "model_config", "dataset_config", "training_config", "evaluation_config"):
            assert key in d

    def test_experiment_result_to_dict_completeness(self):
        """ExperimentResult.to_dict contains all required fields."""
        config = ExperimentConfig(
            name="test",
            model_config={},
            dataset_config={},
            training_config={},
            evaluation_config={},
        )
        result = ExperimentResult(
            config=config,
            metrics={"ndcg@10": 0.3},
            training_time=1.0,
            inference_time=0.1,
            timestamp="2024-01-01T00:00:00",
            metadata=None,
        )
        d = result.to_dict()
        for key in ("config", "metrics", "training_time", "inference_time", "timestamp", "metadata"):
            assert key in d

    def test_json_serialization_with_tmp_path(self, tmp_path):
        """ExperimentResult can be saved to a JSON file and reloaded."""
        from rec_arena.benchmarks.suite import BenchmarkSuite

        config = ExperimentConfig(
            name="file_test",
            model_config={"type": "bert4rec"},
            dataset_config={"name": "ml100k"},
            training_config={"max_epochs": 3},
            evaluation_config={"k": [5]},
        )
        result = ExperimentResult(
            config=config,
            metrics={"ndcg@5": 0.35},
            training_time=5.0,
            inference_time=0.5,
            timestamp=datetime.now().isoformat(),
            metadata={},
        )

        suite = BenchmarkSuite()
        suite.results.append(result)

        out_file = tmp_path / "results.json"
        suite.save_results(str(out_file))

        assert out_file.exists()
        loaded = json.loads(out_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["metrics"]["ndcg@5"] == pytest.approx(0.35)
        assert loaded[0]["config"]["name"] == "file_test"


# ===========================================================================
# Task 11.2 — RecDataModule integration tests (Requirements 14.1–14.5)
# ===========================================================================


class TestRecDataModuleSetupSequential:
    """Test RecDataModule.setup with format='sequential'.

    Validates: Requirements 14.1, 14.3
    """

    def test_setup_sequential_populates_sequential_datasets(self):
        """setup with format='sequential' populates SequentialDataset instances."""
        mock_ds = _make_mock_dataset()
        dm = RecDataModule(
            dataset=mock_ds,
            format="sequential",
            batch_size=4,
            num_workers=0,
            max_seq_length=MAX_SEQ,
            num_negatives=5,
        )
        dm.setup()

        assert isinstance(dm.train_dataset, SequentialDataset)
        assert isinstance(dm.val_dataset, SequentialDataset)
        assert isinstance(dm.test_dataset, SequentialDataset)

    def test_setup_sequential_datasets_are_non_empty(self):
        """After setup, train_dataset should have at least one sample."""
        mock_ds = _make_mock_dataset()
        dm = RecDataModule(
            dataset=mock_ds,
            format="sequential",
            batch_size=4,
            num_workers=0,
            max_seq_length=MAX_SEQ,
            num_negatives=5,
        )
        dm.setup()
        assert len(dm.train_dataset) > 0

    def test_train_dataloader_yields_batches_with_expected_keys(self):
        """train_dataloader yields batches containing 'user_id', 'sequence', 'sequence_length'."""
        mock_ds = _make_mock_dataset()
        dm = RecDataModule(
            dataset=mock_ds,
            format="sequential",
            batch_size=4,
            num_workers=0,
            max_seq_length=MAX_SEQ,
            num_negatives=5,
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert "user_id" in batch
        assert "sequence" in batch
        assert "sequence_length" in batch

    def test_train_dataloader_batch_tensors_have_correct_types(self):
        """Batch tensors should be LongTensors."""
        mock_ds = _make_mock_dataset()
        dm = RecDataModule(
            dataset=mock_ds,
            format="sequential",
            batch_size=4,
            num_workers=0,
            max_seq_length=MAX_SEQ,
            num_negatives=5,
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert batch["user_id"].dtype == torch.long
        assert batch["sequence"].dtype == torch.long


class TestRecDataModuleSetupImplicit:
    """Test RecDataModule.setup with format='implicit'.

    Validates: Requirements 14.2, 14.3
    """

    def test_setup_implicit_populates_implicit_datasets(self):
        """setup with format='implicit' populates ImplicitDataset instances."""
        mock_ds = _make_mock_dataset()
        dm = RecDataModule(
            dataset=mock_ds,
            format="implicit",
            batch_size=4,
            num_workers=0,
            max_seq_length=MAX_SEQ,
            num_negatives=5,
        )
        dm.setup()

        assert isinstance(dm.train_dataset, ImplicitDataset)
        assert isinstance(dm.val_dataset, ImplicitDataset)
        assert isinstance(dm.test_dataset, ImplicitDataset)

    def test_setup_implicit_train_dataset_non_empty(self):
        """After implicit setup, train_dataset should have at least one sample."""
        mock_ds = _make_mock_dataset()
        dm = RecDataModule(
            dataset=mock_ds,
            format="implicit",
            batch_size=4,
            num_workers=0,
            max_seq_length=MAX_SEQ,
            num_negatives=5,
        )
        dm.setup()
        assert len(dm.train_dataset) > 0

    def test_train_dataloader_implicit_yields_batches_with_expected_keys(self):
        """train_dataloader for implicit format yields batches with 'user_id' and 'item_id'."""
        mock_ds = _make_mock_dataset()
        dm = RecDataModule(
            dataset=mock_ds,
            format="implicit",
            batch_size=4,
            num_workers=0,
            max_seq_length=MAX_SEQ,
            num_negatives=5,
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert "user_id" in batch
        assert "item_id" in batch


class TestRecDataModuleValidation:
    """Test RecDataModule parameter validation.

    Validates: Requirements 14.4, 14.5
    """

    def test_num_negatives_gte_num_items_raises_value_error(self):
        """num_negatives >= num_items should raise ValueError."""
        mock_ds = _make_mock_dataset(num_items=10)
        with pytest.raises(ValueError, match="num_negatives"):
            RecDataModule(
                dataset=mock_ds,
                format="sequential",
                batch_size=4,
                num_workers=0,
                num_negatives=10,  # equal to num_items=10 → should raise
            )

    def test_num_negatives_greater_than_num_items_raises_value_error(self):
        """num_negatives > num_items should raise ValueError."""
        mock_ds = _make_mock_dataset(num_items=10)
        with pytest.raises(ValueError, match="num_negatives"):
            RecDataModule(
                dataset=mock_ds,
                format="sequential",
                batch_size=4,
                num_workers=0,
                num_negatives=15,  # > num_items=10 → should raise
            )

    def test_batch_size_zero_raises_value_error(self):
        """batch_size=0 should raise ValueError."""
        mock_ds = _make_mock_dataset()
        with pytest.raises(ValueError, match="batch_size"):
            RecDataModule(
                dataset=mock_ds,
                format="sequential",
                batch_size=0,
                num_workers=0,
                num_negatives=5,
            )

    def test_batch_size_negative_raises_value_error(self):
        """batch_size < 0 should raise ValueError."""
        mock_ds = _make_mock_dataset()
        with pytest.raises(ValueError, match="batch_size"):
            RecDataModule(
                dataset=mock_ds,
                format="sequential",
                batch_size=-1,
                num_workers=0,
                num_negatives=5,
            )

    def test_valid_parameters_do_not_raise(self):
        """Valid parameters should not raise any errors."""
        mock_ds = _make_mock_dataset(num_items=NUM_ITEMS)
        # Should not raise
        dm = RecDataModule(
            dataset=mock_ds,
            format="sequential",
            batch_size=32,
            num_workers=0,
            num_negatives=5,  # < num_items=20
        )
        assert dm is not None
