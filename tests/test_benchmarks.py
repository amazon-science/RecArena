"""Tests for rec_arena.benchmarks module.

Covers: BenchmarkSuite, Experiment, ExperimentConfig, ExperimentResult —
unit tests and property-based correctness checks.
"""

import json
import tempfile
import os
from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rec_arena.benchmarks.experiment import ExperimentConfig, ExperimentResult
from rec_arena.benchmarks.suite import BenchmarkSuite

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_config(name: str = "test_exp") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        model_config={"type": "sasrec", "embedding_dim": 64},
        dataset_config={"name": "ml100k"},
        training_config={"max_epochs": 10, "lr": 0.001},
        evaluation_config={"k": [5, 10]},
    )


def _make_result(name: str = "test_exp", metric_val: float = 0.5) -> ExperimentResult:
    return ExperimentResult(
        config=_make_config(name),
        metrics={"ndcg@10": metric_val, "recall@10": metric_val * 0.8},
        training_time=1.0,
        inference_time=0.1,
        timestamp=datetime.now().isoformat(),
        metadata={"note": "unit test"},
    )


def _suite_with_results(*metric_vals) -> BenchmarkSuite:
    """Return a BenchmarkSuite whose .results list is pre-populated."""
    suite = BenchmarkSuite()
    for i, val in enumerate(metric_vals):
        suite.results.append(_make_result(name=f"exp_{i}", metric_val=val))
    return suite


# ---------------------------------------------------------------------------
# 9.1 — BenchmarkSuite unit tests
# ---------------------------------------------------------------------------


class TestAddExperiment:
    def test_add_experiment_appends_to_list(self):
        suite = BenchmarkSuite()
        assert len(suite.experiments) == 0
        suite.add_experiment(_make_config("a"))
        assert len(suite.experiments) == 1

    def test_add_multiple_experiments(self):
        suite = BenchmarkSuite()
        for i in range(5):
            suite.add_experiment(_make_config(f"exp_{i}"))
        assert len(suite.experiments) == 5

    def test_added_experiment_has_correct_config_name(self):
        suite = BenchmarkSuite()
        suite.add_experiment(_make_config("my_exp"))
        assert suite.experiments[0].config.name == "my_exp"


class TestGetLeaderboard:
    def test_leaderboard_sorted_descending(self):
        suite = _suite_with_results(0.3, 0.7, 0.5)
        lb = suite.get_leaderboard("ndcg@10")
        scores = [entry["score"] for entry in lb]
        assert scores == sorted(scores, reverse=True)

    def test_leaderboard_returns_all_entries(self):
        suite = _suite_with_results(0.1, 0.9, 0.5)
        lb = suite.get_leaderboard("ndcg@10")
        assert len(lb) == 3

    def test_leaderboard_missing_metric_excluded(self):
        suite = _suite_with_results(0.5)
        lb = suite.get_leaderboard("nonexistent_metric")
        assert lb == []

    def test_leaderboard_entry_has_name_and_score(self):
        suite = _suite_with_results(0.8)
        lb = suite.get_leaderboard("ndcg@10")
        assert "name" in lb[0]
        assert "score" in lb[0]


class TestSaveResults:
    def test_save_results_writes_valid_json(self, tmp_path):
        suite = _suite_with_results(0.5, 0.7)
        out_file = tmp_path / "results.json"
        suite.save_results(str(out_file))
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_save_results_json_contains_metrics(self, tmp_path):
        suite = _suite_with_results(0.6)
        out_file = tmp_path / "results.json"
        suite.save_results(str(out_file))
        data = json.loads(out_file.read_text())
        assert "metrics" in data[0]
        assert "ndcg@10" in data[0]["metrics"]

    def test_save_results_path_traversal_raises_runtime_error(self, tmp_path):
        suite = _suite_with_results(0.5)
        bad_path = str(tmp_path / ".." / "evil.json")
        with pytest.raises(RuntimeError):
            suite.save_results(bad_path)

    def test_save_results_empty_suite(self, tmp_path):
        suite = BenchmarkSuite()
        out_file = tmp_path / "empty.json"
        suite.save_results(str(out_file))
        data = json.loads(out_file.read_text())
        assert data == []


class TestExperimentConfigToDict:
    def test_to_dict_contains_all_keys(self):
        cfg = _make_config()
        d = cfg.to_dict()
        for key in ("name", "model_config", "dataset_config", "training_config", "evaluation_config"):
            assert key in d

    def test_to_dict_name_matches(self):
        cfg = _make_config("my_model")
        assert cfg.to_dict()["name"] == "my_model"

    def test_to_dict_model_config_matches(self):
        cfg = _make_config()
        assert cfg.to_dict()["model_config"] == cfg.model_config


class TestExperimentResultToDict:
    def test_to_dict_contains_all_keys(self):
        result = _make_result()
        d = result.to_dict()
        for key in ("config", "metrics", "training_time", "inference_time", "timestamp", "metadata"):
            assert key in d

    def test_to_dict_metrics_match(self):
        result = _make_result(metric_val=0.42)
        d = result.to_dict()
        assert d["metrics"]["ndcg@10"] == pytest.approx(0.42)

    def test_to_dict_config_is_dict(self):
        result = _make_result()
        d = result.to_dict()
        assert isinstance(d["config"], dict)

    def test_to_dict_metadata_defaults_to_empty_dict(self):
        result = ExperimentResult(
            config=_make_config(),
            metrics={"ndcg@10": 0.5},
            training_time=1.0,
            inference_time=0.1,
            timestamp=datetime.now().isoformat(),
            metadata=None,
        )
        d = result.to_dict()
        assert d["metadata"] == {}


# ---------------------------------------------------------------------------
# Hypothesis strategies for benchmark tests
# ---------------------------------------------------------------------------


def _experiment_config_strategy():
    """Strategy that generates ExperimentConfig instances."""
    return st.builds(
        ExperimentConfig,
        name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-")),
        model_config=st.fixed_dictionaries({"type": st.just("sasrec")}),
        dataset_config=st.fixed_dictionaries({"name": st.just("ml100k")}),
        training_config=st.fixed_dictionaries({"max_epochs": st.integers(min_value=1, max_value=100)}),
        evaluation_config=st.fixed_dictionaries({"k": st.just([10])}),
    )


def _experiment_result_strategy():
    """Strategy that generates ExperimentResult instances with a single metric."""
    return st.builds(
        ExperimentResult,
        config=_experiment_config_strategy(),
        metrics=st.fixed_dictionaries({
            "ndcg@10": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        }),
        training_time=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        inference_time=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        timestamp=st.just(datetime.now().isoformat()),
        metadata=st.none() | st.fixed_dictionaries({}),
    )


# ---------------------------------------------------------------------------
# 9.2 — Property 35: BenchmarkSuite.add_experiment grows list
# ---------------------------------------------------------------------------

# Feature: comprehensive-test-suite, Property 35: BenchmarkSuite.add_experiment grows list


@given(configs=st.lists(_experiment_config_strategy(), min_size=1, max_size=20))
@settings(max_examples=100)
def test_property_35_add_experiment_grows_list(configs):
    """Property 35: BenchmarkSuite.add_experiment grows list.

    Validates: Requirements 15.1
    """
    suite = BenchmarkSuite()
    for i, cfg in enumerate(configs):
        before = len(suite.experiments)
        suite.add_experiment(cfg)
        assert len(suite.experiments) == before + 1
    assert len(suite.experiments) == len(configs)


# ---------------------------------------------------------------------------
# 9.3 — Property 36: Leaderboard is sorted descending
# ---------------------------------------------------------------------------

# Feature: comprehensive-test-suite, Property 36: Leaderboard is sorted descending


@given(
    metric_vals=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_property_36_leaderboard_sorted_descending(metric_vals):
    """Property 36: Leaderboard is sorted descending.

    Validates: Requirements 15.2
    """
    suite = BenchmarkSuite()
    for i, val in enumerate(metric_vals):
        suite.results.append(_make_result(name=f"exp_{i}", metric_val=val))

    lb = suite.get_leaderboard("ndcg@10")
    scores = [entry["score"] for entry in lb]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 9.4 — Property 37: Benchmark save/load round-trip
# ---------------------------------------------------------------------------

# Feature: comprehensive-test-suite, Property 37: Benchmark save/load round-trip


@given(results=st.lists(_experiment_result_strategy(), min_size=1, max_size=10))
@settings(max_examples=50)
def test_property_37_benchmark_save_load_roundtrip(results):
    """Property 37: Benchmark save/load round-trip.

    Validates: Requirements 15.3
    """
    suite = BenchmarkSuite()
    suite.results = results

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "results.json")
        suite.save_results(out_file)

        assert os.path.exists(out_file)
        with open(out_file) as f:
            loaded = json.load(f)

    assert len(loaded) == len(results)
    for original, saved in zip(results, loaded):
        assert saved["metrics"] == original.metrics
        assert saved["config"]["name"] == original.config.name


# ---------------------------------------------------------------------------
# 9.5 — Property 38: Experiment dataclass to_dict completeness
# ---------------------------------------------------------------------------

# Feature: comprehensive-test-suite, Property 38: Experiment dataclass to_dict completeness


@given(cfg=_experiment_config_strategy())
@settings(max_examples=100)
def test_property_38_experiment_config_to_dict_completeness(cfg):
    """Property 38: ExperimentConfig.to_dict completeness.

    Validates: Requirements 15.5
    """
    d = cfg.to_dict()
    for key in ("name", "model_config", "dataset_config", "training_config", "evaluation_config"):
        assert key in d
    assert d["name"] == cfg.name
    assert d["model_config"] == cfg.model_config
    assert d["dataset_config"] == cfg.dataset_config
    assert d["training_config"] == cfg.training_config
    assert d["evaluation_config"] == cfg.evaluation_config


@given(result=_experiment_result_strategy())
@settings(max_examples=100)
def test_property_38_experiment_result_to_dict_completeness(result):
    """Property 38: ExperimentResult.to_dict completeness.

    Validates: Requirements 15.6
    """
    d = result.to_dict()
    for key in ("config", "metrics", "training_time", "inference_time", "timestamp", "metadata"):
        assert key in d
    assert d["metrics"] == result.metrics
    assert d["training_time"] == result.training_time
    assert d["inference_time"] == result.inference_time
    assert d["timestamp"] == result.timestamp
    assert isinstance(d["config"], dict)


# ---------------------------------------------------------------------------
# 9.6 — Property 40: ExperimentResult JSON round-trip
# ---------------------------------------------------------------------------

# Feature: comprehensive-test-suite, Property 40: ExperimentResult JSON round-trip


@given(result=_experiment_result_strategy())
@settings(max_examples=100)
def test_property_40_experiment_result_json_roundtrip(result):
    """Property 40: ExperimentResult JSON round-trip.

    Validates: Requirements 17.3
    """
    original_dict = result.to_dict()
    serialized = json.dumps(original_dict)
    deserialized = json.loads(serialized)

    assert deserialized == original_dict
    assert deserialized["metrics"] == original_dict["metrics"]
    assert deserialized["training_time"] == original_dict["training_time"]
    assert deserialized["inference_time"] == original_dict["inference_time"]
    assert deserialized["timestamp"] == original_dict["timestamp"]
    assert deserialized["config"] == original_dict["config"]


# ===========================================================================
# Coverage tests merged from tests/test_benchmark_coverage.py
#
# Additional tests for benchmarks/experiment.py and benchmarks/suite.py.
# Covers: Experiment.run, Experiment._evaluate_model, Experiment.save_result,
# BenchmarkSuite.run_all, run_single, load_config, export_leaderboard.
# ===========================================================================

from unittest.mock import MagicMock

import torch

from rec_arena.benchmarks.experiment import Experiment


def _make_config_cov(name="exp1"):
    return ExperimentConfig(
        name=name,
        model_config={"name": "sasrec"},
        dataset_config={"name": "ml100k"},
        training_config={"epochs": 10},
        evaluation_config={"k": 10},
    )


def _make_result_cov(name="exp1", ndcg=0.5):
    return ExperimentResult(
        config=_make_config_cov(name),
        metrics={"ndcg@10": ndcg, "recall@10": ndcg * 0.8},
        training_time=1.0,
        inference_time=0.5,
        timestamp="2025-01-01T00:00:00",
    )


# ===================================================================
# Experiment
# ===================================================================


class TestExperimentRun:
    def test_run_returns_result(self):
        config = _make_config_cov()
        exp = Experiment(config)

        model = MagicMock()
        dataset = MagicMock()
        dataset.test_dataloader.return_value = []  # empty test loader
        metric_calc = MagicMock()

        result = exp.run(model, dataset, metric_calc)
        assert isinstance(result, ExperimentResult)
        assert result.config.name == "exp1"
        model.fit.assert_called_once()

    def test_run_stores_result(self):
        config = _make_config_cov()
        exp = Experiment(config)

        model = MagicMock()
        dataset = MagicMock()
        dataset.test_dataloader.return_value = []
        metric_calc = MagicMock()

        exp.run(model, dataset, metric_calc)
        assert exp.result is not None


class TestExperimentSaveResult:
    def test_save_result_writes_json(self, tmp_path):
        config = _make_config_cov()
        exp = Experiment(config)
        exp.result = _make_result_cov()

        path = str(tmp_path / "result.json")
        exp.save_result(path)

        with open(path) as f:
            data = json.load(f)
        assert "metrics" in data
        assert data["metrics"]["ndcg@10"] == 0.5

    def test_save_result_no_result_does_nothing(self, tmp_path):
        config = _make_config_cov()
        exp = Experiment(config)
        path = str(tmp_path / "result.json")
        exp.save_result(path)
        assert not (tmp_path / "result.json").exists()


class TestExperimentEvaluateModel:
    def test_evaluate_implicit_model(self):
        config = _make_config_cov()
        exp = Experiment(config)

        model = MagicMock()
        model.eval = MagicMock()

        batch = {
            "user_id": torch.tensor([0, 1, 2, 3]),
            "item_id": torch.tensor([5, 6, 7, 8]),
        }
        model.predict.return_value = torch.randn(4, 20)

        dataset = MagicMock()
        dataset.test_dataloader.return_value = [batch]

        metric_calc = MagicMock()
        metric_calc.calculate_batch.return_value = {"ndcg@10": 0.5}

        # Use a plain object that is NOT a SequentialModel instance
        metrics = exp._evaluate_model(model, dataset, metric_calc)
        assert isinstance(metrics, dict)

    def test_evaluate_empty_test_loader(self):
        config = _make_config_cov()
        exp = Experiment(config)
        model = MagicMock()
        model.eval = MagicMock()
        dataset = MagicMock()
        dataset.test_dataloader.return_value = []
        metric_calc = MagicMock()

        metrics = exp._evaluate_model(model, dataset, metric_calc)
        assert metrics == {}


# ===================================================================
# BenchmarkSuite — run_all, run_single, load_config, export_leaderboard
# ===================================================================


class TestBenchmarkSuiteRunAll:
    def test_run_all_returns_results(self):
        suite = BenchmarkSuite()
        suite.add_experiment(_make_config_cov("a"))
        suite.add_experiment(_make_config_cov("b"))

        model = MagicMock()
        dataset = MagicMock()
        dataset.test_dataloader.return_value = []

        models = {"sasrec": model}
        datasets = {"ml100k": dataset}

        results = suite.run_all(models, datasets)
        assert len(results) == 2
        assert all(isinstance(r, ExperimentResult) for r in results)


class TestBenchmarkSuiteRunSingle:
    def test_run_single_found(self):
        suite = BenchmarkSuite()
        suite.add_experiment(_make_config_cov("target"))
        suite.add_experiment(_make_config_cov("other"))

        model = MagicMock()
        dataset = MagicMock()
        dataset.test_dataloader.return_value = []

        result = suite.run_single("target", {"sasrec": model}, {"ml100k": dataset})
        assert result is not None
        assert result.config.name == "target"

    def test_run_single_not_found(self):
        suite = BenchmarkSuite()
        suite.add_experiment(_make_config_cov("other"))

        result = suite.run_single("missing", {}, {})
        assert result is None


class TestBenchmarkSuiteLoadConfig:
    def test_load_config_from_file(self, tmp_path):
        config_data = {
            "experiments": [
                {
                    "name": "loaded_exp",
                    "model_config": {"name": "sasrec"},
                    "dataset_config": {"name": "ml100k"},
                    "training_config": {},
                    "evaluation_config": {},
                }
            ]
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        suite = BenchmarkSuite(config_path=str(config_path))
        assert len(suite.experiments) == 1
        assert suite.experiments[0].config.name == "loaded_exp"

    def test_load_config_path_traversal_raises(self):
        with pytest.raises(RuntimeError):
            BenchmarkSuite(config_path="../etc/passwd")

    def test_load_config_nonexistent_raises(self):
        with pytest.raises(RuntimeError):
            BenchmarkSuite(config_path="/nonexistent/path.json")


class TestBenchmarkSuiteExportLeaderboard:
    def _suite_with_results(self):
        suite = BenchmarkSuite()
        suite.results = [_make_result_cov("a", 0.8), _make_result_cov("b", 0.6)]
        return suite

    def test_export_json(self, tmp_path):
        suite = self._suite_with_results()
        path = str(tmp_path / "lb.json")
        suite.export_leaderboard(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["score"] >= data[1]["score"]

    def test_export_csv(self, tmp_path):
        suite = self._suite_with_results()
        path = str(tmp_path / "lb.csv")
        suite.export_leaderboard(path)
        assert (tmp_path / "lb.csv").exists()

    def test_export_unsupported_format_raises(self, tmp_path):
        suite = self._suite_with_results()
        path = str(tmp_path / "lb.txt")
        with pytest.raises(RuntimeError, match="Unsupported"):
            suite.export_leaderboard(path)

    def test_export_path_traversal_raises(self):
        suite = self._suite_with_results()
        with pytest.raises(RuntimeError):
            suite.export_leaderboard("../evil/lb.json")
