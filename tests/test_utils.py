"""Tests for rec_arena.utils module.

Covers: TensorCache, set_seed, ReproducibilityContext, get_device,
memory_efficient_mode, clear_memory, profile_function, MemoryTracker,
validate_path, safe_torch_load.
"""

import os
import random
import tempfile

import numpy as np
import pytest
import torch

from rec_arena.utils.cache import TensorCache, cached_computation
from rec_arena.utils.reproducibility import (
    ReproducibilityContext,
    get_device,
    set_seed,
)
from rec_arena.utils.memory import clear_memory, memory_efficient_mode
from rec_arena.utils.performance import MemoryTracker, profile_function
from rec_arena.utils.security import validate_path, safe_torch_load


# ===================================================================
# TensorCache
# ===================================================================


class TestTensorCache:
    def test_get_missing_key_returns_none(self):
        cache = TensorCache()
        assert cache.get("nonexistent") is None

    def test_put_and_get(self):
        cache = TensorCache()
        t = torch.tensor([1.0, 2.0])
        cache.put("k1", t)
        result = cache.get("k1")
        assert result is not None
        assert torch.equal(result, t)

    def test_put_returns_detached_clone(self):
        cache = TensorCache()
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        cache.put("k1", t)
        cached = cache.get("k1")
        assert not cached.requires_grad

    def test_eviction_when_full(self):
        cache = TensorCache(max_size=2)
        cache.put("a", torch.tensor([1.0]))
        cache.put("b", torch.tensor([2.0]))
        # Access "b" more to make "a" least accessed
        cache.get("b")
        cache.get("b")
        cache.put("c", torch.tensor([3.0]))
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_clear(self):
        cache = TensorCache()
        cache.put("k1", torch.tensor([1.0]))
        cache.clear()
        assert cache.get("k1") is None
        assert len(cache.cache) == 0
        assert len(cache.access_count) == 0

    def test_access_count_increments(self):
        cache = TensorCache()
        cache.put("k1", torch.tensor([1.0]))
        cache.get("k1")
        cache.get("k1")
        assert cache.access_count["k1"] == 3  # 1 from put + 2 from get


# ===================================================================
# cached_computation decorator
# ===================================================================


class TestCachedComputation:
    def test_caches_tensor_result(self):
        call_count = 0

        @cached_computation("test_key")
        def compute():
            nonlocal call_count
            call_count += 1
            return torch.tensor([42.0])

        result1 = compute()
        result2 = compute()
        assert torch.equal(result1, result2)
        assert call_count == 1


# ===================================================================
# Reproducibility
# ===================================================================


class TestSetSeed:
    def test_deterministic_random(self):
        set_seed(123)
        a = random.random()
        set_seed(123)
        b = random.random()
        assert a == b

    def test_deterministic_numpy(self):
        set_seed(123)
        a = np.random.rand()
        set_seed(123)
        b = np.random.rand()
        assert a == b

    def test_deterministic_torch(self):
        set_seed(123)
        a = torch.rand(5)
        set_seed(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            set_seed(-1)

    def test_non_int_seed_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            set_seed(3.14)

    def test_sets_pythonhashseed(self):
        set_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"


class TestGetDevice:
    def test_explicit_cpu(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_explicit_cuda(self):
        device = get_device("cuda")
        assert device == torch.device("cuda")

    def test_auto_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)


class TestReproducibilityContext:
    def test_restores_state(self):
        random.seed(0)
        before = random.random()

        random.seed(0)
        with ReproducibilityContext(seed=999):
            inside = random.random()

        after = random.random()
        assert before == after  # state restored
        assert inside != before  # different seed inside

    def test_deterministic_inside_context(self):
        with ReproducibilityContext(seed=42):
            a = torch.rand(3)
        with ReproducibilityContext(seed=42):
            b = torch.rand(3)
        assert torch.equal(a, b)


# ===================================================================
# Memory utilities
# ===================================================================


class TestMemoryUtils:
    def test_memory_efficient_mode_restores_settings(self):
        original_benchmark = torch.backends.cudnn.benchmark
        original_det = torch.backends.cudnn.deterministic
        with memory_efficient_mode():
            assert torch.backends.cudnn.benchmark is False
            assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark == original_benchmark
        assert torch.backends.cudnn.deterministic == original_det

    def test_clear_memory_runs_without_error(self):
        clear_memory()  # Should not raise


# ===================================================================
# Performance utilities
# ===================================================================


class TestProfileFunction:
    def test_decorated_function_returns_result(self):
        @profile_function
        def add(a, b):
            return a + b

        assert add(2, 3) == 5


class TestMemoryTracker:
    def test_start_stop_returns_dict(self):
        tracker = MemoryTracker()
        tracker.start()
        result = tracker.stop()
        assert "peak_memory_mb" in result
        assert "current_memory_mb" in result
        assert "memory_increase_mb" in result


# ===================================================================
# Security utilities
# ===================================================================


class TestValidatePath:
    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_path("")

    def test_valid_path_returns_path(self):
        # Use a path within cwd
        result = validate_path("tests/conftest.py")
        assert result.exists()

    def test_allowed_extensions_filter(self):
        with pytest.raises(ValueError, match="extension"):
            validate_path("tests/conftest.py", allowed_extensions=[".txt"])

    def test_allowed_extensions_pass(self):
        result = validate_path("tests/conftest.py", allowed_extensions=[".py"])
        assert result.exists()


class TestSafeTorchLoad:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            safe_torch_load("nonexistent_model.pt")

    def test_wrong_extension_raises(self):
        with pytest.raises(ValueError):
            safe_torch_load("tests/conftest.py")

    def test_loads_valid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", dir=".", delete=False) as f:
            torch.save({"weight": torch.tensor([1.0])}, f.name)
            try:
                data = safe_torch_load(f.name)
                assert "weight" in data
            finally:
                os.unlink(f.name)


# ===========================================================================
# Additional coverage tests (from test_utils_coverage.py)
# Covers: logging setup, memory utils, performance profiling.
# ===========================================================================

from rec_arena.utils.logging import setup_logger


class TestSetupLogger:
    def test_returns_logger(self):
        logger = setup_logger("test_module_cov")
        assert logger is not None
        assert logger.name == "test_module_cov"

    def test_logger_has_handlers(self):
        logger = setup_logger("test_module_cov2")
        assert len(logger.handlers) >= 1
