"""Utilities package for RecArena."""

from .reproducibility import set_seed, get_device, ReproducibilityContext
from .security import validate_path, safe_torch_load
from .logging import setup_logger
from .performance import (
    profile_function,
    torch_profile,
    optimize_tensor_ops,
    MemoryTracker,
)
from .sparse_optim import (
    HybridOptim,
    build_optimizer,
    sparse_embeddings_eligible,
    split_sparse_dense_params,
    clip_dense_grads_only,
)

__all__ = [
    "set_seed",
    "get_device",
    "ReproducibilityContext",
    "validate_path",
    "safe_torch_load",
    "setup_logger",
    "profile_function",
    "torch_profile",
    "optimize_tensor_ops",
    "MemoryTracker",
    "HybridOptim",
    "build_optimizer",
    "sparse_embeddings_eligible",
    "split_sparse_dense_params",
    "clip_dense_grads_only",
]
