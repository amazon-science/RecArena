"""Sparse-embedding optimizer plumbing.

Sparse embedding tables are a single-GPU memory/throughput win for large item
catalogs, but they force three constraints that this module encapsulates:

1. ``torch.optim.SparseAdam`` is the only built-in optimizer that consumes the
   sparse gradients ``nn.Embedding(sparse=True)`` produces, and it has **no
   weight_decay**. Every other (dense) parameter still wants ``AdamW``.
2. Lightning forbids returning multiple optimizers under *automatic*
   optimization. ``HybridOptim`` wraps the two optimizers so Lightning sees a
   single optimizer and keeps stepping every batch automatically.
3. Lightning's automatic gradient clipping calls ``linalg_vector_norm`` which
   has no sparse kernel; callers must clip dense params only (see
   ``clip_dense_grads_only``).

The eligibility rules ("sparse is only safe for sampled losses on a plain tied
table") live in ``sparse_embeddings_eligible`` so models and the benchmark
harness share one source of truth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
import torch.nn as nn

from .logging import setup_logger

logger = setup_logger(__name__)


# Losses that only score the positive target + sampled negatives. These never
# materialize the full [vocab, dim] logit matrix, so embedding gradients stay
# sparse. Full-softmax cross_entropy is intentionally excluded.
SAMPLED_LOSSES = frozenset({"sampled_softmax", "bce", "gbce", "bpr"})


class HybridOptim(torch.optim.Optimizer):
    """Step several optimizers as one so Lightning stays in automatic mode.

    Modified from https://github.com/Lightning-AI/lightning/issues/3346.
    Presents the union of the wrapped optimizers' ``param_groups`` / ``state``
    / ``defaults`` so Lightning utilities (LR logging, checkpointing) work.
    """

    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)

    @property
    def state(self):  # type: ignore[override]
        combined: dict = {}
        for optimizer in self.optimizers:
            combined.update(optimizer.state)
        return combined

    @property
    def param_groups(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return [g for opt in self.optimizers for g in opt.param_groups]

    @property
    def defaults(self) -> dict[str, Any]:  # type: ignore[override]
        merged: dict[str, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.defaults)
        return merged

    def __getstate__(self):  # type: ignore[override]
        return self.optimizers

    def __setstate__(self, optimizers):  # type: ignore[override]
        self.optimizers = optimizers

    def __repr__(self) -> str:
        body = "\n".join(repr(o) for o in self.optimizers)
        return f"HybridOptim({len(self.optimizers)} optimizers):\n{body}"

    def state_dict(self) -> list[dict]:
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict: list[dict]) -> None:
        for state, opt in zip(state_dict, self.optimizers):
            opt.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss


def sparse_embeddings_eligible(config, model_cls=None) -> tuple[bool, str]:
    """Decide whether a model with this `config` may use sparse embedding tables.

    Pure function of the config (+ optional model class), evaluated in
    ``__init__`` before the module tree exists. Returns (eligible, reason);
    when not eligible the reason explains the dense fallback so callers log a
    single clear warning. Eligibility requires:
      - config.sparse_embeddings is truthy, AND
      - the model class opts in (``SUPPORTS_SPARSE``; some models only have a
        full-logit loss path), AND
      - the effective loss is a sampled loss (full softmax needs the dense
        full-table matmul), AND
      - the output table is a plain tied/standard table (untied / LoRA /
        hierarchical output paths reconstruct .weight densely).
    """
    if not getattr(config, "sparse_embeddings", False):
        return False, "sparse_embeddings disabled"

    if model_cls is not None and not getattr(model_cls, "SUPPORTS_SPARSE", False):
        return (
            False,
            f"{model_cls.__name__} has no sparse-safe loss path "
            "(full-vocab logits produce dense gradients)",
        )

    loss_type = getattr(config, "loss_type", None)
    if loss_type not in SAMPLED_LOSSES:
        return (
            False,
            f"loss '{loss_type}' is not a sampled loss "
            f"({sorted(SAMPLED_LOSSES)}); full-table grads are dense",
        )

    # Untied / LoRA output projections rebuild the full table via matmul ->
    # dense gradient on the sparse param, which SparseAdam can't consume.
    if getattr(config, "tie_embeddings", True) is False:
        return False, "untied output embeddings require a dense output table"
    if getattr(config, "output_lora_rank", 0):
        return False, "LoRA output adapter rebuilds the table densely"

    emb_cfg = getattr(config, "embedding_config", None)
    if isinstance(emb_cfg, dict) and emb_cfg.get("type", "standard") != "standard":
        return False, f"embedding type '{emb_cfg.get('type')}' is not sparse-safe"

    return True, "eligible"


def split_sparse_dense_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition parameters into (sparse, dense) by their owning module.

    A parameter is "sparse" iff it belongs to an ``nn.Embedding`` with
    ``sparse=True``. Everything else (including dense embeddings, position
    tables, transformer weights) is dense.
    """
    sparse_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.sparse:
            for p in module.parameters(recurse=False):
                sparse_param_ids.add(id(p))

    sparse_params, dense_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (sparse_params if id(p) in sparse_param_ids else dense_params).append(p)
    return sparse_params, dense_params


def clip_dense_grads_only(
    model: nn.Module,
    clip_val: float,
    algorithm: str = "norm",
) -> None:
    """Clip gradients of dense params only.

    Sparse gradients have no ``linalg_vector_norm`` kernel, so Lightning's
    automatic clipping crashes on the sparse table. Models override
    ``configure_gradient_clipping`` to call this instead, leaving sparse table
    gradients unclipped (their per-step updates are already row-local and
    well-scaled).
    """
    if not clip_val:
        return
    _, dense_params = split_sparse_dense_params(model)
    if not dense_params:
        return
    if algorithm == "value":
        torch.nn.utils.clip_grad_value_(dense_params, clip_val)
    else:
        torch.nn.utils.clip_grad_norm_(dense_params, clip_val)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Build the top-level optimizer for a neural model's configure_optimizers.

    - Dense-only model (no sparse embedding params): a single ``AdamW`` over all
      parameters (unchanged behavior).
    - Sparse model (>=1 ``nn.Embedding(sparse=True)``): ``SparseAdam`` for the
      sparse table(s) + ``AdamW`` for every dense parameter, wrapped in
      ``HybridOptim`` so Lightning stays in automatic optimization.

    The returned optimizer can be wrapped directly by any torch LR scheduler:
    ``HybridOptim.param_groups`` exposes the underlying optimizers' real param
    groups, so a scheduler decays both the dense and sparse learning rates.
    """
    sparse_params, dense_params = split_sparse_dense_params(model)

    if not sparse_params:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if weight_decay and weight_decay > 0:
        logger.warning(
            "weight_decay=%s is not applied to sparse embedding tables "
            "(SparseAdam has no weight_decay). Add L2 in the loss if needed.",
            weight_decay,
        )
    sparse_opt = torch.optim.SparseAdam(sparse_params, lr=lr)
    # Pure-embedding models (e.g. BPR-MF) have no dense params -> SparseAdam only.
    if not dense_params:
        return sparse_opt
    dense_opt = torch.optim.AdamW(dense_params, lr=lr, weight_decay=weight_decay)
    return HybridOptim([dense_opt, sparse_opt])
