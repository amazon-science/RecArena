"""SASRecPack: Efficient Hyperparameter Ensembles for Sequential Recommendation.

Applies the TabPack pattern (Gorishniy et al., ICML 2026) to SASRec. Trains
multiple SASRec variants simultaneously with shared item/position embeddings
but per-member transformer depth, FFN width, dropout, learning rate, and weight
decay. Uses custom optimizer packs (AdamWPack / MuonAdamWPack) with per-member
hyperparameters. Greedy online ensemble selection runs during training.

Key differences from standard ensembling:
  - Single batched forward pass for all members (bmm-based, GPU-parallel)
  - Per-member early stopping with dynamic pack shrinking
  - Online greedy ensemble selection during training (not post-hoc)
  - Shared item embeddings → better representations, less memory
"""

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from ...modules.pack.nn import (
    PACK_DIM,
    BATCH_DIM,
    ParameterPack,
    get_pack_size,
    make_keep_pack_idx,
    LinearPack,
    DropoutPack,
    LayerNormPack,
    TransformerBlockPack,
    module_pack_remove,
)
from ...modules.pack.optim import (
    AdamWPack,
    MuonAdamWPack,
    optimizer_pack_remove,
    zeropower_via_newtonschulz5,
)


# =============================================================================
# Config
# =============================================================================


@dataclass
class SASRecPackConfig:
    """Full configuration for SASRecPack training."""

    # Data
    vocab_size: int = None
    max_seq_length: int = 200
    pad_token: int = 0

    # Shared architecture (fixed across pack members)
    embedding_dim: int = 128
    num_heads: int = 2
    scale_embeddings: bool = True

    # Pack
    pack_size: int = 24
    seed: int = 0

    # Per-member hyperparameter ranges. GPU memory scales with the PADDED max
    # (every member's FFN is padded to max_ffn, every layer slot allocated), so
    # max_num_layers/max_ffn_multiplier are held at the proven-safe level (4 /
    # 4x = the 32-member config that fit 24GB). Diversity comes from WIDE ranges
    # (members span 0.5x-4x FFN, 1-4 layers) plus the FREE levers below
    # (activation is zero extra memory; lr/wd/dropout are per-member scalars).
    min_num_layers: int = 1
    max_num_layers: int = 4
    min_ffn_multiplier: float = 0.5
    max_ffn_multiplier: float = 4.0
    min_dropout: float = 0.0
    max_dropout: float = 0.6
    min_lr: float = 5e-5
    max_lr: float = 1e-2
    min_weight_decay: float = 1e-6
    max_weight_decay: float = 3e-1
    # Per-member FFN activation (categorical diversity lever, ZERO extra memory).
    activations: tuple = ("gelu", "relu", "silu", "tanh")

    # Optimizer
    optimizer_type: str = "AdamWPack"  # "AdamWPack" or "MuonAdamWPack"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # Shared embedding optimizer (separate from pack)
    embedding_lr: float = 1e-3
    embedding_weight_decay: float = 0.0

    # Training
    batch_size: int = 256
    n_epochs: int = -1  # -1 = unlimited (rely on patience/timeout)
    patience: int = 16  # Per-member early stopping
    eval_batch_size: int = 256  # small: eval runs all seq positions x P members
    pred_topk: int = 200  # top-K items stored per member (>= max metric cutoff)
    timeout: int = None  # Max training time in seconds
    amp_dtype: str = None  # "bfloat16" or None
    grad_checkpoint: bool = True  # REQUIRED: no-ckpt fwd OOMs 24GB A10G at pack=32

    # Ensemble
    ensemble_type: str = "greedy"  # "greedy", "topk"
    ensemble_patience: int = 32
    max_ensemble_size: int = 16
    ensemble_with_replacement: bool = True  # TabPack default: weighted greedy
    ensemble_update_type: str = "best"  # "best", "latest"

    # Loss
    num_negatives: int = 0  # 0 = full cross-entropy, >0 = sampled softmax
    temperature: float = 1.0

    # Reporting
    eval_every_n_epochs: int = 1

    # Hyperparameter sampler: "qmc" (Sobol low-discrepancy, TabPack default),
    # "tpe" (Optuna TPE; requires optuna), or "random".
    sampler_type: str = "qmc"


# =============================================================================
# Hyperparameter sampler
# =============================================================================


class HyperparameterSampler:
    """Samples P joint HP configs for the pack, following TabPack.

    TabPack draws each member's hyperparameters from a shared search space using
    an Optuna sampler (QMCSampler / TPESampler / RandomSampler). We reproduce
    that here: when Optuna is importable we use its samplers directly (identical
    behavior to TabPack); otherwise we fall back to scipy's Sobol sequence for
    QMC (the same low-discrepancy construction Optuna's QMCSampler uses) and to
    numpy for random. TPE without Optuna is not available, so it degrades to QMC.

    The space is a dict of param -> ("int"|"uniform"|"loguniform", low, high).
    Sampling is joint: the i-th member gets the i-th point of a P-point
    low-discrepancy set over the whole space, giving better coverage than
    per-axis independent uniforms.
    """

    def __init__(self, space: dict[str, tuple], *, seed: int, sampler_type: str):
        self.space = space
        self.seed = seed
        self.sampler_type = sampler_type
        self._optuna = None
        if sampler_type in ("qmc", "tpe"):
            try:
                import optuna  # noqa: F401

                self._optuna = optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            except ImportError:
                self._optuna = None

    def _unit_to_value(self, name: str, u: float):
        spec = self.space[name]
        kind = spec[0]
        if kind == "categorical":
            choices = spec[1]
            idx = min(int(u * len(choices)), len(choices) - 1)
            return choices[idx]
        _, low, high = spec
        if kind == "int":
            # inclusive integer range
            return int(np.floor(low + u * (high + 1 - low)).clip(low, high))
        if kind == "loguniform":
            return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
        # uniform
        return float(low + u * (high - low))

    def sample(self, n: int) -> list[dict]:
        names = list(self.space)
        d = len(names)

        if self._optuna is not None:
            optuna = self._optuna
            if self.sampler_type == "tpe":
                sampler = optuna.samplers.TPESampler(seed=self.seed)
            else:
                sampler = optuna.samplers.QMCSampler(
                    qmc_type="sobol", scramble=True, seed=self.seed
                )
            study = optuna.create_study(sampler=sampler, direction="maximize")
            configs = []
            for _ in range(n):
                trial = study.ask()
                cfg = {}
                for name in names:
                    spec = self.space[name]
                    kind = spec[0]
                    if kind == "categorical":
                        cfg[name] = trial.suggest_categorical(name, spec[1])
                    elif kind == "int":
                        cfg[name] = trial.suggest_int(name, int(spec[1]), int(spec[2]))
                    elif kind == "loguniform":
                        cfg[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
                    else:
                        cfg[name] = trial.suggest_float(name, spec[1], spec[2])
                # Tell a neutral value; we don't optimize the study online, we
                # only use it as a joint low-discrepancy generator over the space
                # (same role QMCSampler plays in TabPack's conservative protocol).
                study.tell(trial, 0.0)
                configs.append(cfg)
            return configs

        # --- Fallback: scipy Sobol for QMC, numpy for random ---
        if self.sampler_type == "random":
            rng = np.random.default_rng(self.seed)
            unit = rng.random((n, d))
        else:
            from scipy.stats import qmc

            # Sobol' balance requires a power-of-2 count: draw 2^ceil(log2 n)
            # points (via random_base2, avoids the warning) and take the first n.
            sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
            m_bits = max(1, int(np.ceil(np.log2(n))))
            unit = sampler.random_base2(m_bits)[:n]  # (n, d) in [0,1)

        return [
            {name: self._unit_to_value(name, unit[i, j]) for j, name in enumerate(names)}
            for i in range(n)
        ]


# =============================================================================
# Model
# =============================================================================


class SASRecPackModel(nn.Module):
    """The SASRecPack model — a pack of SASRec variants.

    Shared: item_embedding, pos_embedding (all members score same item space).
    Per-member (packed): transformer blocks, final layer norm.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        max_seq_length: int,
        pack_size: int,
        num_layers: list[int],
        ffn_dims: list[int],
        dropout_rates: list[float],
        activations: list[str] | None = None,
        scale_embeddings: bool = True,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        assert len(num_layers) == pack_size
        assert len(ffn_dims) == pack_size
        assert len(dropout_rates) == pack_size
        assert activations is None or len(activations) == pack_size

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.grad_checkpoint = grad_checkpoint
        self.scale_embeddings = scale_embeddings
        self.max_num_layers = max(num_layers)

        # Shared embeddings (NOT ParameterPack — standard parameters)
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Per-member number of layers (stored as float ParameterPack for removal support)
        self.member_num_layers = ParameterPack(
            torch.tensor(num_layers, dtype=torch.float)
        )
        self.member_num_layers.requires_grad_(False)

        # Per-member input dropout (ParameterPack, no grad)
        self.input_dropout_rates = ParameterPack(
            torch.tensor(dropout_rates, dtype=torch.float)
        )
        self.input_dropout_rates.requires_grad_(False)

        # Transformer block packs (max_num_layers of them)
        self.blocks = nn.ModuleList([
            TransformerBlockPack(
                dim=embedding_dim,
                feedforward_dim=ffn_dims,
                dropout=dropout_rates,
                num_heads=num_heads,
                max_feedforward_dim=max(ffn_dims),
                activations=activations,
                pack_size=pack_size,
            )
            for _ in range(self.max_num_layers)
        ])

        # Final layer norm (per-member)
        self.final_norm = LayerNormPack(embedding_dim, pack_size=pack_size)

        self._init_weights()

    @property
    def pack_size(self) -> int:
        return self.member_num_layers.shape[0]

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        # Zero out padding
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(
        self,
        sequences: Tensor,
        sequence_lengths: Tensor,
    ) -> Tensor:
        """Compute hidden states for all pack members.

        Args:
            sequences: (batch_size, seq_len) item IDs
            sequence_lengths: (batch_size,) actual lengths

        Returns:
            (pack_size, batch_size, seq_len, dim) hidden states
        """
        B, S = sequences.shape
        P = self.pack_size
        D = self.embedding_dim

        # Shared embeddings
        item_embs = self.item_embedding(sequences)  # (B, S, D)
        if self.scale_embeddings:
            item_embs = item_embs * (D ** 0.5)

        positions = torch.arange(S, device=sequences.device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=self.max_seq_length - 1)
        x = item_embs + self.pos_embedding(positions)  # (B, S, D)

        # Expand to pack: (P, B, S, D)
        x = x.unsqueeze(PACK_DIM).expand(P, -1, -1, -1).contiguous()

        # Per-member input dropout
        if self.training:
            p_keep = 1.0 - self.input_dropout_rates  # (P,)
            mask = torch.bernoulli(p_keep[:, None, None, None].expand_as(x))
            x = x * mask / p_keep[:, None, None, None].clamp(min=1e-8)

        # Transformer blocks with variable depth. Gradient checkpointing trades
        # compute for memory: block activations are recomputed in the backward
        # pass instead of stored, which is essential for large packs (the pack
        # holds P copies of every (B,S,D)/(B,S,ffn) activation across all layers).
        use_ckpt = self.grad_checkpoint and self.training
        for layer_idx, block in enumerate(self.blocks):
            active = self.member_num_layers > layer_idx  # (P,)
            if not active.any():
                break
            if active.all():
                x = (
                    checkpoint(block, x, use_reentrant=False)
                    if use_ckpt else block(x)
                )
            else:
                active_idx = active.nonzero(as_tuple=True)[0]
                x_active = (
                    checkpoint(block.forward_subset, x[active_idx], active_idx,
                               use_reentrant=False)
                    if use_ckpt else block.forward_subset(x[active_idx], active_idx)
                )
                x = x.index_copy(PACK_DIM, active_idx, x_active)

        # Final norm
        x = self.final_norm(x)
        return x

    def compute_logits(self, hidden_states: Tensor) -> Tensor:
        """Compute logits from hidden states using shared item embeddings.

        Args:
            hidden_states: (pack_size, batch_size, seq_len, dim)

        Returns:
            (pack_size, batch_size, seq_len, vocab_size)
        """
        return torch.matmul(hidden_states, self.item_embedding.weight.T)

    def get_pack_parameters(self) -> list[nn.Parameter]:
        """Return only the pack parameters (not shared embeddings)."""
        pack_params = []
        for block in self.blocks:
            pack_params.extend(p for p in block.parameters())
        pack_params.extend(self.final_norm.parameters())
        return pack_params

    def get_shared_parameters(self) -> list[nn.Parameter]:
        """Return shared parameters (embeddings)."""
        return list(self.item_embedding.parameters()) + list(
            self.pos_embedding.parameters()
        )


# =============================================================================
# State tracking (analogous to TabPack's StatePack)
# =============================================================================


class PackState:
    """Tracks per-member training state: steps, best scores, patience."""

    def __init__(self, pack_size: int, member_configs: list[dict]):
        self.ids = np.arange(pack_size)
        self.configs = deepcopy(member_configs)
        self.steps = np.zeros(pack_size, dtype=np.int64)
        self.n_bad_updates = np.zeros(pack_size, dtype=np.int64)
        # Early-stopping tracker (NDCG@10): gates patience / member removal.
        self.best_scores = np.full(pack_size, -np.inf)
        # Checkpoint tracker (log-loss): gates WHICH prediction/checkpoint is
        # stored in the pool. The ensemble selects on log-loss, so the pooled
        # checkpoint must be the log-loss-best one, not the NDCG-best one.
        self.best_ckpt_scores = np.full(pack_size, -np.inf)
        self.best_steps = np.zeros(pack_size, dtype=np.int64)
        self.best_predictions: dict[str, np.ndarray] = {}
        self.best_model_state: dict[str, Tensor] = {}

    @property
    def pack_size(self) -> int:
        return len(self.ids)

    def step(self):
        self.steps += 1

    def update(
        self,
        stop_scores: np.ndarray,
        ckpt_scores: np.ndarray,
        predictions: dict[str, np.ndarray],
        model_state: dict[str, Tensor],
    ) -> np.ndarray:
        """Update trackers with two separate signals.

        - stop_scores (NDCG@10): resets patience on improvement -> early stop.
        - ckpt_scores (log-loss): gates which prediction/checkpoint is stored,
          matching the ensemble's selection objective.

        Returns indices whose early-stopping score improved (for logging).
        """
        assert len(stop_scores) == self.pack_size
        assert len(ckpt_scores) == self.pack_size

        # --- Checkpoint tracker (log-loss): store best-by-log-loss preds/state.
        ckpt_improved = ckpt_scores > self.best_ckpt_scores
        ckpt_idx = np.nonzero(ckpt_improved)[0]
        if len(ckpt_idx) > 0:
            self.best_ckpt_scores[ckpt_idx] = ckpt_scores[ckpt_idx]
            self.best_steps[ckpt_idx] = self.steps[ckpt_idx]
            for k, v in predictions.items():
                if k not in self.best_predictions:
                    self.best_predictions[k] = deepcopy(v)
                else:
                    self.best_predictions[k][ckpt_idx] = v[ckpt_idx]
            for k, v in model_state.items():
                if k not in self.best_model_state:
                    self.best_model_state[k] = v.clone()
                else:
                    ckpt_idx_torch = torch.tensor(
                        ckpt_idx, device=v.device, dtype=torch.long
                    )
                    self.best_model_state[k][ckpt_idx_torch] = v[ckpt_idx_torch]

        # --- Early-stopping tracker (NDCG@10): patience bookkeeping.
        stop_improved = stop_scores > self.best_scores
        stop_idx = np.nonzero(stop_improved)[0]
        self.best_scores[stop_idx] = stop_scores[stop_idx]
        self.n_bad_updates[stop_idx] = 0
        self.n_bad_updates[~stop_improved] += 1
        return stop_idx

    def remove(self, pack_idx: np.ndarray):
        """Remove members at given indices."""
        keep = np.setdiff1d(np.arange(self.pack_size), pack_idx)
        keep_torch = torch.tensor(keep, dtype=torch.long)

        self.ids = self.ids[keep]
        self.configs = [self.configs[i] for i in keep]
        self.steps = self.steps[keep]
        self.n_bad_updates = self.n_bad_updates[keep]
        self.best_scores = self.best_scores[keep]
        self.best_ckpt_scores = self.best_ckpt_scores[keep]
        self.best_steps = self.best_steps[keep]
        for k in list(self.best_predictions):
            self.best_predictions[k] = self.best_predictions[k][keep]
        for k in list(self.best_model_state):
            device = self.best_model_state[k].device
            self.best_model_state[k] = self.best_model_state[k][
                keep_torch.to(device)
            ]


# =============================================================================
# Online Ensemble
# =============================================================================


def _greedy_ensemble_weighted(
    predictions: np.ndarray,
    score_fn,
    targets: np.ndarray,
    *,
    max_ensemble_size: int | None,
    with_replacement: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Caruana-style greedy ensemble selection, faithful to TabPack.

    Mirrors project.ensemble_utils_torch.greedy_ensemble:
      - init with the single best member (init_top_k=1),
      - at each step add the candidate that most improves the ensemble score,
      - candidates are ALL members when with_replacement (weights accumulate),
        else only unused members (weight-1),
      - stop when no candidate improves or max_ensemble_size reached.

    Returns (ensemble_idx, weights_or_None, score). `ensemble_idx` are indices
    into `predictions`; `weights` is None for unweighted selection, else the
    per-selected-member integer weight (summing to the number of picks).
    """
    # Predictions may be stored float16 (host-RAM budget); upcast for the
    # incremental-mean arithmetic and scoring to avoid precision loss.
    if predictions.dtype != np.float32:
        predictions = predictions.astype(np.float32)
    n = len(predictions)
    prediction_scores = np.array([score_fn(predictions[i], targets) for i in range(n)])

    # init_top_k = 1: start from the single best member.
    best_idx = int(np.argmax(prediction_scores))
    weights = np.zeros(n, dtype=np.float64)
    weights[best_idx] = 1.0
    ensemble_pred = predictions[best_idx].copy()
    ensemble_score = float(score_fn(ensemble_pred, targets))
    ensemble_size = 1
    cap = n if max_ensemble_size is None else max_ensemble_size

    while ensemble_size < cap:
        if with_replacement:
            candidate_idx = np.arange(n)
        else:
            candidate_idx = np.nonzero(weights == 0.0)[0]
            if candidate_idx.size == 0:
                break

        # Incremental weighted mean: new = old*(k/(k+1)) + cand*(1/(k+1)).
        cand_preds = (
            ensemble_pred[None] * (ensemble_size / (ensemble_size + 1))
            + predictions[candidate_idx] * (1.0 / (ensemble_size + 1))
        )
        cand_scores = np.array([
            score_fn(cand_preds[j], targets) for j in range(len(candidate_idx))
        ])

        best_local = int(np.argmax(cand_scores))
        best_cand_score = float(cand_scores[best_local])
        # Strict improvement required (TabPack stops otherwise).
        if best_cand_score <= ensemble_score:
            break

        # Tie-break by best individual member score, like TabPack.
        best_mask = cand_scores == best_cand_score
        if best_mask.sum() > 1:
            tied = candidate_idx[best_mask]
            best_global = int(tied[np.argmax(prediction_scores[tied])])
            best_local = int(np.nonzero(candidate_idx == best_global)[0][0])
        best_global = int(candidate_idx[best_local])

        weights[best_global] += 1.0
        ensemble_pred = cand_preds[best_local]
        ensemble_score = best_cand_score
        ensemble_size += 1

    ens_idx = np.nonzero(weights > 0)[0]
    ens_weights = weights[ens_idx] if with_replacement else None
    return ens_idx, ens_weights, ensemble_score


class OnlineEnsemble:
    """Weighted greedy online ensemble selection, following TabPack.

    Maintains the best ensemble found so far (by val score). Each update runs a
    fresh greedy selection over the current prediction pool and keeps it only if
    it beats the incumbent, decrementing patience otherwise.
    """

    def __init__(
        self,
        *,
        max_size: int = 16,
        patience: int = 32,
        with_replacement: bool = True,
        score_fn,  # (prediction[N,V], targets[N]) -> float
    ):
        self.max_size = max_size
        self.patience = patience
        self.with_replacement = with_replacement
        self._score_fn = score_fn
        self._remaining_patience = patience
        self.ids: list[int] = []
        self.weights: list[float] | None = None
        self.score: float = -float("inf")

    @property
    def is_running(self) -> bool:
        return self._remaining_patience >= 0

    def update(
        self,
        pool_ids: np.ndarray,
        pool_predictions: np.ndarray,
        targets: np.ndarray,
    ) -> bool:
        """Greedy-select over the pool; keep if it beats the incumbent."""
        if len(pool_ids) == 0:
            return False

        ens_idx, ens_weights, ens_score = _greedy_ensemble_weighted(
            pool_predictions, self._score_fn, targets,
            max_ensemble_size=self.max_size,
            with_replacement=self.with_replacement,
        )

        improved = ens_score > self.score
        if improved:
            self.ids = pool_ids[ens_idx].tolist()
            self.weights = None if ens_weights is None else ens_weights.tolist()
            self.score = ens_score
            self._remaining_patience = self.patience
        else:
            self._remaining_patience -= 1

        return improved


# =============================================================================
# Training
# =============================================================================


def _compute_stop_idx(state: PackState, *, patience: int, epoch_size: int,
                      n_epochs: int) -> np.ndarray | None:
    """Find members that should be stopped (early stopping or max epochs)."""
    early_mask = state.n_bad_updates > patience if patience >= 0 else None
    epoch_mask = (
        state.steps // epoch_size >= n_epochs if n_epochs > 0 else None
    )
    if early_mask is None and epoch_mask is None:
        return None
    stop_mask = (
        early_mask if epoch_mask is None
        else epoch_mask if early_mask is None
        else (early_mask | epoch_mask)
    )
    stop_idx = np.nonzero(stop_mask)[0]
    return stop_idx if len(stop_idx) > 0 else None


def _loss_fn_pack(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Per-member cross-entropy loss. Returns (pack_size,) losses."""
    # logits: (P, B, S, V), targets: (B, S), mask: (B, S)
    P, B, S, V = logits.shape
    # Flatten pack and batch for CE
    logits_flat = logits.reshape(P * B * S, V)
    targets_flat = targets.unsqueeze(0).expand(P, -1, -1).reshape(P * B * S)
    losses = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    losses = losses.reshape(P, B * S)
    mask_flat = mask.reshape(1, B * S).expand(P, -1).float()
    # Per-member mean loss
    return (losses * mask_flat).sum(dim=1) / mask_flat.sum(dim=1).clamp(min=1)


def _sampled_loss_fn_pack(
    hidden: Tensor, item_emb_weight: Tensor, targets: Tensor,
    mask: Tensor, neg_items: Tensor, temperature: float = 1.0,
) -> Tensor:
    """Per-member sampled softmax loss. Returns (pack_size,) losses."""
    # hidden: (P, B, S, D), targets: (B, S), mask: (B, S), neg_items: (B, S, num_neg)
    P, B, S, D = hidden.shape
    num_neg = neg_items.size(-1)

    # Positive scores
    pos_emb = item_emb_weight[targets]  # (B, S, D)
    pos_emb = pos_emb.unsqueeze(0).expand(P, -1, -1, -1)  # (P, B, S, D)
    pos_scores = (hidden * pos_emb).sum(-1, keepdim=True)  # (P, B, S, 1)

    # Negative scores
    neg_emb = item_emb_weight[neg_items]  # (B, S, num_neg, D)
    neg_emb = neg_emb.unsqueeze(0).expand(P, -1, -1, -1, -1)  # (P, B, S, num_neg, D)
    neg_scores = torch.einsum("pbsd,pbsnd->pbsn", hidden, neg_emb)  # (P, B, S, num_neg)

    # Concatenate and apply CE (target is index 0)
    all_scores = torch.cat([pos_scores, neg_scores], dim=-1) / temperature  # (P,B,S,1+neg)
    all_scores_flat = all_scores.reshape(P * B * S, 1 + num_neg)
    ce_targets = torch.zeros(P * B * S, dtype=torch.long, device=hidden.device)
    losses = F.cross_entropy(all_scores_flat, ce_targets, reduction="none")
    losses = losses.reshape(P, B * S)
    mask_flat = mask.reshape(1, B * S).expand(P, -1).float()
    return (losses * mask_flat).sum(dim=1) / mask_flat.sum(dim=1).clamp(min=1)


def _hit_rate_at_k(predictions: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """Compute Hit@k score."""
    if predictions.ndim == 1:
        return 0.0
    top_k = np.argpartition(-predictions, k, axis=-1)[:, :k]
    hits = np.any(top_k == targets[:, None], axis=1)
    return float(hits.mean())


def _neg_log_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Negative cross-entropy on target probs (higher = better).

    predictions are probability rows (sum~1). This is the SMOOTH selection
    objective TabPack uses for classification ensembles: averaging member
    probabilities reliably lowers log-loss (variance reduction) even when a
    coarse top-k metric like NDCG@10 does not move, so greedy selection keeps
    adding useful members instead of collapsing to a single best model.
    """
    if predictions.ndim == 1:
        return -1e9
    idx = np.arange(len(predictions))
    p = predictions[idx, targets]
    return float(np.log(np.clip(p, 1e-12, 1.0)).mean())


def _ndcg_at_k(predictions: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k (single relevant item per query)."""
    if predictions.ndim == 1:
        return 0.0
    top_k = np.argpartition(-predictions, k, axis=-1)[:, :k]
    # Sort the top_k by score (descending)
    rows = np.arange(len(predictions))[:, None]
    top_k_scores = predictions[rows, top_k]
    sorted_order = np.argsort(-top_k_scores, axis=-1)
    top_k_sorted = np.take_along_axis(top_k, sorted_order, axis=-1)
    # Find rank of target in sorted top-k (1-indexed), 0 if not found
    match = (top_k_sorted == targets[:, None])
    # DCG = 1/log2(rank+1) if hit, else 0. IDCG = 1 (single relevant item at rank 1)
    ranks = match.argmax(axis=-1) + 1  # 1-indexed rank
    has_hit = match.any(axis=-1)
    dcg = np.where(has_hit, 1.0 / np.log2(ranks + 1), 0.0)
    return float(dcg.mean())


def _ndcg_from_topk(topk_idx: np.ndarray, topk_prob: np.ndarray,
                    targets: np.ndarray, k: int = 10) -> float:
    """NDCG@k from a (N,K) top-K index/prob representation (single relevant)."""
    order = np.argsort(-topk_prob.astype(np.float32), axis=-1)  # rank within stored K
    ranked = np.take_along_axis(topk_idx, order, axis=-1)[:, :k]  # (N,k) top-k item ids
    match = ranked == targets[:, None]
    ranks = match.argmax(axis=-1) + 1
    has_hit = match.any(axis=-1)
    return float(np.where(has_hit, 1.0 / np.log2(ranks + 1), 0.0).mean())


def _neg_log_loss_from_target(target_prob: np.ndarray) -> float:
    """Mean log(target_prob): exact log-loss selection signal (higher=better)."""
    return float(np.log(np.clip(target_prob.astype(np.float32), 1e-12, 1.0)).mean())


@torch.no_grad()
def _predict_last_compact(
    model: "SASRecPackModel",
    sequences: Tensor,
    lengths: Tensor,
    targets_np: np.ndarray,
    *,
    topk: int,
    eval_batch_size: int,
    device: torch.device,
) -> dict:
    """Compact per-member last-position predictions for large catalogs.

    Storing full (P, N, V) softmax matrices in HOST RAM (best + finished + pool
    copies) OOMs for large vocabularies (ml_1m: 32*6034*3125 ~= 2.4GB EACH, and
    several live at once). Instead we store, per member:

      - target_prob (P, N)      : prob mass on each user's held-out target.
          The log-loss of ANY probability-averaged ensemble depends only on the
          averaged target mass, so this is EXACT for ensemble selection.
      - topk_idx  (P, N, K)     : indices of the top-K items (K >= max metric
          cutoff) for NDCG/ranking (early-stop signal + final report).
      - topk_prob (P, N, K)     : their probabilities, for weighted-mean ranking.

    Memory: ~2*P*N*K vs P*N*V (here K=200 << V), i.e. ~15x smaller for ml_1m.
    """
    model.eval()
    tgt = torch.as_tensor(targets_np, device=device)
    tp_chunks, ti_chunks, tk_chunks = [], [], []
    for start in range(0, sequences.size(0), eval_batch_size):
        seq = sequences[start : start + eval_batch_size]
        lens = lengths[start : start + eval_batch_size]
        tgt_b = tgt[start : start + eval_batch_size]  # (B,)
        hidden = model(seq, lens)  # (P, B, S, D)
        b_idx = torch.arange(seq.size(0), device=device)
        last = (lens - 1).clamp(min=0, max=seq.size(1) - 1)
        last_hidden = hidden[:, b_idx, last, :]  # (P, B, D)
        logits = torch.matmul(last_hidden, model.item_embedding.weight.T)  # (P,B,V)
        logits[:, :, :3] = float("-inf")  # mask special tokens
        probs = torch.softmax(logits, dim=-1)  # (P, B, V)
        # target prob: gather column tgt_b for every member
        P_, B_, _ = probs.shape
        tgt_exp = tgt_b.view(1, B_, 1).expand(P_, B_, 1)
        tp = probs.gather(2, tgt_exp).squeeze(2)  # (P, B)
        tk_prob, tk_idx = torch.topk(probs, topk, dim=-1)  # (P, B, K)
        tp_chunks.append(tp.float().cpu())
        ti_chunks.append(tk_idx.to(torch.int32).cpu())
        tk_chunks.append(tk_prob.half().cpu())
    return {
        "target_prob": torch.cat(tp_chunks, dim=1).numpy(),   # (P, N) f32
        "topk_idx": torch.cat(ti_chunks, dim=1).numpy(),      # (P, N, K) i32
        "topk_prob": torch.cat(tk_chunks, dim=1).numpy(),     # (P, N, K) f16
    }


def train(
    config: SASRecPackConfig,
    *,
    train_sequences: Tensor,
    train_lengths: Tensor,
    val_sequences: Tensor,
    val_lengths: Tensor,
    val_targets: Tensor,
    test_sequences: Tensor | None = None,
    test_lengths: Tensor | None = None,
    test_targets: Tensor | None = None,
    neg_sampler=None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Full TabPack-style training loop for SASRecPack.

    Returns a report dict with ensemble results, member configs, metrics.
    """
    device = torch.device(device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # --- Sample member configs (TabPack: shared space, low-discrepancy draw) ---
    P = config.pack_size
    space = {
        "num_layers": ("int", config.min_num_layers, config.max_num_layers),
        "ffn_multiplier": ("uniform", config.min_ffn_multiplier, config.max_ffn_multiplier),
        "dropout": ("uniform", config.min_dropout, config.max_dropout),
        "lr": ("loguniform", config.min_lr, config.max_lr),
        "weight_decay": ("loguniform", config.min_weight_decay, config.max_weight_decay),
    }
    # Per-member activation as a categorical diversity lever (TabPack varies
    # model hyperparameters; activation is a strong architectural axis).
    if config.activations:
        space["activation"] = ("categorical", list(config.activations))
    sampler = HyperparameterSampler(
        space, seed=config.seed, sampler_type=config.sampler_type
    )
    sampled = sampler.sample(P)

    num_layers = [c["num_layers"] for c in sampled]
    ffn_dims = [max(1, int(c["ffn_multiplier"] * config.embedding_dim)) for c in sampled]
    dropout_rates = [c["dropout"] for c in sampled]
    learning_rates = [c["lr"] for c in sampled]
    weight_decays = [c["weight_decay"] for c in sampled]
    activations = (
        [c["activation"] for c in sampled] if config.activations else None
    )

    member_configs = [
        {
            "num_layers": num_layers[i],
            "ffn_dim": ffn_dims[i],
            "dropout": dropout_rates[i],
            "lr": learning_rates[i],
            "weight_decay": weight_decays[i],
            **({"activation": activations[i]} if activations else {}),
        }
        for i in range(P)
    ]

    # --- Create model ---
    model = SASRecPackModel(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        max_seq_length=config.max_seq_length,
        pack_size=P,
        num_layers=num_layers,
        ffn_dims=ffn_dims,
        dropout_rates=dropout_rates,
        activations=activations,
        scale_embeddings=config.scale_embeddings,
        grad_checkpoint=config.grad_checkpoint,
    ).to(device)

    # --- Create optimizer ---
    pack_params = model.get_pack_parameters()
    shared_params = model.get_shared_parameters()

    if config.optimizer_type == "MuonAdamWPack":
        # Muon for the 2D block weight matrices (QKV/out/FFN); AdamW for the
        # 1D params (biases, LayerNorm weight/bias) and the final norm. Muon's
        # Newton-Schulz orthogonalization only applies to matrices, so 1D
        # tensors must go through the AdamW path.
        # Per-member Muon spectral scale max(1, out/in)**0.5 from ACTUAL
        # (unpadded) dims, following TabPack. qkv/out have fixed dims across
        # members, so the default padded-shape scale is already exact. The FFN
        # matrices are zero-padded to max_ffn, so member i's true fan differs:
        #   up:   (ffn_dims[i], D)   -> max(1, ffn_dims[i] / D)**0.5
        #   down: (D, ffn_dims[i])   -> max(1, D / ffn_dims[i])**0.5
        D = config.embedding_dim
        ffn_arr = np.asarray(ffn_dims, dtype=np.float64)
        up_scale = torch.tensor(
            np.sqrt(np.maximum(1.0, ffn_arr / D)), dtype=torch.float32, device=device
        )
        down_scale = torch.tensor(
            np.sqrt(np.maximum(1.0, D / ffn_arr)), dtype=torch.float32, device=device
        )

        muon_params = []
        adamw_pack_params = []
        for block in model.blocks:
            muon_params.append({"params": [block.qkv_weight], "muon": True})
            muon_params.append({"params": [block.out_weight], "muon": True})
            muon_params.append({
                "params": [block.ffn_up_weight], "muon": True,
                "muon_scale": up_scale,
            })
            muon_params.append({
                "params": [block.ffn_down_weight], "muon": True,
                "muon_scale": down_scale,
            })
            adamw_pack_params.extend([
                block.qkv_bias, block.out_bias,
                block.ffn_up_bias, block.ffn_down_bias,
                block.norm1.weight, block.norm1.bias,
                block.norm2.weight, block.norm2.bias,
            ])
        adamw_pack_params.extend(list(model.final_norm.parameters()))

        optimizer = MuonAdamWPack(
            muon_params + [{"params": adamw_pack_params, "muon": False}],
            lr=learning_rates,
            weight_decay=weight_decays,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            muon_lr=config.muon_lr,
            muon_momentum=config.muon_momentum,
            muon_ns_steps=config.muon_ns_steps,
            pack_size=P,
        )
    else:
        optimizer = AdamWPack(
            [{"params": pack_params}],
            lr=learning_rates,
            weight_decay=weight_decays,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            pack_size=P,
        )

    # Shared embedding optimizer (standard AdamW)
    emb_optimizer = torch.optim.AdamW(
        shared_params, lr=config.embedding_lr, weight_decay=config.embedding_weight_decay
    )

    # --- State ---
    state = PackState(P, member_configs)
    finished_ids = []
    finished_predictions: dict[str, list[np.ndarray]] = {}
    n_finished = 0

    # --- Ensemble ---
    # Selection operates on per-member VAL TARGET-PROB vectors (n_pool, N). The
    # greedy averages these rows; the log-loss of the prob-averaged ensemble is
    # exactly mean(log(mean_of_target_probs)), so selection is exact without
    # ever materializing full (N, V) distributions. `targets` arg is unused here
    # (target identity is already baked into the target_prob vector).
    def ensemble_select_fn(target_prob_row, targets):
        return _neg_log_loss_from_target(target_prob_row)

    ensemble = OnlineEnsemble(
        max_size=config.max_ensemble_size,
        patience=config.ensemble_patience,
        with_replacement=config.ensemble_with_replacement,
        score_fn=ensemble_select_fn,
    )

    # --- Training data ---
    train_size = train_sequences.size(0)
    epoch_size = math.ceil(train_size / config.batch_size)

    # --- AMP ---
    autocast = (
        torch.autocast(device.type, torch.bfloat16)
        if config.amp_dtype == "bfloat16" and device.type in ("cuda", "mps")
        else None
    )

    # --- Training loop ---
    start_time = time.time()
    step = 0
    report = {"n_models": 0, "member_configs": member_configs}

    print(f"SASRecPack: {P} members, max_layers={config.max_num_layers}, "
          f"dim={config.embedding_dim}, vocab={config.vocab_size}", flush=True)
    print(f"Depths: {num_layers}", flush=True)
    print(f"FFN dims: {ffn_dims}", flush=True)
    print(flush=True)

    while (
        n_finished < config.pack_size
        and (config.timeout is None or time.time() - start_time < config.timeout)
        and ensemble.is_running
    ):
        # --- Training epoch ---
        model.train()
        perm = torch.randperm(train_size, device=device)
        epoch_loss = 0.0
        n_batches = 0

        n_total_batches = math.ceil(train_size / config.batch_size)
        for batch_i, batch_start in enumerate(range(0, train_size, config.batch_size)):
            if batch_i == 0:
                print(f"  epoch {step // epoch_size + 1}: batch 1/{n_total_batches}...", end="", flush=True)
            batch_idx = perm[batch_start : batch_start + config.batch_size]
            seq = train_sequences[batch_idx]  # (B, S)
            lengths = train_lengths[batch_idx]

            # Targets: next-item prediction (causal shift)
            targets = seq[:, 1:]  # (B, S-1)
            B, S_full = seq.shape
            mask = torch.zeros(B, S_full - 1, device=device, dtype=torch.bool)
            for i, l in enumerate(lengths):
                if l > 1:
                    mask[i, :l - 1] = True

            # Forward — hidden states for all members at once (blocks are
            # gradient-checkpointed, so only layer-boundary inputs are stored).
            hidden = model(seq, lengths)  # (P, B, S, D)

            # Chunked loss: the per-member full-vocab logits (B,S,V) are huge, so
            # accumulating all P graphs before a single backward OOMs for large
            # catalogs. Instead we detach a leaf copy of `hidden`, backward each
            # member's loss into it (freeing that member's logit graph), sum the
            # leaf grads, then do ONE backward through the (checkpointed) pack
            # forward. Mathematically identical to summing per-member CE.
            item_weight = model.item_embedding.weight  # (V, D)
            mask_flat = mask.reshape(-1).float()
            mask_sum = mask_flat.sum().clamp(min=1)

            hidden_leaf = hidden.detach().requires_grad_(True)
            shifted_leaf = hidden_leaf[:, :, :-1, :]  # (P, B, S-1, D)

            optimizer.zero_grad()
            emb_optimizer.zero_grad()

            total_loss_val = 0.0
            for p_idx in range(model.pack_size):
                h = shifted_leaf[p_idx]  # (B, S-1, D)
                logits_p = torch.matmul(h, item_weight.T)  # (B, S-1, V)
                ce = F.cross_entropy(
                    logits_p.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                )
                loss_p = (ce * mask_flat).sum() / mask_sum
                # No retain_graph: item_weight and hidden_leaf are leaves, so
                # grads accumulate into them while THIS member's (B,S,V) logit
                # graph is freed immediately. retain_graph would keep all P
                # logit graphs alive at once -> the ml_1m OOM.
                loss_p.backward()
                total_loss_val += loss_p.item()

            # Propagate the accumulated hidden-grad through the pack forward once.
            hidden.backward(hidden_leaf.grad)
            optimizer.step()
            emb_optimizer.step()
            total_loss = total_loss_val

            step += 1
            state.step()
            epoch_loss += total_loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1) / model.pack_size
        print(f" done. loss={avg_loss:.4f}, evaluating...", end="", flush=True)

        # --- Evaluation (skip if not eval epoch) ---
        current_epoch = step // epoch_size
        if current_epoch % config.eval_every_n_epochs != 0:
            best_str = f" [best] {state.best_scores.max():.4f}" if state.pack_size > 0 and state.best_scores.max() > -np.inf else ""
            print(
                f"  [E] {current_epoch:<3}"
                f" [T] {time.time() - start_time:.0f}s"
                f" [L] {avg_loss:.4f}"
                f" [P] {model.pack_size}"
                f"{best_str}",
                flush=True,
            )
            continue

        # Per-member last-position COMPACT preds (target_prob + top-K) on val
        # (and test). Full (P,N,V) matrices OOM the host for large catalogs;
        # compact keeps ensemble selection exact (log-loss uses target_prob) and
        # ranking accurate (top-K covers all metric cutoffs).
        val_targets_np = val_targets.cpu().numpy()
        vc = _predict_last_compact(
            model, val_sequences, val_lengths, val_targets_np,
            topk=config.pred_topk, eval_batch_size=config.eval_batch_size, device=device,
        )
        eval_preds = {
            "val_target_prob": vc["target_prob"],
            "val_topk_idx": vc["topk_idx"],
            "val_topk_prob": vc["topk_prob"],
        }
        if test_sequences is not None:
            test_targets_np = test_targets.cpu().numpy()
            tc = _predict_last_compact(
                model, test_sequences, test_lengths, test_targets_np,
                topk=config.pred_topk, eval_batch_size=config.eval_batch_size, device=device,
            )
            eval_preds["test_target_prob"] = tc["target_prob"]
            eval_preds["test_topk_idx"] = tc["topk_idx"]
            eval_preds["test_topk_prob"] = tc["topk_prob"]

        # Two per-member signals on val:
        #  - NDCG@10 gates early stopping (the metric we ultimately report),
        #  - log-loss gates which checkpoint is pooled (matches ensemble select).
        member_ndcg = np.array([
            _ndcg_from_topk(vc["topk_idx"][i], vc["topk_prob"][i], val_targets_np, k=10)
            for i in range(model.pack_size)
        ])
        member_logloss = np.array([
            _neg_log_loss_from_target(vc["target_prob"][i])
            for i in range(model.pack_size)
        ])

        # Update state (val + test predictions tracked in lockstep)
        pack_state_dict = {}
        for name, param in model.named_parameters():
            if isinstance(param, ParameterPack):
                pack_state_dict[name] = param.data

        state.update(
            stop_scores=member_ndcg,
            ckpt_scores=member_logloss,
            predictions=eval_preds,
            model_state=pack_state_dict,
        )

        # --- Early stopping: find members to stop ---
        stop_idx = _compute_stop_idx(
            state, patience=config.patience, epoch_size=epoch_size,
            n_epochs=config.n_epochs,
        )

        have_test = test_sequences is not None
        # Compact prediction keys carried per member through state/finished/pool.
        _pred_keys = ["val_target_prob", "val_topk_idx", "val_topk_prob"]
        if have_test:
            _pred_keys += ["test_target_prob", "test_topk_idx", "test_topk_prob"]

        if stop_idx is not None and len(stop_idx) > 0:
            # Record finished members' best (log-loss) compact predictions.
            for idx in stop_idx:
                finished_ids.append(int(state.ids[idx]))
                for k in _pred_keys:
                    finished_predictions.setdefault(k, []).append(
                        state.best_predictions[k][idx]
                    )

            stop_idx_torch = torch.tensor(stop_idx, device=device, dtype=torch.long)
            old_to_new = module_pack_remove(model, stop_idx_torch)
            optimizer_pack_remove(optimizer, stop_idx_torch, old_to_new)
            state.remove(stop_idx)
            n_finished += len(stop_idx)

        # --- Online ensemble update ---
        # Pool = finished + still-running members' best checkpoints. The ensemble
        # selects on val target-prob (exact log-loss under prob averaging).
        pool_ids_list = []
        pool_tp_list = []       # val target_prob per member (N_val,)
        if finished_ids:
            pool_ids_list.extend(finished_ids)
            pool_tp_list.extend(finished_predictions.get("val_target_prob", []))
        if state.pack_size > 0 and "val_target_prob" in state.best_predictions:
            for i in range(state.pack_size):
                pool_ids_list.append(int(state.ids[i]))
                pool_tp_list.append(state.best_predictions["val_target_prob"][i])

        if pool_tp_list:
            pool_ids = np.array(pool_ids_list)
            pool_target_prob = np.stack(pool_tp_list)  # (n_pool, N_val)
            ensemble_improved = ensemble.update(pool_ids, pool_target_prob, val_targets_np)
        else:
            ensemble_improved = False

        # --- Print progress ---
        elapsed = time.time() - start_time
        best_str = f" [best] {state.best_scores.max():.4f}" if state.pack_size > 0 else ""
        print(
            f"{'$' if ensemble_improved else ' '}"
            f" [E] {step // epoch_size:<3}"
            f" [T] {elapsed:.0f}s"
            f" [L] {avg_loss:.4f}"
            f" [M] {n_finished}/{config.pack_size}"
            f" [P] {model.pack_size}"
            f"{best_str}"
            f" [ens] {ensemble.score:.4f} (size={len(ensemble.ids)})",
            flush=True,
        )

    # --- Final report ---
    elapsed = time.time() - start_time
    report.update({
        "n_models": n_finished,
        "time": elapsed,
        "steps": step,
        "ensemble": {
            "ids": ensemble.ids,
            "weights": ensemble.weights,
            "val_logloss_score": ensemble.score,  # selection objective (neg log-loss)
            "size": len(ensemble.ids),
        },
    })

    # Reconstruct a dense (N, V) test score matrix for the selected ensemble by
    # scattering each member's top-K probs (weighted) into a zero matrix. Done
    # once, only for the ~<=max_ensemble_size selected members, so memory is a
    # single (N, V). Items outside every member's top-K get 0 mass — fine for
    # ranking metrics at cutoffs <= K.
    def _dense_from_topk(idx_key, prob_key, source_state, source_finished, n_rows):
        sel = [pool_ids_list.index(eid) for eid in ensemble.ids
               if eid in pool_ids_list]
        if not sel:
            return None
        w = (np.array(ensemble.weights, dtype=np.float32)
             if ensemble.weights is not None
             else np.ones(len(ensemble.ids), dtype=np.float32))
        w = w / w.sum()
        dense = np.zeros((n_rows, config.vocab_size), dtype=np.float32)
        for j, wj in zip(sel, w):
            idx = pool_idx_arrays[idx_key][j]   # (N, K) int32
            prob = pool_idx_arrays[prob_key][j].astype(np.float32)  # (N, K)
            np.add.at(
                dense,
                (np.arange(n_rows)[:, None], idx.astype(np.int64)),
                prob * wj,
            )
        return torch.from_numpy(dense)

    # Assemble the final pool's compact arrays keyed for reconstruction.
    pool_idx_arrays = {}
    for key in ["val_topk_idx", "val_topk_prob", "test_topk_idx", "test_topk_prob"]:
        if key.startswith("test") and test_sequences is None:
            continue
        arrs = []
        if finished_ids:
            arrs.extend(finished_predictions.get(key, []))
        if state.pack_size > 0 and key in state.best_predictions:
            for i in range(state.pack_size):
                arrs.append(state.best_predictions[key][i])
        pool_idx_arrays[key] = arrs

    test_score_matrix = None
    if test_sequences is not None and ensemble.ids:
        test_score_matrix = _dense_from_topk(
            "test_topk_idx", "test_topk_prob", state, finished_predictions,
            test_sequences.size(0),
        )

    # Ensemble val NDCG@10 for a human-readable, baseline-comparable number.
    ens_val_ndcg = float("nan")
    if ensemble.ids and pool_idx_arrays.get("val_topk_idx"):
        dense_val = _dense_from_topk(
            "val_topk_idx", "val_topk_prob", state, finished_predictions,
            val_sequences.size(0),
        )
        if dense_val is not None:
            ens_val_ndcg = _ndcg_at_k(dense_val.numpy(), val_targets_np, k=10)
    report["ensemble"]["val_ndcg@10"] = ens_val_ndcg

    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Ensemble: {len(ensemble.ids)} members "
          f"(weights={ensemble.weights}), val NDCG@10={ens_val_ndcg:.4f}")
    print(f"Member IDs: {ensemble.ids}", flush=True)

    report["_test_score_matrix"] = test_score_matrix  # (N, V) tensor or None
    return report
