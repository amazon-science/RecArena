"""Tests for the modular sparse-embedding integration.

Sparse embedding tables are a single-GPU memory/throughput win for large item
catalogs. They are only safe for *sampled* losses on a *tied standard* table,
because the full-vocab `.weight @ hᵀ` matmul (full softmax) and advanced
indexing of `.weight` both produce dense gradients that `torch.optim.SparseAdam`
cannot consume.

This module verifies:
  1. sparse_embeddings_eligible gates correctly (loss type, untied/LoRA output,
     model SUPPORTS_SPARSE, embedding type).
  2. build_optimizer routes params: dense-only -> AdamW; sparse+dense ->
     HybridOptim(AdamW, SparseAdam); pure-embedding -> SparseAdam.
  3. split_sparse_dense_params partitions by owning module.
  4. Sequential sampled losses keep the item-table gradient SPARSE via the
     embedding_lookup path, and dense full-softmax stays dense.
  5. Eligible models build sparse tables and fall back to dense (with a
     warning) when ineligible; HybridOptim steps both param sets under
     Lightning automatic optimization.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rec_arena.utils.sparse_optim import (  # noqa: E402
    SAMPLED_LOSSES,
    HybridOptim,
    build_optimizer,
    clip_dense_grads_only,
    sparse_embeddings_eligible,
    split_sparse_dense_params,
)
from rec_arena.models import BPRMF, NCF, SASRec, GRU4Rec  # noqa: E402
from rec_arena.configs.defaults.sasrec import SASRecConfig  # noqa: E402
from rec_arena.configs.defaults.gru4rec import GRU4RecConfig  # noqa: E402
from rec_arena.configs.defaults.bprmf import BPRMFConfig  # noqa: E402
from rec_arena.configs.defaults.ncf import NCFConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _seq_batch(vocab=200, seq=12, bs=8, num_neg=4):
    s = torch.randint(3, vocab, (bs, seq))
    return {
        "sequence": s,
        "sequence_length": torch.full((bs,), seq, dtype=torch.long),
        "target": s[:, -1],
        "neg_items": torch.randint(3, vocab, (bs, seq, num_neg)),
    }


def _implicit_batch(nu=40, ni=60, bs=16, num_neg=4):
    return {
        "user_id": torch.randint(0, nu, (bs,)),
        "item_id": torch.randint(0, ni, (bs,)),
        "neg_items": torch.randint(0, ni, (bs, num_neg)),
    }


# --------------------------------------------------------------------------- #
# 1. Eligibility gating
# --------------------------------------------------------------------------- #
class TestEligibility:
    def test_disabled_by_default(self):
        cfg = SASRecConfig(vocab_size=100, loss_type="sampled_softmax")
        ok, reason = sparse_embeddings_eligible(cfg)
        assert ok is False
        assert "disabled" in reason

    def test_sampled_loss_eligible(self):
        cfg = SASRecConfig(
            vocab_size=100, loss_type="sampled_softmax", sparse_embeddings=True
        )
        ok, _ = sparse_embeddings_eligible(cfg, SASRec)
        assert ok is True

    @pytest.mark.parametrize("loss", sorted(SAMPLED_LOSSES))
    def test_all_sampled_losses_eligible(self, loss):
        cfg = SASRecConfig(vocab_size=100, loss_type=loss, sparse_embeddings=True)
        ok, _ = sparse_embeddings_eligible(cfg, SASRec)
        assert ok is True

    def test_full_softmax_ineligible(self):
        cfg = SASRecConfig(
            vocab_size=100, loss_type="cross_entropy", sparse_embeddings=True
        )
        ok, reason = sparse_embeddings_eligible(cfg, SASRec)
        assert ok is False
        assert "sampled" in reason

    def test_untied_output_ineligible(self):
        cfg = SASRecConfig(
            vocab_size=100,
            loss_type="sampled_softmax",
            sparse_embeddings=True,
            tie_embeddings=False,
        )
        ok, reason = sparse_embeddings_eligible(cfg, SASRec)
        assert ok is False
        assert "untied" in reason

    def test_lora_output_ineligible(self):
        cfg = SASRecConfig(
            vocab_size=100,
            loss_type="sampled_softmax",
            sparse_embeddings=True,
            output_lora_rank=8,
        )
        ok, reason = sparse_embeddings_eligible(cfg, SASRec)
        assert ok is False
        assert "LoRA" in reason

    def test_model_without_sparse_support_ineligible(self):
        # A model class that opts out via SUPPORTS_SPARSE=False.
        class _NoSparse:
            SUPPORTS_SPARSE = False

        cfg = SASRecConfig(
            vocab_size=100, loss_type="sampled_softmax", sparse_embeddings=True
        )
        ok, reason = sparse_embeddings_eligible(cfg, _NoSparse)
        assert ok is False
        assert "sparse-safe loss path" in reason


# --------------------------------------------------------------------------- #
# 2 & 3. Optimizer routing + param split
# --------------------------------------------------------------------------- #
class TestOptimizerRouting:
    def test_dense_only_returns_adamw(self):
        m = nn.Sequential(nn.Linear(8, 8), nn.Embedding(10, 8))  # dense embedding
        opt = build_optimizer(m, lr=1e-3, weight_decay=1e-2)
        assert isinstance(opt, torch.optim.AdamW)

    def test_mixed_returns_hybrid(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 8, sparse=True)
                self.lin = nn.Linear(8, 8)

        opt = build_optimizer(M(), lr=1e-3, weight_decay=0.0)
        assert isinstance(opt, HybridOptim)
        kinds = {type(o).__name__ for o in opt.optimizers}
        assert kinds == {"AdamW", "SparseAdam"}

    def test_pure_sparse_returns_sparseadam(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Embedding(10, 8, sparse=True)
                self.b = nn.Embedding(12, 8, sparse=True)

        opt = build_optimizer(M(), lr=1e-3, weight_decay=0.0)
        assert isinstance(opt, torch.optim.SparseAdam)

    def test_split_partitions_by_module(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.sparse_emb = nn.Embedding(10, 8, sparse=True)
                self.dense_emb = nn.Embedding(10, 8, sparse=False)
                self.lin = nn.Linear(8, 4)

        sparse, dense = split_sparse_dense_params(M())
        assert len(sparse) == 1  # only sparse_emb.weight
        # dense_emb.weight + lin.weight + lin.bias
        assert len(dense) == 3


# --------------------------------------------------------------------------- #
# 4. Gradient stays sparse on the lookup path
# --------------------------------------------------------------------------- #
class TestGradientSparsity:
    @pytest.mark.parametrize("loss", ["sampled_softmax", "bce", "gbce", "bpr"])
    def test_sequential_sampled_loss_keeps_sparse_grad(self, loss):
        torch.manual_seed(0)
        cfg = SASRecConfig(
            vocab_size=200,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            max_seq_length=12,
            loss_type=loss,
            sparse_embeddings=True,
        )
        m = SASRec(cfg)
        assert m._sparse_embeddings is True
        out = m.compute_loss(_seq_batch())
        out.backward()
        assert m.item_embedding.weight.grad.is_sparse
        # position table is dense and must receive a normal dense grad
        assert not m.pos_embedding.weight.grad.is_sparse

    def test_dense_full_softmax_grad_is_dense(self):
        torch.manual_seed(0)
        cfg = SASRecConfig(
            vocab_size=200,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            max_seq_length=12,
            loss_type="cross_entropy",
            sparse_embeddings=True,  # requested, but CE -> dense fallback
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SASRec(cfg)
        assert m._sparse_embeddings is False
        b = _seq_batch()
        out = m.compute_loss(b)
        out.backward()
        assert not m.item_embedding.weight.grad.is_sparse

    def test_implicit_bpr_keeps_sparse_grad(self):
        torch.manual_seed(0)
        cfg = BPRMFConfig(
            num_users=40,
            num_items=60,
            embedding_dim=16,
            loss_type="bpr",
            sparse_embeddings=True,
        )
        m = BPRMF(cfg)
        out = m.compute_loss(_implicit_batch())
        out.backward()
        assert m.item_embedding.weight.grad.is_sparse
        assert m.user_embedding.weight.grad.is_sparse


# --------------------------------------------------------------------------- #
# 5. Model construction + end-to-end Lightning step
# --------------------------------------------------------------------------- #
class TestModelConstruction:
    def test_eligible_model_builds_sparse_table(self):
        cfg = GRU4RecConfig(
            vocab_size=200,
            embedding_dim=32,
            max_seq_length=12,
            loss_type="sampled_softmax",
            sparse_embeddings=True,
        )
        m = GRU4Rec(cfg)
        assert m.item_embedding.sparse is True
        assert m._sparse_embeddings is True

    def test_ineligible_model_falls_back_with_warning(self):
        cfg = SASRecConfig(
            vocab_size=200,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            max_seq_length=12,
            loss_type="cross_entropy",
            sparse_embeddings=True,
        )
        with pytest.warns(
            UserWarning, match="sparse_embeddings requested but disabled"
        ):
            m = SASRec(cfg)
        assert m.item_embedding.sparse is False

    def test_dense_default_unchanged(self):
        cfg = SASRecConfig(
            vocab_size=200,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            max_seq_length=12,
            loss_type="sampled_softmax",
        )
        m = SASRec(cfg)
        assert m.item_embedding.sparse is False
        assert m._sparse_embeddings is False
        assert isinstance(m.configure_optimizers(), torch.optim.AdamW)

    def test_clip_dense_only_skips_sparse(self):
        # clip should not raise on a model holding sparse grads
        torch.manual_seed(0)
        cfg = SASRecConfig(
            vocab_size=200,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            max_seq_length=12,
            loss_type="sampled_softmax",
            sparse_embeddings=True,
        )
        m = SASRec(cfg)
        m.compute_loss(_seq_batch()).backward()
        # must not raise (sparse table is skipped)
        clip_dense_grads_only(m, 1.0, "norm")

    def test_lightning_step_updates_both_param_sets(self):
        import lightning as pl
        from torch.utils.data import DataLoader, Dataset

        class DS(Dataset):
            def __init__(self, n=64):
                g = torch.Generator().manual_seed(0)
                self.s = torch.randint(3, 200, (n, 12), generator=g)

            def __len__(self):
                return len(self.s)

            def __getitem__(self, i):
                seq = self.s[i]
                return {
                    "sequence": seq,
                    "sequence_length": torch.tensor(12),
                    "target": seq[-1],
                    "neg_items": torch.randint(3, 200, (12, 4)),
                    "user_id": torch.tensor(i),
                }

        torch.manual_seed(0)
        cfg = SASRecConfig(
            vocab_size=200,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            max_seq_length=12,
            loss_type="sampled_softmax",
            sparse_embeddings=True,
            gradient_clip_val=1.0,
        )
        m = SASRec(cfg)
        emb_before = m.item_embedding.weight.detach().clone()
        lin_before = (
            next(p for n, p in m.named_parameters() if "transformer_blocks" in n)
            .detach()
            .clone()
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            gradient_clip_val=1.0,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )
        trainer.fit(m, DataLoader(DS(), batch_size=16))

        assert not torch.allclose(emb_before, m.item_embedding.weight.detach())
        lin_after = next(
            p for n, p in m.named_parameters() if "transformer_blocks" in n
        ).detach()
        assert not torch.allclose(lin_before, lin_after)
