"""Tests for rec_arena.modules.

Covers: TransformerBlock, RMSNorm, CausalSelfAttention, MultiHeadAttention,
SwiGLU, FeedForward, normalization layers, embeddings, embedding factory,
LinearBatchEnsembleLayer.
"""

import pytest
import torch
import torch.nn as nn

from rec_arena.modules.transformer_layers.transformer_block import (
    TransformerBlock,
    RMSNorm,
)
from rec_arena.modules.transformer_layers.mha import (
    CausalSelfAttention,
    MultiHeadAttention,
)
from rec_arena.modules.layer_utils.swiglu import FeedForward, SwiGLU
from rec_arena.modules.layer_utils.normalization_layers import (
    LayerNorm,
    BatchNorm,
    LearnableLayerScaling,
)
from rec_arena.modules.layer_utils.embeddings import (
    HierarchicalLoRAEmbedding,
    RotaryPositionalEmbedding,
)
from rec_arena.modules.layer_utils.embedding_factory import create_embedding
from rec_arena.modules.layer_utils.batch_ensembling import (
    LinearBatchEnsembleLayer,
)

BATCH = 2
SEQ_LEN = 8
DIM = 32
NUM_HEADS = 4
HIDDEN_DIM = 64


# ===================================================================
# CausalSelfAttention
# ===================================================================


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(DIM, NUM_HEADS, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = attn(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_no_nan_in_output(self):
        attn = CausalSelfAttention(DIM, NUM_HEADS, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = attn(x)
        assert not torch.isnan(out).any()

    def test_with_padding_mask(self):
        attn = CausalSelfAttention(DIM, NUM_HEADS, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
        mask[:, -2:] = False
        out = attn(x, attn_mask=mask, is_causal=False)
        assert out.shape == (BATCH, SEQ_LEN, DIM)


# ===================================================================
# MultiHeadAttention
# ===================================================================


class TestMultiHeadAttention:
    def test_self_attention_shape(self):
        mha = MultiHeadAttention(DIM, NUM_HEADS, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = mha(x, x, x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_cross_attention_shape(self):
        mha = MultiHeadAttention(DIM, NUM_HEADS, dropout_rate=0.0)
        q = torch.randn(BATCH, 4, DIM)
        kv = torch.randn(BATCH, SEQ_LEN, DIM)
        out = mha(q, kv, kv)
        assert out.shape == (BATCH, 4, DIM)


# ===================================================================
# TransformerBlock
# ===================================================================


class TestTransformerBlock:
    def test_output_shape_default(self):
        block = TransformerBlock(DIM, NUM_HEADS, HIDDEN_DIM, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_output_shape_swiglu(self):
        block = TransformerBlock(
            DIM, NUM_HEADS, HIDDEN_DIM, dropout_rate=0.0, use_swiglu=True
        )
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_output_shape_rms_norm(self):
        block = TransformerBlock(
            DIM, NUM_HEADS, HIDDEN_DIM, dropout_rate=0.0, use_rms_norm=True
        )
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_output_shape_gated_residual(self):
        block = TransformerBlock(
            DIM, NUM_HEADS, HIDDEN_DIM, dropout_rate=0.0, use_gated_residual=True
        )
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_post_norm(self):
        block = TransformerBlock(
            DIM, NUM_HEADS, HIDDEN_DIM, dropout_rate=0.0, norm_first=False
        )
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_gradient_flows(self):
        block = TransformerBlock(DIM, NUM_HEADS, HIDDEN_DIM, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_output(self):
        block = TransformerBlock(DIM, NUM_HEADS, HIDDEN_DIM, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = block(x)
        assert not torch.isnan(out).any()


# ===================================================================
# RMSNorm
# ===================================================================


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(DIM)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_rms_near_one(self):
        norm = RMSNorm(DIM)
        x = torch.randn(BATCH, DIM) * 10
        out = norm(x)
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.2)


# ===================================================================
# SwiGLU and FeedForward
# ===================================================================


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(DIM, HIDDEN_DIM, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = ffn(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_gradient_flows(self):
        ffn = SwiGLU(DIM, HIDDEN_DIM, dropout_rate=0.0)
        x = torch.randn(BATCH, SEQ_LEN, DIM, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(DIM, HIDDEN_DIM, dropout_rate=0.0, activation=nn.GELU())
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = ffn(x)
        assert out.shape == (BATCH, SEQ_LEN, DIM)


# ===================================================================
# Normalization layers
# ===================================================================


class TestLayerNorm:
    def test_output_shape(self):
        norm = LayerNorm(DIM)
        x = torch.randn(BATCH, DIM)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_mean_near_zero(self):
        norm = LayerNorm(DIM)
        x = torch.randn(BATCH, DIM) * 5 + 3
        out = norm(x)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(BATCH), atol=1e-4)


class TestBatchNorm:
    def test_output_shape(self):
        norm = BatchNorm(DIM)
        norm.train()
        x = torch.randn(BATCH, DIM)
        out = norm(x)
        assert out.shape == x.shape

    def test_eval_mode(self):
        norm = BatchNorm(DIM)
        norm.train()
        x = torch.randn(8, DIM)
        norm(x)  # Update running stats
        norm.eval()
        out = norm(x)
        assert out.shape == x.shape


class TestLearnableLayerScaling:
    def test_output_shape(self):
        lls = LearnableLayerScaling(DIM)
        x = torch.randn(BATCH, DIM)
        out = lls(x)
        assert out.shape == x.shape

    def test_initial_identity(self):
        lls = LearnableLayerScaling(DIM)
        x = torch.randn(BATCH, DIM)
        out = lls(x)
        assert torch.allclose(out, x)


# ===================================================================
# Embeddings
# ===================================================================


class TestHierarchicalLoRAEmbedding:
    def test_output_shape(self):
        mapping = torch.tensor([0, 0, 1, 1, 2])
        emb = HierarchicalLoRAEmbedding(
            num_items=5, num_parents=3, embedding_dim=16,
            item_to_parent_mapping=mapping, lora_rank=4,
        )
        ids = torch.tensor([0, 2, 4])
        out = emb(ids)
        assert out.shape == (3, 16)

    def test_batch_shape(self):
        mapping = torch.tensor([0, 0, 1, 1, 2])
        emb = HierarchicalLoRAEmbedding(
            num_items=5, num_parents=3, embedding_dim=16,
            item_to_parent_mapping=mapping, lora_rank=4,
        )
        ids = torch.tensor([[0, 1], [2, 3]])
        out = emb(ids)
        assert out.shape == (2, 2, 16)

    def test_weight_property(self):
        mapping = torch.tensor([0, 0, 1])
        emb = HierarchicalLoRAEmbedding(
            num_items=3, num_parents=2, embedding_dim=8,
            item_to_parent_mapping=mapping, lora_rank=2,
        )
        w = emb.weight
        assert w.shape == (3, 8)


class TestRotaryPositionalEmbedding:
    def test_output_shape(self):
        rope = RotaryPositionalEmbedding(dim=16, max_seq_len=64)
        q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, 16)
        k = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, 16)
        q_out, k_out = rope(q, k, SEQ_LEN)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


# ===================================================================
# Embedding factory
# ===================================================================


class TestEmbeddingFactory:
    def test_standard_embedding(self):
        emb = create_embedding("standard", num_embeddings=100, embedding_dim=32)
        assert isinstance(emb, nn.Embedding)
        assert emb.num_embeddings == 100

    def test_standard_with_padding(self):
        emb = create_embedding("standard", num_embeddings=100, embedding_dim=32, padding_idx=0)
        assert emb.padding_idx == 0

    def test_hierarchical_lora(self):
        mapping = torch.arange(10) % 3
        emb = create_embedding(
            "hierarchical_lora", num_embeddings=10, embedding_dim=16,
            num_parents=3, item_to_parent_mapping=mapping,
        )
        assert isinstance(emb, HierarchicalLoRAEmbedding)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding type"):
            create_embedding("unknown", num_embeddings=10, embedding_dim=16)


# ===================================================================
# LinearBatchEnsembleLayer
# ===================================================================


class TestLinearBatchEnsembleLayer:
    def test_2d_input(self):
        layer = LinearBatchEnsembleLayer(
            in_features=DIM, out_features=16, ensemble_size=3
        )
        x = torch.randn(BATCH, DIM)
        out = layer(x)
        assert out.shape == (BATCH, 3, 16)

    def test_3d_input(self):
        layer = LinearBatchEnsembleLayer(
            in_features=DIM, out_features=16, ensemble_size=3
        )
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        out = layer(x)
        assert out.shape == (BATCH, SEQ_LEN, 3, 16)

    def test_no_scaling(self):
        layer = LinearBatchEnsembleLayer(
            in_features=DIM, out_features=16, ensemble_size=3,
            ensemble_scaling_in=False, ensemble_scaling_out=False,
        )
        x = torch.randn(BATCH, DIM)
        out = layer(x)
        assert out.shape == (BATCH, 3, 16)

    def test_random_signs_init(self):
        layer = LinearBatchEnsembleLayer(
            in_features=DIM, out_features=16, ensemble_size=3,
            scaling_init="random-signs",
        )
        x = torch.randn(BATCH, DIM)
        out = layer(x)
        assert out.shape == (BATCH, 3, 16)

    def test_normal_init(self):
        layer = LinearBatchEnsembleLayer(
            in_features=DIM, out_features=16, ensemble_size=3,
            scaling_init="normal",
        )
        x = torch.randn(BATCH, DIM)
        out = layer(x)
        assert out.shape == (BATCH, 3, 16)
