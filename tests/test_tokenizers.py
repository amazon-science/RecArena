"""Tests for rec_arena.tokenizers module.

Covers: RecTokenizer (fit, encode, decode, pad, batch),
SemanticIDTokenizer (forward, encode_vqvae, encode_hierarchical, decode_vqvae).
"""

import numpy as np
import pytest
import torch

from rec_arena.tokenizers.unified import RecTokenizer
from rec_arena.tokenizers.semantic_id import SemanticIDTokenizer


# ===================================================================
# RecTokenizer
# ===================================================================


class TestRecTokenizer:
    @pytest.fixture
    def tokenizer(self):
        t = RecTokenizer()
        t.fit([[1, 2, 3], [2, 3, 4, 5]])
        return t

    def test_fit_sets_vocab_size(self, tokenizer):
        # Items 1-5 + 3 special tokens
        assert tokenizer.vocab_size == 8

    def test_special_tokens(self, tokenizer):
        assert tokenizer.pad_token == 0
        assert tokenizer.unk_token == 1
        assert tokenizer.mask_token == 2

    def test_encode_known_items(self, tokenizer):
        encoded = tokenizer.encode([1, 2, 3])
        assert len(encoded) == 3
        assert all(t >= 3 for t in encoded)  # All above special tokens

    def test_encode_unknown_item_returns_unk(self, tokenizer):
        encoded = tokenizer.encode([999])
        assert encoded == [tokenizer.unk_token]

    def test_decode_roundtrip(self, tokenizer):
        original = [1, 2, 3]
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_decode_skips_special_tokens(self, tokenizer):
        decoded = tokenizer.decode([0, 1, 2])
        assert decoded == []

    def test_encode_batch(self, tokenizer):
        batch = [[1, 2], [3, 4, 5]]
        encoded = tokenizer.encode_batch(batch)
        assert len(encoded) == 2
        assert len(encoded[0]) == 2
        assert len(encoded[1]) == 3

    def test_pad_sequences_shorter(self, tokenizer):
        seqs = [[3, 4], [5]]
        padded = tokenizer.pad_sequences(seqs, max_length=4)
        assert len(padded[0]) == 4
        assert len(padded[1]) == 4
        assert padded[1][-1] == tokenizer.pad_token

    def test_pad_sequences_truncates(self, tokenizer):
        seqs = [[3, 4, 5, 6, 7]]
        padded = tokenizer.pad_sequences(seqs, max_length=3)
        assert len(padded[0]) == 3

    def test_get_vocab_size(self, tokenizer):
        assert tokenizer.get_vocab_size() == tokenizer.vocab_size

    def test_fit_empty_sequences(self):
        t = RecTokenizer()
        t.fit([[]])
        assert t.vocab_size == 3  # Only special tokens


# ===================================================================
# SemanticIDTokenizer
# ===================================================================


class TestSemanticIDTokenizer:
    def test_forward_identity_mapping(self):
        tok = SemanticIDTokenizer(num_items=10, method="kmeans")
        ids = torch.tensor([0, 1, 5, 9])
        result = tok(ids)
        assert torch.equal(result, ids)

    def test_forward_clamps_out_of_range(self):
        tok = SemanticIDTokenizer(num_items=10, method="kmeans")
        ids = torch.tensor([15])
        result = tok(ids)
        assert result.item() == 9  # Clamped to num_items - 1

    def test_forward_batch_shape(self):
        tok = SemanticIDTokenizer(num_items=20, method="kmeans")
        ids = torch.randint(0, 20, (4, 8))
        result = tok(ids)
        assert result.shape == (4, 8)

    def test_fit_kmeans(self):
        tok = SemanticIDTokenizer(num_items=50, num_codes=5, method="kmeans")
        features = np.random.randn(50, 16).astype(np.float32)
        tok.fit_kmeans(features)
        # After fitting, semantic IDs should be in [0, num_codes)
        all_ids = tok(torch.arange(50))
        assert all_ids.min() >= 0
        assert all_ids.max() < 5

    def test_fit_kmeans_wrong_method_raises(self):
        tok = SemanticIDTokenizer(num_items=10, method="vqvae")
        with pytest.raises(ValueError, match="kmeans"):
            tok.fit_kmeans(np.random.randn(10, 16))

    def test_encode_vqvae_2d(self):
        tok = SemanticIDTokenizer(num_items=10, num_codes=8, method="vqvae", feature_dim=16)
        embeddings = torch.randn(4, 16)
        ids = tok.encode_vqvae(embeddings)
        assert ids.shape == (4,)
        assert ids.min() >= 0
        assert ids.max() < 8

    def test_encode_vqvae_3d(self):
        tok = SemanticIDTokenizer(num_items=10, num_codes=8, method="vqvae", feature_dim=16)
        embeddings = torch.randn(2, 5, 16)
        ids = tok.encode_vqvae(embeddings)
        assert ids.shape == (2, 5)

    def test_decode_vqvae(self):
        tok = SemanticIDTokenizer(num_items=10, num_codes=8, method="vqvae", feature_dim=16)
        ids = torch.tensor([0, 3, 7])
        decoded = tok.decode_vqvae(ids)
        assert decoded.shape == (3, 16)

    def test_encode_hierarchical(self):
        tok = SemanticIDTokenizer(
            num_items=10, num_codes=16, method="hierarchical", feature_dim=16, num_levels=2
        )
        embeddings = torch.randn(4, 16)
        ids_list = tok.encode_hierarchical(embeddings)
        assert len(ids_list) == 2
        assert ids_list[0].shape == (4,)

    def test_fit_from_embeddings(self):
        tok = SemanticIDTokenizer(num_items=20, num_codes=4, method="kmeans", feature_dim=8)
        embeddings = torch.randn(20, 8)
        tok.fit_from_embeddings(embeddings)
        result = tok(torch.arange(20))
        assert result.min() >= 0
        assert result.max() < 4

    def test_get_semantic_embedding(self):
        tok = SemanticIDTokenizer(num_items=10, method="kmeans")
        emb_layer = torch.nn.Embedding(10, 32)
        ids = torch.tensor([0, 1, 2])
        result = tok.get_semantic_embedding(ids, emb_layer)
        assert result.shape == (3, 32)
