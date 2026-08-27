"""Tests for sequential model implementations."""

from unittest.mock import MagicMock

import pytest
import torch

from rec_arena.losses import get_loss_function

BATCH = 4
SEQ_LEN = 8
VOCAB = 50
EMB_DIM = 16


# ===================================================================
# SASRec
# ===================================================================


class TestSASRec:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.sasrec import SASRecConfig
        from rec_arena.models.sequential_models.sasrec import SASRec
        config = SASRecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, feedforward_dim=32, dropout_rate=0.0,
            max_seq_length=SEQ_LEN, loss_type="cross_entropy",
        )
        return SASRec(config)

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_forward_no_nan(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert not torch.isnan(logits).any()

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_predict_next_sums_to_one(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seqs, lengths)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_get_sequence_embedding_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        emb = model.get_sequence_embedding(seqs, lengths)
        assert emb.shape == (BATCH, EMB_DIM)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        loss = logits.sum()
        loss.backward()
        assert model.item_embedding.weight.grad is not None

    def test_get_targets_and_mask(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([5, 8, 3, 6])
        batch = {"sequence": seqs, "sequence_length": lengths}
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape == (BATCH, SEQ_LEN)
        # Mask should be True for positions 1..length-1
        assert mask[0, 0].item() is False
        assert mask[0, 1].item() is True

    def test_variable_lengths(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([3, 5, 2, 8])
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_with_ligr(self):
        from rec_arena.configs.defaults.sasrec import SASRecConfig
        from rec_arena.models.sequential_models.sasrec import SASRec
        config = SASRecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, feedforward_dim=32, dropout_rate=0.0,
            max_seq_length=SEQ_LEN, loss_type="cross_entropy",
            use_ligr=True,
        )
        model = SASRec(config)
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_untied_embeddings(self):
        from rec_arena.configs.defaults.sasrec import SASRecConfig
        from rec_arena.models.sequential_models.sasrec import SASRec
        config = SASRecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, feedforward_dim=32, dropout_rate=0.0,
            max_seq_length=SEQ_LEN, loss_type="cross_entropy",
            tie_embeddings=False,
        )
        model = SASRec(config)
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_output_lora(self):
        from rec_arena.configs.defaults.sasrec import SASRecConfig
        from rec_arena.models.sequential_models.sasrec import SASRec
        config = SASRecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, feedforward_dim=32, dropout_rate=0.0,
            max_seq_length=SEQ_LEN, loss_type="cross_entropy",
            output_lora_rank=4,
        )
        model = SASRec(config)
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)



# ===================================================================
# GRU4Rec
# ===================================================================


class TestGRU4Rec:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.gru4rec import GRU4RecConfig
        from rec_arena.models.sequential_models.gru4rec import GRU4Rec
        config = GRU4RecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, hidden_size=EMB_DIM,
            num_layers=1, dropout_rate=0.0, max_seq_length=SEQ_LEN,
            loss_type="cross_entropy",
        )
        return GRU4Rec(config)

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_forward_no_nan(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert not torch.isnan(logits).any()

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_predict_next_sums_to_one(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seqs, lengths)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_get_sequence_embedding_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        emb = model.get_sequence_embedding(seqs, lengths)
        assert emb.shape == (BATCH, EMB_DIM)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        loss = logits.sum()
        loss.backward()
        assert model.item_embedding.weight.grad is not None

    def test_variable_lengths(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([3, 5, 2, 8])
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_get_targets_and_mask(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([5, 8, 3, 6])
        batch = {"sequence": seqs, "sequence_length": lengths}
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape == (BATCH, SEQ_LEN)
        assert mask[0, 0].item() is False


# ===================================================================
# BERT4Rec
# ===================================================================


class TestBERT4Rec:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.bert4rec import BERT4RecConfig
        from rec_arena.models.sequential_models.bert4rec import BERT4Rec
        config = BERT4RecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, dropout_rate=0.0, max_seq_length=SEQ_LEN,
            loss_type="cross_entropy",
        )
        return BERT4Rec(config)

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        logits = model.forward(seqs)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_mask_sequences(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        masked, labels = model.mask_sequences(seqs)
        assert masked.shape == seqs.shape
        assert labels.shape == seqs.shape

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        hidden = model.get_hidden_states(seqs)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        logits = model.forward(seqs)
        logits.sum().backward()
        assert model.item_embedding.weight.grad is not None


# ===================================================================
# Caser
# ===================================================================


class TestCaser:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.caser import CaserConfig
        from rec_arena.models.sequential_models.caser import Caser
        config = CaserConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM,
            num_horizontal_filters=4, num_vertical_filters=2,
            horizontal_filter_sizes=[2, 3], vertical_filter_size=SEQ_LEN,
            dropout_rate=0.0, max_seq_length=SEQ_LEN,
            loss_type="cross_entropy",
        )
        return Caser(config)

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        logits = model.forward(seqs)
        assert logits.shape == (BATCH, VOCAB)

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        hidden = model.get_hidden_states(seqs)
        assert hidden.shape == (BATCH, EMB_DIM)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        logits = model.forward(seqs)
        logits.sum().backward()
        assert model.item_embedding.weight.grad is not None


class TestCaserComputeLoss:
    @pytest.fixture
    def model(self):
        from rec_arena.models.sequential_models.caser import Caser
        from rec_arena.configs.defaults.caser import CaserConfig
        config = CaserConfig(vocab_size=23, embedding_dim=16, max_seq_length=10, dropout_rate=0.1)
        m = Caser(config)
        m.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
        return m

    def test_compute_loss_finite(self, model):
        batch = {"sequence": torch.randint(1, 20, (2, 10)), "sequence_length": torch.tensor([10, 8])}
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_get_targets_and_mask(self, model):
        batch = {"sequence": torch.randint(1, 20, (2, 10)), "sequence_length": torch.tensor([10, 8])}
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape[0] == 2



# ===================================================================
# FMLPRec
# ===================================================================


class TestFMLPRec:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.fmlprec import FMLPRecConfig
        from rec_arena.models.sequential_models.fmlprec import FMLPRec
        config = FMLPRecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM,
            num_blocks=1, dropout_rate=0.0, max_seq_length=SEQ_LEN,
            loss_type="cross_entropy",
        )
        return FMLPRec(config)

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seqs, lengths)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        logits.sum().backward()
        assert model.item_embedding.weight.grad is not None


# ===================================================================
# HSTU
# ===================================================================


class TestHSTU:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.hstu import HSTUConfig
        from rec_arena.models.sequential_models.hstu import HSTU
        config = HSTUConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, dropout_rate=0.0, max_seq_length=SEQ_LEN,
            loss_type="cross_entropy",
        )
        return HSTU(config)

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        logits.sum().backward()
        assert model.item_embedding.weight.grad is not None

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seqs, lengths)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_get_sequence_embedding_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        emb = model.get_sequence_embedding(seqs, lengths)
        assert emb.shape == (BATCH, EMB_DIM)

    def test_variable_lengths(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([3, 5, 2, 8])
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_predict_next_sums_to_one(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_forward_no_nan(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert not torch.isnan(logits).any()


class TestHSTUComputeLoss:
    @pytest.fixture
    def model(self):
        from rec_arena.models.sequential_models.hstu import HSTU
        from rec_arena.configs.defaults.hstu import HSTUConfig
        config = HSTUConfig(vocab_size=23, embedding_dim=16, max_seq_length=10, num_heads=2, num_layers=1, dropout_rate=0.1)
        m = HSTU(config)
        m.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
        return m

    def test_compute_loss_finite(self, model):
        batch = {"sequence": torch.randint(1, 20, (2, 10)), "sequence_length": torch.tensor([10, 8])}
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_get_targets_and_mask(self, model):
        batch = {"sequence": torch.randint(1, 20, (2, 10)), "sequence_length": torch.tensor([10, 8])}
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape == (2, 10)



# ===================================================================
# Mamba4Rec
# ===================================================================


try:
    from rec_arena.configs.defaults.mamba4rec import Mamba4RecConfig
    from rec_arena.models.sequential_models.mamba4rec import Mamba4Rec
    # Try to actually instantiate to check if mamba2 is available
    _test_cfg = Mamba4RecConfig(vocab_size=10, embedding_dim=8, d_model=8, d_state=4,
                                 d_conv=2, expand_factor=2, num_layers=1, max_seq_length=4)
    Mamba4Rec(_test_cfg)
    _HAS_MAMBA = True
    del _test_cfg
except (ImportError, Exception):
    _HAS_MAMBA = False


@pytest.mark.skipif(not _HAS_MAMBA, reason="Mamba4Rec requires mamba2 library")
class TestMamba4Rec:
    @pytest.fixture
    def model(self):
        config = Mamba4RecConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, d_model=EMB_DIM,
            d_state=8, d_conv=4, expand_factor=2, num_layers=1,
            dropout_rate=0.0, max_seq_length=SEQ_LEN,
            loss_type="cross_entropy",
        )
        return Mamba4Rec(config)

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seqs, lengths)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        logits = model.forward(seqs, lengths)
        logits.sum().backward()
        assert model.item_embedding.weight.grad is not None


# ===================================================================
# MLP4Rec
# ===================================================================


class TestMLP4Rec:
    @pytest.fixture
    def model(self):
        from rec_arena.models.sequential_models.mlp4rec import MLP4Rec
        from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
        config = MLP4RecConfig(
            vocab_size=23, embedding_dim=16, max_seq_length=10,
            num_layers=2, hidden_multiplier=2, dropout_rate=0.1,
        )
        return MLP4Rec(config)

    def test_forward_output_shape(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        out = model.forward(seq, lengths)
        assert out.shape[0] == 2

    def test_predict_next_shape(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        probs = model.predict_next(seq, lengths)
        assert probs.shape == (2, 23)

    def test_get_hidden_states_shape(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        hidden = model.get_hidden_states(seq, lengths)
        assert hidden.shape[0] == 2
        assert hidden.shape[2] == 16

    def test_get_sequence_embedding_shape(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        emb = model.get_sequence_embedding(seq, lengths)
        assert emb.shape == (2, 16)

    def test_gradient_flows(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        out = model.forward(seq, lengths)
        loss = out.sum()
        loss.backward()
        assert model.item_embedding.weight.grad is not None

    def test_get_targets_and_mask(self, model):
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
        }
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape == (2, 10)
        assert mask.shape == (2, 10)

    def test_mean_pooling(self):
        from rec_arena.models.sequential_models.mlp4rec import MLP4Rec
        from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
        config = MLP4RecConfig(
            vocab_size=23, embedding_dim=16, max_seq_length=10,
            pooling="mean", dropout_rate=0.1,
        )
        model = MLP4Rec(config)
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        probs = model.predict_next(seq, lengths)
        assert probs.shape == (2, 23)

    def test_max_pooling(self):
        from rec_arena.models.sequential_models.mlp4rec import MLP4Rec
        from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
        config = MLP4RecConfig(
            vocab_size=23, embedding_dim=16, max_seq_length=10,
            pooling="max", dropout_rate=0.1,
        )
        model = MLP4Rec(config)
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        probs = model.predict_next(seq, lengths)
        assert probs.shape == (2, 23)


class TestMLP4RecComputeLoss:
    @pytest.fixture
    def model(self):
        from rec_arena.models.sequential_models.mlp4rec import MLP4Rec
        from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
        config = MLP4RecConfig(vocab_size=23, embedding_dim=16, max_seq_length=10, dropout_rate=0.1)
        m = MLP4Rec(config)
        m.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
        return m

    def test_compute_loss_finite(self, model):
        batch = {"sequence": torch.randint(1, 20, (2, 10)), "sequence_length": torch.tensor([10, 8])}
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_compute_loss_with_neg_items(self, model):
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "neg_items": torch.randint(3, 20, (2, 10, 4)),
        }
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_attention_pooling(self):
        from rec_arena.models.sequential_models.mlp4rec import MLP4Rec
        from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
        config = MLP4RecConfig(vocab_size=23, embedding_dim=16, max_seq_length=10, pooling="attention", dropout_rate=0.1)
        m = MLP4Rec(config)
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        probs = m.predict_next(seq, lengths)
        assert probs.shape == (2, 23)

    def test_get_item_embedding(self):
        from rec_arena.models.sequential_models.mlp4rec import MLP4Rec
        from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
        config = MLP4RecConfig(vocab_size=23, embedding_dim=16, max_seq_length=10, dropout_rate=0.1)
        m = MLP4Rec(config)
        emb = m.get_item_embedding(torch.tensor([1, 5, 10]))
        assert emb.shape == (3, 16)



# ===================================================================
# FuXiGamma
# ===================================================================


class TestFuXiGamma:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig
        from rec_arena.models.sequential_models.fuxi_gamma import FuXiGamma
        config = FuXiGammaConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, max_seq_length=SEQ_LEN,
            num_heads=2, num_layers=1, linear_dim=EMB_DIM // 2, attention_dim=EMB_DIM // 2,
            ffn_multiply=1.0, dropout_rate=0.0,
        )
        return FuXiGamma(config)

    def test_get_hidden_states_shape(self, model):
        seq = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seq, lengths)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_predict_next_shape(self, model):
        seq = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seq, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_get_sequence_embedding_shape(self, model):
        seq = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        emb = model.get_sequence_embedding(seq, lengths)
        assert emb.shape == (BATCH, EMB_DIM)

    def test_gradient_flows(self, model):
        seq = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seq, lengths)
        loss = hidden.sum()
        loss.backward()
        assert model.item_embedding.weight.grad is not None

    def test_get_targets_and_mask(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([5, 8, 3, 6])
        batch = {"sequence": seqs, "sequence_length": lengths}
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape == (BATCH, SEQ_LEN)
        assert mask.shape == (BATCH, SEQ_LEN)
        assert mask[0, 0].item() is False

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden_states = model.get_hidden_states(seqs, lengths)
        logits = hidden_states @ model.item_embedding.weight.T
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_variable_lengths(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([3, 5, 2, 8])
        hidden_states = model.get_hidden_states(seqs, lengths)
        logits = hidden_states @ model.item_embedding.weight.T
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_predict_next_sums_to_one(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_forward_no_nan(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden_states = model.get_hidden_states(seqs, lengths)
        logits = hidden_states @ model.item_embedding.weight.T
        assert not torch.isnan(logits).any()


class TestFuXiGammaComputeLoss:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig
        from rec_arena.models.sequential_models.fuxi_gamma import FuXiGamma
        config = FuXiGammaConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, max_seq_length=SEQ_LEN,
            num_heads=2, num_layers=1, linear_dim=EMB_DIM // 2, attention_dim=EMB_DIM // 2,
            ffn_multiply=1.0, dropout_rate=0.0,
        )
        m = FuXiGamma(config)
        m.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
        return m

    def test_compute_loss_finite(self, model):
        batch = {"sequence": torch.randint(3, VOCAB, (BATCH, SEQ_LEN)), "sequence_length": torch.full((BATCH,), SEQ_LEN, dtype=torch.long)}
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)


# ===================================================================
# FuXi
# ===================================================================


class TestFuXi:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.fuxi import FuXiConfig
        from rec_arena.models.sequential_models.fuxi import FuXi
        config = FuXiConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, attention_dim=EMB_DIM, linear_dim=EMB_DIM,
            dropout_rate=0.0, max_seq_length=SEQ_LEN,
            loss_type="cross_entropy",
        )
        return FuXi(config)

    def test_get_hidden_states_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seqs, lengths)
        assert hidden.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_predict_next_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert probs.shape == (BATCH, VOCAB)

    def test_get_sequence_embedding_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        emb = model.get_sequence_embedding(seqs, lengths)
        assert emb.shape == (BATCH, EMB_DIM)

    def test_gradient_flows(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden = model.get_hidden_states(seqs, lengths)
        logits = torch.matmul(hidden, model.item_embedding.weight.T)
        logits.sum().backward()
        assert model.item_embedding.weight.grad is not None

    def test_get_targets_and_mask(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([5, 8, 3, 6])
        batch = {"sequence": seqs, "sequence_length": lengths}
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape == (BATCH, SEQ_LEN)
        assert mask.shape == (BATCH, SEQ_LEN)
        assert mask[0, 0].item() is False  # Position 0 not predicted

    def test_forward_output_shape(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden_states = model.get_hidden_states(seqs, lengths)
        logits = hidden_states @ model.item_embedding.weight.T
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_variable_lengths(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.tensor([3, 5, 2, 8])
        hidden_states = model.get_hidden_states(seqs, lengths)
        logits = hidden_states @ model.item_embedding.weight.T
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_predict_next_sums_to_one(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        probs = model.predict_next(seqs, lengths)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_forward_no_nan(self, model):
        seqs = torch.randint(3, VOCAB, (BATCH, SEQ_LEN))
        lengths = torch.full((BATCH,), SEQ_LEN, dtype=torch.long)
        hidden_states = model.get_hidden_states(seqs, lengths)
        logits = hidden_states @ model.item_embedding.weight.T
        assert not torch.isnan(logits).any()


class TestFuXiComputeLoss:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.fuxi import FuXiConfig
        from rec_arena.models.sequential_models.fuxi import FuXi
        config = FuXiConfig(
            vocab_size=VOCAB, embedding_dim=EMB_DIM, num_heads=2,
            num_layers=1, attention_dim=EMB_DIM, linear_dim=EMB_DIM,
            dropout_rate=0.0, max_seq_length=SEQ_LEN,
        )
        m = FuXi(config)
        m.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
        return m

    def test_compute_loss_finite(self, model):
        batch = {"sequence": torch.randint(3, VOCAB, (BATCH, SEQ_LEN)), "sequence_length": torch.full((BATCH,), SEQ_LEN, dtype=torch.long)}
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)


# ===================================================================
# LLaDA4Rec
# ===================================================================


class TestLLaDA4Rec:
    @pytest.fixture
    def model(self):
        from rec_arena.models.sequential_models.llada4rec import LLaDA4Rec
        from rec_arena.configs.defaults.llada4rec import LLaDA4RecConfig
        config = LLaDA4RecConfig(
            vocab_size=23, embedding_dim=16, max_seq_length=10,
            num_heads=2, num_layers=1, dropout_rate=0.1,
            diffusion_steps=5, eps=0.1,
        )
        return LLaDA4Rec(config)

    def test_forward_process(self, model):
        seq = torch.randint(3, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        masked_seq, mask_indices, p_mask = model.forward_process(seq, lengths)
        assert masked_seq.shape == (2, 10)
        assert mask_indices.shape == (2, 10)
        assert p_mask.shape == (2, 10)

    def test_predict_next_shape(self, model):
        seq = torch.randint(3, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        probs = model.predict_next(seq, lengths)
        assert probs.shape == (2, 23)

    def test_compute_loss(self, model):
        from rec_arena.losses.sequential.llada_loss import LLaDALoss
        model.set_loss_fn(LLaDALoss())
        batch = {
            "sequence": torch.randint(3, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
        }
        loss = model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flows(self, model):
        from rec_arena.losses.sequential.llada_loss import LLaDALoss
        model.set_loss_fn(LLaDALoss())
        batch = {
            "sequence": torch.randint(3, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
        }
        loss = model.compute_loss(batch)
        loss.backward()
        assert model.item_embedding.weight.grad is not None

    def test_generate_sequence(self, model):
        prompt = torch.randint(3, 20, (1, 5))
        result = model.generate_sequence(prompt, gen_length=3, steps=2)
        assert result.shape[1] == 5 + 3


class TestLLaDA4RecValidationStep:
    @pytest.fixture
    def model(self):
        from rec_arena.models.sequential_models.llada4rec import LLaDA4Rec
        from rec_arena.configs.defaults.llada4rec import LLaDA4RecConfig
        from rec_arena.losses.sequential.llada_loss import LLaDALoss
        config = LLaDA4RecConfig(vocab_size=23, embedding_dim=16, max_seq_length=10, num_heads=2, num_layers=1, dropout_rate=0.1, diffusion_steps=3, eps=0.1)
        m = LLaDA4Rec(config)
        m.set_loss_fn(LLaDALoss())
        return m

    def test_validation_with_target(self, model):
        model.log = MagicMock()
        model._trainer = MagicMock()
        model._trainer.callback_metrics = {}
        batch = {
            "sequence": torch.randint(3, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
            "target": torch.randint(3, 20, (2,)),
        }
        result = model.validation_step(batch, 0)
        assert result is not None

    def test_validation_without_target(self, model):
        model.log = MagicMock()
        batch = {"sequence": torch.randint(3, 20, (2, 10)), "sequence_length": torch.tensor([10, 8])}
        loss = model.validation_step(batch, 0)
        assert torch.isfinite(loss)

    def test_generate_with_temperature(self, model):
        model.config = model.config  # ensure config accessible
        prompt = torch.randint(3, 20, (1, 5))
        result = model.generate_sequence(prompt, gen_length=2, steps=2)
        assert result.shape[1] == 7



# ===================================================================
# RecM
# ===================================================================


def _make_recm(ensemble_size=1, ensemble_loss_functions=None):
    from rec_arena.models.sequential_models.recm import RecM
    from rec_arena.configs.defaults.recm import RecMConfig
    config = RecMConfig(
        vocab_size=23, embedding_dim=16, max_seq_length=10,
        num_heads=2, num_layers=1, dropout_rate=0.1,
        ensemble_size=ensemble_size,
        ensemble_loss_functions=ensemble_loss_functions,
        loss_type="cross_entropy",
        embedding_config={"type": "standard"},
        position_config={"type": "learnable"},
    )
    model = RecM(config)
    if ensemble_loss_functions is None:
        model.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
    return model


def _make_recm_batch():
    return {
        "sequence": torch.randint(1, 20, (2, 10)),
        "sequence_length": torch.tensor([10, 8]),
    }


class TestRecM:
    @pytest.fixture
    def model(self):
        return _make_recm(ensemble_size=1)

    def test_forward_output_shape(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        out = model.forward(seq, lengths)
        assert out.shape[0] == 2

    def test_predict_next_shape(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        probs = model.predict_next(seq, lengths)
        assert probs.shape == (2, 23)

    def test_get_hidden_states(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        hidden = model.get_hidden_states(seq, lengths)
        assert hidden.shape[0] == 2
        assert hidden.shape[-1] == 16

    def test_get_sequence_embedding(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        emb = model.get_sequence_embedding(seq, lengths)
        assert emb.shape[0] == 2
        assert emb.shape[-1] == 16

    def test_gradient_flows(self, model):
        seq = torch.randint(1, 20, (2, 10))
        lengths = torch.tensor([10, 8])
        out = model.forward(seq, lengths)
        loss = out.sum()
        loss.backward()
        assert model.item_embedding.weight.grad is not None

    def test_get_targets_and_mask(self, model):
        batch = {
            "sequence": torch.randint(1, 20, (2, 10)),
            "sequence_length": torch.tensor([10, 8]),
        }
        targets, mask = model.get_targets_and_mask(batch)
        assert targets.shape == (2, 10)


class TestRecMComputeLossDefault:
    def test_default_loss_produces_finite_scalar(self):
        model = _make_recm(ensemble_size=1)
        loss = model.compute_loss(_make_recm_batch())
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_default_loss_with_neg_items_3d(self):
        model = _make_recm(ensemble_size=1)
        batch = _make_recm_batch()
        batch["neg_items"] = torch.randint(3, 20, (2, 10, 4))
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_default_loss_with_neg_items_2d(self):
        model = _make_recm(ensemble_size=1)
        batch = _make_recm_batch()
        batch["neg_items"] = torch.randint(3, 20, (2, 10))
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)


class TestRecMComputeLossEnsemble:
    def test_ensemble_single_loss_type(self):
        model = _make_recm(ensemble_size=2, ensemble_loss_functions=["cross_entropy"])
        loss = model.compute_loss(_make_recm_batch())
        assert torch.isfinite(loss)

    def test_ensemble_multiple_loss_types(self):
        model = _make_recm(ensemble_size=2, ensemble_loss_functions=["cross_entropy", "bce"])
        batch = _make_recm_batch()
        # Add per-ensemble neg items
        batch["neg_items_0"] = torch.randint(3, 20, (2, 10, 4))
        batch["neg_items_1"] = torch.randint(3, 20, (2, 10, 4))
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_ensemble_with_bpr_loss(self):
        model = _make_recm(ensemble_size=2, ensemble_loss_functions=["bpr", "bpr"])
        batch = _make_recm_batch()
        batch["neg_items_0"] = torch.randint(3, 20, (2, 10, 4))
        batch["neg_items_1"] = torch.randint(3, 20, (2, 10, 4))
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)


class TestRecMValidationStep:
    def test_with_target(self):
        model = _make_recm(ensemble_size=1)
        model.log = MagicMock()
        batch = _make_recm_batch()
        batch["target"] = torch.randint(3, 20, (2,))
        result = model.validation_step(batch, 0)
        assert torch.isfinite(result)

    def test_without_target(self):
        model = _make_recm(ensemble_size=1)
        model.log = MagicMock()
        loss = model.validation_step(_make_recm_batch(), 0)
        assert torch.isfinite(loss)


class TestRecMTestStep:
    def test_with_target(self):
        model = _make_recm(ensemble_size=1)
        model.log = MagicMock()
        batch = _make_recm_batch()
        batch["target"] = torch.randint(3, 20, (2,))
        result = model.test_step(batch, 0)
        assert "predictions" in result
        assert "targets" in result

    def test_without_target(self):
        model = _make_recm(ensemble_size=1)
        model.log = MagicMock()
        result = model.test_step(_make_recm_batch(), 0)
        assert "predictions" in result


class TestRecMGetLossMask:
    def test_returns_mask(self):
        model = _make_recm(ensemble_size=1)
        batch = _make_recm_batch()
        mask = model.get_loss_mask(batch)
        assert mask.shape == (2, 10)


# ===================================================================
# Normalization layers
# ===================================================================


class TestNormalizationLayersCoverage:
    def test_learnable_layer_scaling(self):
        from rec_arena.modules.layer_utils.normalization_layers import LearnableLayerScaling
        lls = LearnableLayerScaling(64)
        x = torch.randn(2, 10, 64)
        out = lls(x)
        assert out.shape == (2, 10, 64)


# ===================================================================
# Sampler base
# ===================================================================


class TestSamplerBaseCoverage:
    def test_random_sampler_sample(self):
        from rec_arena.samplers.random_sampler import RandomSampler
        sampler = RandomSampler(num_items=100, num_negatives=4)
        positives = {1, 2, 3}
        result = sampler.sample(positives)
        assert len(result) == 4


# ===================================================================
# Property-Based Tests for HSTU, FuXi-α, and FuXi-γ
# ===================================================================

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st


class TestHSTUFuXiProperties:
    """Property-based tests for HSTU, FuXi-α, and FuXi-γ using Hypothesis."""

    PBT_EMB_DIM = 16
    PBT_VOCAB = 50

    @staticmethod
    def _make_hstu(vocab, emb_dim, seq_len):
        from rec_arena.configs.defaults.hstu import HSTUConfig
        from rec_arena.models.sequential_models.hstu import HSTU
        config = HSTUConfig(
            vocab_size=vocab, embedding_dim=emb_dim, num_heads=2,
            num_layers=1, dropout_rate=0.0, max_seq_length=seq_len,
            loss_type="cross_entropy",
        )
        return HSTU(config)

    @staticmethod
    def _make_fuxi(vocab, emb_dim, seq_len):
        from rec_arena.configs.defaults.fuxi import FuXiConfig
        from rec_arena.models.sequential_models.fuxi import FuXi
        config = FuXiConfig(
            vocab_size=vocab, embedding_dim=emb_dim, num_heads=2,
            num_layers=1, attention_dim=emb_dim, linear_dim=emb_dim,
            dropout_rate=0.0, max_seq_length=seq_len,
            loss_type="cross_entropy",
        )
        return FuXi(config)

    @staticmethod
    def _make_fuxi_gamma(vocab, emb_dim, seq_len):
        from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig
        from rec_arena.models.sequential_models.fuxi_gamma import FuXiGamma
        config = FuXiGammaConfig(
            vocab_size=vocab, embedding_dim=emb_dim, max_seq_length=seq_len,
            num_heads=2, num_layers=1, linear_dim=emb_dim // 2,
            attention_dim=emb_dim // 2, ffn_multiply=1.0, dropout_rate=0.0,
            loss_type="cross_entropy",
        )
        return FuXiGamma(config)

    def _all_models(self, seq_len):
        v, e = self.PBT_VOCAB, self.PBT_EMB_DIM
        return [
            self._make_hstu(v, e, seq_len),
            self._make_fuxi(v, e, seq_len),
            self._make_fuxi_gamma(v, e, seq_len),
        ]

    # Feature: hstu-integration, Property 1: predict_next returns a valid probability distribution
    @given(batch_size=st.integers(1, 8), seq_len=st.integers(2, 16))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_predict_next_valid_distribution(self, batch_size, seq_len):
        for model in self._all_models(seq_len):
            seqs = torch.randint(3, self.PBT_VOCAB, (batch_size, seq_len))
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            probs = model.predict_next(seqs, lengths)
            assert probs.shape == (batch_size, self.PBT_VOCAB)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)

    # Feature: hstu-integration, Property 2: get_hidden_states returns correct shape
    @given(batch_size=st.integers(1, 8), seq_len=st.integers(2, 16))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_get_hidden_states_shape(self, batch_size, seq_len):
        for model in self._all_models(seq_len):
            seqs = torch.randint(3, self.PBT_VOCAB, (batch_size, seq_len))
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            hidden = model.get_hidden_states(seqs, lengths)
            assert hidden.shape == (batch_size, seq_len, self.PBT_EMB_DIM)

    # Feature: hstu-integration, Property 3: get_sequence_embedding returns correct shape
    @given(batch_size=st.integers(1, 8), seq_len=st.integers(2, 16))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_get_sequence_embedding_shape(self, batch_size, seq_len):
        for model in self._all_models(seq_len):
            seqs = torch.randint(3, self.PBT_VOCAB, (batch_size, seq_len))
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            emb = model.get_sequence_embedding(seqs, lengths)
            assert emb.shape == (batch_size, self.PBT_EMB_DIM)

    # Feature: hstu-integration, Property 4: forward output has correct shape with variable lengths
    @given(batch_size=st.integers(1, 8), seq_len=st.integers(2, 16))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_shape_variable_lengths(self, batch_size, seq_len):
        for model in self._all_models(seq_len):
            seqs = torch.randint(3, self.PBT_VOCAB, (batch_size, seq_len))
            lengths = torch.randint(1, seq_len + 1, (batch_size,))
            hidden = model.get_hidden_states(seqs, lengths)
            logits = hidden @ model.item_embedding.weight.T
            assert logits.shape == (batch_size, seq_len, self.PBT_VOCAB)

    # Feature: hstu-integration, Property 5: forward output contains no NaN values
    @given(batch_size=st.integers(1, 8), seq_len=st.integers(2, 16))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_no_nan(self, batch_size, seq_len):
        for model in self._all_models(seq_len):
            seqs = torch.randint(3, self.PBT_VOCAB, (batch_size, seq_len))
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            hidden = model.get_hidden_states(seqs, lengths)
            logits = hidden @ model.item_embedding.weight.T
            assert not torch.isnan(logits).any()

    # Feature: hstu-integration, Property 6: compute_loss returns a finite scalar
    @given(batch_size=st.integers(1, 8), seq_len=st.integers(2, 16))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_compute_loss_finite(self, batch_size, seq_len):
        for model in self._all_models(seq_len):
            model.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
            batch = {
                "sequence": torch.randint(3, self.PBT_VOCAB, (batch_size, seq_len)),
                "sequence_length": torch.full((batch_size,), seq_len, dtype=torch.long),
            }
            loss = model.compute_loss(batch)
            assert torch.isfinite(loss)
