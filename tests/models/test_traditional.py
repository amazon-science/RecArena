"""Tests for traditional model implementations (EASE, ItemKNN, SLIM, BPRMF)."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

NUM_USERS = 10
NUM_ITEMS = 15
BATCH = 4
EMB_DIM = 16


def _make_sparse_matrix():
    """Create a small sparse user-item interaction matrix."""
    rows, cols, data = [], [], []
    for u in range(NUM_USERS):
        for i in range(3, 8):  # Each user interacts with items 3-7
            rows.append(u)
            cols.append(i)
            data.append(1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(NUM_USERS, NUM_ITEMS))


# ===================================================================
# TraditionalModel base class
# ===================================================================


class TestTraditionalModelBase:
    def test_predict_converts_tensors(self):
        """TraditionalModel.predict should convert tensors to numpy and back."""
        from rec_arena.configs.defaults.ease import EASEConfig
        from rec_arena.models.traditional_models.ease import EASE
        config = EASEConfig(num_users=NUM_USERS, num_items=NUM_ITEMS)
        model = EASE(config)
        X = _make_sparse_matrix()
        model.fit(X)
        users = torch.tensor([0, 1])
        items = torch.tensor([3, 4])
        result = model.predict(users, items)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)

    def test_recommend_converts_tensors(self):
        from rec_arena.configs.defaults.ease import EASEConfig
        from rec_arena.models.traditional_models.ease import EASE
        config = EASEConfig(num_users=NUM_USERS, num_items=NUM_ITEMS)
        model = EASE(config)
        X = _make_sparse_matrix()
        model.fit(X)
        users = torch.tensor([0, 1])
        top_items, top_scores = model.recommend(users, k=3)
        assert isinstance(top_items, torch.Tensor)
        assert top_items.shape == (2, 3)



# ===================================================================
# EASE
# ===================================================================


class TestEASE:
    @pytest.fixture
    def trained_model(self):
        from rec_arena.configs.defaults.ease import EASEConfig
        from rec_arena.models.traditional_models.ease import EASE
        config = EASEConfig(num_users=NUM_USERS, num_items=NUM_ITEMS, reg_lambda=100.0)
        model = EASE(config)
        X = _make_sparse_matrix()
        model.fit(X)
        return model

    def test_fit_creates_B_matrix(self, trained_model):
        assert trained_model.B is not None
        assert trained_model.B.shape == (NUM_ITEMS, NUM_ITEMS)

    def test_B_diagonal_is_zero(self, trained_model):
        assert np.allclose(np.diag(trained_model.B), 0.0)

    def test_predict_numpy(self, trained_model):
        users = np.array([0, 1])
        items = np.array([3, 4])
        scores = trained_model._predict_numpy(users, items)
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))

    def test_recommend_numpy(self, trained_model):
        users = np.array([0, 1])
        top_items, top_scores = trained_model._recommend_numpy(users, k=3)
        assert top_items.shape == (2, 3)
        assert top_scores.shape == (2, 3)

    def test_predict_before_fit_raises(self):
        from rec_arena.configs.defaults.ease import EASEConfig
        from rec_arena.models.traditional_models.ease import EASE
        config = EASEConfig(num_users=NUM_USERS, num_items=NUM_ITEMS)
        model = EASE(config)
        with pytest.raises(RuntimeError, match="not trained"):
            model._predict_numpy(np.array([0]), np.array([0]))

    def test_recommend_before_fit_raises(self):
        from rec_arena.configs.defaults.ease import EASEConfig
        from rec_arena.models.traditional_models.ease import EASE
        config = EASEConfig(num_users=NUM_USERS, num_items=NUM_ITEMS)
        model = EASE(config)
        with pytest.raises(RuntimeError, match="not trained"):
            model._recommend_numpy(np.array([0]), k=3)

    def test_fit_invalid_input_raises(self):
        from rec_arena.configs.defaults.ease import EASEConfig
        from rec_arena.models.traditional_models.ease import EASE
        config = EASEConfig(num_users=NUM_USERS, num_items=NUM_ITEMS)
        model = EASE(config)
        with pytest.raises(ValueError):
            model.fit("not a matrix")


# ===================================================================
# ItemKNN
# ===================================================================


class TestItemKNN:
    @pytest.fixture
    def trained_model(self):
        from rec_arena.configs.defaults.itemknn import ItemKNNConfig
        from rec_arena.models.traditional_models.itemknn import ItemKNN
        config = ItemKNNConfig(num_users=NUM_USERS, num_items=NUM_ITEMS, k=5)
        model = ItemKNN(config)
        X = _make_sparse_matrix()
        model.fit(X)
        return model

    def test_fit_creates_similarity_matrix(self, trained_model):
        assert trained_model.similarity_matrix is not None
        assert trained_model.similarity_matrix.shape == (NUM_ITEMS, NUM_ITEMS)

    def test_diagonal_is_zero(self, trained_model):
        assert np.allclose(np.diag(trained_model.similarity_matrix), 0.0)

    def test_predict_numpy(self, trained_model):
        users = np.array([0, 1])
        items = np.array([3, 4])
        scores = trained_model._predict_numpy(users, items)
        assert scores.shape == (2,)

    def test_recommend_numpy(self, trained_model):
        users = np.array([0, 1])
        top_items, top_scores = trained_model._recommend_numpy(users, k=3)
        assert top_items.shape == (2, 3)

    def test_jaccard_similarity(self):
        from rec_arena.configs.defaults.itemknn import ItemKNNConfig
        from rec_arena.models.traditional_models.itemknn import ItemKNN
        config = ItemKNNConfig(num_users=NUM_USERS, num_items=NUM_ITEMS, k=5, similarity="jaccard")
        model = ItemKNN(config)
        X = _make_sparse_matrix()
        model.fit(X)
        assert model.similarity_matrix is not None

    def test_predict_before_fit_raises(self):
        from rec_arena.configs.defaults.itemknn import ItemKNNConfig
        from rec_arena.models.traditional_models.itemknn import ItemKNN
        config = ItemKNNConfig(num_users=NUM_USERS, num_items=NUM_ITEMS)
        model = ItemKNN(config)
        with pytest.raises(RuntimeError, match="not trained"):
            model._predict_numpy(np.array([0]), np.array([0]))


# ===================================================================
# SLIM
# ===================================================================


class TestSLIM:
    @pytest.fixture
    def trained_model(self):
        from rec_arena.configs.defaults.slim import SLIMConfig
        from rec_arena.models.traditional_models.slim import SLIM
        config = SLIMConfig(num_users=NUM_USERS, num_items=NUM_ITEMS, alpha=1.0, l1_ratio=0.5)
        model = SLIM(config)
        X = _make_sparse_matrix()
        model.fit(X)
        return model

    def test_fit_creates_W_matrix(self, trained_model):
        assert trained_model.W is not None

    def test_predict_numpy(self, trained_model):
        users = np.array([0, 1])
        items = np.array([3, 4])
        scores = trained_model._predict_numpy(users, items)
        assert scores.shape == (2,)

    def test_recommend_numpy(self, trained_model):
        users = np.array([0, 1])
        top_items, top_scores = trained_model._recommend_numpy(users, k=3)
        assert top_items.shape == (2, 3)

    def test_predict_before_fit_raises(self):
        from rec_arena.configs.defaults.slim import SLIMConfig
        from rec_arena.models.traditional_models.slim import SLIM
        config = SLIMConfig(num_users=NUM_USERS, num_items=NUM_ITEMS)
        model = SLIM(config)
        with pytest.raises(RuntimeError, match="not trained"):
            model._predict_numpy(np.array([0]), np.array([0]))


# ===================================================================
# BPRMF
# ===================================================================


class TestBPRMF:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.bprmf import BPRMFConfig
        from rec_arena.models.traditional_models.bpr_mf import BPRMF
        config = BPRMFConfig(
            num_users=NUM_USERS, num_items=NUM_ITEMS,
            embedding_dim=EMB_DIM, loss_type="bpr",
        )
        return BPRMF(config)

    def test_forward_output_shape(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        assert scores.shape == (BATCH,)

    def test_predict_output_shape(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.predict(users, items)
        assert scores.shape == (BATCH,)

    def test_recommend_shapes(self, model):
        users = torch.tensor([0, 1])
        top_items, top_scores = model.recommend(users, k=5)
        assert top_items.shape == (2, 5)
        assert top_scores.shape == (2, 5)

    def test_get_user_embedding(self, model):
        users = torch.tensor([0, 1, 2])
        emb = model.get_user_embedding(users)
        assert emb.shape == (3, EMB_DIM)

    def test_get_item_embedding(self, model):
        items = torch.tensor([0, 1, 2])
        emb = model.get_item_embedding(items)
        assert emb.shape == (3, EMB_DIM)

    def test_gradient_flows(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        scores.sum().backward()
        assert model.user_embedding.weight.grad is not None

    def test_get_hidden_states(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        hs = model.get_hidden_states(users, items)
        assert hs.shape == (BATCH, EMB_DIM * 2)

    def test_prediction_from_hidden_states(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        hs = model.get_hidden_states(users, items)
        pred = model.prediction(hs)
        assert pred.shape == (BATCH, 1)
