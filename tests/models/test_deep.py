"""Tests for deep model implementations (NCF, TwoTower, SimpleX)."""

import pytest
import torch

BATCH = 4
EMB_DIM = 16
NUM_USERS = 20
NUM_ITEMS = 30


# ===================================================================
# NCF
# ===================================================================


class TestNCF:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.ncf import NCFConfig
        from rec_arena.models.deep_models.ncf import NCF
        config = NCFConfig(
            num_users=NUM_USERS, num_items=NUM_ITEMS,
            embedding_dim=EMB_DIM, hidden_dims=[32, 16],
            dropout_rate=0.0, loss_type="bce",
        )
        return NCF(config)

    def test_forward_output_shape(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        assert scores.shape == (BATCH,)

    def test_forward_no_nan(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        assert not torch.isnan(scores).any()

    def test_predict_returns_probabilities(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        probs = model.predict(users, items)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_recommend_shapes(self, model):
        users = torch.tensor([0, 1])
        top_items, top_scores = model.recommend(users, k=5)
        assert top_items.shape == (2, 5)
        assert top_scores.shape == (2, 5)

    def test_get_hidden_states_shape(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        hidden = model.get_hidden_states(users, items)
        assert hidden.shape[0] == BATCH
        # hidden dim = embedding_dim + last hidden dim
        assert hidden.shape[1] == EMB_DIM + 16

    def test_gradient_flows(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        loss = scores.sum()
        loss.backward()
        assert model.user_embedding_mf.weight.grad is not None

    def test_batch_norm_variant(self):
        from rec_arena.configs.defaults.ncf import NCFConfig
        from rec_arena.models.deep_models.ncf import NCF
        config = NCFConfig(
            num_users=NUM_USERS, num_items=NUM_ITEMS,
            embedding_dim=EMB_DIM, hidden_dims=[32, 16],
            dropout_rate=0.0, loss_type="bce", use_batch_norm=True,
        )
        model = NCF(config)
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        assert scores.shape == (BATCH,)


# ===================================================================
# TwoTower
# ===================================================================


class TestTwoTower:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.twotower import TwoTowerConfig
        from rec_arena.models.deep_models.twotower import TwoTower
        config = TwoTowerConfig(
            num_users=NUM_USERS, num_items=NUM_ITEMS,
            embedding_dim=EMB_DIM, user_tower_dims=[32, 16],
            item_tower_dims=[32, 16], dropout_rate=0.0,
            loss_type="bpr",
        )
        return TwoTower(config)

    def test_forward_output_shape(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        assert scores.shape == (BATCH,)

    def test_predict_returns_probabilities(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        probs = model.predict(users, items)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_encode_user_shape(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        emb = model.encode_user(users)
        assert emb.shape == (BATCH, 16)

    def test_encode_item_shape(self, model):
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        emb = model.encode_item(items)
        assert emb.shape == (BATCH, 16)

    def test_recommend_shapes(self, model):
        users = torch.tensor([0, 1])
        top_items, top_scores = model.recommend(users, k=5)
        assert top_items.shape == (2, 5)
        assert top_scores.shape == (2, 5)

    def test_gradient_flows(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        scores.sum().backward()
        assert model.user_embedding.weight.grad is not None


# ===================================================================
# SimpleX
# ===================================================================


class TestSimpleX:
    @pytest.fixture
    def model(self):
        from rec_arena.configs.defaults.simplex import SimpleXConfig
        from rec_arena.models.deep_models.simplex import SimpleX
        config = SimpleXConfig(
            num_users=NUM_USERS, num_items=NUM_ITEMS,
            embedding_dim=EMB_DIM, loss_type="bpr",
        )
        return SimpleX(config)

    def test_forward_output_shape(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        assert scores.shape == (BATCH,)

    def test_predict_returns_probabilities(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        probs = model.predict(users, items)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_recommend_shapes(self, model):
        users = torch.tensor([0, 1])
        top_items, top_scores = model.recommend(users, k=5)
        assert top_items.shape == (2, 5)

    def test_cosine_similarity_bounded(self, model):
        users = torch.randint(0, NUM_USERS, (BATCH,))
        items = torch.randint(0, NUM_ITEMS, (BATCH,))
        scores = model.forward(users, items)
        assert (scores >= -1).all() and (scores <= 1).all()

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


class TestSimpleXCoverage:
    @pytest.fixture
    def model(self):
        from rec_arena.models.deep_models.simplex import SimpleX
        from rec_arena.configs.defaults.simplex import SimpleXConfig
        config = SimpleXConfig(num_users=5, num_items=20, embedding_dim=16)
        return SimpleX(config)

    def test_compute_loss(self, model):
        batch = {
            "user_id": torch.tensor([0, 1]),
            "item_id": torch.tensor([3, 5]),
            "neg_items": torch.randint(0, 20, (2, 4)),
        }
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_forward_shape(self, model):
        user_ids = torch.tensor([0, 1])
        item_ids = torch.tensor([3, 5])
        out = model(user_ids, item_ids)
        assert out.shape[0] == 2
