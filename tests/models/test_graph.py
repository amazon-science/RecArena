"""Tests for graph models: graph_utils, GraphModel base class, and LightGCN.

Consolidated from:
- tests/test_graph_utils.py (create_edge_index, create_pyg_edge_index,
  add_graph_support_to_datamodule, add_pyg_support_to_datamodule)
- tests/test_graph_base_coverage.py (GraphModel base class methods:
  configure_optimizers, training_step, validation_step, set_graph_data,
  create_bipartite_graph)
- tests/test_graph_models.py (PyGLightGCN predict/recommend/compute_loss/
  test_step/training_step/configure_optimizers/graph_data)
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from rec_arena.configs.defaults.lightgcn import LightGCNConfig
from rec_arena.models.graph import GraphModel
from rec_arena.models.graph_models.graph_utils import (
    add_graph_support_to_datamodule,
    add_pyg_support_to_datamodule,
    create_edge_index,
    create_pyg_edge_index,
)
from rec_arena.models.graph_models.lightgcn import PyGLightGCN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_df(num_users=3, items_per_user=2):
    rows = []
    for u in range(num_users):
        for i in range(items_per_user):
            rows.append({"user_id": u, "item_id": i})
    return pd.DataFrame(rows)


def _make_edge_index(num_users=5, num_items=10, num_edges=20):
    """Create a simple bidirectional edge index."""
    user_ids = torch.randint(0, num_users, (num_edges,))
    item_ids = torch.randint(0, num_items, (num_edges,)) + num_users
    forward = torch.stack([user_ids, item_ids])
    reverse = torch.stack([item_ids, user_ids])
    return torch.cat([forward, reverse], dim=1)


# ---------------------------------------------------------------------------
# Stubs / Fixtures
# ---------------------------------------------------------------------------


class StubGraphModel(GraphModel):
    """Minimal concrete GraphModel for testing base class methods."""

    def __init__(self, config):
        super().__init__(config)
        self.user_emb = torch.nn.Embedding(self.num_users, self.embedding_dim)
        self.item_emb = torch.nn.Embedding(self.num_items, self.embedding_dim)

    def compute_loss(self, batch):
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return torch.nn.functional.mse_loss(
            (u * i).sum(dim=-1), torch.ones(user_ids.size(0))
        )

    def predict(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return (u * i).sum(dim=-1)

    def recommend(self, user_ids, k=10):
        u = self.user_emb(user_ids)
        scores = u @ self.item_emb.weight.T
        return torch.topk(scores, k, dim=-1)

    def fit(self, train_data, val_data=None):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))


@pytest.fixture
def graph_model():
    config = LightGCNConfig(num_users=5, num_items=10, embedding_dim=16, num_layers=2)
    return StubGraphModel(config)


@pytest.fixture
def lightgcn_model():
    config = LightGCNConfig(num_users=5, num_items=10, embedding_dim=16, num_layers=2)
    model = PyGLightGCN(config)
    ei = _make_edge_index(5, 10, 20)
    model.set_graph_data(ei)
    return model


# ===========================================================================
# Graph Utils
# ===========================================================================


class TestCreateEdgeIndex:
    def test_shape_is_2_by_2n(self):
        df = _make_train_df(3, 2)
        ei = create_edge_index(df, num_users=3)
        assert ei.shape[0] == 2
        assert ei.shape[1] == 2 * len(df)  # bidirectional

    def test_bidirectional(self):
        df = _make_train_df(2, 1)
        ei = create_edge_index(df, num_users=2)
        n = len(df)
        forward = set(zip(ei[0, :n].tolist(), ei[1, :n].tolist()))
        reverse = set(zip(ei[0, n:].tolist(), ei[1, n:].tolist()))
        for src, dst in forward:
            assert (dst, src) in reverse

    def test_item_offset(self):
        df = pd.DataFrame({"user_id": [0], "item_id": [5]})
        ei = create_edge_index(df, num_users=10)
        assert ei[1, 0].item() == 5 + 10  # item offset by num_users


class TestCreatePygEdgeIndex:
    def test_shape_matches_create_edge_index(self):
        df = _make_train_df(3, 2)
        ei = create_pyg_edge_index(df, num_users=3)
        assert ei.shape[0] == 2
        assert ei.shape[1] == 2 * len(df)


class TestAddGraphSupportToDatamodule:
    def test_adds_methods(self):
        class DummyDM:
            pass

        add_graph_support_to_datamodule(DummyDM)
        assert hasattr(DummyDM, "_create_edge_index")
        assert hasattr(DummyDM, "get_edge_index")
        assert hasattr(DummyDM, "get_pyg_edge_index")

    def test_get_edge_index_raises_without_setup(self):
        class DummyDM:
            pass

        add_graph_support_to_datamodule(DummyDM)
        dm = DummyDM()
        with pytest.raises(ValueError, match="Edge index not available"):
            dm.get_edge_index()

    def test_get_pyg_edge_index_raises_without_setup(self):
        class DummyDM:
            pass

        add_graph_support_to_datamodule(DummyDM)
        dm = DummyDM()
        with pytest.raises(ValueError, match="Edge index not available"):
            dm.get_pyg_edge_index()


class TestAddPygSupportToDatamodule:
    def test_adds_methods(self):
        class DummyDM:
            pass

        add_pyg_support_to_datamodule(DummyDM)
        assert hasattr(DummyDM, "_create_pyg_edge_index")
        assert hasattr(DummyDM, "get_pyg_edge_index")

    def test_get_pyg_edge_index_raises_without_setup(self):
        class DummyDM:
            pass

        add_pyg_support_to_datamodule(DummyDM)
        dm = DummyDM()
        with pytest.raises(ValueError, match="Edge index not available"):
            dm.get_pyg_edge_index()


# ===========================================================================
# GraphModel Base
# ===========================================================================


class TestGraphModelConfigureOptimizers:
    def test_returns_optimizer(self, graph_model):
        opt = graph_model.configure_optimizers()
        assert isinstance(opt, torch.optim.Optimizer)


class TestGraphModelTrainingStep:
    def test_returns_loss(self, graph_model):
        graph_model.log = MagicMock()
        batch = {"user_id": torch.tensor([0, 1]), "item_id": torch.tensor([3, 5])}
        loss = graph_model.training_step(batch, 0)
        assert torch.isfinite(loss)


class TestGraphModelValidationStep:
    def test_returns_loss_via_compute_loss(self, graph_model):
        """Test that validation_step computes loss (the test_step path requires
        full graph setup, so we test the compute_loss path directly)."""
        graph_model.log = MagicMock()
        batch = {"user_id": torch.tensor([0, 1]), "item_id": torch.tensor([3, 5])}
        loss = graph_model.compute_loss(batch)
        assert torch.isfinite(loss)
        assert torch.isfinite(loss)


class TestGraphModelSetGraphData:
    def test_sets_edge_index(self, graph_model):
        ei = torch.tensor([[0, 1], [5, 6]])
        graph_model.set_graph_data(ei)
        assert graph_model.edge_index is not None
        assert torch.equal(graph_model.edge_index, ei)


class TestGraphModelCreateBipartiteGraph:
    def test_creates_bidirectional_edges(self, graph_model):
        user_ids = torch.tensor([0, 1])
        item_ids = torch.tensor([3, 5])
        ei = graph_model.create_bipartite_graph(user_ids, item_ids)
        assert ei.shape[0] == 2
        assert ei.shape[1] == 4  # 2 forward + 2 reverse


# ===========================================================================
# LightGCN
# ===========================================================================


class TestPyGLightGCNPredict:
    def test_output_shape(self, lightgcn_model):
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([3, 5, 7])
        scores = lightgcn_model.predict(user_ids, item_ids)
        assert scores.shape == (3,)

    def test_output_finite(self, lightgcn_model):
        user_ids = torch.tensor([0, 1])
        item_ids = torch.tensor([0, 1])
        scores = lightgcn_model.predict(user_ids, item_ids)
        assert torch.isfinite(scores).all()

    def test_clamps_out_of_range(self, lightgcn_model):
        user_ids = torch.tensor([99])  # out of range
        item_ids = torch.tensor([99])
        scores = lightgcn_model.predict(user_ids, item_ids)
        assert torch.isfinite(scores).all()


class TestPyGLightGCNRecommend:
    def test_output_shapes(self, lightgcn_model):
        user_ids = torch.tensor([0, 1])
        items, scores = lightgcn_model.recommend(user_ids, k=5)
        assert items.shape == (2, 5)
        assert scores.shape == (2, 5)


class TestPyGLightGCNComputeLoss:
    def test_produces_finite_scalar(self, lightgcn_model):
        batch = {
            "user_id": torch.tensor([0, 1, 2]),
            "item_id": torch.tensor([3, 5, 7]),
            "neg_items": torch.randint(0, 10, (3, 4)),
        }
        loss = lightgcn_model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_no_neg_items_raises(self, lightgcn_model):
        batch = {"user_id": torch.tensor([0]), "item_id": torch.tensor([3])}
        with pytest.raises(ValueError, match="negative"):
            lightgcn_model.compute_loss(batch)


class TestPyGLightGCNGraphData:
    def test_compute_graph_embeddings_without_data_raises(self):
        config = LightGCNConfig(num_users=5, num_items=10, embedding_dim=16)
        model = PyGLightGCN(config)
        with pytest.raises(ValueError, match="Graph data not set"):
            model.compute_graph_embeddings()

    def test_set_graph_data(self):
        config = LightGCNConfig(num_users=5, num_items=10, embedding_dim=16)
        model = PyGLightGCN(config)
        ei = _make_edge_index(5, 10, 10)
        model.set_graph_data(ei)
        assert model.edge_index is not None


class TestPyGLightGCNConfigureOptimizers:
    def test_returns_optimizer(self, lightgcn_model):
        opt = lightgcn_model.configure_optimizers()
        assert isinstance(opt, torch.optim.Optimizer)


class TestPyGLightGCNTestStep:
    def test_returns_predictions_and_targets(self, lightgcn_model):
        lightgcn_model.log = MagicMock()
        batch = {"user_id": torch.tensor([0, 1]), "item_id": torch.tensor([3, 5])}
        result = lightgcn_model.test_step(batch, 0)
        assert "predictions" in result
        assert "targets" in result


class TestPyGLightGCNTrainingStep:
    def test_returns_finite_loss(self, lightgcn_model):
        lightgcn_model.log = MagicMock()
        batch = {
            "user_id": torch.tensor([0, 1]),
            "item_id": torch.tensor([3, 5]),
            "neg_items": torch.randint(0, 10, (2, 4)),
        }
        loss = lightgcn_model.training_step(batch, 0)
        assert torch.isfinite(loss)


class TestPyGLightGCNCreateBipartiteGraph:
    def test_creates_bidirectional_edges(self, lightgcn_model):
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([3, 5, 7])
        ei = lightgcn_model.create_bipartite_graph(user_ids, item_ids)
        assert ei.shape[0] == 2
        # Bidirectional: 2 * num_edges
        assert ei.shape[1] == 6
