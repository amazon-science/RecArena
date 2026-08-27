import torch
import torch.nn as nn
import lightning as pl
from typing import Dict, Any, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from .base import BaseModel


class GraphModel(BaseModel, pl.LightningModule):
    """Base class for graph-based recommendation models using PyTorch Lightning."""

    def __init__(self, config):
        BaseModel.__init__(self, config)
        pl.LightningModule.__init__(self)

        self.num_users = getattr(config, "num_users", None)
        self.num_items = getattr(config, "num_items", None)
        self.embedding_dim = getattr(config, "embedding_dim", 64)
        self.lr = getattr(config, "lr", 1e-3)
        self.gradient_clip_val = getattr(config, "gradient_clip_val", 1.0)
        
        # Graph-specific parameters
        self.num_layers = getattr(config, "num_layers", 3)
        self.dropout_rate = getattr(config, "dropout_rate", 0.1)
        
        # Graph structure (to be set by subclasses or data loading)
        self.edge_index: Optional[torch.Tensor] = None
        self.num_nodes = self.num_users + self.num_items
        
        # Optional metrics during validation
        self.compute_val_metrics = getattr(config, "compute_val_metrics", False)
        self.val_k_values = getattr(config, "val_k_values", [10])
        self._metric_calculator = None
        # Per-user TRAIN-history table [num_users, Lmax] of 3-indexed item ids
        # (pad 0), populated by the harness via set_eval_history. Used to mask a
        # user's already-seen items during VALIDATION so the monitored val NDCG
        # matches the reported (history-masked) test full-sort protocol.
        self._val_hist_ids = None

        # Set up loss function
        self.loss_fn = None
        loss_type = getattr(config, "loss_type", None)
        if loss_type:
            from ..losses.factory import get_loss_function
            self.loss_fn = get_loss_function(loss_type, model_type="implicit")

    def configure_optimizers(self):
        """Configure optimizer for Lightning."""
        weight_decay = getattr(self.config, "weight_decay", 1e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        loss = self.compute_loss(batch)
        batch_size = batch["user_id"].size(0)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def set_eval_history(self, hist_ids):
        """Register a per-user TRAIN-history table for validation masking.

        hist_ids: LongTensor [num_users, Lmax] of 3-indexed item ids (pad 0).
        Mirrors DeepModel/SequentialModel.set_eval_history so the harness wires
        all families identically. TRAIN items only -- never the val target.
        """
        self._val_hist_ids = hist_ids

    def validation_step(self, batch, batch_idx):
        """Validation step: masked full-sort ranking metrics (val_ndcg@10 etc.).

        Masks special-token columns (0/1/2) and each user's already-seen TRAIN
        history before ranking, mirroring the reported TEST full-sort protocol
        (core.py) so the monitored val NDCG selects the same epoch the benchmark
        would reward. Uses the shared MetricCalculator (tie-safe rank), not a
        raw argsort that assumed a unique, untied target.
        """
        with torch.no_grad():
            result = self.test_step(batch, batch_idx)
            predictions = result['predictions']
            targets = result['targets']
            batch_size = predictions.size(0)

            # Mask special tokens and already-seen TRAIN items (see docstring).
            if predictions.size(1) >= 3:
                predictions[:, :3] = float("-inf")
            if self._val_hist_ids is not None and "user_id" in batch:
                hist = self._val_hist_ids.to(predictions.device)[batch["user_id"]]
                hist = hist.clamp_(max=predictions.size(1) - 1)
                predictions.scatter_(1, hist, float("-inf"))
                predictions[:, 0] = float("-inf")  # undo PAD-col unmask (pad=0)

            if self._metric_calculator is None:
                from ..metrics import MetricCalculator

                self._metric_calculator = MetricCalculator(k_values=self.val_k_values)
            metrics = self._metric_calculator.calculate_all(
                predictions.detach().cpu(), targets.detach().cpu()
            )
            for metric_name, value in metrics.items():
                self.log(
                    f"val_{metric_name}",
                    float(value),
                    prog_bar=True,
                    batch_size=batch_size,
                )
            # Keep val_loss logged too (some monitors/schedulers may use it), as
            # -NDCG@10 so lower stays better on that key.
            ndcg10 = metrics.get("ndcg@10", 0.0)
            self.log("val_loss", -float(ndcg10), prog_bar=True, batch_size=batch_size)
            return -float(ndcg10)

    def set_graph_data(self, edge_index: torch.Tensor) -> None:
        """Set graph structure for the model."""
        self.edge_index = edge_index
    
    def create_bipartite_graph(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Create bipartite graph edge_index from user-item interactions."""
        # Users: 0 to num_users-1, Items: num_users to num_users+num_items-1
        item_nodes = item_ids + self.num_users
        edge_index = torch.stack([user_ids, item_nodes], dim=0)
        # Add reverse edges for undirected bipartite graph
        reverse_edge_index = torch.stack([item_nodes, user_ids], dim=0)
        return torch.cat([edge_index, reverse_edge_index], dim=1)

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch. To be implemented by subclasses."""
        raise NotImplementedError

    def fit(self, train_data, val_data=None) -> None:
        """Train using Lightning Trainer."""
        from lightning import Trainer
        from lightning.pytorch.callbacks import EarlyStopping

        callbacks = []
        if getattr(self.config, "early_stopping", True):
            early_stop = EarlyStopping(
                monitor="val_loss", patience=getattr(self.config, "patience", 10), mode="min"
            )
            callbacks.append(early_stop)

        trainer = Trainer(
            max_epochs=getattr(self.config, "max_epochs", 100),
            accelerator="auto",
            devices="auto",
            gradient_clip_val=self.gradient_clip_val,
            callbacks=callbacks,
        )
        trainer.fit(self, train_data, val_data)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        self.load_state_dict(torch.load(path, map_location="cpu"))