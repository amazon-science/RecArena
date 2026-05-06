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

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning - use ranking metrics instead of cross-entropy loss."""
        with torch.no_grad():
            # Use test_step for proper ranking evaluation
            result = self.test_step(batch, batch_idx)
            predictions = result['predictions']
            targets = result['targets']
            
            # Compute ranking-based validation metrics
            batch_size = predictions.size(0)
            
            # 1. Hit Rate@10 (most interpretable)
            _, top10_items = torch.topk(predictions, k=10, dim=1)
            hits = (top10_items == targets.unsqueeze(1)).any(dim=1).float()
            hit_rate = hits.mean()
            
            # 2. Reciprocal Rank (for validation loss - lower is better)
            sorted_indices = torch.argsort(predictions, dim=1, descending=True)
            target_ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
            reciprocal_ranks = 1.0 / target_ranks.float()
            mrr = reciprocal_ranks.mean()
            
            # Use negative MRR as validation loss (so lower is better for early stopping)
            val_loss = -mrr
            
            # Log multiple metrics
            self.log("val_loss", val_loss, prog_bar=True, batch_size=batch_size)  # For early stopping
            self.log("val_hit_rate@10", hit_rate, prog_bar=True, batch_size=batch_size)
            self.log("val_mrr", mrr, prog_bar=True, batch_size=batch_size)
            
            # Debug: Print metrics for first batch
            if batch_idx == 0:
                print(f"Val metrics: Hit@10={hit_rate:.4f}, MRR={mrr:.4f}, Loss={val_loss:.6f}")
            
            return val_loss

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