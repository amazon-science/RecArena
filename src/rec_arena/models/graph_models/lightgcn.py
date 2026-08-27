"""Clean PyTorch Geometric LightGCN implementation."""
import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN as TorchGeoLightGCN
from ..graph import GraphModel
from ...configs.defaults.lightgcn import LightGCNConfig


class PyGLightGCN(GraphModel):
    """Standard PyTorch Geometric LightGCN implementation."""
    
    def __init__(self, config: LightGCNConfig):
        super().__init__(config)
        self.save_hyperparameters()
        
        # Initialize PyG LightGCN
        self.num_nodes = self.num_users + self.num_items
        self.pyg_model = TorchGeoLightGCN(
            num_nodes=self.num_nodes,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            alpha=None  # Use default alpha (uniform weighting)
        )
        
    def compute_graph_embeddings(self):
        """Compute embeddings using PyG LightGCN."""
        if self.edge_index is None:
            raise ValueError("Graph data not set. Call set_graph_data() first.")
        
        # Ensure edge_index is on the same device as model parameters
        edge_index = self.edge_index.to(self.device)
        
        node_embeddings = self.pyg_model.get_embedding(edge_index)
        user_emb = node_embeddings[:self.num_users]
        item_emb = node_embeddings[self.num_users:]
        
        return user_emb, item_emb
    
    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Post-convolution user embeddings for `user_ids`.

        Lets the implicit benchmark adapter score users via a single
        user x all-items matmul (its fast full-sort path) instead of the
        per-pair predict() expansion.
        """
        user_emb, _ = self.compute_graph_embeddings()
        return user_emb[torch.clamp(user_ids, 0, self.num_users - 1)]

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Post-convolution item embeddings for `item_ids` (3-indexed == column)."""
        _, item_emb = self.compute_graph_embeddings()
        return item_emb[torch.clamp(item_ids, 0, self.num_items - 1)]

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict scores for user-item pairs."""
        user_emb, item_emb = self.compute_graph_embeddings()
        
        user_ids = torch.clamp(user_ids, 0, self.num_users - 1)
        item_ids = torch.clamp(item_ids, 0, self.num_items - 1)
        
        user_repr = user_emb[user_ids]
        item_repr = item_emb[item_ids]
        
        return torch.sum(user_repr * item_repr, dim=-1)
    
    def recommend(self, user_ids: torch.Tensor, k: int = 10) -> tuple:
        """Generate top-k recommendations."""
        user_emb, item_emb = self.compute_graph_embeddings()
        
        user_ids = torch.clamp(user_ids, 0, self.num_users - 1)
        user_repr = user_emb[user_ids]
        scores = torch.matmul(user_repr, item_emb.t())
        
        top_scores, top_items = torch.topk(scores, k, dim=-1)
        return top_items, top_scores
    
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute BPR loss with optimized graph computation."""
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        neg_items = batch.get("neg_items")
        
        if neg_items is None:
            raise ValueError("No negative items provided in batch")
        
        # Compute graph embeddings ONCE for the entire batch
        user_emb, item_emb = self.compute_graph_embeddings()
        
        # Get positive embeddings
        user_ids_clamped = torch.clamp(user_ids, 0, self.num_users - 1)
        item_ids_clamped = torch.clamp(item_ids, 0, self.num_items - 1)
        user_repr = user_emb[user_ids_clamped]
        pos_item_repr = item_emb[item_ids_clamped]
        
        # Get negative embeddings
        batch_size, num_neg = neg_items.size()
        neg_items_clamped = torch.clamp(neg_items, 0, self.num_items - 1)
        neg_item_repr = item_emb[neg_items_clamped.view(-1)].view(batch_size, num_neg, -1)
        
        # Compute BPR loss manually (avoiding redundant graph computation)
        pos_scores = torch.sum(user_repr * pos_item_repr, dim=-1)  # [batch_size]
        neg_scores = torch.sum(user_repr.unsqueeze(1) * neg_item_repr, dim=-1)  # [batch_size, num_neg]
        
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        score_diff = pos_scores.unsqueeze(1) - neg_scores  # [batch_size, num_neg]
        bpr_loss = -torch.log(torch.sigmoid(score_diff) + 1e-8).mean()
        
        # Regularization on base embeddings (before convolution)
        user_base_emb = self.pyg_model.embedding.weight[user_ids_clamped]
        pos_item_base_emb = self.pyg_model.embedding.weight[item_ids_clamped + self.num_users]
        neg_item_base_emb = self.pyg_model.embedding.weight[neg_items_clamped.view(-1) + self.num_users]
        

        # Regularization: L2 norm per sample, then sum
        user_reg = user_base_emb.norm(2, dim=1).pow(2).sum()
        pos_item_reg = pos_item_base_emb.norm(2, dim=1).pow(2).sum()
        neg_item_reg = neg_item_base_emb.norm(2, dim=1).pow(2).sum()
        
        # Total number of embeddings: batch_size users + batch_size pos items + (batch_size * num_neg) neg items
        total_embeddings = batch_size + batch_size + batch_size * num_neg
        
        weight_decay = getattr(self.hparams.config, "weight_decay", 1e-4)
        reg_loss = weight_decay * (user_reg + pos_item_reg + neg_item_reg) / (2 * total_embeddings)
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        """Test step for evaluation."""
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        
        batch_size = user_ids.size(0)
        all_items = torch.arange(self.num_items, device=user_ids.device)
        user_expanded = user_ids.unsqueeze(1).expand(-1, self.num_items)
        item_expanded = all_items.unsqueeze(0).expand(batch_size, -1)
        
        predictions = self.predict(user_expanded.flatten(), item_expanded.flatten())
        predictions = predictions.view(batch_size, self.num_items)
        targets = item_ids
        
        return {'predictions': predictions, 'targets': targets}
