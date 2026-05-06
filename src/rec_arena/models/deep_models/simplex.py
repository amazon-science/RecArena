"""SimpleX: Simple but Effective model for implicit feedback."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..deep import DeepModel
from ...configs.defaults.simplex import SimpleXConfig


class SimpleX(DeepModel):
    """SimpleX: A Simple and Strong Baseline for Collaborative Filtering.
    
    A simple yet effective model that uses cosine similarity between user and item
    embeddings with history aggregation.
    
    Paper: "SimpleX: A Simple and Strong Baseline for Collaborative Filtering" (CIKM 2021)
    Link: https://arxiv.org/abs/2109.12613
    
    Model ID: simplex
    Model Type: Implicit Feedback
    
    Key Features:
        - Cosine similarity for user-item matching
        - Normalized embeddings
        - History aggregation (mean/sum/max)
        - Simple yet effective
    
    Args:
        config (SimpleXConfig): Model configuration with embedding parameters
    
    Example:
        >>> config = SimpleXConfig(num_users=1000, num_items=500, embedding_dim=64)
        >>> model = SimpleX(config)
        >>> scores = model.forward(user_ids, item_ids)
    """

    def __init__(self, config: SimpleXConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # Embeddings
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.embedding_dim)
        self.item_embedding = nn.Embedding(self.config.num_items, self.config.embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def aggregate_history(self, item_ids):
        """Aggregate user's historical items."""
        item_embs = self.item_embedding(item_ids)
        
        if self.config.history_aggregation == "mean":
            return item_embs.mean(dim=1)
        elif self.config.history_aggregation == "sum":
            return item_embs.sum(dim=1)
        else:  # max
            return item_embs.max(dim=1)[0]

    def forward(self, user_ids, item_ids):
        """Compute cosine similarity scores."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Cosine similarity
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        
        return (user_emb * item_emb).sum(dim=-1)

    def predict(self, user_ids, item_ids):
        """Predict scores for user-item pairs."""
        return torch.sigmoid(self.forward(user_ids, item_ids))

    def recommend(self, user_ids, k=10):
        """Generate top-k recommendations."""
        batch_size = len(user_ids)
        user_embs = F.normalize(self.user_embedding(user_ids), p=2, dim=-1)
        
        # Normalize all item embeddings
        all_item_embs = F.normalize(self.item_embedding.weight, p=2, dim=-1)
        
        # Cosine similarity
        scores = torch.matmul(user_embs, all_item_embs.t())
        
        top_scores, top_items = torch.topk(scores, k, dim=-1)
        return top_items, top_scores

    def get_hidden_states(self, user_ids, item_ids):
        """Unified interface: returns concatenated normalized embeddings."""
        user_emb = F.normalize(self.user_embedding(user_ids), p=2, dim=-1)
        item_emb = F.normalize(self.item_embedding(item_ids), p=2, dim=-1)
        return torch.cat([user_emb, item_emb], dim=-1)

    def prediction(self, hidden_states):
        """Unified prediction interface: computes cosine similarity."""
        dim = hidden_states.size(-1) // 2
        user_emb = hidden_states[..., :dim]
        item_emb = hidden_states[..., dim:]
        return (user_emb * item_emb).sum(dim=-1, keepdim=True)

    def compute_loss(self, batch):
        """Compute loss using unified loss function interface."""
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        
        hidden_states = self.get_hidden_states(user_ids, item_ids)
        return self.loss_fn(self, batch, hidden_states)
