import torch
import torch.nn as nn
import numpy as np
from ..deep import DeepModel
from ...configs.defaults.bprmf import BPRMFConfig


class BPRMF(DeepModel):
    """BPR-MF: Bayesian Personalized Ranking Matrix Factorization.

    A classic matrix factorization model optimized with Bayesian Personalized Ranking (BPR)
    for implicit feedback recommendation.

    Paper: "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
    Link: https://arxiv.org/abs/1205.2618

    Model ID: bprmf
    Model Type: Implicit Feedback

    Key Features:
        - Matrix factorization with user/item embeddings
        - Pairwise ranking optimization
        - Simple and interpretable
        - Efficient for large-scale data

    Args:
        config (BPRMFConfig): Model configuration with embedding parameters

    Example:
        >>> config = BPRMFConfig(num_users=1000, num_items=500, embedding_dim=64)
        >>> model = BPRMF(config)
        >>> scores = model.forward(user_ids, item_ids)
    """

    def __init__(self, config: BPRMFConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # User and item embeddings
        self.user_embedding = nn.Embedding(
            self.config.num_users, self.config.embedding_dim
        )
        self.item_embedding = nn.Embedding(
            self.config.num_items, self.config.embedding_dim
        )

        # Initialize embeddings with Xavier/Glorot initialization for better convergence
        # Xavier initialization: std = sqrt(1/n) where n is embedding_dim
        # This is much more stable than config.init_std (0.1) which is too large
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, user_ids, item_ids):
        """Forward pass for user-item pairs."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        return (user_emb * item_emb).sum(dim=-1)

    def predict(self, user_ids, item_ids):
        """Predict ratings for user-item pairs."""
        return self.forward(user_ids, item_ids)

    def recommend(self, user_ids, k=10):
        """Generate top-k recommendations for users."""
        batch_size = len(user_ids)

        # Get user embeddings
        user_embs = self.user_embedding(user_ids)  # [batch_size, emb_dim]

        # Get all item embeddings
        all_item_embs = self.item_embedding.weight  # [num_items, emb_dim]

        # Compute scores for all items
        scores = torch.matmul(user_embs, all_item_embs.t())  # [batch_size, num_items]

        # Get top-k items
        top_scores, top_items = torch.topk(scores, k, dim=-1)

        return top_items, top_scores

    def get_user_embedding(self, user_ids):
        """Get user embeddings."""
        return self.user_embedding(user_ids)

    def get_item_embedding(self, item_ids):
        """Get item embeddings."""
        return self.item_embedding(item_ids)

    def get_hidden_states(self, user_ids, item_ids):
        """Unified interface: returns concatenated user-item embeddings."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return torch.cat([user_emb, item_emb], dim=-1)

    def prediction(self, hidden_states):
        """Unified prediction interface: computes dot product from concatenated embeddings."""
        dim = hidden_states.size(-1) // 2
        user_emb = hidden_states[..., :dim]
        item_emb = hidden_states[..., dim:]
        return (user_emb * item_emb).sum(dim=-1, keepdim=True)

    def compute_loss(self, batch):
        """Compute loss using unified loss function interface."""
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]

        # Get hidden states for unified interface
        hidden_states = self.get_hidden_states(user_ids, item_ids)

        # Use unified loss function
        return self.loss_fn(self, batch, hidden_states)
