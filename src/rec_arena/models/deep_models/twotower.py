"""Two-Tower recommendation model."""
import torch
import torch.nn as nn
from ..deep import DeepModel
from ...configs.defaults.twotower import TwoTowerConfig


class TwoTower(DeepModel):
    """TwoTower: Dual Encoder Architecture for Retrieval.
    
    A dual encoder model with separate neural networks (towers) for users and items.
    Similarity is computed via dot product in a shared embedding space.
    
    Paper: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (RecSys 2019)
    Link: https://research.google/pubs/pub48840/
    
    Model ID: twotower
    Model Type: Implicit Feedback
    
    Key Features:
        - Separate user and item encoders
        - Shared embedding space
        - Efficient for large-scale retrieval
        - Configurable tower architectures
    
    Args:
        config (TwoTowerConfig): Model configuration with tower parameters
    
    Example:
        >>> config = TwoTowerConfig(num_users=1000, num_items=500, embedding_dim=64)
        >>> model = TwoTower(config)
        >>> scores = model.forward(user_ids, item_ids)
    """

    def __init__(self, config: TwoTowerConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # User tower
        if self.user_embedding_config["type"] == "standard":
            self.user_embedding = nn.Embedding(self.config.num_users, self.config.embedding_dim)
        else:
            from ...modules.layer_utils.embedding_factory import create_embedding
            self.user_embedding = create_embedding(
                embedding_type=self.user_embedding_config["type"],
                num_embeddings=self.config.num_users,
                embedding_dim=self.config.embedding_dim,
                **self.user_embedding_config.get("kwargs", {})
            )
        
        self.user_tower = self._build_tower(
            self.config.embedding_dim,
            self.config.user_tower_dims
        )

        # Item tower
        if self.item_embedding_config["type"] == "standard":
            self.item_embedding = nn.Embedding(self.config.num_items, self.config.embedding_dim)
        else:
            from ...modules.layer_utils.embedding_factory import create_embedding
            self.item_embedding = create_embedding(
                embedding_type=self.item_embedding_config["type"],
                num_embeddings=self.config.num_items,
                embedding_dim=self.config.embedding_dim,
                **self.item_embedding_config.get("kwargs", {})
            )
        
        self.item_tower = self._build_tower(
            self.config.embedding_dim,
            self.config.item_tower_dims
        )

        self._init_weights()

    def _build_tower(self, input_dim: int, hidden_dims: list) -> nn.Module:
        """Build a tower (MLP) with specified dimensions."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(self.config.dropout_rate))
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _get_activation(self):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }
        return activations[self.config.activation]

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=self.config.init_std)
        nn.init.normal_(self.item_embedding.weight, std=self.config.init_std)

        for tower in [self.user_tower, self.item_tower]:
            for layer in tower:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def encode_user(self, user_ids):
        """Encode users into embedding space."""
        user_emb = self.user_embedding(user_ids)
        return self.user_tower(user_emb)

    def encode_item(self, item_ids):
        """Encode items into embedding space."""
        item_emb = self.item_embedding(item_ids)
        return self.item_tower(item_emb)

    def forward(self, user_ids, item_ids):
        """Compute similarity scores for user-item pairs."""
        user_vec = self.encode_user(user_ids)
        item_vec = self.encode_item(item_ids)
        return (user_vec * item_vec).sum(dim=-1)

    def predict(self, user_ids, item_ids):
        """Predict scores for user-item pairs."""
        return torch.sigmoid(self.forward(user_ids, item_ids))

    def recommend(self, user_ids, k=10):
        """Generate top-k recommendations for users."""
        batch_size = len(user_ids)
        user_vecs = self.encode_user(user_ids)

        # Encode all items
        all_items = torch.arange(self.config.num_items, device=user_ids.device)
        all_item_vecs = self.encode_item(all_items)

        # Compute scores
        scores = torch.matmul(user_vecs, all_item_vecs.t())

        # Get top-k
        top_scores, top_items = torch.topk(scores, k, dim=-1)
        return top_items, top_scores

    def get_hidden_states(self, user_ids, item_ids):
        """Get concatenated user-item representations for loss computation."""
        user_vec = self.encode_user(user_ids)
        item_vec = self.encode_item(item_ids)
        return torch.cat([user_vec, item_vec], dim=-1)

    def prediction(self, hidden_states):
        """Compute dot product from concatenated hidden states."""
        # Split concatenated user-item vectors
        dim = hidden_states.size(-1) // 2
        user_vec = hidden_states[..., :dim]
        item_vec = hidden_states[..., dim:]
        return (user_vec * item_vec).sum(dim=-1, keepdim=True)

    def compute_loss(self, batch):
        """Compute loss using implicit loss functions."""
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        
        hidden_states = self.get_hidden_states(user_ids, item_ids)
        return self.loss_fn(self, batch, hidden_states)
