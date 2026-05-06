import torch
import torch.nn as nn
from ..deep import DeepModel
from ...configs.defaults.ncf import NCFConfig


class NCF(DeepModel):
    """NCF: Neural Collaborative Filtering.

    A neural network model that combines Generalized Matrix Factorization (GMF) and
    Multi-Layer Perceptron (MLP) for implicit feedback recommendation.

    Paper: "Neural Collaborative Filtering" (WWW 2017)
    Link: https://arxiv.org/abs/1708.05031

    Model ID: ncf
    Model Type: Implicit Feedback

    Key Features:
        - GMF component for linear user-item interactions
        - MLP component for non-linear interactions
        - Fusion of GMF and MLP outputs
        - Configurable MLP architecture

    Args:
        config (NCFConfig): Model configuration with GMF and MLP parameters

    Example:
        >>> config = NCFConfig(num_users=1000, num_items=500, embedding_dim=64)
        >>> model = NCF(config)
        >>> scores = model.forward(user_ids, item_ids)
    """

    def __init__(self, config: NCFConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # GMF (Generalized Matrix Factorization) embeddings
        self.user_embedding_mf = nn.Embedding(
            self.config.num_users, self.config.embedding_dim
        )
        self.item_embedding_mf = nn.Embedding(
            self.config.num_items, self.config.embedding_dim
        )

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(
            self.config.num_users, self.config.hidden_dims[0] // 2
        )
        self.item_embedding_mlp = nn.Embedding(
            self.config.num_items, self.config.hidden_dims[0] // 2
        )

        # MLP layers
        mlp_layers = []
        for i in range(len(self.config.hidden_dims) - 1):
            mlp_layers.append(
                nn.Linear(self.config.hidden_dims[i], self.config.hidden_dims[i + 1])
            )
            mlp_layers.append(self._get_activation())
            if self.config.use_batch_norm:
                mlp_layers.append(nn.BatchNorm1d(self.config.hidden_dims[i + 1]))
            mlp_layers.append(nn.Dropout(self.config.dropout_rate))

        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        self.prediction_layer = nn.Linear(
            self.config.embedding_dim + self.config.hidden_dims[-1], 1
        )

        # Initialize weights
        self._init_weights()

    def _get_activation(self):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations[self.config.activation]

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding_mf.weight, std=self.config.init_std)
        nn.init.normal_(self.item_embedding_mf.weight, std=self.config.init_std)
        nn.init.normal_(self.user_embedding_mlp.weight, std=self.config.init_std)
        nn.init.normal_(self.item_embedding_mlp.weight, std=self.config.init_std)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.prediction_layer.weight)
        nn.init.zeros_(self.prediction_layer.bias)

    def forward(self, user_ids, item_ids):
        """Forward pass through NCF."""
        # Get hidden states
        hidden_states = self.get_hidden_states(user_ids, item_ids)

        # Final prediction
        return self.prediction(hidden_states).squeeze(-1)

    def prediction(self, hidden_states):
        """Unified prediction interface: takes hidden states, returns scores."""
        return self.prediction_layer(hidden_states)

    def predict(self, user_ids, item_ids):
        """Predict ratings for user-item pairs."""
        return torch.sigmoid(self.forward(user_ids, item_ids))

    def recommend(self, user_ids, k=10):
        """Generate top-k recommendations for users."""
        batch_size = len(user_ids)

        # Create all user-item pairs for scoring
        all_items = torch.arange(self.config.num_items, device=user_ids.device)
        user_ids_expanded = user_ids.unsqueeze(1).repeat(1, self.config.num_items)
        all_items_expanded = all_items.unsqueeze(0).repeat(batch_size, 1)

        # Compute scores for all items
        scores = self.predict(user_ids_expanded.flatten(), all_items_expanded.flatten())
        scores = scores.view(batch_size, self.config.num_items)

        # Get top-k items
        top_scores, top_items = torch.topk(scores, k, dim=-1)

        return top_items, top_scores

    def get_hidden_states(self, user_ids, item_ids):
        """Get user and item embeddings - NCF's 'hidden states'."""
        # For NCF, hidden states are the concatenated user/item embeddings
        user_emb_mf = self.user_embedding_mf(user_ids)
        item_emb_mf = self.item_embedding_mf(item_ids)
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)

        # GMF part
        mf_output = user_emb_mf * item_emb_mf

        # MLP part
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate GMF and MLP outputs
        hidden_states = torch.cat([mf_output, mlp_output], dim=-1)

        return hidden_states

    # Removed redundant test_step - now uses base class DeepModel.test_step

    def compute_loss(self, batch):
        """Compute loss using self.loss_fn - provide hidden states."""
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]

        # Get hidden states (user-item representations)
        hidden_states = self.get_hidden_states(user_ids, item_ids)

        # Let implicit loss function handle the rest
        return self.loss_fn(self, batch, hidden_states)
