"""Semantic ID tokenizer for learning meaningful item representations."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
from sklearn.cluster import KMeans


class SemanticIDTokenizer(nn.Module):
    """Semantic ID tokenizer that learns meaningful item IDs.
    
    Instead of random IDs, items with similar features get similar IDs,
    improving generalization and cold-start performance.
    
    Methods:
        - Clustering-based: Group similar items via k-means
        - Embedding-based: Learn semantic codes via VQ-VAE
        - Hierarchical: Multi-level semantic codes
    
    Args:
        num_items: Total number of items
        num_codes: Number of semantic codes (vocabulary size)
        method: 'kmeans', 'vqvae', or 'hierarchical'
        feature_dim: Dimension of item features (if available)
    """

    def __init__(
        self,
        num_items: int,
        num_codes: int = 1024,
        method: str = "kmeans",
        feature_dim: int = 64,
        num_levels: int = 2,  # For hierarchical
    ):
        super().__init__()
        self.num_items = num_items
        self.num_codes = num_codes
        self.method = method
        self.feature_dim = feature_dim
        self.num_levels = num_levels

        # Item to semantic ID mapping
        self.register_buffer(
            "item_to_semantic_id", torch.arange(num_items, dtype=torch.long)
        )

        if method == "vqvae":
            # Learnable codebook for VQ-VAE
            self.codebook = nn.Embedding(num_codes, feature_dim)
            nn.init.normal_(self.codebook.weight, std=0.02)
        elif method == "hierarchical":
            # Multi-level codebooks
            codes_per_level = int(num_codes ** (1 / num_levels))
            self.codebooks = nn.ModuleList(
                [
                    nn.Embedding(codes_per_level, feature_dim)
                    for _ in range(num_levels)
                ]
            )
            for codebook in self.codebooks:
                nn.init.normal_(codebook.weight, std=0.02)

    def fit_kmeans(self, item_features: np.ndarray):
        """Fit k-means clustering on item features.
        
        Args:
            item_features: [num_items, feature_dim] numpy array
        """
        if self.method != "kmeans":
            raise ValueError("fit_kmeans only works with method='kmeans'")

        kmeans = KMeans(n_clusters=self.num_codes, random_state=42, n_init=10)
        semantic_ids = kmeans.fit_predict(item_features)
        self.item_to_semantic_id = torch.tensor(semantic_ids, dtype=torch.long)

    def fit_from_embeddings(self, item_embeddings: torch.Tensor):
        """Fit semantic IDs from learned embeddings (e.g., from pretrained model).
        
        Args:
            item_embeddings: [num_items, embedding_dim] tensor
        """
        if self.method != "kmeans":
            raise ValueError("fit_from_embeddings only works with method='kmeans'")

        item_features = item_embeddings.detach().cpu().numpy()
        self.fit_kmeans(item_features)

    def encode_vqvae(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode items using VQ-VAE quantization.
        
        Args:
            item_embeddings: [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
        
        Returns:
            semantic_ids: [batch_size] or [batch_size, seq_len]
        """
        original_shape = item_embeddings.shape
        if item_embeddings.dim() == 3:
            batch_size, seq_len, _ = item_embeddings.shape
            item_embeddings = item_embeddings.view(-1, self.feature_dim)
        
        # Compute distances to codebook
        distances = torch.cdist(item_embeddings, self.codebook.weight)
        semantic_ids = distances.argmin(dim=-1)
        
        if len(original_shape) == 3:
            semantic_ids = semantic_ids.view(batch_size, seq_len)
        
        return semantic_ids

    def encode_hierarchical(self, item_embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Encode items using hierarchical codes.
        
        Args:
            item_embeddings: [batch_size, feature_dim]
        
        Returns:
            List of semantic_ids for each level
        """
        semantic_ids = []
        for codebook in self.codebooks:
            distances = torch.cdist(item_embeddings, codebook.weight)
            level_ids = distances.argmin(dim=-1)
            semantic_ids.append(level_ids)
        return semantic_ids

    def decode_vqvae(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """Decode semantic IDs back to embeddings.
        
        Args:
            semantic_ids: [batch_size] or [batch_size, seq_len]
        
        Returns:
            embeddings: [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
        """
        return self.codebook(semantic_ids)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Convert item IDs to semantic IDs.
        
        Args:
            item_ids: [batch_size] or [batch_size, seq_len]
        
        Returns:
            semantic_ids: [batch_size] or [batch_size, seq_len]
        """
        # Clamp to valid range
        item_ids = torch.clamp(item_ids, 0, self.num_items - 1)
        return self.item_to_semantic_id[item_ids]

    def get_semantic_embedding(
        self, item_ids: torch.Tensor, embedding_layer: nn.Embedding
    ) -> torch.Tensor:
        """Get embeddings using semantic IDs.
        
        Args:
            item_ids: [batch_size] or [batch_size, seq_len]
            embedding_layer: Embedding layer to use
        
        Returns:
            embeddings: [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        semantic_ids = self.forward(item_ids)
        return embedding_layer(semantic_ids)


class SemanticIDModel(nn.Module):
    """Wrapper to add semantic ID support to any sequential model.
    
    Example:
        >>> base_model = SASRec(config)
        >>> semantic_model = SemanticIDModel(base_model, num_items=1000, num_codes=512)
        >>> # Train with semantic IDs
        >>> logits = semantic_model(sequences, sequence_lengths)
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_items: int,
        num_codes: int = 1024,
        method: str = "kmeans",
    ):
        super().__init__()
        self.base_model = base_model
        self.semantic_tokenizer = SemanticIDTokenizer(
            num_items=num_items, num_codes=num_codes, method=method
        )

        # Replace base model's item embedding with semantic embedding
        if hasattr(base_model, "item_embedding"):
            embedding_dim = base_model.item_embedding.embedding_dim
            self.base_model.item_embedding = nn.Embedding(
                num_codes, embedding_dim, padding_idx=0
            )

    def forward(self, sequences, sequence_lengths):
        """Forward pass with semantic IDs."""
        # Convert to semantic IDs
        semantic_sequences = self.semantic_tokenizer(sequences)
        return self.base_model.forward(semantic_sequences, sequence_lengths)

    def fit_semantic_ids(self, item_features: np.ndarray):
        """Fit semantic IDs from item features."""
        self.semantic_tokenizer.fit_kmeans(item_features)

    def __getattr__(self, name):
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
