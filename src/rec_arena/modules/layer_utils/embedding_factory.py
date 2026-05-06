"""Factory for creating different embedding types."""

import torch
import torch.nn as nn
from .embeddings import HierarchicalLoRAEmbedding


def create_embedding(
    embedding_type: str,
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int = None,
    **kwargs
) -> nn.Module:
    """Create embedding layer based on type."""
    if embedding_type == "standard":
        return nn.Embedding(
            num_embeddings, 
            embedding_dim, 
            padding_idx=padding_idx,
            scale_grad_by_freq=kwargs.get("scale_grad_by_freq", False)
        )
    
    elif embedding_type == "hierarchical_lora":
        return HierarchicalLoRAEmbedding(
            num_items=num_embeddings,
            num_parents=kwargs["num_parents"],
            embedding_dim=embedding_dim,
            item_to_parent_mapping=kwargs["item_to_parent_mapping"],
            lora_rank=kwargs.get("lora_rank", 16),
            scale_grad_by_freq=kwargs.get("scale_grad_by_freq", False)
        )
    
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")