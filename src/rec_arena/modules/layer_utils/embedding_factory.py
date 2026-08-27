"""Factory + protocol for item/user embedding layers.

All models in RecArena depend on a small, uniform embedding contract so that any
embedding implementation (plain, hierarchical+LoRA, quantized, or a future
TorchRec-backed table) can be swapped in without touching model code:

    emb(ids)          -> Tensor [*, embedding_dim]      (dense lookup)
    emb.weight        -> Tensor [num_embeddings, dim]   (tied-output / full table)
    emb.num_embeddings, emb.embedding_dim               (int dims)

Models must read the full table via ``get_output_embeddings()`` (defined on the
base sequential model) rather than touching ``.weight`` directly, so the output
projection stays implementation-agnostic. ``nn.Embedding`` and
``HierarchicalLoRAEmbedding`` already satisfy this contract.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from .embeddings import HierarchicalLoRAEmbedding


@runtime_checkable
class ItemEmbedding(Protocol):
    """Structural contract every embedding layer must satisfy.

    This is a typing Protocol (duck-typed): any module exposing these members
    works, no subclassing required. ``nn.Embedding`` satisfies it natively.
    """

    num_embeddings: int
    embedding_dim: int

    def __call__(self, ids: torch.Tensor) -> torch.Tensor: ...

    @property
    def weight(self) -> torch.Tensor: ...


def create_embedding(
    embedding_type: str,
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int = None,
    sparse: bool = False,
    **kwargs,
) -> nn.Module:
    """Create an embedding layer satisfying the ItemEmbedding protocol.

    Registered types:
        - "standard":          torch.nn.Embedding
        - "hierarchical_lora":  HierarchicalLoRAEmbedding (parent + per-item LoRA)

    Args:
        sparse: if True, the table produces sparse gradients (lookup-only paths)
            for use with ``torch.optim.SparseAdam``. Only honored by the
            "standard" type; hierarchical_lora rebuilds the table densely and
            ignores the flag.

    To add a new backend (e.g. quantized or TorchRec-backed), implement the
    ItemEmbedding protocol and add a branch here -- no model changes needed.
    """
    if embedding_type == "standard":
        return nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            sparse=sparse,
            scale_grad_by_freq=kwargs.get("scale_grad_by_freq", False),
        )

    elif embedding_type == "hierarchical_lora":
        return HierarchicalLoRAEmbedding(
            num_items=num_embeddings,
            num_parents=kwargs["num_parents"],
            embedding_dim=embedding_dim,
            item_to_parent_mapping=kwargs["item_to_parent_mapping"],
            lora_rank=kwargs.get("lora_rank", 16),
            scale_grad_by_freq=kwargs.get("scale_grad_by_freq", False),
        )

    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
