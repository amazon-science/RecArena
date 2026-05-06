"""Clustered Cross-Entropy Loss for scalable sequential recommendation.

Partitions items into K clusters and computes separate CE losses per cluster.
Includes cross-cluster negatives and anchor items for proper calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class ClusteredCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with clustered softmax + cross-cluster negatives.

    Args:
        cluster_item_ids: List of tensors, each containing item IDs for a cluster
        vocab_size: Total vocabulary size
        num_cross_negatives: Number of negatives to sample from OTHER clusters
        anchor_items: Items that appear in ALL clusters' softmax (for calibration)
    """

    def __init__(
        self,
        cluster_item_ids: List[torch.Tensor],
        vocab_size: int,
        num_cross_negatives: int = 64,
        anchor_items: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_clusters = len(cluster_item_ids)
        self.vocab_size = vocab_size
        self.num_cross_negatives = num_cross_negatives
        self._logged = False

        # Register cluster item IDs as buffers
        for i, items in enumerate(cluster_item_ids):
            self.register_buffer(f"cluster_{i}_items", items.long())

        # Anchor items (shared across all clusters)
        if anchor_items is not None:
            self.register_buffer("anchor_items", anchor_items.long())
        else:
            self.anchor_items = None

        # Create reverse mapping: global_item_id -> (cluster_id, local_id)
        item_to_cluster = torch.full((vocab_size,), -1, dtype=torch.long)
        item_to_local = torch.full((vocab_size,), -1, dtype=torch.long)

        for cluster_id, items in enumerate(cluster_item_ids):
            for local_id, global_id in enumerate(items):
                item_to_cluster[global_id] = cluster_id
                item_to_local[global_id] = local_id

        self.register_buffer("item_to_cluster", item_to_cluster)
        self.register_buffer("item_to_local", item_to_local)

        # Precompute items NOT in each cluster (for cross-cluster sampling)
        all_items = set(range(vocab_size))
        for i, items in enumerate(cluster_item_ids):
            other_items = list(all_items - set(items.tolist()))
            self.register_buffer(f"other_{i}_items", torch.tensor(other_items, dtype=torch.long))

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        neg_items: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute clustered CE loss with cross-cluster negatives."""
        # Validation path
        if logits is not None:
            mask_bool = mask.bool() if mask is not None else torch.ones_like(targets, dtype=torch.bool)
            return F.cross_entropy(logits[mask_bool], targets[mask_bool], reduction="mean")

        device = hidden_states.device
        batch_size, seq_len, dim = hidden_states.shape

        # Flatten
        hidden_flat = hidden_states.reshape(-1, dim)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1).bool()

        # Apply mask early - only process valid positions
        valid_hidden = hidden_flat[mask_flat]  # [N_valid, D]
        valid_targets = targets_flat[mask_flat]  # [N_valid]
        
        if valid_hidden.size(0) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Get cluster assignments for valid targets only
        target_clusters = self.item_to_cluster[valid_targets]  # [N_valid]

        # Vectorized: process all clusters in parallel using scatter
        losses = []
        counts = []
        
        for cluster_id in range(self.num_clusters):
            cluster_mask = target_clusters == cluster_id
            if not cluster_mask.any():
                continue

            cluster_hidden = valid_hidden[cluster_mask]
            cluster_targets_global = valid_targets[cluster_mask]
            n_samples = cluster_hidden.size(0)

            # Get cluster items + anchors + cross-negatives
            cluster_items = getattr(self, f"cluster_{cluster_id}_items")
            
            # Sample cross-cluster negatives (vectorized)
            if self.num_cross_negatives > 0:
                other_items = getattr(self, f"other_{cluster_id}_items")
                num_neg = min(self.num_cross_negatives, len(other_items))
                neg_idx = torch.randint(0, len(other_items), (num_neg,), device=device)
                cross_neg = other_items[neg_idx]
            else:
                cross_neg = torch.tensor([], dtype=torch.long, device=device)

            # Combine: cluster + anchors + cross-neg
            parts = [cluster_items]
            if self.anchor_items is not None:
                parts.append(self.anchor_items)
            if cross_neg.numel() > 0:
                parts.append(cross_neg)
            combined_items = torch.unique(torch.cat(parts))

            # Compute logits
            combined_emb = item_embeddings[combined_items]
            cluster_logits = cluster_hidden @ combined_emb.T  # [n_samples, combined_size]

            # Find target indices in combined_items using searchsorted (faster than broadcasting)
            sorted_combined, sort_idx = combined_items.sort()
            target_pos_in_sorted = torch.searchsorted(sorted_combined, cluster_targets_global)
            combined_targets = sort_idx[target_pos_in_sorted]

            # CE loss
            loss = F.cross_entropy(cluster_logits, combined_targets, reduction="sum")
            losses.append(loss)
            counts.append(n_samples)

            if not self._logged:
                print(f"    [ClusteredCE] Cluster {cluster_id}: {n_samples} targets, {len(combined_items)} items in softmax")

        if not self._logged and counts:
            print(f"    [ClusteredCE] Total: {sum(counts)} targets across {self.num_clusters} clusters")
            self._logged = True

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).sum() / sum(counts)


def create_item_clusters(
    train_df,
    num_clusters: int = 4,
    strategy: str = "popularity",
    num_anchors: int = 100,
) -> tuple[List[torch.Tensor], torch.Tensor]:
    """Create item clusters and anchor items from training data.

    Args:
        train_df: DataFrame with 'item_id' column
        num_clusters: Number of clusters to create
        strategy: "popularity" (sort by frequency) or "random"
        num_anchors: Number of most popular items to use as anchors

    Returns:
        Tuple of (cluster_item_lists, anchor_items)
    """
    item_counts = train_df["item_id"].value_counts()
    items_sorted = item_counts.index.tolist()

    # Anchor items = most popular (appear in all clusters)
    anchor_items = torch.tensor(items_sorted[:num_anchors], dtype=torch.long)

    # Remaining items split into clusters
    remaining_items = items_sorted[num_anchors:]

    if strategy != "popularity":
        np.random.shuffle(remaining_items)

    cluster_size = len(remaining_items) // num_clusters
    clusters = []

    for i in range(num_clusters):
        start = i * cluster_size
        end = None if i == num_clusters - 1 else (i + 1) * cluster_size
        cluster_items = remaining_items[start:end]
        clusters.append(torch.tensor(cluster_items, dtype=torch.long))

    return clusters, anchor_items
