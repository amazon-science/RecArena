"""Dynamic negative sampling collate functions for all model types."""

import torch
import numpy as np
from typing import List, Dict, Set

# Default offset for item indices (PAD=0, UNK=1, MASK=2, items start at 3)
DEFAULT_ITEM_OFFSET = 3


class SequentialNegativeSamplingCollate:
    """Collate function with dynamic per-batch negative sampling for sequential models."""

    def __init__(
        self,
        num_items: int,
        num_negatives: int,
        user_histories: Dict[int, Set[int]] = None,
        sampler=None,
        item_offset: int = DEFAULT_ITEM_OFFSET,
    ):
        self.num_items = num_items
        self.num_negatives = (
            sampler.num_negatives if sampler is not None else num_negatives
        )
        self.user_histories = user_histories or {}
        self.sampler = sampler
        self.item_offset = item_offset
        self.rng = np.random.default_rng()
        # Items are in range [item_offset, num_items + item_offset - 1]
        self.all_items = np.arange(item_offset, num_items + item_offset)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic per-position negative sampling."""
        sequences = torch.stack([item["sequence"] for item in batch])
        sequence_lengths = torch.stack([item["sequence_length"] for item in batch])
        user_ids = torch.stack([item["user_id"] for item in batch])

        batch_size, seq_len = sequences.shape
        neg_items = torch.zeros(batch_size, seq_len, self.num_negatives, dtype=torch.long)

        if self.sampler is not None:
            for i, user_id in enumerate(user_ids.tolist()):
                user_positives = self.user_histories.get(user_id, set())
                total_samples = seq_len * self.num_negatives
                sampled_flat = self.sampler.sample(user_positives, total_samples)
                neg_items[i] = torch.from_numpy(sampled_flat.reshape(seq_len, self.num_negatives))
        else:
            # Lazy per-batch candidate computation (faster than pre-computing all users)
            for i, user_id in enumerate(user_ids.tolist()):
                user_positives = self.user_histories.get(user_id, set())
                # Fast set difference using numpy
                candidates = np.setdiff1d(self.all_items, list(user_positives), assume_unique=True)
                
                if len(candidates) >= self.num_negatives:
                    # Sample all positions at once
                    total_samples = seq_len * self.num_negatives
                    sampled_flat = self.rng.choice(candidates, size=total_samples, replace=True)
                    neg_items[i] = torch.from_numpy(sampled_flat.reshape(seq_len, self.num_negatives))
                elif len(candidates) > 0:
                    # Fallback for users with few candidates
                    for pos in range(seq_len):
                        sampled = self.rng.choice(candidates, size=min(self.num_negatives, len(candidates)), replace=False)
                        neg_items[i, pos, :len(sampled)] = torch.from_numpy(sampled)

        result = {
            "sequence": sequences,
            "sequence_length": sequence_lengths,
            "user_id": user_ids,
            "neg_items": neg_items,
        }

        if "target" in batch[0]:
            result["target"] = torch.stack([item["target"] for item in batch])

        return result


class ImplicitNegativeSamplingCollate:
    """Collate function with dynamic negative sampling for implicit models (NCF, BPRMF)."""

    def __init__(
        self,
        num_items: int,
        num_negatives: int,
        user_histories: Dict[int, Set[int]] = None,
        sampler=None,
        item_offset: int = DEFAULT_ITEM_OFFSET,
    ):
        self.num_items = num_items
        self.num_negatives = (
            sampler.num_negatives if sampler is not None else num_negatives
        )
        self.user_histories = user_histories or {}
        self.sampler = sampler
        self.item_offset = item_offset
        self.rng = np.random.default_rng()
        # Items are in range [item_offset, num_items + item_offset - 1]
        self.all_items = np.arange(item_offset, num_items + item_offset)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic negative sampling for implicit feedback."""
        user_ids = torch.stack([item["user_id"] for item in batch])
        item_ids = torch.stack([item["item_id"] for item in batch])  # 3-indexed

        batch_size = len(batch)
        neg_items = torch.zeros(batch_size, self.num_negatives, dtype=torch.long)

        for i, user_id in enumerate(user_ids.tolist()):
            user_positives = self.user_histories.get(user_id, set())  # 3-indexed

            if self.sampler is not None:
                sampled = self.sampler.sample(user_positives, user_id)
                neg_items[i, : len(sampled)] = torch.tensor(sampled, dtype=torch.long)
            else:
                # Sample from correct range [item_offset, num_items + item_offset - 1]
                candidates = np.setdiff1d(self.all_items, list(user_positives), assume_unique=True)
                sampled = self.rng.choice(
                    candidates,
                    size=min(self.num_negatives, len(candidates)),
                    replace=False,
                )
                neg_items[i, : len(sampled)] = torch.from_numpy(sampled)

        return {"user_id": user_ids, "item_id": item_ids, "neg_items": neg_items}


class BatchSharedNegativeSamplingCollate:
    """Collate function that samples one shared set of negatives per batch.

    Instead of sampling num_neg negatives per sequence position (shape [B, S, N]),
    this samples num_neg negatives once per batch item (shape [B, N]).
    The loss function broadcasts these across all sequence positions.
    This allows using a larger N (e.g. 1024) without the memory cost of per-position sampling.
    """

    def __init__(
        self,
        num_items: int,
        num_negatives: int,
        user_histories: Dict[int, Set[int]] = None,
        sampler=None,
        item_offset: int = DEFAULT_ITEM_OFFSET,
    ):
        self.num_items = num_items
        self.num_negatives = sampler.num_negatives if sampler is not None else num_negatives
        self.user_histories = user_histories or {}
        self.sampler = sampler
        self.item_offset = item_offset
        self.rng = np.random.default_rng()
        self.all_items = np.arange(item_offset, num_items + item_offset)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        sequences = torch.stack([item["sequence"] for item in batch])
        sequence_lengths = torch.stack([item["sequence_length"] for item in batch])
        user_ids = torch.stack([item["user_id"] for item in batch])

        batch_size = len(batch)
        neg_items = torch.zeros(batch_size, self.num_negatives, dtype=torch.long)

        for i, user_id in enumerate(user_ids.tolist()):
            user_positives = self.user_histories.get(user_id, set())
            if self.sampler is not None:
                sampled = self.sampler.sample(user_positives, self.num_negatives)
            else:
                candidates = np.setdiff1d(self.all_items, list(user_positives), assume_unique=True)
                sampled = self.rng.choice(candidates, size=self.num_negatives, replace=len(candidates) < self.num_negatives)
            neg_items[i] = torch.from_numpy(sampled)

        result = {
            "sequence": sequences,
            "sequence_length": sequence_lengths,
            "user_id": user_ids,
            "neg_items": neg_items,  # [batch, num_neg] — 2D, broadcast across positions in loss
        }
        if "target" in batch[0]:
            result["target"] = torch.stack([item["target"] for item in batch])
        return result


class GraphNegativeSamplingCollate:
    """Collate function with dynamic per-batch negative sampling for graph models."""

    def __init__(
        self,
        num_items: int,
        num_negatives: int,
        user_histories: Dict[int, Set[int]] = None,
        sampler=None,
        item_offset: int = DEFAULT_ITEM_OFFSET,
    ):
        self.num_items = num_items
        self.num_negatives = (
            sampler.num_negatives if sampler is not None else num_negatives
        )
        self.user_histories = user_histories or {}
        self.sampler = sampler
        self.item_offset = item_offset
        self.rng = np.random.default_rng()
        # Items are in range [item_offset, num_items + item_offset - 1]
        self.all_items = np.arange(item_offset, num_items + item_offset)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic negative sampling for graph models."""
        user_ids = torch.stack([item["user_id"] for item in batch])
        item_ids = torch.stack([item["item_id"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        edge_index = batch[0]["edge_index"]  # Same for all items in batch

        batch_size = len(batch)
        neg_items = torch.zeros(batch_size, self.num_negatives, dtype=torch.long)

        for i, user_id in enumerate(user_ids.tolist()):
            user_positives = self.user_histories.get(user_id, set())

            if self.sampler is not None:
                sampled = self.sampler.sample(user_positives, user_id)
                neg_items[i, : len(sampled)] = torch.tensor(sampled, dtype=torch.long)
            else:
                # Sample from correct range [item_offset, num_items + item_offset - 1]
                candidates = np.setdiff1d(self.all_items, list(user_positives), assume_unique=True)
                sampled = self.rng.choice(
                    candidates,
                    size=min(self.num_negatives, len(candidates)),
                    replace=False,
                )
                neg_items[i, : len(sampled)] = torch.from_numpy(sampled)

        return {
            "user_id": user_ids,
            "item_id": item_ids,
            "label": labels,
            "edge_index": edge_index,
            "neg_items": neg_items,
        }
