"""Dynamic negative sampling collate functions for all model types."""

import torch
import numpy as np
from typing import List, Dict, Set

# Default offset for item indices (PAD=0, UNK=1, MASK=2, items start at 3)
DEFAULT_ITEM_OFFSET = 3


def _sample_negatives_fast(
    rng: np.random.Generator,
    lo: int,
    hi: int,
    positives,
    n: int,
) -> np.ndarray:
    """Draw ``n`` negative item ids in ``[lo, hi)`` excluding ``positives``.

    Rejection sampling: oversample uniformly from the full range and drop the
    (rare, thanks to sparsity) collisions with the user's positives. This is
    O(n + |positives|) per call vs the old np.setdiff1d(all_items, positives)
    which allocated + sorted a full catalog-sized array O(num_items log num_items)
    for EVERY batch row -- the dominant cost for MF / sampled-loss models
    (BPR-MF, NCF, SimpleX). Negatives may repeat (uniform-with-replacement,
    matching RecBole); the only invariant preserved is "no positive is a
    negative". Falls back to plain uniform when the user has no history.
    """
    if not positives:
        return rng.integers(lo, hi, size=n, dtype=np.int64)

    # Ensure a real set for O(1) membership (histories are already sets, but a
    # caller might pass a list). Rejection via Python `in` on a set is ~8x
    # faster here than np.isin, which is O(n * |positives|) and was the actual
    # hot spot (measured 27us vs 3us per draw at catalog size ~14k).
    pos = positives if isinstance(positives, (set, frozenset)) else set(positives)
    out = np.empty(n, dtype=np.int64)
    filled = 0
    # Oversample uniformly, then reject collisions. Collision prob per draw =
    # |positives| / (hi-lo) is tiny for sparse data, so 1-2 rounds suffice.
    for _ in range(32):
        need = n - filled
        if need <= 0:
            break
        cand = rng.integers(lo, hi, size=need + 8, dtype=np.int64)
        for x in cand.tolist():
            if x not in pos:
                out[filled] = x
                filled += 1
                if filled == n:
                    break
    if filled < n:
        # Degenerate case (catalog almost entirely positives): top up with plain
        # uniform draws so the shape is always satisfied.
        out[filled:] = rng.integers(lo, hi, size=n - filled, dtype=np.int64)
    return out


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
        neg_items = torch.zeros(
            batch_size, seq_len, self.num_negatives, dtype=torch.long
        )

        if self.sampler is not None:
            for i, user_id in enumerate(user_ids.tolist()):
                user_positives = self.user_histories.get(user_id, set())
                total_samples = seq_len * self.num_negatives
                sampled_flat = self.sampler.sample(user_positives, total_samples)
                neg_items[i] = torch.from_numpy(
                    sampled_flat.reshape(seq_len, self.num_negatives)
                )
        else:
            # Fast rejection sampling per user (see _sample_negatives_fast):
            # draw all seq_len*num_neg negatives in one call, no per-row
            # catalog-sized setdiff1d.
            lo, hi = self.item_offset, self.num_items + self.item_offset
            total_samples = seq_len * self.num_negatives
            for i, user_id in enumerate(user_ids.tolist()):
                user_positives = self.user_histories.get(user_id, set())
                sampled_flat = _sample_negatives_fast(
                    self.rng, lo, hi, user_positives, total_samples
                )
                neg_items[i] = torch.from_numpy(
                    sampled_flat.reshape(seq_len, self.num_negatives)
                )

        result = {
            "sequence": sequences,
            "sequence_length": sequence_lengths,
            "user_id": user_ids,
            "neg_items": neg_items,
        }

        if "target" in batch[0]:
            result["target"] = torch.stack([item["target"] for item in batch])

        # Preserve real interaction timestamps for time-aware models
        # (HSTU time bias, FuXi-alpha, FuXi-gamma). Aligned 1:1 with `sequence`.
        if "timestamps" in batch[0]:
            result["timestamps"] = torch.stack([item["timestamps"] for item in batch])

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

        lo, hi = self.item_offset, self.num_items + self.item_offset
        for i, user_id in enumerate(user_ids.tolist()):
            user_positives = self.user_histories.get(user_id, set())  # 3-indexed

            if self.sampler is not None:
                # sampler.sample(user_positives, n): 2nd arg is the sample COUNT,
                # not the user id (that bug silently sampled `user_id` negatives).
                sampled = self.sampler.sample(user_positives, self.num_negatives)
                neg_items[i, : len(sampled)] = torch.tensor(sampled, dtype=torch.long)
            else:
                # Fast rejection sampling (see _sample_negatives_fast): avoids the
                # old per-row np.setdiff1d over the full catalog, which dominated
                # runtime for MF / sampled-loss models.
                sampled = _sample_negatives_fast(
                    self.rng, lo, hi, user_positives, self.num_negatives
                )
                neg_items[i] = torch.from_numpy(sampled)

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
        self.num_negatives = (
            sampler.num_negatives if sampler is not None else num_negatives
        )
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

        lo, hi = self.item_offset, self.num_items + self.item_offset
        for i, user_id in enumerate(user_ids.tolist()):
            user_positives = self.user_histories.get(user_id, set())
            if self.sampler is not None:
                sampled = self.sampler.sample(user_positives, self.num_negatives)
            else:
                sampled = _sample_negatives_fast(
                    self.rng, lo, hi, user_positives, self.num_negatives
                )
            neg_items[i] = torch.from_numpy(sampled)

        result = {
            "sequence": sequences,
            "sequence_length": sequence_lengths,
            "user_id": user_ids,
            "neg_items": neg_items,  # [batch, num_neg] — 2D, broadcast across positions in loss
        }
        if "target" in batch[0]:
            result["target"] = torch.stack([item["target"] for item in batch])
        if "timestamps" in batch[0]:
            result["timestamps"] = torch.stack([item["timestamps"] for item in batch])
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

        lo, hi = self.item_offset, self.num_items + self.item_offset
        for i, user_id in enumerate(user_ids.tolist()):
            user_positives = self.user_histories.get(user_id, set())

            if self.sampler is not None:
                # 2nd arg to sampler.sample is the sample COUNT, not the user id.
                sampled = self.sampler.sample(user_positives, self.num_negatives)
                neg_items[i, : len(sampled)] = torch.tensor(sampled, dtype=torch.long)
            else:
                sampled = _sample_negatives_fast(
                    self.rng, lo, hi, user_positives, self.num_negatives
                )
                neg_items[i] = torch.from_numpy(sampled)

        return {
            "user_id": user_ids,
            "item_id": item_ids,
            "label": labels,
            "edge_index": edge_index,
            "neg_items": neg_items,
        }
