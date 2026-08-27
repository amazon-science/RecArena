"""Clean sequential dataset without pre-computed negatives."""

import torch
import pandas as pd
from typing import Dict, List, Set
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    """Sequential dataset - stores only sequences, no negatives."""

    def __init__(self, sequences: List[Dict], max_seq_length: int):
        self.sequences = sequences
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        result = {
            "user_id": torch.tensor(seq["user_id"], dtype=torch.long),
            "sequence": torch.tensor(seq["sequence"], dtype=torch.long),
            "sequence_length": torch.tensor(seq["sequence_length"], dtype=torch.long),
        }

        if "target" in seq:
            result["target"] = torch.tensor(seq["target"], dtype=torch.long)
        
        if "timestamps" in seq:
            ts = seq["timestamps"]
            if ts and hasattr(ts[0], 'timestamp'):
                ts = [int(t.timestamp()) if hasattr(t, 'timestamp') else int(t) for t in ts]
            result["timestamps"] = torch.tensor(ts, dtype=torch.long)

        return result


def build_user_histories(df: pd.DataFrame) -> Dict[int, Set[int]]:
    """Build user interaction histories for negative sampling."""
    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def prepare_sequences(
    df: pd.DataFrame,
    max_seq_length: int,
    model_type: str = "sasrec",
    for_val_loo: bool = False,
    train_df: pd.DataFrame = None,
    append_df: pd.DataFrame = None,
) -> List[Dict]:
    """Prepare sequences. For LOO val/test, builds full sequences from train_df.

    `df`        : the rows whose item_id is the prediction TARGET (val or test).
    `train_df`  : the pure history to build the input sequence from.
    `append_df` : OPTIONAL extra held-out-history rows to APPEND at the end of
                  each user's input, in split order (used for TEST: the input is
                  train history + the validation item as the most-recent item).

    LOO semantics: the split already DEFINES the order -- train is the history,
    val/test are the held-out "next" interactions. So we build the input from
    the pure sorted train history and APPEND the val item at the end explicitly,
    rather than concatenating val into train and re-sorting by timestamp. Sorting
    the boundary item back in is fragile: with 1-second-resolution timestamps
    (MovieLens) the val item routinely TIES the last train item, and any reorder
    can bury it mid-sequence -- which fed SASRec a train item as the most-recent
    input for ~34% of ml-100k users and deflated test NDCG. Append-at-end is
    order-preserving and timestamp-independent, matching the LOO definition (and
    RecBole, whose synthetic timestamps force train<val<test).
    """
    sequences = []

    if for_val_loo and train_df is not None:
        # Pure train history, stable-sorted by time (deterministic; ties keep
        # input row order). The held-out boundary item is appended below, NOT
        # sorted in, so its position never depends on a timestamp tie.
        train_sorted = train_df.sort_values("timestamp", kind="stable")
        train_grouped_items = train_sorted.groupby("user_id")["item_id"].apply(list).to_dict()
        train_grouped_ts = train_sorted.groupby("user_id")["timestamp"].apply(list).to_dict()

        # Optional per-user items to append at the END of the input (the val
        # item for the TEST split), in split order.
        append_items = {}
        append_ts = {}
        if append_df is not None:
            _ap = append_df.sort_values("timestamp", kind="stable")
            append_items = _ap.groupby("user_id")["item_id"].apply(list).to_dict()
            append_ts = _ap.groupby("user_id")["timestamp"].apply(list).to_dict()

        for _, row in df.iterrows():
            user_id = row["user_id"]
            if user_id not in train_grouped_items:
                continue
            items = list(train_grouped_items[user_id])
            timestamps = [int(t.timestamp()) if hasattr(t, 'timestamp') else int(t) for t in train_grouped_ts[user_id]]
            # Append the held-out-history item(s) at the end (most recent).
            if user_id in append_items:
                items = items + list(append_items[user_id])
                timestamps = timestamps + [
                    int(t.timestamp()) if hasattr(t, 'timestamp') else int(t)
                    for t in append_ts[user_id]
                ]
            if len(items) > max_seq_length:
                items = items[-max_seq_length:]
                timestamps = timestamps[-max_seq_length:]
            seq_padded = items + [0] * (max_seq_length - len(items))
            ts_padded = timestamps + [0] * (max_seq_length - len(timestamps))
            sequences.append(
                {
                    "user_id": user_id,
                    "sequence": seq_padded,
                    "sequence_length": len(items),
                    "target": row["item_id"],
                    "timestamps": ts_padded,
                }
            )
    else:
        # Stable sort for deterministic, tie-safe recency ordering (see the
        # for_val_loo branch above for why unstable sorts corrupt tied-timestamp
        # sequences).
        df_sorted = df.sort_values("timestamp", kind="stable")
        grouped = df_sorted.groupby("user_id")
        for user_id, group in grouped:
            items = group["item_id"].tolist()
            timestamps = [int(t.timestamp()) if hasattr(t, 'timestamp') else int(t) for t in group["timestamp"].tolist()]
            if len(items) < 1:
                continue
            if len(items) > max_seq_length:
                items = items[-max_seq_length:]
                timestamps = timestamps[-max_seq_length:]
            seq_padded = items + [0] * (max_seq_length - len(items))
            ts_padded = timestamps + [0] * (max_seq_length - len(timestamps))
            seq_dict = {
                "user_id": user_id,
                "sequence": seq_padded,
                "sequence_length": len(items),
                "timestamps": ts_padded,
            }
            # Caser/FMLPRec are single-next-item models: drop the last item from
            # the input and expose it as `target`. GRU4Rec is INTENTIONALLY NOT
            # truncated here: with its default per-position causal-shift loss
            # (last_position_loss=False) the base compute_loss uses the sequence
            # itself as targets and NEVER reads batch["target"], so truncating
            # the input just discarded the most-recent transition and cost one
            # supervised target/user (L-2 vs SASRec's L-1) -- a GRU-specific
            # handicap vs SASRec with no upside. GRU4Rec now gets the FULL
            # sequence like SASRec. Its opt-in last_position_loss=True path
            # builds its own prefix in compute_loss, so it needs no dataset-level
            # truncation either.
            if model_type in ["caser", "fmlprec"]:
                seq_dict["target"] = items[-1]
                seq_dict["sequence"] = items[:-1] + [0] * (
                    max_seq_length - len(items) + 1
                )
                seq_dict["sequence_length"] = len(items) - 1
                seq_dict["timestamps"] = ts_padded[:-1] + [0] * (
                    max_seq_length - len(ts_padded) + 1
                )
            sequences.append(seq_dict)

    return sequences
