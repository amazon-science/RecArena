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
) -> List[Dict]:
    """Prepare sequences. For LOO val/test, builds full sequences from train_df."""
    sequences = []

    if for_val_loo and train_df is not None:
        # Build full sequences from train_df for each user in val/test
        train_sorted = train_df.sort_values("timestamp")
        train_grouped_items = train_sorted.groupby("user_id")["item_id"].apply(list).to_dict()
        train_grouped_ts = train_sorted.groupby("user_id")["timestamp"].apply(list).to_dict()
        
        for _, row in df.iterrows():
            user_id = row["user_id"]
            if user_id not in train_grouped_items:
                continue
            items = train_grouped_items[user_id]
            timestamps = [int(t.timestamp()) if hasattr(t, 'timestamp') else int(t) for t in train_grouped_ts[user_id]]
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
        df_sorted = df.sort_values("timestamp")
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
            if model_type in ["gru4rec", "caser", "fmlprec"]:
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
