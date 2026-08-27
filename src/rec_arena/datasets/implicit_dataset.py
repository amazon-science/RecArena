"""Clean implicit dataset without pre-computed negatives."""

import torch
import pandas as pd
from typing import Dict, List, Set
from torch.utils.data import Dataset


class ImplicitDataset(Dataset):
    """Implicit dataset - stores only positive interactions."""

    def __init__(self, interactions: List[Dict]):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        return {
            "user_id": torch.tensor(interaction["user_id"], dtype=torch.long),
            "item_id": torch.tensor(interaction["item_id"], dtype=torch.long),
        }


def prepare_implicit_interactions(df: pd.DataFrame) -> List[Dict]:
    """Prepare positive interactions only.

    If 'implicit' column exists, filter to implicit==1.
    Otherwise, treat all interactions as positive (already implicit feedback).

    Item ids are kept exactly as produced by the dataset loader, which maps
    items to a 3-indexed space ([3, N+2]; PAD/UNK/MASK occupy 0/1/2) -- the same
    convention sequential models use. Implicit models therefore size their item
    embedding table to N+3 and use the id directly as the embedding/column
    index, keeping positives, negatives (sampled over [3, N+2]) and held-out
    targets on one consistent convention.
    """
    interactions = []

    # Filter to positive interactions if implicit column exists
    if "implicit" in df.columns:
        positive_df = df[df["implicit"] == 1]
    else:
        # All interactions are positive (implicit feedback dataset)
        positive_df = df

    for _, row in positive_df.iterrows():
        interactions.append(
            {
                "user_id": int(row["user_id"]),
                "item_id": int(row["item_id"]),  # 3-indexed (direct column index)
            }
        )

    return interactions
