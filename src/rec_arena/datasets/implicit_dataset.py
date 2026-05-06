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

    Note: Converts item_ids from 1-indexed (data convention) to 0-indexed (model convention).
    This matches the library standard where:
    - Data stores items as [1, 2, 3, ..., N]
    - Models expect embedding indices [0, 1, 2, ..., N-1]
    - Sequential models handle this via _to_model_indices() during evaluation
    - Implicit models need conversion at dataset level since they embed directly
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
                "item_id": int(row["item_id"])
                - 1,  # Convert to 0-indexed for model embeddings
            }
        )

    return interactions
