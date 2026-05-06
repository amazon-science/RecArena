"""Split strategies for recommendation datasets.

This module provides different strategies for splitting recommendation data
into train/validation/test sets.
"""

import pandas as pd
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class SplitStrategy(ABC):
    """Abstract base class for split strategies."""

    @abstractmethod
    def split(
        self, df: pd.DataFrame, num_users: int, num_items: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train/val/test.

        Args:
            df: Interaction dataframe with columns: user_id, item_id, timestamp
            num_users: Number of users
            num_items: Number of items

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        pass


class LeaveOneOutSplit(SplitStrategy):
    """Leave-One-Out split strategy.

    For each user:
    - Test: Last item in their sequence
    - Validation: Second-to-last item
    - Train: All earlier items

    This is the standard LOO split used in many sequential recommendation papers.
    """

    def __init__(self, min_sequence_length: int = 3):
        """
        Args:
            min_sequence_length: Minimum sequence length per user to include
        """
        self.min_sequence_length = min_sequence_length

    def split(
        self, df: pd.DataFrame, num_users: int, num_items: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform leave-one-out split per user."""
        train_list, val_list, test_list = [], [], []

        for user_id, group in df.groupby("user_id"):
            # Sort by timestamp
            group = group.sort_values("timestamp")
            n = len(group)

            if n >= self.min_sequence_length:
                # Standard case: enough items for train/val/test
                train_list.append(group.iloc[:-2])
                val_list.append(group.iloc[[-2]])
                test_list.append(group.iloc[[-1]])
            elif n == 2:
                # Edge case: only 2 items, skip validation
                train_list.append(group.iloc[:-1])
                test_list.append(group.iloc[[-1]])
            elif n == 1:
                # Edge case: only 1 item, put in train
                train_list.append(group)

        train_df = pd.concat(train_list) if train_list else pd.DataFrame()
        val_df = pd.concat(val_list) if val_list else pd.DataFrame()
        test_df = pd.concat(test_list) if test_list else pd.DataFrame()

        return train_df, val_df, test_df


class TemporalSplit(SplitStrategy):
    """Temporal split strategy.

    Splits data chronologically by timestamp:
    - Train: First 70% of interactions
    - Validation: Next 15% of interactions
    - Test: Last 15% of interactions

    This preserves temporal ordering and simulates realistic deployment scenarios.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self, df: pd.DataFrame, num_users: int, num_items: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform temporal split."""
        # Sort by timestamp
        df_sorted = df.sort_values("timestamp")
        n = len(df_sorted)

        # Calculate split points
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]

        return train_df, val_df, test_df


class UserBasedSplit(SplitStrategy):
    """User-based random split strategy.

    Randomly assigns users to train/validation/test sets:
    - Train: 80% of users (all their interactions)
    - Validation: 10% of users (all their interactions)
    - Test: 10% of users (all their interactions)

    This tests model generalization to new users (cold-start scenario).
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Args:
            train_ratio: Proportion of users for training
            val_ratio: Proportion of users for validation
            test_ratio: Proportion of users for testing
            random_seed: Random seed for reproducibility
        """
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(
        self, df: pd.DataFrame, num_users: int, num_items: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform user-based random split."""
        import numpy as np

        # Get unique users
        unique_users = df["user_id"].unique()
        n_users = len(unique_users)

        # Shuffle users
        np.random.seed(self.random_seed)
        shuffled_users = np.random.permutation(unique_users)

        # Split users
        train_end = int(n_users * self.train_ratio)
        val_end = int(n_users * (self.train_ratio + self.val_ratio))

        train_users = set(shuffled_users[:train_end])
        val_users = set(shuffled_users[train_end:val_end])
        test_users = set(shuffled_users[val_end:])

        # Split data based on user assignment
        train_df = df[df["user_id"].isin(train_users)]
        val_df = df[df["user_id"].isin(val_users)]
        test_df = df[df["user_id"].isin(test_users)]

        return train_df, val_df, test_df


# Registry for easy lookup
SPLIT_STRATEGIES = {
    "leave_one_out": LeaveOneOutSplit,
    "loo": LeaveOneOutSplit,  # Alias
    "temporal": TemporalSplit,
    "time": TemporalSplit,  # Alias
    "time_split": TemporalSplit,  # Alias
    "user_based": UserBasedSplit,
    "random_user_split": UserBasedSplit,  # Alias
}


def get_split_strategy(name: str, **kwargs) -> SplitStrategy:
    """Get split strategy by name.

    Args:
        name: Strategy name ('leave_one_out', 'temporal', 'user_based')
        **kwargs: Additional arguments for the strategy

    Returns:
        SplitStrategy instance

    Raises:
        ValueError: If strategy name is not recognized
    """
    if name not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown split strategy: {name}. "
            f"Available strategies: {list(SPLIT_STRATEGIES.keys())}"
        )

    return SPLIT_STRATEGIES[name](**kwargs)
