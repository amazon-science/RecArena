"""Base dataset class with unified architecture.

This module provides the base class for all recommendation datasets,
with common functionality for loading, filtering, ID remapping, and splitting.
"""

import pandas as pd
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod
from .split_strategies import get_split_strategy, SplitStrategy
from .preprocessing import PreprocessingPipeline


class BaseDataset(ABC):
    """Abstract base class for recommendation datasets.

    Provides common functionality:
    - Filtering by minimum interactions
    - ID remapping (1-indexed items with 0=padding)
    - Split strategy dispatch
    - Smart defaults for split strategies
    """

    def __init__(
        self,
        min_interactions: int = 5,
        implicit_threshold: Optional[float] = None,
        split_type: Optional[str] = None,
        split_kwargs: Optional[Dict] = None,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
    ):
        """Initialize base dataset.

        Args:
            min_interactions: Minimum interactions per user/item to keep
            implicit_threshold: Threshold for implicit feedback (e.g., rating >= 4)
                If None, assumes all interactions are positive
            split_type: Split strategy ('leave_one_out', 'temporal', 'user_based')
                If None, uses smart default based on dataset type
            split_kwargs: Additional arguments for split strategy
            preprocessing_pipeline: Optional preprocessing pipeline to apply
                between raw data loading and ID remapping. If provided, replaces
                the default inline preprocessing logic.
        """
        self.min_interactions = min_interactions
        self.implicit_threshold = implicit_threshold
        self._split_type = split_type
        self.split_kwargs = split_kwargs or {}
        self.preprocessing_pipeline = preprocessing_pipeline

        # Will be set after loading
        self.num_users = None
        self.num_items = None
        self.interactions_df = None
        self._user_map = {}
        self._item_map = {}

    @property
    def split_type(self) -> str:
        """Get split type with smart defaults."""
        if self._split_type is not None:
            return self._split_type
        return self._get_default_split_type()

    @abstractmethod
    def _get_default_split_type(self) -> str:
        """Get default split type for this dataset.

        Subclasses should override to provide smart defaults.
        For example, LOO datasets should return 'leave_one_out'.
        """
        pass

    @abstractmethod
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw interaction data.

        Should return a DataFrame with at minimum:
        - user_id: Original user IDs
        - item_id: Original item IDs
        - timestamp: Interaction timestamp

        Optional columns:
        - rating: Explicit rating (if applicable)
        - implicit: Binary implicit feedback (if pre-computed)

        Returns:
            DataFrame with raw interactions
        """
        pass

    def load_data(self):
        """Load and preprocess data with filtering and ID remapping."""
        # Load raw data
        df = self._load_raw_data()

        if self.preprocessing_pipeline is not None:
            # Use the provided pipeline
            df = self.preprocessing_pipeline.transform(df)
            if df.empty:
                raise ValueError(
                    "Preprocessing pipeline produced an empty DataFrame. "
                    "Check your pipeline configuration and input data."
                )
        else:
            # Apply default inline preprocessing (backward compatibility)
            if self.implicit_threshold is not None and "rating" in df.columns:
                if "implicit" not in df.columns:
                    df["implicit"] = (df["rating"] >= self.implicit_threshold).astype(int)
                # Filter to keep only positive implicit feedback
                df = df[df["implicit"] == 1].copy()
            elif "implicit" not in df.columns:
                # No implicit column and no threshold: assume all positive
                df["implicit"] = 1

            # Filter by minimum interactions
            df = self._filter_by_min_interactions(df)

        # Remap IDs (1-indexed items with 0=padding)
        df = self._remap_ids(df)

        self.interactions_df = df.sort_values("timestamp").reset_index(drop=True)

        print(
            f"Dataset loaded: {self.num_users} users, {self.num_items} items, "
            f"{len(df)} interactions"
        )

    def _filter_by_min_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and items by minimum interaction count."""
        if self.min_interactions <= 1:
            return df

        # Iteratively filter until convergence
        prev_size = len(df) + 1
        while len(df) < prev_size:
            prev_size = len(df)

            # Count interactions
            user_counts = df["user_id"].value_counts()
            item_counts = df["item_id"].value_counts()

            # Get valid users/items
            valid_users = user_counts[user_counts >= self.min_interactions].index
            valid_items = item_counts[item_counts >= self.min_interactions].index

            # Filter
            df = df[
                df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)
            ].copy()

        return df

    def _remap_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remap user and item IDs to be contiguous.

        Users: 0-indexed (0, 1, 2, ...)
        Items: 1-indexed (1, 2, 3, ...) with 0 reserved for padding
        """
        # Get unique IDs
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())

        # Create mappings
        # Users: 0-indexed
        self._user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

        # Items: 1-indexed (0 reserved for padding)
        self._item_map = {
            old_id: new_id + 1 for new_id, old_id in enumerate(unique_items)
        }

        # Apply mappings
        df["user_id"] = df["user_id"].map(self._user_map)
        df["item_id"] = df["item_id"].map(self._item_map)

        # Set counts
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)  # Does NOT include padding (0)

        return df

    def split(
        self, split_type: Optional[str] = None, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test.

        Args:
            split_type: Override default split type
            **kwargs: Additional arguments for split strategy

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure data is loaded
        if self.interactions_df is None:
            self.load_data()

        # Determine split type
        split_name = split_type or self.split_type

        # Merge kwargs
        split_params = {**self.split_kwargs, **kwargs}

        # Get strategy and split
        strategy = get_split_strategy(split_name, **split_params)
        return strategy.split(self.interactions_df, self.num_users, self.num_items)

    def get_id_mappings(self) -> Tuple[Dict, Dict]:
        """Get user and item ID mappings.

        Returns:
            Tuple of (user_map, item_map) where maps are {old_id: new_id}
        """
        return self._user_map.copy(), self._item_map.copy()


class LocalDataset(BaseDataset):
    """Dataset loaded from local files.

    Supports various file formats: CSV, TSV, Parquet, DAT (MovieLens format).
    """

    def __init__(
        self,
        data_path: str,
        file_format: str = "auto",
        separator: str = "\t",
        column_names: Optional[list] = None,
        min_interactions: int = 5,
        implicit_threshold: Optional[float] = None,
        split_type: Optional[str] = None,
        split_kwargs: Optional[Dict] = None,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
    ):
        """Initialize local dataset.

        Args:
            data_path: Path to data file
            file_format: File format ('csv', 'tsv', 'parquet', 'dat', 'auto')
                'auto' detects from file extension
            separator: Column separator for text files (default: tab)
            column_names: Column names if file has no header
                Expected order: [user_id, item_id, rating, timestamp]
            min_interactions: Minimum interactions per user/item
            implicit_threshold: Rating threshold for implicit feedback
            split_type: Split strategy (None for smart default)
            split_kwargs: Additional split strategy arguments
            preprocessing_pipeline: Optional preprocessing pipeline to apply
                between raw data loading and ID remapping.
        """
        super().__init__(
            min_interactions=min_interactions,
            implicit_threshold=implicit_threshold,
            split_type=split_type,
            split_kwargs=split_kwargs,
            preprocessing_pipeline=preprocessing_pipeline,
        )

        self.data_path = data_path
        self.file_format = file_format
        self.separator = separator
        self.column_names = column_names

    def _get_default_split_type(self) -> str:
        """Default to leave-one-out for local datasets."""
        return "leave_one_out"

    def _detect_format(self) -> str:
        """Detect file format from extension."""
        if self.file_format != "auto":
            return self.file_format

        path_lower = self.data_path.lower()
        if path_lower.endswith(".csv"):
            return "csv"
        elif path_lower.endswith(".tsv"):
            return "tsv"
        elif path_lower.endswith(".parquet"):
            return "parquet"
        elif path_lower.endswith(".dat"):
            return "dat"
        else:
            # Default to CSV
            return "csv"

    def _load_raw_data(self) -> pd.DataFrame:
        """Load data from local file."""
        fmt = self._detect_format()

        if fmt == "parquet":
            df = pd.read_parquet(self.data_path)
        elif fmt in ["csv", "tsv", "dat"]:
            # For DAT files (MovieLens), use :: separator
            sep = "::" if fmt == "dat" else self.separator

            if self.column_names:
                df = pd.read_csv(
                    self.data_path,
                    sep=sep,
                    names=self.column_names,
                    engine="python",
                )
            else:
                df = pd.read_csv(self.data_path, sep=sep, engine="python")
        else:
            raise ValueError(f"Unsupported file format: {fmt}")

        # Ensure required columns exist
        if "user_id" not in df.columns or "item_id" not in df.columns:
            raise ValueError(
                "Data must contain 'user_id' and 'item_id' columns. "
                f"Found columns: {list(df.columns)}"
            )

        if "timestamp" not in df.columns:
            # Create dummy timestamps if not present
            df["timestamp"] = range(len(df))

        return df
