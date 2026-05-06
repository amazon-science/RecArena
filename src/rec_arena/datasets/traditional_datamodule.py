"""DataModule for traditional models (EASE, SLIM, ItemKNN)."""

import lightning as pl
import pandas as pd
import numpy as np
import scipy.sparse as sp


class TraditionalDataModule(pl.LightningDataModule):
    """DataModule for traditional models that need full interaction matrices.

    Traditional models (EASE, SLIM, ItemKNN) don't use batched training.
    They need the full user-item interaction matrix upfront.

    Args:
        dataset: Dataset object (e.g., ML100K)
        implicit_threshold: Threshold for implicit feedback (default: 4.0)

    Example:
        >>> dataset = ML100K("./data/ml-100k/")
        >>> dataset.load_data()
        >>> datamodule = TraditionalDataModule(dataset)
        >>> datamodule.setup()
        >>>
        >>> # Get interaction matrix
        >>> train_matrix = datamodule.get_train_matrix()
        >>>
        >>> # Or get as DataFrame
        >>> train_df = datamodule.get_train_df()
    """

    def __init__(self, dataset, implicit_threshold: float = 4.0):
        super().__init__()
        self.dataset = dataset
        self.implicit_threshold = implicit_threshold

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.train_matrix = None
        self.val_matrix = None
        self.test_matrix = None

    def setup(self, stage=None):
        """Setup train/val/test splits and create sparse matrices."""
        # Get splits from dataset
        self.train_df, self.val_df, self.test_df = self.dataset.split()

        # Add implicit column if not present
        if "implicit" not in self.train_df.columns:
            self.train_df["implicit"] = (
                self.train_df["rating"] >= self.implicit_threshold
            ).astype(int)
        if "implicit" not in self.val_df.columns:
            self.val_df["implicit"] = (
                self.val_df["rating"] >= self.implicit_threshold
            ).astype(int)
        if "implicit" not in self.test_df.columns:
            self.test_df["implicit"] = (
                self.test_df["rating"] >= self.implicit_threshold
            ).astype(int)

        # Filter to positive interactions only
        train_positive = self.train_df[self.train_df["implicit"] == 1]
        val_positive = self.val_df[self.val_df["implicit"] == 1]
        test_positive = self.test_df[self.test_df["implicit"] == 1]

        # Create sparse matrices
        self.train_matrix = self._create_sparse_matrix(train_positive)
        self.val_matrix = self._create_sparse_matrix(val_positive)
        self.test_matrix = self._create_sparse_matrix(test_positive)

    def _create_sparse_matrix(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Create sparse user-item interaction matrix.

        Note: Items are 1-indexed in the dataset, but matrix indices are 0-indexed.
        We convert item IDs to 0-indexed before creating the matrix.
        """
        user_ids = df["user_id"].values  # Already 0-indexed
        item_ids = df["item_id"].values - 1  # Convert 1-indexed to 0-indexed
        data = np.ones(len(df))

        matrix = sp.csr_matrix(
            (data, (user_ids, item_ids)),
            shape=(self.dataset.num_users, self.dataset.num_items),
            dtype=np.float32,
        )

        return matrix

    def get_train_matrix(self) -> sp.csr_matrix:
        """Get training interaction matrix."""
        if self.train_matrix is None:
            raise RuntimeError("Call setup() first")
        return self.train_matrix

    def get_val_matrix(self) -> sp.csr_matrix:
        """Get validation interaction matrix."""
        if self.val_matrix is None:
            raise RuntimeError("Call setup() first")
        return self.val_matrix

    def get_test_matrix(self) -> sp.csr_matrix:
        """Get test interaction matrix."""
        if self.test_matrix is None:
            raise RuntimeError("Call setup() first")
        return self.test_matrix

    def get_train_df(self) -> pd.DataFrame:
        """Get training DataFrame."""
        if self.train_df is None:
            raise RuntimeError("Call setup() first")
        return self.train_df

    def get_val_df(self) -> pd.DataFrame:
        """Get validation DataFrame."""
        if self.val_df is None:
            raise RuntimeError("Call setup() first")
        return self.val_df

    def get_test_df(self) -> pd.DataFrame:
        """Get test DataFrame."""
        if self.test_df is None:
            raise RuntimeError("Call setup() first")
        return self.test_df
