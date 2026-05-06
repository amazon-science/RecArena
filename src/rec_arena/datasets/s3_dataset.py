"""S3 Dataset - Load pre-split datasets from S3 with unified architecture."""

import io

import boto3
import os
import pandas as pd
from typing import Optional, Literal, Tuple, Dict
from .base_dataset import BaseDataset


class S3Dataset(BaseDataset):
    """Load pre-split recommendation datasets from S3.

    This dataset loads pre-split data from S3 (train/val/test already separated).
    It inherits from BaseDataset but uses pre-split data when available.
    Items are 1-indexed (0 reserved for padding).
    """

    # Class-level cache to share data across instances
    _cache = {}

    def __init__(
        self,
        dataset_name: str,
        split_type: Literal["leave_one_out", "random_user_split", "time_split"],
        s3_bucket: str = "music-ml-berlin-recarena",
        s3_prefix: str = "recarena",
        cache_dir: Optional[str] = None,
        min_interactions: int = 0,  # Pre-split data already filtered
        local_data_dir: Optional[str] = None,
    ):
        """Initialize S3 dataset.

        Args:
            dataset_name: Dataset name (e.g., "ml-100k", "amazon-books")
            split_type: Split strategy used for pre-split data
                ('leave_one_out', 'random_user_split', 'time_split')
            s3_bucket: S3 bucket name (if None, falls back to local_data_dir)
            s3_prefix: S3 prefix path
            cache_dir: Local cache directory (default: ~/.cache/recarena)
            min_interactions: Minimum interactions (not used for pre-split data)
            local_data_dir: Local directory containing pre-split parquet files.
                If None, auto-detected from RECARENA_DATA_DIR env var or ./data/
        """
        # Initialize with split_type since data is pre-split
        super().__init__(
            min_interactions=min_interactions,
            implicit_threshold=None,  # Pre-split data already processed
            split_type=split_type,
            split_kwargs={},
        )

        self.dataset_name = dataset_name
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/recarena")
        self.local_data_dir = local_data_dir or os.environ.get("RECARENA_DATA_DIR")

        # Create cache key for this dataset configuration
        self.cache_key = f"{s3_bucket}/{s3_prefix}/{dataset_name}/{split_type}"

        # Pre-split data
        self.train_df = None
        self.val_df = None
        self.test_df = None

        # Only create S3 client if we actually need S3
        if self.s3_bucket:
            self.s3_client = self._create_s3_client()
        else:
            self.s3_client = None

    def _create_s3_client(self):
        """Create S3 client using AWS profile from environment or default credentials."""
        aws_profile = os.environ.get("AWS_PROFILE")
        if aws_profile:
            try:
                session = boto3.Session(profile_name=aws_profile)
                return session.client("s3")
            except Exception as e:
                print(f"⚠ Profile '{aws_profile}' not found: {e}")
                print("  Falling back to default credentials (IAM role)")
                # Unset AWS_PROFILE to prevent boto3 from trying to use it
                del os.environ["AWS_PROFILE"]
        # Use default credentials (IAM role on SageMaker/EC2)
        return boto3.client("s3")

    def _get_default_split_type(self) -> str:
        """Return the split type specified at initialization.

        For S3 datasets, the split type is determined by what's stored in S3.
        """
        return "leave_one_out"

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw data - not used for S3 pre-split datasets.

        S3 datasets override load_data() to load pre-split data directly.
        """
        raise NotImplementedError(
            "S3Dataset uses pre-split data. Use load_data() instead."
        )

    def _read_parquet(self, key: str) -> pd.DataFrame:
        """Read a parquet file from S3 via boto3 client (respects AWS_PROFILE)."""
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        return df

    def _resolve_local_dir(self) -> Optional[str]:
        """Find local data directory for this dataset."""
        candidates = [
            self.local_data_dir,
            os.path.join("data", self.dataset_name, self._split_type),
            os.path.join("data"),
        ]
        for base in candidates:
            if base is None:
                continue
            # Check if base already includes dataset/split path
            train_path = os.path.join(base, "train.parquet")
            if os.path.exists(train_path):
                return base
            # Check nested path
            nested = os.path.join(base, self.dataset_name, self._split_type)
            if os.path.exists(os.path.join(nested, "train.parquet")):
                return nested
        return None

    def load_data(self):
        """Load pre-split train/test data from S3 or local files (cached after first load)."""
        if self.train_df is not None and self.test_df is not None:
            return

        if self.cache_key in S3Dataset._cache:
            cached_data = S3Dataset._cache[self.cache_key]
            self.train_df = cached_data["train_df"]
            self.test_df = cached_data["test_df"]
            self.val_df = cached_data["val_df"]
            self.num_users = cached_data["num_users"]
            self.num_items = cached_data["num_items"]
            self.interactions_df = cached_data["interactions_df"]
            return

        # Try local files first if no S3 bucket configured
        local_dir = self._resolve_local_dir()
        if not self.s3_bucket and local_dir:
            self._load_from_local(local_dir)
        elif self.s3_bucket:
            self._load_from_s3()
        elif local_dir:
            self._load_from_local(local_dir)
        else:
            raise RuntimeError(
                f"Cannot load dataset '{self.dataset_name}': no S3 bucket configured "
                f"(set RECARENA_S3_BUCKET env var) and no local data found "
                f"(run: python -m rec_arena.experiments.prepare_datasets "
                f"--datasets {self.dataset_name} --output-dir data/)"
            )

        # Finalize: remap IDs, build interactions_df, cache
        self._finalize_load()

    def _load_from_local(self, local_dir: str):
        """Load pre-split parquet files from local directory."""
        from pathlib import Path
        base = Path(local_dir)

        print(f"Loading train data from: {base / 'train.parquet'}")
        self.train_df = pd.read_parquet(base / "train.parquet")

        print(f"Loading test data from: {base / 'test.parquet'}")
        self.test_df = pd.read_parquet(base / "test.parquet")

        val_path = base / "val.parquet"
        if val_path.exists():
            print(f"Loading val data from: {val_path}")
            self.val_df = pd.read_parquet(val_path)
            print(f"✓ Val data loaded: {len(self.val_df)} interactions")
        else:
            print("⚠ No val split found (will create from train)")
            self.val_df = None

        # Normalize column names
        for df in [self.train_df, self.val_df, self.test_df]:
            if df is not None and "datetime" in df.columns:
                df.rename(columns={"datetime": "timestamp"}, inplace=True)

    def _load_from_s3(self):
        """Load pre-split parquet files from S3."""
        prefix = f"{self.s3_prefix}/{self.dataset_name}/{self._split_type}"

        print(f"Loading train data from: s3://{self.s3_bucket}/{prefix}/train.parquet")
        self.train_df = self._read_parquet(f"{prefix}/train.parquet")

        print(f"Loading test data from: s3://{self.s3_bucket}/{prefix}/test.parquet")
        self.test_df = self._read_parquet(f"{prefix}/test.parquet")

        try:
            print(f"Loading val data from: s3://{self.s3_bucket}/{prefix}/val.parquet")
            self.val_df = self._read_parquet(f"{prefix}/val.parquet")
            print(f"✓ Val data loaded: {len(self.val_df)} interactions")
        except Exception as e:
            print(f"⚠ No val split found (will create from train): {e}")
            self.val_df = None

    def _finalize_load(self):
        """Remap IDs, build interactions_df, and cache."""
        self._remap_presplit_ids()

        # Combine all data for interactions_df (used by some models)
        dfs = [self.train_df]
        if self.val_df is not None:
            dfs.append(self.val_df)
        dfs.append(self.test_df)
        self.interactions_df = pd.concat(dfs, ignore_index=True).sort_values(
            "timestamp"
        )

        # Cache the loaded data at class level
        S3Dataset._cache[self.cache_key] = {
            "train_df": self.train_df,
            "test_df": self.test_df,
            "val_df": self.val_df,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "interactions_df": self.interactions_df,
        }

        print(f"Loaded {self.dataset_name} ({self._split_type})")
        print(f"  Users: {self.num_users}, Items: {self.num_items}")
        print(f"  Train: {len(self.train_df)}, Test: {len(self.test_df)}")
        if self.val_df is not None:
            print(f"  Val: {len(self.val_df)}")

    def _remap_presplit_ids(self):
        """Remap IDs for pre-split data with GPT-style special tokens.

        Special tokens:
        - 0: PAD (padding)
        - 1: UNK (unknown)
        - 2: MASK (for BERT4Rec)

        Items start at index 3.
        """
        # Collect all unique users and items across splits
        all_users = set(self.train_df["user_id"].unique())
        all_items = set(self.train_df["item_id"].unique())

        if self.test_df is not None:
            all_users.update(self.test_df["user_id"].unique())
            all_items.update(self.test_df["item_id"].unique())

        if self.val_df is not None:
            all_users.update(self.val_df["user_id"].unique())
            all_items.update(self.val_df["item_id"].unique())

        all_users = sorted(all_users)
        all_items = sorted(all_items)

        # Create mappings: users 0-indexed, items start at 3 (after PAD=0, UNK=1, MASK=2)
        user_map = {old_id: new_id for new_id, old_id in enumerate(all_users)}
        item_map = {old_id: new_id + 3 for new_id, old_id in enumerate(all_items)}

        # Apply mappings to all splits
        self.train_df["user_id"] = self.train_df["user_id"].map(user_map)
        self.train_df["item_id"] = self.train_df["item_id"].map(item_map)

        if self.test_df is not None:
            self.test_df["user_id"] = self.test_df["user_id"].map(user_map)
            self.test_df["item_id"] = self.test_df["item_id"].map(item_map)

        if self.val_df is not None:
            self.val_df["user_id"] = self.val_df["user_id"].map(user_map)
            self.val_df["item_id"] = self.val_df["item_id"].map(item_map)

        # Set counts
        self.num_users = len(all_users)
        self.num_items = len(all_items)  # Does NOT include special tokens

        # Store mappings
        self._user_map = user_map
        self._item_map = item_map

    def split(
        self, split_type: Optional[str] = None, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return pre-split data.

        Args:
            split_type: Ignored for S3 datasets (uses pre-split)
            **kwargs: Ignored

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Only load data if not already loaded
        if self.train_df is None or self.test_df is None:
            self.load_data()

        if self.val_df is not None:
            return self.train_df, self.val_df, self.test_df
        else:
            # No validation split: for leave-one-out, use last item per user as validation
            if self._split_type == "leave_one_out":
                # Sort by timestamp and take last item per user for validation
                df_sorted = self.train_df.sort_values(["user_id", "timestamp"])
                val_indices = df_sorted.groupby("user_id").tail(1).index
                val_split = self.train_df.loc[val_indices].copy()
                train_split = self.train_df.drop(val_indices).copy()
                return train_split, val_split, self.test_df
            else:
                # For other splits, use 90/10 split
                n = len(self.train_df)
                train_end = int(n * 0.9)
                train_split = self.train_df.iloc[:train_end].copy()
                val_split = self.train_df.iloc[train_end:].copy()
                return train_split, val_split, self.test_df
