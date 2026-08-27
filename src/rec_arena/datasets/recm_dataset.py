"""RecM Dataset - Load pre-split datasets from S3."""

import pandas as pd
import boto3
import os
from pathlib import Path
from typing import Optional, Literal


class RecMDataset:
    """Load pre-split recommendation datasets from S3."""

    # Class-level cache to share data across instances
    _cache = {}

    def __init__(
        self,
        dataset_name: str,
        split_type: Literal[
            "leave_one_out", "random_user_split", "time_split"
        ] = "time_split",
        s3_bucket: str = "example-account",
        s3_prefix: str = "recarena",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_name: Dataset name (e.g., "ml-100k", "amazon-books")
            split_type: Split strategy - "leave_one_out", "random_user_split", or "time_split"
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix path
            cache_dir: Local cache directory (default: ~/.cache/recarena)
        """
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/recarena")

        # Create cache key for this dataset configuration
        self.cache_key = f"{s3_bucket}/{s3_prefix}/{dataset_name}/{split_type}"

        self.num_users = None
        self.num_items = None
        self.train_df = None
        self.test_df = None
        self.val_df = None

        self.s3_client = self._create_s3_client()

    def _create_s3_client(self):
        """Create S3 client using AWS profile from environment or default credentials."""
        aws_profile = os.environ.get("AWS_PROFILE")
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            return session.client("s3")
        else:
            return boto3.client("s3")

    def load_data(self):
        """Load train/test splits directly from S3 (cached in memory after first load)."""
        # Check if data is already loaded in this instance
        if self.train_df is not None and self.test_df is not None:
            return

        # Check class-level cache first
        if self.cache_key in RecMDataset._cache:
            cached_data = RecMDataset._cache[self.cache_key]
            self.train_df = cached_data["train_df"]
            self.test_df = cached_data["test_df"]
            self.val_df = cached_data["val_df"]
            self.num_users = cached_data["num_users"]
            self.num_items = cached_data["num_items"]
            return

        # Load dataframes directly from S3
        train_s3_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.dataset_name}/{self.split_type}/train.parquet"
        test_s3_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.dataset_name}/{self.split_type}/test.parquet"

        print(f"Loading train data from: {train_s3_path}")
        self.train_df = pd.read_parquet(train_s3_path)

        print(f"Loading test data from: {test_s3_path}")
        self.test_df = pd.read_parquet(test_s3_path)

        # Standardize column names (datetime -> timestamp for compatibility)
        if "datetime" in self.train_df.columns:
            self.train_df = self.train_df.rename(columns={"datetime": "timestamp"})
        if "datetime" in self.test_df.columns:
            self.test_df = self.test_df.rename(columns={"datetime": "timestamp"})

        # Try to load validation split if exists
        try:
            val_s3_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.dataset_name}/{self.split_type}/val.parquet"
            self.val_df = pd.read_parquet(val_s3_path)
            if "datetime" in self.val_df.columns:
                self.val_df = self.val_df.rename(columns={"datetime": "timestamp"})
        except:
            self.val_df = None

        # Calculate num_users and num_items
        all_users = set(self.train_df["user_id"].unique())
        all_items = set(self.train_df["item_id"].unique())

        if self.test_df is not None:
            all_users.update(self.test_df["user_id"].unique())
            all_items.update(self.test_df["item_id"].unique())

        if self.val_df is not None:
            all_users.update(self.val_df["user_id"].unique())
            all_items.update(self.val_df["item_id"].unique())

        self.num_users = len(all_users)
        self.num_items = len(all_items)

        # Cache the loaded data at class level
        RecMDataset._cache[self.cache_key] = {
            "train_df": self.train_df,
            "test_df": self.test_df,
            "val_df": self.val_df,
            "num_users": self.num_users,
            "num_items": self.num_items,
        }

        print(f"Loaded {self.dataset_name} ({self.split_type})")
        print(f"  Users: {self.num_users}, Items: {self.num_items}")
        print(f"  Train: {len(self.train_df)}, Test: {len(self.test_df)}")
        if self.val_df is not None:
            print(f"  Val: {len(self.val_df)}")

    def split(self):
        """Return pre-split data (for compatibility with RecDataModule)."""
        # Only load data if not already loaded in this instance
        if self.train_df is None or self.test_df is None:
            self.load_data()
        if self.val_df is not None:
            return self.train_df, self.val_df, self.test_df
        else:
            # No validation split available, use test as validation
            return self.train_df, self.test_df, self.test_df

    def _download_split(self, split_name: str) -> str:
        """Download split from S3 to local cache."""
        s3_key = f"{self.s3_prefix}/{self.dataset_name}/{self.split_type}/{split_name}.parquet"
        local_path = (
            Path(self.cache_dir)
            / self.dataset_name
            / self.split_type
            / f"{split_name}.parquet"
        )

        # Create cache directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download if not cached
        if not local_path.exists():
            print(f"Downloading {split_name} from S3...")
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))

        return str(local_path)
