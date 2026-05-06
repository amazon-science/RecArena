"""RecDataModule with dynamic negative sampling for all model types."""

import pandas as pd
import lightning as pl
from torch.utils.data import DataLoader
from .sequential_dataset import (
    SequentialDataset,
    prepare_sequences,
    build_user_histories,
)
from .implicit_dataset import ImplicitDataset, prepare_implicit_interactions
from .graph_dataset import GraphDataset, to_graph
from .collate import (
    SequentialNegativeSamplingCollate,
    BatchSharedNegativeSamplingCollate,
    ImplicitNegativeSamplingCollate,
    GraphNegativeSamplingCollate,
)


class RecDataModule(pl.LightningDataModule):
    """RecDataModule for all model types with dynamic negative sampling."""

    def __init__(
        self,
        dataset,
        format: str = "sequential",  # 'sequential', 'implicit' or 'graph'
        model_type: str = "sasrec",
        batch_size: int = 256,
        num_workers: int = 4,  # Default for local datasets
        max_seq_length: int = 200,
        num_negatives: int = 50,
        negative_sampler=None,
        negative_scope: str = "per_position",  # "per_position" or "batch_shared"
    ):
        super().__init__()

        # Force num_workers=0 for S3 datasets, regardless of user input
        if hasattr(dataset, "s3_bucket"):
            if num_workers > 0:
                import warnings

                warnings.warn(
                    "S3 datasets require num_workers=0. Overriding user setting."
                )
            self.num_workers = 0  # Always set to 0 for S3, regardless of input
        else:
            self.num_workers = num_workers  # Use user's setting for local datasets

        # Validate parameters
        if num_negatives < 0:
            raise ValueError(f"num_negatives must be non-negative, got {num_negatives}")

        if num_negatives > 0 and num_negatives >= dataset.num_items:
            raise ValueError(
                f"num_negatives ({num_negatives}) must be less than "
                f"num_items ({dataset.num_items})"
            )

        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        self.dataset = dataset
        self.format = format
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.negative_sampler = negative_sampler
        self.negative_scope = negative_scope

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_collate = None
        self.val_collate = None
        self.test_collate = None

    def setup(self, stage=None):
        """Setup datasets with user histories for negative sampling."""
        # Load data (S3Dataset has class-level caching, no need for instance cache)
        if not hasattr(self.dataset, "interactions_df") or self.dataset.interactions_df is None:
            self.dataset.load_data()
        
        train_df, val_df, test_df = self.dataset.split()

        # Build histories for negative sampling
        train_histories = build_user_histories(train_df)
        val_histories = build_user_histories(train_df)
        # CRITICAL FIX: Test histories must include train+val since test sequences use train+val
        train_val_df = pd.concat([train_df, val_df])
        test_histories = build_user_histories(train_val_df)

        if self.format == "sequential":
            train_data = prepare_sequences(
                train_df, self.max_seq_length, self.model_type
            )
            val_data = prepare_sequences(
                val_df,
                self.max_seq_length,
                self.model_type,
                for_val_loo=True,
                train_df=train_df,
            )
            # For test data, use train+val combined to build sequences
            train_val_df = pd.concat([train_df, val_df])
            test_data = prepare_sequences(
                test_df,
                self.max_seq_length,
                self.model_type,
                for_val_loo=True,
                train_df=train_val_df,
            )

            self.train_dataset = SequentialDataset(train_data, self.max_seq_length)
            self.val_dataset = SequentialDataset(val_data, self.max_seq_length)
            self.test_dataset = SequentialDataset(test_data, self.max_seq_length)

            CollateClass = (
                BatchSharedNegativeSamplingCollate
                if self.negative_scope == "batch_shared"
                else SequentialNegativeSamplingCollate
            )
            self.train_collate = CollateClass(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )
            self.val_collate = CollateClass(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )
            self.test_collate = CollateClass(
                self.dataset.num_items,
                self.num_negatives,
                test_histories,
                self.negative_sampler,
            )

        elif self.format == "implicit":
            # For training: use all training interactions
            train_data = prepare_implicit_interactions(train_df)

            # For val/test in LOO: use the same data as sequential models
            # Val/test df contains the held-out items, so we just use them directly
            val_data = prepare_implicit_interactions(val_df)
            test_data = prepare_implicit_interactions(test_df)

            self.train_dataset = ImplicitDataset(train_data)
            self.val_dataset = ImplicitDataset(val_data)
            self.test_dataset = ImplicitDataset(test_data)

            self.train_collate = ImplicitNegativeSamplingCollate(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )
            self.val_collate = ImplicitNegativeSamplingCollate(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )
            self.test_collate = ImplicitNegativeSamplingCollate(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )

        elif self.format == "graph":
            # Import graph utilities only when needed
            from ..models.graph_models.graph_utils import (
                add_graph_support_to_datamodule,
            )

            add_graph_support_to_datamodule(self.__class__)

            # Convert to graph format
            train_interactions = to_graph(train_df)
            val_interactions = to_graph(val_df)
            test_interactions = to_graph(test_df)

            # Create edge_index from ONLY training interactions (no data leakage)
            self.edge_index = self._create_edge_index(train_df)

            # Create datasets without negative sampler (handled in collate)
            self.train_dataset = GraphDataset(train_interactions, self.edge_index)
            self.val_dataset = GraphDataset(val_interactions, self.edge_index)
            self.test_dataset = GraphDataset(test_interactions, self.edge_index)

            # Use collate functions for negative sampling
            self.train_collate = GraphNegativeSamplingCollate(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )
            self.val_collate = GraphNegativeSamplingCollate(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )
            self.test_collate = GraphNegativeSamplingCollate(
                self.dataset.num_items,
                self.num_negatives,
                train_histories,
                self.negative_sampler,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
