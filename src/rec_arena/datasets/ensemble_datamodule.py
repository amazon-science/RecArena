"""Ensemble-specific DataModule for models like RecM that need multiple negative samplers."""

import torch
from typing import List, Optional
from .rec_datamodule import RecDataModule
from .sequential_dataset import SequentialDataset


class EnsembleRecDataModule(RecDataModule):
    """DataModule that generates multiple negative samples for ensemble models."""

    def __init__(
        self,
        dataset,
        ensemble_samplers: List[str],  # e.g., ["random", "popularity"]
        ensemble_size: int = 4,
        ensemble_num_negatives: List[int] = None,  # e.g., [25, 75] different neg counts
        format: str = "sequential",
        model_type: str = "recm",
        batch_size: int = 256,
        num_workers: int = 4,
        max_seq_length: int = 50,
    ):
        # Initialize parent with no negative sampler (we'll handle it ourselves)
        super().__init__(
            dataset=dataset,
            format=format,
            model_type=model_type,
            batch_size=batch_size,
            num_workers=num_workers,
            max_seq_length=max_seq_length,
            num_negatives=50,  # Default, will be overridden by ensemble
        )

        self.ensemble_samplers = ensemble_samplers
        self.ensemble_size = ensemble_size
        self.ensemble_num_negatives = ensemble_num_negatives or [50] * len(
            ensemble_samplers
        )

        # Validate configuration
        if (
            len(ensemble_samplers) not in [1, ensemble_size]
            and ensemble_size % len(ensemble_samplers) != 0
        ):
            raise ValueError(
                f"ensemble_samplers length ({len(ensemble_samplers)}) must be 1, "
                f"equal to ensemble_size ({ensemble_size}), or evenly divisible"
            )

        if len(self.ensemble_num_negatives) != len(ensemble_samplers):
            raise ValueError(
                f"ensemble_num_negatives length ({len(self.ensemble_num_negatives)}) "
                f"must match ensemble_samplers length ({len(ensemble_samplers)})"
            )

        # Create multiple negative samplers
        self.negative_samplers = self._create_ensemble_samplers()

    def _get_sampler_idx(self, k):
        """Get sampler index for ensemble member k."""
        if len(self.ensemble_samplers) == 1:
            return 0
        elif len(self.ensemble_samplers) == self.ensemble_size:
            return k
        else:
            members_per_sampler = self.ensemble_size // len(self.ensemble_samplers)
            return k // members_per_sampler

    def _create_sampler(self, sampler_type, num_negatives, user_id, k):
        """Create a sampler instance for an ensemble member."""
        from ..samplers import RandomSampler, PopularitySampler

        if sampler_type == "random":
            return RandomSampler(
                num_items=self.dataset.num_items,
                num_negatives=num_negatives,
                seed=1000 + user_id * self.ensemble_size + k,
            )
        elif sampler_type == "popularity":
            return PopularitySampler(
                num_items=self.dataset.num_items,
                num_negatives=num_negatives,
                seed=2000 + user_id * self.ensemble_size + k,
            )
        else:
            return RandomSampler(
                num_items=self.dataset.num_items,
                num_negatives=num_negatives,
                seed=3000 + user_id * self.ensemble_size + k,
            )

    def _create_ensemble_samplers(self):
        """Create different negative samplers for ensemble members."""
        from ..samplers import RandomSampler, PopularitySampler
        from ..samplers.genre_sampler import GenreDiverseSampler, GenreSimilarSampler

        samplers = []
        for i, sampler_type in enumerate(self.ensemble_samplers):
            num_negatives = self.ensemble_num_negatives[i]

            if sampler_type == "random":
                sampler = RandomSampler(
                    num_items=self.dataset.num_items,
                    num_negatives=num_negatives,
                    seed=42 + len(samplers),
                )
            elif sampler_type == "popularity":
                sampler = PopularitySampler(
                    num_items=self.dataset.num_items,
                    num_negatives=num_negatives,
                    seed=100 + len(samplers),
                )
            elif sampler_type == "genre_diverse":
                # Requires dataset with item_genres attribute
                if hasattr(self.dataset, 'item_genres'):
                    sampler = GenreDiverseSampler(
                        num_items=self.dataset.num_items,
                        item_genres=self.dataset.item_genres,
                        num_negatives=num_negatives,
                        seed=300 + len(samplers),
                    )
                else:
                    sampler = RandomSampler(self.dataset.num_items, num_negatives, seed=300 + len(samplers))
            elif sampler_type == "genre_similar":
                if hasattr(self.dataset, 'item_genres'):
                    sampler = GenreSimilarSampler(
                        num_items=self.dataset.num_items,
                        item_genres=self.dataset.item_genres,
                        num_negatives=num_negatives,
                        seed=400 + len(samplers),
                    )
                else:
                    sampler = RandomSampler(self.dataset.num_items, num_negatives, seed=400 + len(samplers))
            else:
                # Default to random
                sampler = RandomSampler(
                    num_items=self.dataset.num_items,
                    num_negatives=num_negatives,
                    seed=200 + len(samplers),
                )
            samplers.append(sampler)

        return samplers

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with multiple negative samples."""
        # Import here to avoid issues
        from .sequential_dataset import prepare_sequences, build_user_histories
        import pandas as pd

        # Split the unified dataset
        train_df, val_df, test_df = self.dataset.split()

        if self.format == "sequential":
            # Use standard prepare_sequences (NO pre-generated negatives)
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

            # For test: use train+val combined
            train_val_df = pd.concat([train_df, val_df])
            test_data = prepare_sequences(
                test_df,
                self.max_seq_length,
                self.model_type,
                for_val_loo=True,
                train_df=train_val_df,
            )

            # Use standard SequentialDataset (no ensemble-specific dataset needed)
            self.train_dataset = SequentialDataset(train_data, self.max_seq_length)
            self.val_dataset = SequentialDataset(val_data, self.max_seq_length)
            self.test_dataset = SequentialDataset(test_data, self.max_seq_length)

            # Build user histories for dynamic sampling
            train_histories = build_user_histories(train_df)
            val_histories = build_user_histories(val_df)
            test_histories = build_user_histories(test_df)

            # Create ensemble collate functions with reusable samplers
            self.train_collate = EnsembleSequentialCollate(
                self.dataset.num_items,
                self.ensemble_samplers,
                self.ensemble_num_negatives,
                self.ensemble_size,
                train_histories,
            )
            self.train_collate.set_dataset(self.dataset)
            
            self.val_collate = EnsembleSequentialCollate(
                self.dataset.num_items,
                self.ensemble_samplers,
                self.ensemble_num_negatives,
                self.ensemble_size,
                val_histories,
            )
            self.val_collate.set_dataset(self.dataset)
            
            self.test_collate = EnsembleSequentialCollate(
                self.dataset.num_items,
                self.ensemble_samplers,
                self.ensemble_num_negatives,
                self.ensemble_size,
                test_histories,
            )
            self.test_collate.set_dataset(self.dataset)
        else:
            raise NotImplementedError(
                "Ensemble DataModule only supports sequential format currently"
            )


class EnsembleSequentialCollate:
    """Wrapper that calls SequentialNegativeSamplingCollate n times for n ensemble members."""

    def __init__(
        self,
        num_items: int,
        ensemble_samplers: List[str],
        ensemble_num_negatives: List[int],
        ensemble_size: int,
        user_histories: dict,
    ):
        from .collate import SequentialNegativeSamplingCollate
        from ..samplers import RandomSampler, PopularitySampler
        from ..samplers.genre_sampler import GenreDiverseSampler, GenreSimilarSampler

        # Store dataset reference for genre samplers
        self.dataset = None
        self.num_items = num_items
        self.ensemble_samplers = ensemble_samplers
        self.ensemble_num_negatives = ensemble_num_negatives
        self.ensemble_size = ensemble_size
        self.user_histories = user_histories
        
        # Create ONE collate function per ensemble member
        self.collate_fns = []
        for k in range(ensemble_size):
            # Determine sampler for this ensemble member
            if len(ensemble_samplers) == 1:
                sampler_type = ensemble_samplers[0]
                num_neg = ensemble_num_negatives[0]
            elif len(ensemble_samplers) == ensemble_size:
                sampler_type = ensemble_samplers[k]
                num_neg = ensemble_num_negatives[k]
            else:
                members_per_sampler = ensemble_size // len(ensemble_samplers)
                sampler_idx = k // members_per_sampler
                sampler_type = ensemble_samplers[sampler_idx]
                num_neg = ensemble_num_negatives[sampler_idx]

            # Create sampler for this ensemble member
            if sampler_type == "random":
                sampler = RandomSampler(num_items, num_neg, seed=1000 + k)
            elif sampler_type == "popularity":
                sampler = PopularitySampler(num_items, num_neg, seed=2000 + k)
            elif sampler_type == "genre_diverse":
                # Will be set later with dataset reference
                sampler = None
            elif sampler_type == "genre_similar":
                # Will be set later with dataset reference
                sampler = None
            else:
                sampler = RandomSampler(num_items, num_neg, seed=3000 + k)

            # Create collate function with this sampler
            collate_fn = SequentialNegativeSamplingCollate(
                num_items, num_neg, user_histories, sampler
            )
            self.collate_fns.append(collate_fn)
    
    def set_dataset(self, dataset):
        """Set dataset reference for genre-based samplers."""
        from ..samplers.genre_sampler import GenreDiverseSampler, GenreSimilarSampler
        
        if not hasattr(dataset, 'item_genres'):
            return
        
        for k in range(self.ensemble_size):
            if len(self.ensemble_samplers) == 1:
                sampler_type = self.ensemble_samplers[0]
                num_neg = self.ensemble_num_negatives[0]
            elif len(self.ensemble_samplers) == self.ensemble_size:
                sampler_type = self.ensemble_samplers[k]
                num_neg = self.ensemble_num_negatives[k]
            else:
                members_per_sampler = self.ensemble_size // len(self.ensemble_samplers)
                sampler_idx = k // members_per_sampler
                sampler_type = self.ensemble_samplers[sampler_idx]
                num_neg = self.ensemble_num_negatives[sampler_idx]
            
            if sampler_type == "genre_diverse":
                sampler = GenreDiverseSampler(
                    num_items=self.num_items,
                    item_genres=dataset.item_genres,
                    num_negatives=num_neg,
                    seed=5000 + k
                )
                self.collate_fns[k].sampler = sampler
            elif sampler_type == "genre_similar":
                sampler = GenreSimilarSampler(
                    num_items=self.num_items,
                    item_genres=dataset.item_genres,
                    num_negatives=num_neg,
                    seed=6000 + k
                )
                self.collate_fns[k].sampler = sampler

    def __call__(self, batch: List[dict]) -> dict:
        """Sample negatives for all ensemble members - OPTIMIZED.
        
        Key optimizations:
        1. Stack tensors once at the start
        2. Batch user history lookups
        3. Reuse sampler instances (no recreation)
        """
        sequences = torch.stack([item["sequence"] for item in batch])
        sequence_lengths = torch.stack([item["sequence_length"] for item in batch])
        user_ids = torch.stack([item["user_id"] for item in batch])
        
        result = {
            "sequence": sequences,
            "sequence_length": sequence_lengths,
            "user_id": user_ids,
        }
        
        batch_size, seq_len = sequences.shape
        user_ids_list = user_ids.tolist()
        
        # Pre-fetch all user histories once
        user_histories_batch = [self.user_histories.get(uid, set()) for uid in user_ids_list]
        
        for k in range(len(self.collate_fns)):
            num_neg = self.collate_fns[k].num_negatives
            
            if num_neg > 0:
                sampler = self.collate_fns[k].sampler
                neg_items_k = torch.zeros(batch_size, seq_len, num_neg, dtype=torch.long)
                
                if sampler is not None:
                    total_samples = seq_len * num_neg
                    for i, user_positives in enumerate(user_histories_batch):
                        sampled_flat = sampler.sample_many(user_positives, total_samples, user_ids_list[i])
                        neg_items_k[i] = torch.from_numpy(sampled_flat.reshape(seq_len, num_neg))
                
                result[f"neg_items_{k}"] = neg_items_k
            else:
                result[f"neg_items_{k}"] = None
        
        if "target" in batch[0]:
            result["target"] = torch.stack([item["target"] for item in batch])
        
        return result
