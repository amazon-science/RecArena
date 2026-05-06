"""Unified dataset handling for RecArena.

This module provides a clean, unified architecture for loading and splitting
recommendation datasets:

- BaseDataset: Abstract base with common functionality
- LocalDataset: Load from local files (CSV, TSV, Parquet, DAT)
- S3Dataset: Load pre-split data from S3
- Split strategies: leave_one_out, temporal, user_based

All datasets use 1-indexed items (0 reserved for padding) for consistency.
"""

from .base_dataset import BaseDataset, LocalDataset
from .s3_dataset import S3Dataset
from .preprocessing import (
    Preprocessor,
    PreprocessingPipeline,
    create_default_pipeline,
    MinInteractionFilter,
    ImplicitThresholdFilter,
    TimestampNormalizer,
    DuplicateInteractionRemover,
    PREPROCESSOR_REGISTRY,
)
from .split_strategies import (
    LeaveOneOutSplit,
    TemporalSplit,
    UserBasedSplit,
    get_split_strategy,
)
from .sequential_dataset import (
    SequentialDataset,
    prepare_sequences,
    build_user_histories,
)
from .implicit_dataset import ImplicitDataset, prepare_implicit_interactions
from .graph_dataset import GraphDataset, to_graph
from .recm_dataset import RecMDataset
from .rec_datamodule import RecDataModule
from .ensemble_datamodule import EnsembleRecDataModule
from .traditional_datamodule import TraditionalDataModule
from .cache import DatasetCache
from .augmentation import SequenceAugmenter
from .samplers import UniformSampler, PopularitySampler
from .collate import BatchSharedNegativeSamplingCollate

__all__ = [
    # Base classes
    "BaseDataset",
    "LocalDataset",
    "S3Dataset",
    # Preprocessing
    "Preprocessor",
    "PreprocessingPipeline",
    "create_default_pipeline",
    "MinInteractionFilter",
    "ImplicitThresholdFilter",
    "TimestampNormalizer",
    "DuplicateInteractionRemover",
    "PREPROCESSOR_REGISTRY",
    # Split strategies
    "LeaveOneOutSplit",
    "TemporalSplit",
    "UserBasedSplit",
    "get_split_strategy",
    # Format-specific datasets
    "SequentialDataset",
    "ImplicitDataset",
    "GraphDataset",
    "RecMDataset",
    # Helper functions
    "prepare_sequences",
    "prepare_implicit_interactions",
    "build_user_histories",
    "to_graph",
    # DataModules
    "RecDataModule",
    "EnsembleDataModule",
    "TraditionalDataModule",
    # Utilities
    "DatasetCache",
    "SequenceAugmenter",
    # Samplers
    "UniformSampler",
    "PopularitySampler",
    # Collate
    "BatchSharedNegativeSamplingCollate",
]
