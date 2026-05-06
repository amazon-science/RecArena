"""Composable dataset preprocessing pipeline.

Provides a base class for preprocessing transformations and a pipeline
to chain them in order. Preprocessors are stateless transformations over
pandas DataFrames that operate between raw data loading and ID remapping.

Usage::

    pipeline = PreprocessingPipeline([
        ImplicitThresholdFilter(threshold=4.0),
        MinInteractionFilter(min_interactions=5),
    ])
    cleaned_df = pipeline.transform(raw_df)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Preprocessor(ABC):
    """Abstract base for all preprocessing transformations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """String identifier for this preprocessor."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform an interactions DataFrame, returning a new DataFrame."""
        ...

    def get_params(self) -> dict:
        """Return serializable parameters for this preprocessor."""
        return {}


# ---------------------------------------------------------------------------
# Concrete Preprocessors
# ---------------------------------------------------------------------------


class MinInteractionFilter(Preprocessor):
    """Remove users and items with fewer than *min_interactions* interactions.

    Iteratively filters until convergence (no more removals possible).
    No-op when ``min_interactions == 1``.
    """

    def __init__(self, min_interactions: int = 5) -> None:
        if min_interactions < 1:
            raise ValueError(
                f"min_interactions must be >= 1, got {min_interactions}"
            )
        self._min_interactions = min_interactions

    @property
    def name(self) -> str:
        return "min_interaction_filter"

    def get_params(self) -> dict:
        return {"min_interactions": self._min_interactions}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._min_interactions <= 1:
            return df

        prev_size = len(df) + 1
        while len(df) < prev_size:
            prev_size = len(df)
            user_counts = df["user_id"].value_counts()
            item_counts = df["item_id"].value_counts()
            valid_users = user_counts[user_counts >= self._min_interactions].index
            valid_items = item_counts[item_counts >= self._min_interactions].index
            df = df[
                df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)
            ].copy()
        return df


class ImplicitThresholdFilter(Preprocessor):
    """Convert explicit ratings to implicit feedback using a threshold.

    If a ``rating`` column exists and *threshold* is not ``None``, adds
    ``implicit`` (1 where rating ≥ threshold, else 0) and keeps only rows
    with ``implicit == 1``.  If no ``rating`` column exists **or** *threshold*
    is ``None``, adds ``implicit = 1`` and returns the DataFrame unchanged in
    size.
    """

    def __init__(self, threshold: float | None = 4.0) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "implicit_threshold_filter"

    def get_params(self) -> dict:
        return {"threshold": self._threshold}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._threshold is not None and "rating" in df.columns:
            df = df.copy()
            df["implicit"] = (df["rating"] >= self._threshold).astype(int)
            df = df[df["implicit"] == 1].copy()
        elif "implicit" not in df.columns:
            df = df.copy()
            df["implicit"] = 1
        return df


class TimestampNormalizer(Preprocessor):
    """Normalize timestamps to a consistent numeric format.

    - Datetime objects → integer Unix epoch seconds.
    - Numeric values → unchanged.
    - Missing ``timestamp`` column → sequential integers ``[0, 1, …, n-1]``.
    """

    @property
    def name(self) -> str:
        return "timestamp_normalizer"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" not in df.columns:
            df["timestamp"] = range(len(df))
            return df

        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].map(lambda t: int(pd.Timestamp(t).timestamp()))
        return df


class DuplicateInteractionRemover(Preprocessor):
    """Remove duplicate ``(user_id, item_id)`` interactions.

    Keeps the row with the latest ``timestamp`` for each pair.
    """

    @property
    def name(self) -> str:
        return "duplicate_interaction_remover"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return (
            df.sort_values("timestamp")
            .drop_duplicates(subset=["user_id", "item_id"], keep="last")
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREPROCESSOR_REGISTRY: dict[str, type[Preprocessor]] = {
    "min_interaction_filter": MinInteractionFilter,
    "implicit_threshold_filter": ImplicitThresholdFilter,
    "timestamp_normalizer": TimestampNormalizer,
    "duplicate_interaction_remover": DuplicateInteractionRemover,
}


def create_default_pipeline(
    min_interactions: int = 5,
    implicit_threshold: float | None = None,
) -> "PreprocessingPipeline":
    """Create a pipeline matching current BaseDataset.load_data() behavior.

    Args:
        min_interactions: Minimum interactions per user/item to keep.
        implicit_threshold: Rating threshold for implicit feedback.
            If ``None``, the ImplicitThresholdFilter still adds an
            ``implicit = 1`` column but does not filter by rating.

    Returns:
        A :class:`PreprocessingPipeline` reproducing the inline logic.
    """
    steps: list[Preprocessor] = [
        ImplicitThresholdFilter(threshold=implicit_threshold),
        MinInteractionFilter(min_interactions=min_interactions),
    ]
    return PreprocessingPipeline(steps)


class PreprocessingPipeline:
    """Ordered sequence of Preprocessor instances."""

    def __init__(self, steps: list[Preprocessor] | None = None):
        self.steps: list[Preprocessor] = steps or []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply each preprocessor in order. Short-circuit on empty DataFrame."""
        for step in self.steps:
            df = step.transform(df)
            if df.empty:
                return df
        return df

    def to_dict(self) -> dict:
        """Serialize pipeline configuration."""
        return {
            "steps": [
                {"name": step.name, "params": step.get_params()}
                for step in self.steps
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> PreprocessingPipeline:
        """Reconstruct pipeline from serialized configuration."""
        steps: list[Preprocessor] = []
        for step_data in data.get("steps", []):
            name = step_data["name"]
            params = step_data.get("params", {})
            if name not in PREPROCESSOR_REGISTRY:
                available = ", ".join(sorted(PREPROCESSOR_REGISTRY.keys())) or "(none)"
                raise ValueError(
                    f"Unknown preprocessor name '{name}'. "
                    f"Available preprocessors: {available}"
                )
            preprocessor_cls = PREPROCESSOR_REGISTRY[name]
            steps.append(preprocessor_cls(**params))
        return cls(steps)
