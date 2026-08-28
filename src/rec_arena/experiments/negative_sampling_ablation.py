"""Negative Sampling Ablation Study.

Tests sampled softmax with:
  - Negative scope:     per_position [B, S, N]  vs  batch_shared [B, N]
  - Sampling strategy:  uniform  vs  popularity
  - Negative count:     128 (per_position)  vs  1024 (batch_shared)

Metrics reported at k = 10, 20, 50.
"""

import argparse
import os

import tempfile
import time
from itertools import product
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from rec_arena.configs.defaults.sasrec import SASRecConfig
from rec_arena.datasets import S3Dataset, RecDataModule, build_user_histories
from rec_arena.datasets.collate import (
    SequentialNegativeSamplingCollate,
    BatchSharedNegativeSamplingCollate,
)
from rec_arena.datasets.samplers import UniformSampler, PopularitySampler
from rec_arena.datasets.sequential_dataset import (
    prepare_sequences,
    SequentialDataset,
)
from rec_arena.losses import get_loss_function
from rec_arena.models import SASRec

import torch
from torch.utils.data import DataLoader

DATASETS = ["ml_100k", "ml_1m", "ratebeer", "amazon_beauty_2014", "netflix", "ml_20m"]
MAX_TRAIN_BATCHES = 200
K_VALUES = [10, 20, 50]

# (scope, strategy, num_negatives)
CONFIGS = [
    ("per_position", "uniform", 128),
    ("per_position", "popularity", 128),
    ("batch_shared", "uniform", 128),  # same N, different scope
    ("batch_shared", "popularity", 128),
    ("batch_shared", "uniform", 1024),  # larger pool, batch-shared
    ("batch_shared", "popularity", 1024),
]


def get_max_seq_length(dataset, max_len: int = 200) -> int:
    seq_lengths = dataset.train_df.groupby("user_id").size().values
    return min(int(np.percentile(seq_lengths, 75)), max_len)


def build_datamodule(
    dataset, max_seq_length, batch_size, scope, strategy, num_negatives
):
    """Build a RecDataModule with the right collate + sampler."""
    train_df, val_df, test_df = dataset.split()

    train_histories = build_user_histories(train_df)
    train_val_df = pd.concat([train_df, val_df])
    test_histories = build_user_histories(train_val_df)

    train_data = prepare_sequences(train_df, max_seq_length, "sasrec")
    val_data = prepare_sequences(
        val_df, max_seq_length, "sasrec", for_val_loo=True, train_df=train_df
    )
    test_data = prepare_sequences(
        test_df, max_seq_length, "sasrec", for_val_loo=True, train_df=train_val_df
    )

    train_ds = SequentialDataset(train_data, max_seq_length)
    val_ds = SequentialDataset(val_data, max_seq_length)
    test_ds = SequentialDataset(test_data, max_seq_length)

    def make_sampler(histories_df):
        if strategy == "popularity":
            return PopularitySampler(dataset.num_items, num_negatives, train_df)
        return UniformSampler(dataset.num_items, num_negatives)

    CollateClass = (
        SequentialNegativeSamplingCollate
        if scope == "per_position"
        else BatchSharedNegativeSamplingCollate
    )

    def make_collate(histories):
        return CollateClass(
            dataset.num_items,
            num_negatives,
            histories,
            sampler=make_sampler(train_df),
        )

    train_collate = make_collate(train_histories)
    val_collate = make_collate(train_histories)
    test_collate = make_collate(test_histories)

    # Wrap in a simple object the Trainer can use via DataLoader
    class _DM:
        def __init__(self):
            self._train = (train_ds, train_collate)
            self._val = (val_ds, val_collate)
            self._test = (test_ds, test_collate)

        def _loader(self, ds, collate, shuffle):
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=collate,
                pin_memory=True,
            )

        def train_dataloader(self):
            return self._loader(*self._train, True)

        def val_dataloader(self):
            return self._loader(*self._val, False)

        def test_dataloader(self):
            return self._loader(*self._test, False)

    return _DM()


def run_experiment(
    dataset_name: str,
    scope: str,
    strategy: str,
    num_negatives: int,
    max_epochs: int = 500,
    batch_size: int = 128,
    embedding_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
) -> dict:
    label = f"{scope}|{strategy}|neg={num_negatives}"
    print(f"\n{'='*60}\nDataset: {dataset_name}  Config: {label}\n{'='*60}")

    dataset = S3Dataset(
        dataset_name=dataset_name,
        split_type="leave_one_out",
        s3_bucket=os.environ.get("RECARENA_S3_BUCKET"),
    )
    dataset.load_data()

    if num_negatives >= dataset.num_items:
        return {
            "dataset": dataset_name,
            "scope": scope,
            "strategy": strategy,
            "num_negatives": num_negatives,
            "skipped": True,
            "reason": f"num_negatives >= num_items ({dataset.num_items})",
        }

    max_seq_length = get_max_seq_length(dataset)
    print(f"  Items: {dataset.num_items}, MaxSeq: {max_seq_length}")

    dm = build_datamodule(
        dataset, max_seq_length, batch_size, scope, strategy, num_negatives
    )

    config = SASRecConfig(
        vocab_size=dataset.num_items + 3,
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        loss_type="sampled_softmax",
        lr=1e-3,
        weight_decay=1e-6,
        metric_compute_interval=10,
        val_k_values=K_VALUES,
    )
    model = SASRec(config)
    model.set_loss_fn(get_loss_function("sampled_softmax", "sequential"))

    checkpoint_dir = tempfile.mkdtemp()
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_ndcg@10",
        mode="max",
        save_top_k=1,
        every_n_epochs=10,
    )

    batches_per_epoch = len(dataset.train_df) // batch_size
    limit_train_batches = (
        min(MAX_TRAIN_BATCHES, batches_per_epoch)
        if batches_per_epoch > MAX_TRAIN_BATCHES
        else None
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision="16-mixed",
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=True,
        limit_train_batches=limit_train_batches,
        callbacks=[
            EarlyStopping(monitor="val_ndcg@10", patience=100, mode="max"),
            checkpoint_callback,
        ],
    )

    start_time = time.time()
    try:
        trainer.fit(model, dm)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                "dataset": dataset_name,
                "scope": scope,
                "strategy": strategy,
                "num_negatives": num_negatives,
                "skipped": True,
                "reason": f"OOM: {e}",
            }
        raise
    train_time = time.time() - start_time

    best_path = checkpoint_callback.best_model_path
    if best_path:
        # PyTorch >= 2.6 defaults torch.load to weights_only=True; allowlist the
        # config object so the best checkpoint can be reloaded before testing.
        torch.serialization.add_safe_globals([SASRecConfig])
        best_model = SASRec.load_from_checkpoint(best_path, config=config)
        test_results = trainer.test(best_model, dm, verbose=False)
    else:
        test_results = trainer.test(model, dm, verbose=False)

    return {
        "dataset": dataset_name,
        "scope": scope,
        "strategy": strategy,
        "num_negatives": num_negatives,
        "num_items": dataset.num_items,
        "train_time_s": round(train_time, 2),
        "epochs_trained": trainer.current_epoch + 1,
        **{k: round(v, 4) for k, v in test_results[0].items()},
    }


def append_result(csv_path: Path, result: dict):
    lock_path = csv_path.with_suffix(".csv.lock")
    with filelock.FileLock(lock_path):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        else:
            df = pd.DataFrame([result])
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Negative Sampling Ablation Study")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/results)",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "negative_sampling_ablation.csv"
    print(f"Results → {csv_path}")

    total = len(args.datasets) * len(CONFIGS)
    i = 0
    for dataset_name, (scope, strategy, num_neg) in product(args.datasets, CONFIGS):
        i += 1
        print(f"\n[{i}/{total}]")
        result = run_experiment(
            dataset_name, scope, strategy, num_neg, max_epochs=args.max_epochs
        )
        append_result(csv_path, result)
        ndcg10 = result.get("test_ndcg@10", 0)
        ndcg20 = result.get("test_ndcg@20", 0)
        ndcg50 = result.get("test_ndcg@50", 0)
        print(f"  → NDCG@10: {ndcg10:.4f}  @20: {ndcg20:.4f}  @50: {ndcg50:.4f}")

    print("\n" + "=" * 80)
    print("NEGATIVE SAMPLING ABLATION RESULTS")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    cols = [
        "dataset",
        "scope",
        "strategy",
        "num_negatives",
        "test_ndcg@10",
        "test_ndcg@20",
        "test_ndcg@50",
        "test_recall@10",
        "test_recall@20",
        "test_recall@50",
    ]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
