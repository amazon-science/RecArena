"""SASRec Ablation Study: Loss Functions & Architecture Enhancements.

Experiment 1: Loss functions with varying negative samples
Experiment 2: Progressive architecture enhancements (baseline → LLaMA-style)
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import filelock
import tempfile

import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from rec_arena.configs.defaults.sasrec import SASRecConfig
from rec_arena.datasets import S3Dataset, RecDataModule
from rec_arena.losses import get_loss_function
from rec_arena.models import SASRec


# Only the missing experiments (completed ones are skipped via COMPLETED set)
DATASETS = [
    "ml_100k",
    "ml_1m",
    "ml_20m",
    "goodreads",
    "netflix",
]

LOSS_CONFIGS = {
    "cross_entropy": {"loss_type": "cross_entropy"},
    "bce": {"loss_type": "bce"},
    "gbce": {"loss_type": "gbce", "loss_kwargs": {"alpha": 0.5, "t": 0.5}},
    "sampled_softmax": {"loss_type": "sampled_softmax"},
}

NEGATIVE_COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]

# Limit training batches per epoch for large datasets
MAX_TRAIN_BATCHES = 200

# Architecture ablation: progressive enhancements
ARCH_CONFIGS = {
    "baseline": {
        "position_config": {"type": "learnable"},
        "use_ligr": False,
        "use_rms_norm": False,
    },
    "+rope": {
        "position_config": {"type": "rope", "base": 10000},
        "use_ligr": False,
        "use_rms_norm": False,
    },
    "+rope+swiglu": {
        "position_config": {"type": "rope", "base": 10000},
        "use_ligr": True,  # Enables SwiGLU + gated residuals
        "use_rms_norm": False,
    },
    "+rope+swiglu+rmsnorm": {
        "position_config": {"type": "rope", "base": 10000},
        "use_ligr": True,
        "use_rms_norm": True,
    },
}


def get_max_seq_length(dataset, max_len: int = 200) -> int:
    """Compute max_seq_length from dataset as 75th percentile, capped at max_len."""
    seq_lengths = dataset.train_df.groupby("user_id").size().values
    return min(int(np.percentile(seq_lengths, 75)), max_len)


def run_experiment(
    dataset_name: str,
    loss_type: str,
    num_negatives: int,
    arch_config: dict,
    arch_name: str,
    max_epochs: int = 500,
    batch_size: int = 128,
    embedding_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    device: str = "auto",
    gpu_id: int = None,
) -> dict:
    """Run single experiment and return results."""
    # Pin to specific GPU if provided
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"

    # Load dataset
    dataset = S3Dataset(dataset_name=dataset_name, split_type="leave_one_out",
                        s3_bucket=os.environ.get("RECARENA_S3_BUCKET"))
    dataset.load_data()

    # Compute max_seq_length from dataset (75th percentile, max 200)
    max_seq_length = get_max_seq_length(dataset)
    print(f"  max_seq_length={max_seq_length} (75th percentile, max 200)")

    # Skip if num_negatives exceeds item count
    if num_negatives >= dataset.num_items:
        return {
            "skipped": True,
            "reason": f"num_negatives ({num_negatives}) >= num_items ({dataset.num_items})",
        }

    # Skip cross_entropy for large vocabularies (OOM risk)
    # Full softmax over vocab requires O(batch * seq * vocab) memory
    MAX_VOCAB_FOR_CROSS_ENTROPY = 100_000
    if loss_type == "cross_entropy" and dataset.num_items > MAX_VOCAB_FOR_CROSS_ENTROPY:
        return {
            "skipped": True,
            "reason": f"cross_entropy skipped: num_items ({dataset.num_items}) > {MAX_VOCAB_FOR_CROSS_ENTROPY} (OOM risk)",
        }

    # Create datamodule
    datamodule = RecDataModule(
        dataset,
        format="sequential",
        model_type="sasrec",
        batch_size=batch_size,
        num_workers=0,
        num_negatives=num_negatives,
        max_seq_length=max_seq_length,
    )
    datamodule.setup("fit")

    # Build config
    loss_cfg = LOSS_CONFIGS[loss_type]
    config = SASRecConfig(
        vocab_size=dataset.num_items + 3,  # +3 for PAD, UNK, MASK
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        loss_type=loss_cfg["loss_type"],
        lr=1e-3,
        weight_decay=1e-6,
        metric_compute_interval=10,  # Compute NDCG every 10 epochs (expensive for large vocabs)
        **arch_config,
    )

    # Create model with loss function
    model = SASRec(config)
    loss_kwargs = loss_cfg.get("loss_kwargs", {})
    model.set_loss_fn(
        get_loss_function(loss_cfg["loss_type"], "sequential", **loss_kwargs)
    )

    # Checkpoint best model based on val_ndcg@10 (only every 10 epochs when it's computed)
    checkpoint_dir = tempfile.mkdtemp()
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_ndcg@10",
        mode="max",
        save_top_k=1,
        every_n_epochs=10,  # Only checkpoint when NDCG is actually computed
    )

    # Limit batches per epoch for large datasets
    num_train_samples = len(dataset.train_df)
    batches_per_epoch = num_train_samples // batch_size
    limit_train_batches = (
        min(MAX_TRAIN_BATCHES, batches_per_epoch)
        if batches_per_epoch > MAX_TRAIN_BATCHES
        else None
    )
    if limit_train_batches:
        print(
            f"  Limiting to {limit_train_batches} train batches/epoch (dataset has {batches_per_epoch})"
        )

    # Train with early stopping on validation metric (not loss - different losses have different scales)
    # patience=100 epochs = 10 actual NDCG computations with interval=10
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        precision="16-mixed",  # Enable Flash Attention + faster training
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
        trainer.fit(model, datamodule)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                "skipped": True,
                "reason": f"OOM: {e}",
                "dataset": dataset_name,
                "loss_type": loss_type,
                "num_negatives": num_negatives,
                "arch": arch_name,
                "num_items": dataset.num_items,
            }
        raise
    train_time = time.time() - start_time

    # Test with best model. Reload the best checkpoint manually (PyTorch >= 2.6
    # defaults torch.load to weights_only=True, which cannot unpickle the config
    # object stored in the checkpoint); allowlist SASRecConfig and load explicitly.
    datamodule.setup("test")
    best_path = checkpoint_callback.best_model_path
    if best_path:
        torch.serialization.add_safe_globals([SASRecConfig])
        best_model = SASRec.load_from_checkpoint(best_path, config=config)
        test_results = trainer.test(best_model, datamodule, verbose=False)
    else:
        test_results = trainer.test(model, datamodule, verbose=False)

    return {
        "dataset": dataset_name,
        "loss_type": loss_type,
        "num_negatives": num_negatives,
        "arch": arch_name,
        "num_items": dataset.num_items,
        "train_time_s": round(train_time, 2),
        "epochs_trained": trainer.current_epoch + 1,
        **{k: round(v, 4) for k, v in test_results[0].items()},
    }


def _run_single(args):
    """Wrapper for parallel execution."""
    return run_experiment(**args)


def run_loss_ablation(
    datasets: list, max_epochs: int, output_dir: Path, device: str, num_gpus: int = 1, shard=None
):
    """Experiment 1: Loss functions × negative counts."""

    # Build all experiment configs. Full cross-entropy is the reference loss
    # (no negatives), so it runs once per dataset; the sampling-based losses run
    # across the full negative-count grid.
    experiments = []

    for ds in datasets:
        experiments.append(
            {
                "dataset_name": ds,
                "loss_type": "cross_entropy",
                "num_negatives": 0,
                "arch_config": ARCH_CONFIGS["baseline"],
                "arch_name": "baseline",
                "max_epochs": max_epochs,
                "device": device,
            }
        )

    for ds, loss, num_neg in product(datasets, ["bce", "gbce", "sampled_softmax"], NEGATIVE_COUNTS):
        experiments.append(
            {
                "dataset_name": ds,
                "loss_type": loss,
                "num_negatives": num_neg,
                "arch_config": ARCH_CONFIGS["baseline"],
                "arch_name": "baseline",
                "max_epochs": max_epochs,
                "device": device,
            }
        )
    if shard is not None:
        shard_idx, num_shards = shard
        experiments = [e for i, e in enumerate(experiments) if i % num_shards == shard_idx]
    print(f"Running {len(experiments)} loss-ablation experiments")

    # Run experiments (parallel if multiple GPUs)
    results = _run_experiments_parallel(
        experiments, output_dir, "loss_ablation", num_gpus
    )
    return pd.DataFrame(results)


def run_arch_ablation(
    datasets: list, max_epochs: int, output_dir: Path, device: str, num_gpus: int = 1
):
    """Experiment 2: Architecture enhancements."""

    experiments = []
    for ds, (arch_name, arch_cfg) in product(datasets, ARCH_CONFIGS.items()):
        experiments.append(
            {
                "dataset_name": ds,
                "loss_type": "cross_entropy",
                "num_negatives": 0,
                "arch_config": arch_cfg,
                "arch_name": arch_name,
                "max_epochs": max_epochs,
                "device": device,
            }
        )

    results = _run_experiments_parallel(
        experiments, output_dir, "arch_ablation", num_gpus
    )
    return pd.DataFrame(results)


def append_result(output_dir: Path, name: str, result: dict):
    """Append a single result to CSV file with file locking for concurrent writes."""
    csv_path = output_dir / f"{name}.csv"
    lock_path = output_dir / f"{name}.csv.lock"

    with filelock.FileLock(lock_path):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        else:
            df = pd.DataFrame([result])
        df.to_csv(csv_path, index=False)


def _run_experiments_parallel(
    experiments: list, output_dir: Path, name: str, num_gpus: int
):
    """Run experiments in parallel across GPUs."""
    results = []

    if num_gpus <= 1:
        # Sequential execution
        for i, exp in enumerate(experiments):
            print(
                f"\n[{i+1}/{len(experiments)}] {exp['dataset_name']} | {exp['loss_type']} | neg={exp['num_negatives']} | {exp['arch_name']}"
            )
            result = run_experiment(**exp)
            results.append(result)
            append_result(output_dir, name, result)
    else:
        # Parallel execution across GPUs
        print(f"\nRunning {len(experiments)} experiments across {num_gpus} GPUs...")

        # Assign GPU IDs round-robin
        for i, exp in enumerate(experiments):
            exp["gpu_id"] = i % num_gpus

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(_run_single, exp): exp for exp in experiments}
            for future in as_completed(futures):
                exp = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    append_result(output_dir, name, result)
                    print(
                        f"✓ {exp['dataset_name']} | {exp['loss_type']} | neg={exp['num_negatives']}"
                    )
                except Exception as e:
                    print(f"✗ {exp['dataset_name']} | {exp['loss_type']} failed: {e}")
                    error_result = {
                        "error": str(e),
                        **{k: v for k, v in exp.items() if k != "arch_config"},
                    }
                    results.append(error_result)
                    append_result(output_dir, name, error_result)

    print(f"\nResults saved to {output_dir / f'{name}.csv'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="SASRec Ablation Study")
    parser.add_argument(
        "--experiment", choices=["loss", "arch", "both"], default="both"
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/results)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs for parallel execution"
    )
    parser.add_argument("--shard", type=str, default=None, help="Shard in format INDEX/TOTAL, e.g. 0/2 or 1/2")
    args = parser.parse_args()

    shard = None
    if args.shard:
        idx, total = args.shard.split("/")
        shard = (int(idx), int(total))

    # Default output dir to experiments/results
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "timestamp": datetime.now().isoformat(),
        "datasets": args.datasets,
        "max_epochs": args.max_epochs,
        "loss_configs": LOSS_CONFIGS,
        "negative_counts": NEGATIVE_COUNTS,
        "arch_configs": {k: str(v) for k, v in ARCH_CONFIGS.items()},
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.experiment in ["loss", "both"]:
        run_loss_ablation(
            args.datasets, args.max_epochs, output_dir, args.device, args.num_gpus, shard=shard
        )

    if args.experiment in ["arch", "both"]:
        run_arch_ablation(
            args.datasets, args.max_epochs, output_dir, args.device, args.num_gpus
        )

    print(f"\n{'='*60}\nAll experiments complete! Results in {output_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
