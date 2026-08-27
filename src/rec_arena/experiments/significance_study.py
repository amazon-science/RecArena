"""Significance study: multi-seed factorial ablation with per-user predictions.

Design principles:
  - Same seed controls ALL randomness within a run (init, shuffle, dropout).
  - Within a seed, all configs share identical data ordering because the
    DataModule is constructed with the same seed -> same negative samples.
  - Different seeds produce different data orderings.
  - Per-user predictions are saved so bootstrap CIs can be computed offline.
  - FLOPs (via torch.profiler) and peak GPU memory are recorded per run.

Usage:
  # Run all datasets on 8 GPUs (round-robin):
  python -m rec_arena.experiments.significance_study \
      --num-gpus 8 --output-dir results/significance

  # Single dataset, single seed (debugging):
  python -m rec_arena.experiments.significance_study \
      --datasets ml_1m --seeds 42 --num-gpus 1
"""

import argparse
import json
import os
import time
import tempfile
from datetime import datetime
from itertools import product
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from rec_arena.configs.defaults.sasrec import SASRecConfig
from rec_arena.datasets import S3Dataset, RecDataModule
from rec_arena.losses import get_loss_function
from rec_arena.metrics import MetricCalculator
from rec_arena.models import SASRec
from rec_arena.utils.reproducibility import set_seed


# ── Experiment grid ───────────────────────────────────────────────────────────

SEEDS = [42, 43, 44]

CONFIGS = {
    "learnable+ligr=False+rms=False": {
        "position_config": {"type": "learnable"},
        "use_ligr": False,
        "use_rms_norm": False,
    },
    "learnable+ligr=False+rms=True": {
        "position_config": {"type": "learnable"},
        "use_ligr": False,
        "use_rms_norm": True,
    },
    "learnable+ligr=True+rms=False": {
        "position_config": {"type": "learnable"},
        "use_ligr": True,
        "use_rms_norm": False,
    },
    "learnable+ligr=True+rms=True": {
        "position_config": {"type": "learnable"},
        "use_ligr": True,
        "use_rms_norm": True,
    },
    "rope+ligr=False+rms=False": {
        "position_config": {"type": "rope", "base": 10000},
        "use_ligr": False,
        "use_rms_norm": False,
    },
    "rope+ligr=False+rms=True": {
        "position_config": {"type": "rope", "base": 10000},
        "use_ligr": False,
        "use_rms_norm": True,
    },
    "rope+ligr=True+rms=False": {
        "position_config": {"type": "rope", "base": 10000},
        "use_ligr": True,
        "use_rms_norm": False,
    },
    "rope+ligr=True+rms=True": {
        "position_config": {"type": "rope", "base": 10000},
        "use_ligr": True,
        "use_rms_norm": True,
    },
}

# Datasets where full epochs are feasible on A10
FULL_EPOCH_DATASETS = ["ml_100k", "ml_1m", "amazon_beauty_2014", "ratebeer", "goodreads"]
# Datasets that need batch-capped epochs
CAPPED_DATASETS = ["ml_20m", "netflix"]
MAX_TRAIN_BATCHES_CAPPED = 400

ALL_DATASETS = FULL_EPOCH_DATASETS + CAPPED_DATASETS

K_VALUES = [10, 20, 50, 100]

# Universal max epochs — early stopping handles convergence
MAX_EPOCHS = 500

# Per-dataset settings: (metric_interval, patience)
# metric_interval: how often to compute full-ranking NDCG (expensive for large test sets)
# patience: epochs without NDCG improvement before stopping
# Effective checks before stop = patience / metric_interval ≈ 10 for all datasets
DATASET_SETTINGS = {
    # Small/cheap validation → check every epoch
    "ml_100k":            {"metric_interval": 1,  "patience": 10},
    "ml_1m":              {"metric_interval": 1,  "patience": 10},
    # Moderate validation cost → check every 5 epochs
    "amazon_beauty_2014": {"metric_interval": 5,  "patience": 50},
    "ratebeer":           {"metric_interval": 5,  "patience": 50},
    "goodreads":          {"metric_interval": 5,  "patience": 50},
    # Capped datasets → check every 10 epochs (same as original paper)
    "ml_20m":             {"metric_interval": 10, "patience": 100},
    "netflix":            {"metric_interval": 10, "patience": 100},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_max_seq_length(dataset, max_len: int = 200) -> int:
    seq_lengths = dataset.train_df.groupby("user_id").size().values
    return min(int(np.percentile(seq_lengths, 85)), max_len)


def _reinit_shared_params(model, seed: int):
    """Re-initialize architecture-shared parameters with a dedicated RNG.

    This ensures that item_embedding, attention projections, and FFN weights
    receive identical values across configs for the same seed, regardless of
    config-specific allocations (pos_embedding, gates, rope, SwiGLU vs
    standard FFN) that would otherwise shift the global RNG.

    Each parameter group gets its own deterministically-seeded generator so
    that the number of draws in one group cannot affect another.

    Config-specific parameters (pos_embedding, gate_attn, gate_ffn, rope
    buffers, lora) are NOT touched.
    """
    def _make_gen(salt: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed * 1000003 + salt)
        return g

    # Item embedding (salt=0)
    g = _make_gen(0)
    model.item_embedding.weight.data.normal_(0, 0.02, generator=g)
    with torch.no_grad():
        model.item_embedding.weight.data[0].zero_()

    # Per-layer shared params: each layer gets its own salt range
    for layer_idx, block in enumerate(model.transformer_blocks):
        base_salt = (layer_idx + 1) * 100

        # Attention QKV and output projections
        block.attention.qkv_proj.weight.data.normal_(0, 0.02, generator=_make_gen(base_salt + 1))
        block.attention.out_proj.weight.data.normal_(0, 0.02, generator=_make_gen(base_salt + 2))

        # Normalization layers (deterministic, no RNG needed)
        if hasattr(block.attn_norm, 'weight'):
            block.attn_norm.weight.data.fill_(1.0)
        if hasattr(block.attn_norm, 'bias') and block.attn_norm.bias is not None:
            block.attn_norm.bias.data.zero_()
        if hasattr(block.ffn_norm, 'weight'):
            block.ffn_norm.weight.data.fill_(1.0)
        if hasattr(block.ffn_norm, 'bias') and block.ffn_norm.bias is not None:
            block.ffn_norm.bias.data.zero_()

        # FFN weights — use same salts regardless of SwiGLU vs standard
        # so the attention weights in the next layer are unaffected
        if hasattr(block, 'ffn'):
            # SwiGLU: linear1, gate, linear2
            block.ffn.linear1.weight.data.normal_(0, 0.02, generator=_make_gen(base_salt + 3))
            block.ffn.linear2.weight.data.normal_(0, 0.02, generator=_make_gen(base_salt + 4))
            block.ffn.gate.weight.data.normal_(0, 0.02, generator=_make_gen(base_salt + 5))
        else:
            # Standard: dense1, dense2
            block.dense1.weight.data.normal_(0, 0.02, generator=_make_gen(base_salt + 3))
            block.dense2.weight.data.normal_(0, 0.02, generator=_make_gen(base_salt + 4))

    # Final layer norm (deterministic)
    if hasattr(model, 'layer_norm'):
        model.layer_norm.weight.data.fill_(1.0)
        if model.layer_norm.bias is not None:
            model.layer_norm.bias.data.zero_()


def profile_flops(model, datamodule, device="cuda"):
    """Profile FLOPs for a single forward pass."""
    model.eval()
    model.to(device)
    batch = next(iter(datamodule.train_dataloader()))
    sequences = batch["sequence"].to(device)
    seq_lengths = batch["sequence_length"].to(device)

    try:
        from torch.profiler import profile, ProfilerActivity, record_function
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with record_function("forward"):
                with torch.no_grad():
                    _ = model(sequences, seq_lengths)

        total_flops = sum(
            e.flops for e in prof.key_averages() if e.flops is not None and e.flops > 0
        )
        return total_flops
    except Exception as e:
        print(f"  FLOPs profiling failed: {e}")
        return -1


# S3 bucket/prefix for persisting predictions (set via --s3-path or env var)
_S3_PATH = None


def _save_predictions_to_s3(
    dataset_name: str,
    arch_name: str,
    seed: int,
    pred_data: dict,
    per_user_metrics: dict,
):
    """Save predictions directly to S3. No local copy is kept."""
    if _S3_PATH is None:
        print("  Skipping prediction save (no --s3-path configured)")
        return

    fname = f"{dataset_name}_{arch_name}_seed{seed}.npz"
    tmp_path = Path(tempfile.gettempdir()) / fname

    try:
        import boto3

        np.savez_compressed(
            tmp_path,
            user_ids=pred_data["user_ids"],
            targets=pred_data["targets"],
            predictions=pred_data["predictions"],
            **{f"per_user_{k}": v for k, v in per_user_metrics.items()},
        )

        s3_path = _S3_PATH.replace("s3://", "")
        bucket = s3_path.split("/")[0]
        prefix = "/".join(s3_path.split("/")[1:])
        key = f"{prefix}/{fname}"

        boto3.client("s3").upload_file(str(tmp_path), bucket, key)
        print(f"  Uploaded predictions to s3://{bucket}/{key}")
    except Exception as e:
        print(f"  S3 upload failed (non-fatal): {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


def collect_per_user_predictions(model, datamodule, device="cuda"):
    """Run inference and collect per-user prediction scores + targets.

    Returns dict with:
      user_ids:    np.array [N]
      targets:     np.array [N]        (ground-truth item id)
      predictions: np.array [N, V]     (full-vocab logit/score per user)
    """
    model.eval()
    model.to(device)

    all_user_ids, all_targets, all_preds = [], [], []

    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            sequences = batch["sequence"].to(device)
            seq_lengths = batch["sequence_length"].to(device)
            targets = batch["target"]
            user_ids = batch["user_id"]

            preds = model.predict_next(sequences, seq_lengths)  # [B, V]
            # Mask special tokens
            preds[:, :3] = float("-inf")

            all_user_ids.append(user_ids.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return {
        "user_ids": np.concatenate(all_user_ids),
        "targets": np.concatenate(all_targets),
        "predictions": np.concatenate(all_preds, axis=0),
    }


def compute_metrics_from_predictions(predictions_np, targets_np, k_values):
    """Compute all metrics from numpy arrays."""
    preds_t = torch.from_numpy(predictions_np).float()
    tgts_t = torch.from_numpy(targets_np).long()
    calc = MetricCalculator(k_values=k_values)
    return calc.calculate_all(preds_t, tgts_t)


def compute_per_user_metrics(predictions_np, targets_np, k_values):
    """Compute per-user NDCG/HR for bootstrap. Returns dict of np arrays."""
    preds_t = torch.from_numpy(predictions_np).float()
    tgts_t = torch.from_numpy(targets_np).long()
    n_users = preds_t.shape[0]

    per_user = {}
    for k in k_values:
        _, topk_indices = torch.topk(preds_t, k, dim=-1)  # [N, k]
        hits = (topk_indices == tgts_t.unsqueeze(1)).float()  # [N, k]

        # HR@k: 1 if target in top-k
        per_user[f"hr@{k}"] = hits.any(dim=1).numpy()

        # NDCG@k
        positions = torch.arange(1, k + 1, dtype=torch.float32).unsqueeze(0)
        dcg = (hits / torch.log2(positions + 1)).sum(dim=1)
        per_user[f"ndcg@{k}"] = dcg.numpy()

        # MRR@k
        hit_positions = hits * positions
        hit_positions[hit_positions == 0] = float("inf")
        first_hit = hit_positions.min(dim=1).values
        rr = torch.where(first_hit == float("inf"), torch.zeros(1), 1.0 / first_hit)
        per_user[f"mrr@{k}"] = rr.numpy()

    return per_user


# ── Single experiment ─────────────────────────────────────────────────────────

def run_single_experiment(
    dataset_name: str,
    arch_name: str,
    arch_config: dict,
    seed: int,
    gpu_id: int = 0,
    output_dir: Path = None,
) -> dict:
    """Run one (dataset, config, seed) experiment. Returns result dict."""

    # Pin GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set seed for full reproducibility
    set_seed(seed)

    is_capped = dataset_name in CAPPED_DATASETS
    ds_settings = DATASET_SETTINGS.get(
        dataset_name, {"metric_interval": 5, "patience": 50}
    )
    metric_interval = ds_settings["metric_interval"]
    patience = ds_settings["patience"]

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = S3Dataset(dataset_name=dataset_name, split_type="leave_one_out",
                        s3_bucket=os.environ.get("RECARENA_S3_BUCKET"))
    dataset.load_data()
    max_seq_length = get_max_seq_length(dataset)

    # Skip if vocab too large for cross-entropy
    if dataset.num_items > 100_000:
        return {"skipped": True, "reason": f"num_items={dataset.num_items} > 100K"}

    datamodule = RecDataModule(
        dataset,
        format="sequential",
        model_type="sasrec",
        batch_size=128,
        num_workers=0,
        num_negatives=0,
        max_seq_length=max_seq_length,
    )
    datamodule.setup("fit")

    # ── Model ─────────────────────────────────────────────────────────────
    config = SASRecConfig(
        vocab_size=dataset.num_items + 3,
        max_seq_length=max_seq_length,
        embedding_dim=64,
        num_heads=2,
        num_layers=2,
        loss_type="cross_entropy",
        lr=1e-3,
        weight_decay=1e-6,
        metric_compute_interval=metric_interval,
        **arch_config,
    )
    model = SASRec(config)
    model.set_loss_fn(get_loss_function("cross_entropy", "sequential"))

    # Re-initialize shared parameters with a config-independent RNG so that
    # item_embedding, attention, and FFN weights are identical across configs
    # for the same seed.  Config-specific params (pos_embedding, gates, rope)
    # keep their original init.
    _reinit_shared_params(model, seed)

    # Re-seed global RNG so that DataLoader shuffle order, dropout masks, etc.
    # are identical across configs for the same seed.  Without this, the
    # different number of RNG draws during model.__init__ (e.g. pos_embedding
    # vs rope buffers) would cause the global state to diverge.
    set_seed(seed)

    # ── Training ──────────────────────────────────────────────────────────
    ckpt_dir = tempfile.mkdtemp()
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_ndcg@10",
        mode="max",
        save_top_k=1,
        every_n_epochs=metric_interval,
    )

    limit_train_batches = None
    if is_capped:
        batches_per_epoch = len(dataset.train_df) // 128
        if batches_per_epoch > MAX_TRAIN_BATCHES_CAPPED:
            limit_train_batches = MAX_TRAIN_BATCHES_CAPPED

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cuda",
        precision="16-mixed",
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=True,
        limit_train_batches=limit_train_batches,
        callbacks=[
            EarlyStopping(monitor="val_ndcg@10", patience=patience, mode="max"),
            ckpt_cb,
        ],
    )

    # Memory tracking
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        trainer.fit(model, datamodule)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                "skipped": True,
                "reason": f"OOM: {e}",
                "dataset": dataset_name,
                "arch": arch_name,
                "seed": seed,
            }
        raise

    train_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    epochs_trained = trainer.current_epoch + 1

    # ── FLOPs profiling (single forward pass) ─────────────────────────────
    flops = profile_flops(model, datamodule, device="cuda")

    # ── Test with best checkpoint ─────────────────────────────────────────
    datamodule.setup("test")
    best_path = ckpt_cb.best_model_path

    if best_path:
        # Load best checkpoint (allowlist our config class for safe deserialization)
        torch.serialization.add_safe_globals([SASRecConfig])
        best_model = SASRec.load_from_checkpoint(best_path, config=config)
        best_model.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
    else:
        best_model = model

    best_model.eval()
    best_model.to("cuda")

    # ── Collect per-user predictions ──────────────────────────────────────
    pred_data = collect_per_user_predictions(best_model, datamodule, device="cuda")

    # ── Compute aggregate metrics ─────────────────────────────────────────
    metrics = compute_metrics_from_predictions(
        pred_data["predictions"], pred_data["targets"], K_VALUES
    )

    # ── Compute per-user metrics (for bootstrap) ──────────────────────────
    per_user_metrics = compute_per_user_metrics(
        pred_data["predictions"], pred_data["targets"], K_VALUES
    )

    # ── Save predictions to S3 only (no local copy) ───────────────────
    _save_predictions_to_s3(
        dataset_name, arch_name, seed, pred_data, per_user_metrics
    )

    # ── Build result row ──────────────────────────────────────────────────
    result = {
        "dataset": dataset_name,
        "arch": arch_name,
        "seed": seed,
        "num_items": dataset.num_items,
        "num_users": dataset.num_users,
        "num_train_interactions": len(dataset.train_df),
        "max_seq_length": max_seq_length,
        "batch_capped": is_capped,
        "train_time_s": round(train_time, 2),
        "epochs_trained": epochs_trained,
        "peak_memory_mb": round(peak_memory_mb, 1),
        "flops_forward": flops,
    }
    for k, v in metrics.items():
        result[f"test_{k}"] = round(v, 6)

    return result


# ── Parallel orchestration ────────────────────────────────────────────────────

def append_result(output_dir: Path, name: str, result: dict):
    csv_path = output_dir / f"{name}.csv"
    lock_path = output_dir / f"{name}.csv.lock"
    with filelock.FileLock(lock_path):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        else:
            df = pd.DataFrame([result])
        df.to_csv(csv_path, index=False)


def _worker(args):
    """Subprocess entry point."""
    # Restore module-level S3 path in spawned child process
    global _S3_PATH
    _S3_PATH = args.pop("_s3_path", None)
    return run_single_experiment(**args)


def build_experiment_list(datasets, configs, seeds):
    """Build flat list of experiments, ordered for good GPU utilisation.

    Strategy: interleave datasets so that expensive ones don't all land on
    the same GPU.  Within a dataset, group by seed so that configs within
    the same seed share the same data ordering.
    """
    experiments = []
    for seed in seeds:
        for ds in datasets:
            for cfg_name, cfg in configs.items():
                experiments.append({
                    "dataset_name": ds,
                    "arch_name": cfg_name,
                    "arch_config": cfg,
                    "seed": seed,
                })
    return experiments


def run_all(
    datasets: list,
    configs: dict,
    seeds: list,
    output_dir: Path,
    num_gpus: int = 1,
):
    experiments = build_experiment_list(datasets, configs, seeds)
    total = len(experiments)
    print(f"\n{'='*60}")
    print(f"Significance study: {total} experiments")
    print(f"  Datasets: {datasets}")
    print(f"  Configs:  {list(configs.keys())}")
    print(f"  Seeds:    {seeds}")
    print(f"  GPUs:     {num_gpus}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}\n")

    # Save experiment manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "datasets": datasets,
        "configs": list(configs.keys()),
        "seeds": seeds,
        "k_values": K_VALUES,
        "max_epochs": MAX_EPOCHS,
        "dataset_settings": DATASET_SETTINGS,
        "batch_size": 128,
        "embedding_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "full_epoch_datasets": FULL_EPOCH_DATASETS,
        "capped_datasets": CAPPED_DATASETS,
        "max_train_batches_capped": MAX_TRAIN_BATCHES_CAPPED,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if num_gpus <= 1:
        # Sequential
        for i, exp in enumerate(experiments):
            exp["gpu_id"] = 0
            exp["output_dir"] = output_dir
            print(f"\n[{i+1}/{total}] {exp['dataset_name']} | {exp['arch_name']} | seed={exp['seed']}")
            result = run_single_experiment(**exp)
            append_result(output_dir, "significance_results", result)
            ndcg = result.get("test_ndcg@10", "skipped")
            print(f"  -> NDCG@10={ndcg}")
    else:
        # Parallel across GPUs via ProcessPoolExecutor with 'spawn' context.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")

        # Round-robin GPU assignment
        for i, exp in enumerate(experiments):
            exp["gpu_id"] = i % num_gpus
            exp["output_dir"] = output_dir
            exp["_s3_path"] = _S3_PATH

        print(f"Dispatching {total} experiments across {num_gpus} GPUs...\n")
        with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            futures = {executor.submit(_worker, exp): exp for exp in experiments}
            done = 0
            for future in as_completed(futures):
                done += 1
                exp = futures[future]
                tag = f"{exp['dataset_name']}|{exp['arch_name']}|seed={exp['seed']}"
                try:
                    result = future.result()
                    append_result(output_dir, "significance_results", result)
                    ndcg = result.get("test_ndcg@10", "skipped")
                    print(f"  [{done}/{total}] ✓ {tag}  NDCG@10={ndcg}")
                except Exception as e:
                    print(f"  [{done}/{total}] ✗ {tag}  ERROR: {e}")
                    error_result = {
                        "dataset": exp["dataset_name"],
                        "arch": exp["arch_name"],
                        "seed": exp["seed"],
                        "error": str(e),
                    }
                    append_result(output_dir, "significance_results", error_result)

    print(f"\n{'='*60}")
    print(f"Done! Results: {output_dir / 'significance_results.csv'}")
    print(f"Predictions:   {output_dir / 'predictions/'}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed significance study for architecture ablation"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Architecture configs to test (default: all 8)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Random seeds",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for parallel execution",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/significance",
        help="Output directory",
    )
    parser.add_argument(
        "--s3-path", type=str, default=None,
        help="S3 path for persisting predictions (e.g. 's3://bucket/prefix/significance')",
    )
    args = parser.parse_args()

    # Configure S3 upload
    global _S3_PATH
    _S3_PATH = args.s3_path or os.environ.get("RECARENA_S3_PREDICTIONS")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = {k: CONFIGS[k] for k in args.configs} if args.configs else CONFIGS

    run_all(
        datasets=args.datasets,
        configs=selected_configs,
        seeds=args.seeds,
        output_dir=output_dir,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
