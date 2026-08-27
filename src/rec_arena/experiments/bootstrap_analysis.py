"""Bootstrap significance analysis from saved per-user predictions.

Reads .npz files produced by significance_study.py and computes:
  1. Per-config mean ± std across seeds
  2. Bootstrap 95% CIs for each (dataset, config, seed)
  3. Paired bootstrap tests: RoPE+LiGR vs baseline (same seed, same users)
  4. Summary table for the paper

Usage:
  python -m rec_arena.experiments.bootstrap_analysis \
      --input-dir results/significance/predictions \
      --output-dir results/significance/analysis
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
ALPHA = 0.05
K_VALUES = [10, 20, 50, 100]
METRICS = ["ndcg", "hr", "mrr"]


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, ci: float = CI_LEVEL):
    """Compute bootstrap confidence interval for the mean of `values`."""
    n = len(values)
    rng = np.random.default_rng(0)  # fixed seed for reproducibility of CIs
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return values.mean(), lo, hi


def paired_bootstrap_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
):
    """Two-sided paired bootstrap test: H0: mean(A) == mean(B).

    Returns (delta, p_value) where delta = mean(A) - mean(B).
    Uses the percentile method.
    """
    assert len(values_a) == len(values_b), "Must have same users"
    n = len(values_a)
    observed_delta = values_a.mean() - values_b.mean()

    rng = np.random.default_rng(0)
    boot_deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_deltas[b] = values_a[idx].mean() - values_b[idx].mean()

    # Two-sided p-value: fraction of bootstrap deltas on the opposite side of 0
    if observed_delta >= 0:
        p_value = (boot_deltas <= 0).mean() * 2
    else:
        p_value = (boot_deltas >= 0).mean() * 2
    p_value = min(p_value, 1.0)

    return observed_delta, p_value


def load_predictions(input_dir_or_s3: str):
    """Load all .npz prediction files into a structured dict.

    input_dir_or_s3: local path OR s3://bucket/prefix
    Downloads S3 files one at a time and deletes immediately to avoid
    filling local disk.

    Returns: dict[(dataset, arch, seed)] -> {per_user_ndcg@10: array, ...}
    """
    data = {}

    def _parse_and_load(f: Path):
        parts = f.stem.rsplit("_seed", 1)
        if len(parts) != 2:
            return None, None, None
        seed = int(parts[1])
        prefix = parts[0]
        for arch in [
            "rope+ligr=True+rms=True",
            "rope+ligr=True+rms=False",
            "rope+ligr=False+rms=True",
            "rope+ligr=False+rms=False",
            "learnable+ligr=True+rms=True",
            "learnable+ligr=True+rms=False",
            "learnable+ligr=False+rms=True",
            "learnable+ligr=False+rms=False",
        ]:
            if prefix.endswith(f"_{arch}"):
                dataset = prefix[: -(len(arch) + 1)]
                return dataset, arch, seed
        return None, None, None

    if input_dir_or_s3.startswith("s3://"):
        for local_path, fname in _stream_s3_predictions(input_dir_or_s3):
            dataset, arch, seed = _parse_and_load(local_path)
            if dataset is None:
                continue
            npz = np.load(local_path)
            # Only load per_user_* arrays (tiny), skip predictions matrix (huge)
            data[(dataset, arch, seed)] = {
                key: npz[key] for key in npz.files if key.startswith("per_user_")
            }
    else:
        for f in sorted(Path(input_dir_or_s3).glob("*.npz")):
            dataset, arch, seed = _parse_and_load(f)
            if dataset is None:
                continue
            npz = np.load(f)
            data[(dataset, arch, seed)] = {
                key: npz[key] for key in npz.files if key.startswith("per_user_")
            }

    print(f"Loaded {len(data)} prediction files")
    return data


def _list_and_download_s3(s3_path: str):
    """List .npz files in S3 prefix and download to /tmp one at a time.
    Returns a generator of (local_path, should_delete) tuples.
    """
    import boto3

    s3_path = s3_path.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:])

    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".npz")]
    print(f"Found {len(keys)} files in s3://{bucket}/{prefix}")
    return bucket, keys


def _stream_s3_predictions(s3_path: str):
    """Download S3 prediction files one at a time, yielding (local_path, fname).
    Deletes each file after yielding to avoid filling disk.
    """
    import boto3

    s3_path = s3_path.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:])

    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".npz")]
    print(f"Streaming {len(keys)} files from s3://{bucket}/{prefix}")

    tmp_dir = Path("/home/sagemaker-user/tmp_bootstrap")
    tmp_dir.mkdir(exist_ok=True)

    for i, key in enumerate(keys):
        fname = Path(key).name
        local = tmp_dir / fname
        try:
            s3.download_file(bucket, key, str(local))
            print(f"  [{i+1}/{len(keys)}] Loaded {fname}", flush=True)
            yield local, fname
        finally:
            local.unlink(missing_ok=True)


def analyze(input_dir_or_s3: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_predictions(input_dir_or_s3)

    if not data:
        print("No prediction files found. Exiting.")
        return

    # Discover datasets, archs, seeds
    datasets = sorted({k[0] for k in data})
    archs = sorted({k[1] for k in data})
    seeds = sorted({k[2] for k in data})
    print(f"Datasets: {datasets}")
    print(f"Archs:    {archs}")
    print(f"Seeds:    {seeds}")

    # ── 1. Aggregate metrics: mean ± std across seeds ─────────────────────
    rows = []
    for ds in datasets:
        for arch in archs:
            seed_metrics = []
            for seed in seeds:
                key = (ds, arch, seed)
                if key not in data:
                    continue
                entry = data[key]
                row = {"dataset": ds, "arch": arch, "seed": seed}
                for metric in METRICS:
                    for k in K_VALUES:
                        col = f"per_user_{metric}@{k}"
                        if col in entry:
                            row[f"{metric}@{k}"] = entry[col].mean()
                seed_metrics.append(row)
                rows.append(row)

    df_all = pd.DataFrame(rows)
    df_all.to_csv(output_dir / "all_seed_metrics.csv", index=False)

    # Mean ± std across seeds
    metric_cols = [f"{m}@{k}" for m in METRICS for k in K_VALUES]
    available_cols = [c for c in metric_cols if c in df_all.columns]
    summary = df_all.groupby(["dataset", "arch"])[available_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "seed_summary.csv", index=False)
    print(f"\nSeed summary saved to {output_dir / 'seed_summary.csv'}")

    # ── 2. Bootstrap CIs per (dataset, arch, seed) ────────────────────────
    ci_rows = []
    for (ds, arch, seed), entry in data.items():
        for metric in METRICS:
            for k in K_VALUES:
                col = f"per_user_{metric}@{k}"
                if col not in entry:
                    continue
                mean, lo, hi = bootstrap_ci(entry[col])
                ci_rows.append({
                    "dataset": ds,
                    "arch": arch,
                    "seed": seed,
                    "metric": f"{metric}@{k}",
                    "mean": round(mean, 6),
                    "ci_lo": round(lo, 6),
                    "ci_hi": round(hi, 6),
                })

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    print(f"Bootstrap CIs saved to {output_dir / 'bootstrap_ci.csv'}")

    # ── 3. Paired bootstrap: each config vs baseline (same seed) ──────────
    paired_rows = []
    baseline_arch = "learnable+ligr=False+rms=False"

    for ds in datasets:
        for seed in seeds:
            key_base = (ds, baseline_arch, seed)
            if key_base not in data:
                continue

            for arch in archs:
                if arch == baseline_arch:
                    continue
                key_comp = (ds, arch, seed)
                if key_comp not in data:
                    continue

                entry_base = data[key_base]
                entry_comp = data[key_comp]

                for metric in METRICS:
                    for k in K_VALUES:
                        col = f"per_user_{metric}@{k}"
                        if col not in entry_base or col not in entry_comp:
                            continue

                        vals_base = entry_base[col]
                        vals_comp = entry_comp[col]

                        # Arrays must be same length (same users, same seed)
                        if len(vals_base) != len(vals_comp):
                            continue

                        delta, p_val = paired_bootstrap_test(vals_comp, vals_base)
                        paired_rows.append({
                            "dataset": ds,
                            "seed": seed,
                            "comparison": f"{arch}_vs_baseline",
                            "metric": f"{metric}@{k}",
                            "delta": round(delta, 6),
                            "p_value": round(p_val, 6),
                            "significant": p_val < ALPHA,
                        })

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    print(f"Paired bootstrap tests saved to {output_dir / 'paired_bootstrap.csv'}")

    # ── 4. Paper-ready summary table ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("PAPER SUMMARY: NDCG@10 mean ± std across seeds")
    print("=" * 80)
    for ds in datasets:
        print(f"\n{ds}:")
        for arch in archs:
            mask = (df_all["dataset"] == ds) & (df_all["arch"] == arch)
            vals = df_all.loc[mask, "ndcg@10"]
            if len(vals) == 0:
                continue
            mean, std = vals.mean(), vals.std()
            print(f"  {arch:20s}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")

    # Significance summary
    if len(df_paired) > 0:
        print("\n" + "=" * 80)
        print("SIGNIFICANCE: RoPE+LiGR vs baseline (paired bootstrap, α=0.05)")
        print("=" * 80)
        sig_ndcg = df_paired[
            (df_paired["comparison"].str.contains("rope.*ligr=True.*rms=False"))
            & (df_paired["metric"] == "ndcg@10")
        ]
        for _, row in sig_ndcg.iterrows():
            star = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
            print(
                f"  {row['dataset']:25s} seed={row['seed']}  "
                f"Δ={row['delta']:+.4f}  p={row['p_value']:.4f}  {star}"
            )


def main():
    parser = argparse.ArgumentParser(description="Bootstrap significance analysis")
    parser.add_argument(
        "--input-dir", type=str, default="results/significance/predictions",
        help="Local directory or S3 path (s3://bucket/prefix) containing .npz prediction files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/significance/analysis",
        help="Output directory for analysis results",
    )
    args = parser.parse_args()

    analyze(args.input_dir, Path(args.output_dir))


if __name__ == "__main__":
    main()
