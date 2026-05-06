# RecArena

> **Note:** This repository is part of a larger reproducibility effort for recommendation research. RecArena is a self-contained snippet of that framework, providing everything needed to reproduce the experiments and results of the two papers below.

A modular recommendation system benchmark framework for reproducible research in sequential recommendation.

This repository contains the code and experiment scripts for two papers submitted to RecSys 2026:

1. **On the Transferability of Modern Transformer Design Choices to Sequential Recommendation** — A factorial ablation of RoPE, LiGR, and RMSNorm applied to SASRec across 7 datasets with multi-seed significance testing.

2. **Revisiting Negative Sampling for Sequential Recommendation: A Systematic Comparison** — A comprehensive comparison of full cross-entropy, sampled softmax, BCE, and gBCE loss functions across 6 datasets.

## Installation

```bash
git clone <repo-url>
cd RecArena

python -m venv .venv
source .venv/bin/activate

pip install -e ".[analysis]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA support
- NVIDIA GPU with ≥ 16 GB VRAM (A10G or better recommended)

## Data Preparation

All datasets must be preprocessed before running experiments. The preparation script downloads raw data, applies k-core filtering, performs leave-one-out splitting, and saves the results.

```bash
# Prepare all auto-downloadable datasets locally
python -m rec_arena.experiments.prepare_datasets \
    --datasets ml_100k ml_1m ml_20m amazon_beauty_2014 gowalla steam \
    --output-dir data/

# Or upload to S3 for multi-machine experiments
python -m rec_arena.experiments.prepare_datasets \
    --datasets ml_100k ml_1m ml_20m amazon_beauty_2014 gowalla steam \
    --s3-path s3://your-bucket/recarena

# List all available datasets and download instructions
python -m rec_arena.experiments.prepare_datasets --list
```

RateBeer, Goodreads, and Twitch require manual download due to licensing. Run `--list` for instructions.

If using S3, set the environment variable:
```bash
export RECARENA_S3_BUCKET="your-bucket/recarena"
```

## Reproducing Paper 1: Architecture Ablation

This paper evaluates all 8 combinations of {RoPE, Learnable} × {LiGR, Standard FFN} × {RMSNorm, LayerNorm} on SASRec. The full experiment grid runs across 7 datasets (ml_100k, ml_1m, ml_20m, amazon_beauty_2014, ratebeer, goodreads, netflix) with 3 seeds per configuration.

### Full experiment (all datasets, all seeds, all configs)

```bash
# Single GPU — runs sequentially
python -m rec_arena.experiments.significance_study \
    --num-gpus 1 \
    --output-dir results/significance

# Multi-GPU — one experiment per GPU, round-robin assignment
python -m rec_arena.experiments.significance_study \
    --num-gpus 8 \
    --output-dir results/significance
```

To run a subset (e.g. for debugging):
```bash
python -m rec_arena.experiments.significance_study \
    --datasets ml_100k ml_1m \
    --seeds 42 43 \
    --num-gpus 1 \
    --output-dir results/significance
```

Per-user predictions are optionally saved to S3 for offline bootstrap analysis:
```bash
python -m rec_arena.experiments.significance_study \
    --num-gpus 8 \
    --s3-path s3://your-bucket/recarena/predictions \
    --output-dir results/significance
```

### Robustness study (model scale ablation)

Tests whether architecture effects hold across different model sizes and depths.

```bash
# Embedding dimension ablation (d ∈ {64, 128, 256, 512})
python -m rec_arena.experiments.robustness_ablation \
    --experiment size --output-dir results/

# Depth ablation (L ∈ {1, 2, 4})
python -m rec_arena.experiments.robustness_ablation \
    --experiment depth --output-dir results/
```

### Bootstrap significance analysis

Computes bootstrap confidence intervals and paired significance tests from saved per-user predictions.

```bash
# From local predictions
python -m rec_arena.experiments.bootstrap_analysis \
    --input-dir results/significance/predictions \
    --output-dir results/significance/analysis

# Or directly from S3
python -m rec_arena.experiments.bootstrap_analysis \
    --input-dir s3://your-bucket/recarena/predictions \
    --output-dir results/significance/analysis
```

## Reproducing Paper 2: Loss Function Study

This paper compares full cross-entropy, sampled softmax, BCE, and gBCE across 6 datasets with negative sample counts ranging from 16 to 2048.

### Loss function ablation

```bash
python -m rec_arena.experiments.sasrec_ablation \
    --experiment loss \
    --datasets ml_100k ml_1m ml_20m amazon_beauty_2014 ratebeer twitch \
    --output-dir results/
```

### Negative sampling strategy ablation

Compares per-position vs. batch-shared negative sampling with uniform and popularity-based strategies.

```bash
python -m rec_arena.experiments.negative_sampling_ablation \
    --output-dir results/
```

## Project Structure

```
src/rec_arena/
├── configs/           # Model configurations (dataclass-based)
├── datasets/          # Dataset loading, splitting, negative sampling
├── experiments/       # Experiment scripts for both papers
│   ├── prepare_datasets.py           # Data download & preprocessing
│   ├── significance_study.py         # Paper 1: factorial architecture ablation
│   ├── robustness_ablation.py        # Paper 1: model scale robustness
│   ├── bootstrap_analysis.py         # Paper 1: bootstrap significance tests
│   ├── sasrec_ablation.py            # Paper 2: loss function ablation
│   └── negative_sampling_ablation.py # Paper 2: sampling strategy comparison
├── losses/            # CE, BCE, sampled softmax, gBCE, BPR
├── metrics/           # NDCG, HR, MRR, Recall, Precision
├── models/            # SASRec (+ other models for general use)
├── modules/           # Transformer blocks, RoPE, SwiGLU, RMSNorm
└── utils/             # Reproducibility, logging, profiling
```

## Key Design Decisions

- **Full-vocabulary cross-entropy** as the default loss (no sampled metrics)
- **Leave-one-out** evaluation with full ranking over all items
- **Deterministic per-parameter initialization** — shared weights (item embeddings, attention, FFN) are identical across architecture configs for the same seed, isolating the effect of each component
- **Paired significance testing** — t-tests across seeds + per-user bootstrap with 10,000 resamples
- **FP16 mixed precision** and Flash Attention for training efficiency

## Hardware

All experiments were conducted on NVIDIA A10G GPUs (24 GB VRAM) with AMD EPYC 7R32 CPUs and 768 GB RAM.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
