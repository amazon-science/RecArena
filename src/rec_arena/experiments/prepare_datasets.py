"""Download, preprocess, and save all datasets used in the architecture ablation study.

Preprocessing protocol (consistent across all datasets):
  - K-core filtering (default 5-core, Gowalla uses 20-core)
  - Leave-one-out split: last item = test, second-to-last = val, rest = train
  - User/item IDs remapped to contiguous integers (0-indexed)
  - Saved as parquet files (train.parquet, val.parquet, test.parquet)

Supports two output modes:
  --output-dir /path/to/local/dir   (saves locally)
  --s3-path s3://bucket/prefix      (uploads to S3)

Usage:
  # Prepare all datasets locally:
  python -m rec_arena.experiments.prepare_datasets --output-dir data/

  # Prepare specific datasets to S3:
  python -m rec_arena.experiments.prepare_datasets \
      --datasets ml_1m gowalla steam \
      --s3-path s3://my-bucket/recarena

  # List available datasets:
  python -m rec_arena.experiments.prepare_datasets --list

Data sources:
  ML-100K, ML-1M, ML-20M: https://grouplens.org/datasets/movielens/
  Beauty: https://jmcauley.ucsd.edu/data/amazon/ (2014 version)
  RateBeer: https://snap.stanford.edu/data/web-RateBeer.html
  Goodreads: https://mengtingwan.github.io/data/goodreads.html
  Gowalla: https://snap.stanford.edu/data/loc-gowalla.html
  Steam: BERT4Rec preprocessed version (Sun et al. 2019)
  Twitch: https://clivecast.github.io/
"""

import argparse
import gzip
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import dateutil.parser
import numpy as np
import pandas as pd


# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "ml_100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "kcore": 5,
        "parser": "parse_ml100k",
    },
    "ml_1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "kcore": 5,
        "parser": "parse_ml1m",
    },
    "ml_20m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "kcore": 5,
        "parser": "parse_ml20m",
    },
    "amazon_beauty_2014": {
        "url": "https://jmcauley.ucsd.edu/data/amazon/links/ratings_Beauty.csv",
        "kcore": 5,
        "parser": "parse_amazon_beauty",
    },
    "ratebeer": {
        "url": None,  # Requires manual download or alternative source
        "kcore": 5,
        "parser": "parse_ratebeer",
        "note": "RateBeer data must be obtained from https://snap.stanford.edu/data/web-RateBeer.html",
    },
    "goodreads": {
        "url": None,  # Large file, requires manual download
        "kcore": 5,
        "parser": "parse_goodreads",
        "note": "Goodreads data must be obtained from https://mengtingwan.github.io/data/goodreads.html",
    },
    "gowalla": {
        "url": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
        "kcore": 20,
        "parser": "parse_gowalla",
    },
    "steam": {
        "url": "https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/steam.txt",
        "kcore": 5,
        "parser": "parse_steam",
    },
    "twitch": {
        "url": None,  # Requires manual download
        "kcore": 5,
        "parser": "parse_twitch",
        "note": "Twitch data must be obtained from https://clivecast.github.io/",
    },
}


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_ml100k(cache_dir):
    url = DATASETS["ml_100k"]["url"]
    zip_path = cache_dir / "ml-100k.zip"
    if not zip_path.exists():
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("ml-100k/u.data") as f:
            df = pd.read_csv(f, sep="\t", header=None,
                             names=["user_id", "item_id", "rating", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["user_id", "item_id", "timestamp"]]


def parse_ml1m(cache_dir):
    url = DATASETS["ml_1m"]["url"]
    zip_path = cache_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("ml-1m/ratings.dat") as f:
            df = pd.read_csv(f, sep="::", header=None, engine="python",
                             names=["user_id", "item_id", "rating", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["user_id", "item_id", "timestamp"]]


def parse_ml20m(cache_dir):
    url = DATASETS["ml_20m"]["url"]
    zip_path = cache_dir / "ml-20m.zip"
    if not zip_path.exists():
        print(f"  Downloading from {url} (this is ~190MB)...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("ml-20m/ratings.csv") as f:
            df = pd.read_csv(f)
    df = df.rename(columns={"userId": "user_id", "movieId": "item_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["user_id", "item_id", "timestamp"]]


def parse_amazon_beauty(cache_dir):
    url = DATASETS["amazon_beauty_2014"]["url"]
    csv_path = cache_dir / "ratings_Beauty.csv"
    if not csv_path.exists():
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, csv_path)
    df = pd.read_csv(csv_path, header=None,
                     names=["user_id", "item_id", "rating", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["user_id", "item_id", "timestamp"]]


def parse_gowalla(cache_dir):
    url = DATASETS["gowalla"]["url"]
    gz_path = cache_dir / "gowalla.txt.gz"
    if not gz_path.exists():
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, gz_path)
    rows = []
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            user_id, timestamp_str, lat, lon, item_id = parts
            try:
                ts = dateutil.parser.isoparse(timestamp_str)
            except Exception:
                continue
            rows.append({"user_id": user_id, "item_id": item_id, "timestamp": ts})
    return pd.DataFrame(rows)


def parse_steam(cache_dir):
    url = DATASETS["steam"]["url"]
    txt_path = cache_dir / "steam.txt"
    if not txt_path.exists():
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, txt_path)
    rows = []
    prev_user = None
    timestamp = 0
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            user_id, item_id = parts
            if user_id != prev_user:
                timestamp = 0
                prev_user = user_id
            timestamp += 1
            rows.append({"user_id": user_id, "item_id": item_id, "timestamp": timestamp})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def parse_ratebeer(cache_dir):
    raise NotImplementedError(
        "RateBeer requires manual download. Please obtain the data from "
        "https://snap.stanford.edu/data/web-RateBeer.html and place the "
        "ratings file in " + str(cache_dir)
    )


def parse_goodreads(cache_dir):
    raise NotImplementedError(
        "Goodreads requires manual download. Please obtain the data from "
        "https://mengtingwan.github.io/data/goodreads.html and place the "
        "interactions file in " + str(cache_dir)
    )


def parse_twitch(cache_dir):
    raise NotImplementedError(
        "Twitch requires manual download. Please obtain the data from "
        "https://clivecast.github.io/ and place the interactions file in "
        + str(cache_dir)
    )


PARSERS = {
    "parse_ml100k": parse_ml100k,
    "parse_ml1m": parse_ml1m,
    "parse_ml20m": parse_ml20m,
    "parse_amazon_beauty": parse_amazon_beauty,
    "parse_gowalla": parse_gowalla,
    "parse_steam": parse_steam,
    "parse_ratebeer": parse_ratebeer,
    "parse_goodreads": parse_goodreads,
    "parse_twitch": parse_twitch,
}


# ── Shared preprocessing ─────────────────────────────────────────────────────

def apply_kcore(df, k):
    print(f"  Applying {k}-core filtering...")
    while True:
        uc = df["user_id"].value_counts()
        ic = df["item_id"].value_counts()
        filtered = df[df["user_id"].isin(uc[uc >= k].index) &
                      df["item_id"].isin(ic[ic >= k].index)]
        if len(filtered) == len(df):
            break
        df = filtered
    print(f"    {len(df):,} interactions, {df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items")
    return df.reset_index(drop=True)


def leave_one_out_split(df):
    print("  Leave-one-out split...")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    user_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item_map = {it: i for i, it in enumerate(sorted(df["item_id"].unique()))}
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)
    df["rating"] = 1.0
    df["weight"] = 1

    train_rows, val_rows, test_rows = [], [], []
    for uid, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            continue
        train_rows.append(group.iloc[:-2])
        val_rows.append(group.iloc[[-2]])
        test_rows.append(group.iloc[[-1]])

    train = pd.concat(train_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True)
    print(f"    Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def save_local(train, val, test, output_dir, dataset_name):
    out = Path(output_dir) / dataset_name / "leave_one_out"
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    print(f"  Saved to {out}/")


def save_s3(train, val, test, s3_path, dataset_name):
    import boto3
    s3_path = s3_path.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:])
    key_prefix = f"{prefix}/{dataset_name}/leave_one_out"

    s3 = boto3.client("s3")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        key = f"{key_prefix}/{name}.parquet"
        s3.upload_fileobj(buf, bucket, key)
    print(f"  Uploaded to s3://{bucket}/{key_prefix}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare_dataset(name, cache_dir, output_dir=None, s3_path=None):
    print(f"\n{'='*60}")
    print(f"Preparing {name}")
    print(f"{'='*60}")

    info = DATASETS[name]
    ds_cache = Path(cache_dir) / name
    ds_cache.mkdir(parents=True, exist_ok=True)

    parser = PARSERS[info["parser"]]
    df = parser(ds_cache)
    print(f"  Raw: {len(df):,} interactions, {df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items")

    df = apply_kcore(df, info["kcore"])
    train, val, test = leave_one_out_split(df)

    if output_dir:
        save_local(train, val, test, output_dir, name)
    if s3_path:
        save_s3(train, val, test, s3_path, name)
    if not output_dir and not s3_path:
        print("  WARNING: No output specified. Use --output-dir or --s3-path.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess datasets for the architecture ablation study."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets to prepare (default: all). Available: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Local directory to save processed datasets",
    )
    parser.add_argument(
        "--s3-path", type=str, default=None,
        help="S3 path to upload processed datasets (e.g. s3://my-bucket/recarena)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/recarena_data",
        help="Directory for caching raw downloads",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            auto = "auto-download" if info["url"] else "manual download required"
            print(f"  {name:25s} ({auto}, {info['kcore']}-core)")
            if "note" in info:
                print(f"    {info['note']}")
        return

    datasets = args.datasets or [d for d in DATASETS if DATASETS[d]["url"] is not None]

    if not args.output_dir and not args.s3_path:
        print("ERROR: Specify --output-dir and/or --s3-path")
        return

    for name in datasets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
            continue
        try:
            prepare_dataset(name, args.cache_dir, args.output_dir, args.s3_path)
        except NotImplementedError as e:
            print(f"  SKIPPED: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("Done!")
    if args.output_dir:
        print(f"Local data: {args.output_dir}/")
    if args.s3_path:
        print(f"S3 data: {args.s3_path}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
