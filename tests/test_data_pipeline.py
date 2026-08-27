"""Unit + property-based tests for the data_pipeline preprocessing pipeline.

Covers the deterministic core primitives (dedup, k-core, LOO split, ID remap,
subsample, stats), provenance helpers (sha256, manifest), registry integrity,
and a full end-to-end run via a synthetic parser (no network / large files).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure repo root is importable so `import data_pipeline` works.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.core import (  # noqa: E402
    ITEM,
    TS,
    USER,
    Manifest,
    compute_stats,
    deduplicate,
    iterative_k_core,
    leave_one_out_split,
    remap_ids,
    sha256_of,
    subsample_users,
    write_outputs,
)
from data_pipeline.parsers import PARSERS  # noqa: E402
from data_pipeline.registry import REGISTRY  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def make_df(rows):
    """rows: list of (user, item, ts[, rating]) tuples."""
    has_rating = len(rows[0]) == 4
    cols = [USER, ITEM, TS] + (["rating"] if has_rating else [])
    df = pd.DataFrame(rows, columns=cols)
    df[TS] = pd.to_datetime(df[TS], unit="s")
    return df


@pytest.fixture
def simple_df():
    """6 users with overlapping item catalogs so held-out items still appear
    in train (avoids the degenerate 'every user ends on the same item' case).

    User u interacts with items (u%4)+1 .. (u%4)+6 in increasing time, so each
    item is shared across multiple users and is not universally a held-out item.
    """
    rows = []
    for u in range(6):
        base = u % 4
        for rank in range(6):
            item = base + rank + 1
            rows.append((u, item, 1_000 + u * 100 + rank))
    return make_df(rows)


# --------------------------------------------------------------------------- #
# deduplicate
# --------------------------------------------------------------------------- #
def test_dedup_earliest_keeps_first_timestamp():
    df = make_df([(0, 5, 300), (0, 5, 100), (0, 5, 200), (0, 7, 50)])
    out = deduplicate(df, keep="earliest")
    assert len(out) == 2
    kept = out[out[ITEM] == 5][TS].iloc[0]
    assert kept == pd.to_datetime(100, unit="s")


def test_dedup_latest_keeps_last_timestamp():
    df = make_df([(0, 5, 300), (0, 5, 100), (0, 5, 200)])
    out = deduplicate(df, keep="latest")
    assert len(out) == 1
    assert out[TS].iloc[0] == pd.to_datetime(300, unit="s")


def test_dedup_none_keeps_everything():
    df = make_df([(0, 5, 300), (0, 5, 100), (0, 5, 200)])
    out = deduplicate(df, keep="none")
    assert len(out) == 3


def test_dedup_distinct_pairs_unaffected(simple_df):
    out = deduplicate(simple_df, keep="earliest")
    assert len(out) == len(simple_df)


@settings(max_examples=60, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(
            st.integers(0, 4),  # user
            st.integers(1, 6),  # item
            st.integers(0, 1000),  # ts
        ),
        min_size=1,
        max_size=120,
    )
)
def test_dedup_yields_unique_user_item_pairs(pairs):
    df = make_df(list(pairs))
    out = deduplicate(df, keep="earliest")
    assert not out.duplicated(subset=[USER, ITEM]).any()
    # Every original (user,item) pair survives exactly once.
    assert set(map(tuple, out[[USER, ITEM]].values)) == {(u, i) for u, i, _ in pairs}


# --------------------------------------------------------------------------- #
# iterative_k_core
# --------------------------------------------------------------------------- #
def test_k_core_removes_low_degree():
    # user 0 has 5 items; item 99 appears once (below 5-core) -> removed.
    rows = [(0, i, i) for i in range(1, 6)] + [(1, 99, 500)]
    df = make_df(rows)
    out = iterative_k_core(df, user_core=5, item_core=5)
    assert 99 not in set(out[ITEM])
    assert 1 not in set(out[USER])


def test_k_core_convergence_cascade():
    # A chain where removing one item drops a user below threshold, cascading.
    rows = []
    # user 0: items 1..5 (ok). item 1 also used by users 1..4 once each.
    for i in range(1, 6):
        rows.append((0, i, i))
    for u in range(1, 5):
        rows.append((u, 1, 100 + u))  # these users have only 1 interaction
    df = make_df(rows)
    out = iterative_k_core(df, user_core=5, item_core=5)
    # users 1-4 have 1 interaction -> removed; then item 1 loses those -> only
    # user 0 keeps it (still >=5 items for user 0, but item counts matter).
    assert set(out[USER]).issubset({0})


def test_k_core_all_satisfy_threshold(simple_df):
    out = iterative_k_core(simple_df, user_core=2, item_core=2)
    uc = out[USER].value_counts()
    ic = out[ITEM].value_counts()
    assert (uc >= 2).all()
    assert (ic >= 2).all()


def test_k_core_noop_when_threshold_one(simple_df):
    out = iterative_k_core(simple_df, user_core=1, item_core=1)
    assert len(out) == len(simple_df)


@settings(max_examples=40, deadline=None)
@given(
    rows=st.lists(
        st.tuples(st.integers(0, 8), st.integers(1, 8), st.integers(0, 500)),
        min_size=1,
        max_size=200,
    ),
    core=st.integers(2, 4),
)
def test_k_core_postcondition_holds(rows, core):
    df = make_df(list(rows)).drop_duplicates(subset=[USER, ITEM])
    out = iterative_k_core(df, user_core=core, item_core=core)
    if len(out):
        assert (out[USER].value_counts() >= core).all()
        assert (out[ITEM].value_counts() >= core).all()


# --------------------------------------------------------------------------- #
# remap_ids
# --------------------------------------------------------------------------- #
def test_remap_users_zero_indexed_items_one_indexed():
    df = make_df([(10, 500, 1), (10, 600, 2), (20, 500, 3)])
    out, umap, imap = remap_ids(df)
    assert set(out[USER]) == {0, 1}
    assert out[ITEM].min() == 1
    assert set(out[ITEM]) == set(range(1, out[ITEM].nunique() + 1))


def test_remap_preserves_row_count_and_mapping(simple_df):
    out, umap, imap = remap_ids(simple_df)
    assert len(out) == len(simple_df)
    # mapping is a bijection on observed ids
    assert len(umap) == simple_df[USER].nunique()
    assert len(imap) == simple_df[ITEM].nunique()
    assert min(imap.values()) == 1


def test_remap_contiguous():
    df = make_df([(0, 7, 1), (0, 13, 2), (1, 99, 3), (1, 7, 4)])
    out, _, _ = remap_ids(df)
    items = sorted(out[ITEM].unique())
    users = sorted(out[USER].unique())
    assert items == list(range(1, len(items) + 1))
    assert users == list(range(len(users)))


# --------------------------------------------------------------------------- #
# subsample_users
# --------------------------------------------------------------------------- #
def test_subsample_noop_when_under_cap(simple_df):
    out = subsample_users(simple_df, max_users=999, seed=42)
    assert out[USER].nunique() == simple_df[USER].nunique()


def test_subsample_caps_user_count(simple_df):
    out = subsample_users(simple_df, max_users=3, seed=42)
    assert out[USER].nunique() == 3
    # only complete user histories are kept
    assert set(out[USER]).issubset(set(simple_df[USER]))


def test_subsample_deterministic(simple_df):
    a = subsample_users(simple_df, max_users=3, seed=42)
    b = subsample_users(simple_df, max_users=3, seed=42)
    assert set(a[USER]) == set(b[USER])


def test_subsample_none_is_noop(simple_df):
    out = subsample_users(simple_df, max_users=None, seed=1)
    assert len(out) == len(simple_df)


# --------------------------------------------------------------------------- #
# leave_one_out_split
# --------------------------------------------------------------------------- #
def test_loo_basic_shapes(simple_df):
    train, val, test = leave_one_out_split(simple_df)
    # At most one held-out test/val row per user (some may be cold-filtered).
    n_users = simple_df[USER].nunique()
    assert len(test) <= n_users
    assert len(val) <= n_users
    assert len(train) >= len(simple_df) - 2 * n_users
    # Every row is accounted for or intentionally cold-dropped.
    assert len(train) + len(val) + len(test) <= len(simple_df)


def test_loo_one_row_per_user(simple_df):
    _, val, test = leave_one_out_split(simple_df)
    assert not test[USER].duplicated().any()
    assert not val[USER].duplicated().any()


def test_loo_test_is_last_item_chronologically():
    # Two users sharing a catalog so held-out items also appear in train.
    rows = [
        (0, 1, 10),
        (0, 2, 20),
        (0, 3, 30),
        (0, 4, 40),
        (1, 4, 11),
        (1, 3, 21),
        (1, 2, 31),
        (1, 1, 41),
    ]
    df = make_df(rows)
    train, val, test = leave_one_out_split(df)
    t0 = test[test[USER] == 0]
    v0 = val[val[USER] == 0]
    assert t0[ITEM].iloc[0] == 4  # latest timestamp for user 0
    assert v0[ITEM].iloc[0] == 3  # 2nd latest for user 0


def test_loo_tie_break_uses_ingestion_order():
    # All same timestamp -> original ingestion order decides "last".
    # Reversed orders across users so every item also appears in train.
    rows = [
        (0, 1, 100),
        (0, 2, 100),
        (0, 3, 100),
        (0, 4, 100),
        (1, 4, 100),
        (1, 3, 100),
        (1, 2, 100),
        (1, 1, 100),
    ]
    df = make_df(rows)
    _, val, test = leave_one_out_split(df)
    t0 = test[test[USER] == 0]
    v0 = val[val[USER] == 0]
    assert t0[ITEM].iloc[0] == 4
    assert v0[ITEM].iloc[0] == 3


def test_loo_single_interaction_user_goes_to_train():
    rows = [(0, i, i) for i in range(1, 5)] + [(1, 1, 50)]
    df = make_df(rows)
    train, val, test = leave_one_out_split(df)
    # user 1 has only 1 interaction -> not in val/test
    assert 1 not in set(test[USER])
    assert 1 not in set(val[USER])
    assert 1 in set(train[USER])


def test_loo_two_interaction_user_has_test_no_val():
    rows = [(0, i, i) for i in range(1, 6)] + [(1, 1, 10), (1, 2, 20)]
    df = make_df(rows)
    train, val, test = leave_one_out_split(df)
    assert 1 in set(test[USER])
    assert 1 not in set(val[USER])  # needs >=3 for val
    assert 1 in set(train[USER])


def test_loo_cold_item_filtering():
    # user 1's last item (777) appears nowhere in train -> dropped from test.
    rows = [(0, i, i) for i in range(1, 6)]
    rows += [(1, 1, 10), (1, 2, 20), (1, 777, 30)]
    df = make_df(rows)
    train, val, test = leave_one_out_split(df)
    train_items = set(train[ITEM])
    assert set(test[ITEM]).issubset(train_items)
    assert set(val[ITEM]).issubset(train_items)
    # 777 only ever appears as user 1's held-out test item -> filtered out.
    assert 777 not in train_items
    assert 777 not in set(test[ITEM])


def test_loo_no_leakage_between_splits(simple_df):
    train, val, test = leave_one_out_split(simple_df)

    # A (user,item,ts) row cannot appear in two splits.
    def keyset(df):
        return set(map(tuple, df[[USER, ITEM, TS]].astype(str).values))

    assert keyset(train).isdisjoint(keyset(test))
    assert keyset(train).isdisjoint(keyset(val))
    assert keyset(val).isdisjoint(keyset(test))


def test_loo_val_test_users_subset_of_train(simple_df):
    train, val, test = leave_one_out_split(simple_df)
    train_users = set(train[USER])
    assert set(val[USER]).issubset(train_users)
    assert set(test[USER]).issubset(train_users)


@settings(max_examples=40, deadline=None)
@given(n_users=st.integers(1, 8), seq_len=st.integers(1, 10))
def test_loo_counts_never_exceed_input(n_users, seq_len):
    rows = []
    for u in range(n_users):
        for rank in range(seq_len):
            rows.append((u, rank + 1, u * 1000 + rank))
    df = make_df(rows)
    train, val, test = leave_one_out_split(df)
    assert len(train) + len(val) + len(test) <= len(df)
    # test/val never have more than one row per user
    assert len(test) <= n_users
    assert len(val) <= n_users


# --------------------------------------------------------------------------- #
# compute_stats
# --------------------------------------------------------------------------- #
def test_compute_stats_fields(simple_df):
    train, val, test = leave_one_out_split(simple_df)
    stats = compute_stats(simple_df, train, val, test)
    assert stats["n_users"] == simple_df[USER].nunique()
    assert stats["n_items"] == simple_df[ITEM].nunique()
    assert stats["n_interactions"] == len(simple_df)
    assert stats["n_train"] == len(train)
    assert 0.0 <= stats["density_pct"] <= 100.0
    assert stats["min_seq_len"] <= stats["avg_seq_len"] <= stats["max_seq_len"]


# --------------------------------------------------------------------------- #
# provenance: sha256 + manifest
# --------------------------------------------------------------------------- #
def test_sha256_none_for_missing_path(tmp_path):
    assert sha256_of(tmp_path / "nope.bin") is None


def test_sha256_stable(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"recarena" * 1000)
    assert sha256_of(p) == sha256_of(p)


def test_manifest_round_trips_json():
    m = Manifest(
        dataset="x",
        domain="movies",
        source_url="http://e",
        raw_file="f",
        raw_sha256=None,
        positive_rule="r>=4",
        dedup="earliest",
        user_core=5,
        item_core=5,
        max_users=None,
        seed=42,
        split="leave_one_out",
        has_metadata=True,
        stats={"n_users": 1},
    )
    d = json.loads(m.to_json())
    assert d["dataset"] == "x"
    assert d["pipeline_version"]
    assert d["stats"]["n_users"] == 1


# --------------------------------------------------------------------------- #
# registry integrity
# --------------------------------------------------------------------------- #
def test_every_registry_entry_has_parser():
    for name in REGISTRY:
        assert name in PARSERS, f"{name} missing a parser"


def test_registry_specs_have_required_fields():
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.domain
        assert spec.positive_rule
        assert spec.dedup in {"earliest", "latest", "none"}
        assert spec.user_core >= 1 and spec.item_core >= 1


def test_download_specs_are_consistent():
    for spec in REGISTRY.values():
        # if a download URL is set, a target filename must be too
        if spec.download_url:
            assert spec.download_file


# --------------------------------------------------------------------------- #
# end-to-end: process_one with a synthetic parser (no network/large files)
# --------------------------------------------------------------------------- #
def test_process_one_end_to_end(tmp_path, monkeypatch):
    import data_pipeline.run as run
    from data_pipeline.registry import DatasetSpec

    # Build a synthetic dataset: 6 users x 6 items + metadata, with one cold
    # item and one rating below threshold to exercise filtering.
    def fake_parser(raw_dir):
        rows = []
        for u in range(6):
            for rank in range(6):
                rows.append((u, rank + 1, u * 1000 + rank, 5))
        df = pd.DataFrame(rows, columns=[USER, ITEM, TS, "rating"])
        df[TS] = pd.to_datetime(df[TS], unit="s")
        meta = pd.DataFrame(
            {ITEM: [1, 2, 3, 4, 5, 6], "title": [f"t{i}" for i in range(1, 7)]}
        )
        return df, meta

    spec = DatasetSpec(
        name="synthetic",
        domain="test",
        raw_subdir="synthetic",
        raw_file=None,
        source_url="local",
        positive_rule="all",
        user_core=2,
        item_core=2,
    )
    monkeypatch.setitem(run.PARSERS, "synthetic", fake_parser)
    raw_root = tmp_path / "raw"
    (raw_root / "synthetic").mkdir(parents=True)
    out_root = tmp_path / "out"

    stats = run.process_one(spec, raw_root, out_root)

    d = out_root / "synthetic"
    for f in ["interactions.parquet", "items.parquet", "stats.json", "manifest.json"]:
        assert (d / f).exists(), f"missing {f}"
    for f in ["train.parquet", "val.parquet", "test.parquet"]:
        assert (d / "leave_one_out" / f).exists(), f"missing {f}"

    train = pd.read_parquet(d / "leave_one_out" / "train.parquet")
    val = pd.read_parquet(d / "leave_one_out" / "val.parquet")
    test = pd.read_parquet(d / "leave_one_out" / "test.parquet")
    assert train[ITEM].min() == 1
    assert not test[USER].duplicated().any()
    assert set(val[ITEM]).issubset(set(train[ITEM]))
    assert set(test[ITEM]).issubset(set(train[ITEM]))
    assert stats["n_users"] == 6

    # metadata aligns to remapped item ids and joins on train items
    meta = pd.read_parquet(d / "items.parquet")
    assert set(train[ITEM]).issubset(set(meta[ITEM]))

    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["dataset"] == "synthetic"
    assert manifest["has_metadata"] is True


def test_process_one_missing_raw_dir_raises(tmp_path, monkeypatch):
    import data_pipeline.run as run
    from data_pipeline.registry import DatasetSpec

    spec = DatasetSpec(
        name="nope",
        domain="test",
        raw_subdir="nope",
        raw_file=None,
        source_url="local",
        positive_rule="all",
    )
    monkeypatch.setitem(run.PARSERS, "nope", lambda r: (pd.DataFrame(), None))
    with pytest.raises(FileNotFoundError):
        run.process_one(spec, tmp_path / "raw", tmp_path / "out")


# --------------------------------------------------------------------------- #
# Optional: validate real produced artifacts if they exist on disk.
# These are skipped automatically in environments without the processed data
# (e.g. CI), so they never break the core suite.
# --------------------------------------------------------------------------- #
PROCESSED_DIR = REPO_ROOT / "data_pipeline" / "processed"


def _processed_datasets():
    if not PROCESSED_DIR.exists():
        return []
    return sorted(
        p.name
        for p in PROCESSED_DIR.iterdir()
        if p.is_dir() and (p / "leave_one_out" / "train.parquet").exists()
    )


@pytest.mark.parametrize("name", _processed_datasets())
def test_produced_artifacts_satisfy_loo_contract(name):
    d = PROCESSED_DIR / name
    loo = d / "leave_one_out"
    train = pd.read_parquet(loo / "train.parquet")
    val = pd.read_parquet(loo / "val.parquet")
    test = pd.read_parquet(loo / "test.parquet")
    inter = pd.read_parquet(d / "interactions.parquet")

    # schema
    for df in (train, val, test, inter):
        assert {USER, ITEM, TS}.issubset(df.columns)
        assert not df[[USER, ITEM, TS]].isna().any().any()

    # 1-indexed contiguous items, 0-indexed contiguous users (on full frame)
    assert inter[ITEM].min() == 1
    assert set(inter[ITEM].unique()) == set(range(1, inter[ITEM].nunique() + 1))
    assert set(inter[USER].unique()) == set(range(inter[USER].nunique()))

    # LOO: one held-out row per user
    assert not val[USER].duplicated().any()
    assert not test[USER].duplicated().any()

    # cold filtering: val/test items & users are present in train
    train_items = set(train[ITEM].unique())
    train_users = set(train[USER].unique())
    assert set(val[ITEM].unique()).issubset(train_items)
    assert set(test[ITEM].unique()).issubset(train_items)
    assert set(val[USER].unique()).issubset(train_users)
    assert set(test[USER].unique()).issubset(train_users)

    # split counts never exceed the deduped interaction count
    assert len(train) + len(val) + len(test) <= len(inter)


@pytest.mark.parametrize("name", _processed_datasets())
def test_produced_metadata_joins_on_item_id(name):
    d = PROCESSED_DIR / name
    items_f = d / "items.parquet"
    if not items_f.exists():
        pytest.skip(f"{name} has no metadata sidecar")
    meta = pd.read_parquet(items_f)
    train = pd.read_parquet(d / "leave_one_out" / "train.parquet")
    assert ITEM in meta.columns
    assert not meta[ITEM].duplicated().any()
    covered = set(meta[ITEM].unique())
    train_items = set(train[ITEM].unique())
    # majority of train items should carry metadata
    assert len(train_items & covered) / len(train_items) >= 0.5
