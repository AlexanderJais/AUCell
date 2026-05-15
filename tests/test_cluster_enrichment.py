"""Unit tests for `compute_cluster_enrichment_stats`.

Covers BH-FDR application, min_cells filtering, NaN handling, the
corrected one-sided Welch t-test (alternative="greater"), and sort order.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from analysis import compute_cluster_enrichment_stats


def _make_scores(rng, n_per: int = 100, n_clusters: int = 5):
    """N clusters of equal size; planted-positive cluster 'A' has higher mean."""
    labels, scores = [], []
    for i, name in enumerate(["A"] + [f"C{i}" for i in range(1, n_clusters)]):
        labels.extend([name] * n_per)
        if name == "A":
            scores.extend(rng.normal(0.5, 0.1, n_per))
        else:
            scores.extend(rng.normal(0.0, 0.1, n_per))
    return np.asarray(scores), np.asarray(labels)


def test_basic_ranking_planted_cluster_first():
    rng = np.random.default_rng(0)
    scores, labels = _make_scores(rng)
    df = compute_cluster_enrichment_stats(scores, labels, min_cells=10, alpha=0.05)
    # Sorted ascending by q — the planted enriched cluster A should rank first
    assert df.iloc[0]["cluster"] == "A"
    assert bool(df.iloc[0]["significant"])
    assert df.iloc[0]["qvalue"] < 0.05


def test_depleted_cluster_returns_high_p():
    """Regression for C1: a depleted cluster should get p≈1 with the
    one-sided "greater" alternative, not 0.5 from the broken p/2 halving."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([
        rng.normal(1.0, 0.1, 100),
        rng.normal(0.0, 0.1, 300),
    ])
    labels = np.array(["enriched"] * 100 + ["depleted"] * 300)
    df = compute_cluster_enrichment_stats(scores, labels, min_cells=10)
    p_dep = float(df.loc[df["cluster"] == "depleted", "pvalue"].iloc[0])
    assert p_dep > 0.99
    assert not bool(df.loc[df["cluster"] == "depleted", "significant"].iloc[0])


def test_min_cells_filter_drops_small_clusters():
    rng = np.random.default_rng(0)
    n_keep, n_drop = 50, 3
    scores = np.concatenate([
        rng.normal(0.3, 0.1, n_keep),
        rng.normal(0.0, 0.1, 200),
        rng.normal(0.0, 0.1, n_drop),
    ])
    labels = np.array(["big"] * n_keep + ["other"] * 200 + ["tiny"] * n_drop)
    df = compute_cluster_enrichment_stats(scores, labels, min_cells=10)
    assert "tiny" not in df["cluster"].values
    assert {"big", "other"} <= set(df["cluster"].values)


def test_bh_fdr_applied_and_sorted():
    rng = np.random.default_rng(0)
    scores, labels = _make_scores(rng, n_per=80, n_clusters=10)
    df = compute_cluster_enrichment_stats(scores, labels, min_cells=10)
    # qvalues are non-decreasing in sort order
    qs = df["qvalue"].to_numpy()
    assert np.all(np.diff(qs) >= -1e-12)
    # qvalue >= pvalue for every row (BH never reduces below raw p)
    assert (df["qvalue"] >= df["pvalue"] - 1e-12).all()


def test_empty_when_no_cluster_passes_min_cells():
    """If every cluster is below min_cells we should return an empty
    DataFrame without crashing on the BH call (n=0 case)."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array(["A", "B", "C", "D", "E"])
    df = compute_cluster_enrichment_stats(scores, labels, min_cells=10)
    assert df.empty
    # Required columns still present so downstream code doesn't KeyError
    for col in ("qvalue", "significant"):
        assert col in df.columns


def test_nan_t_stat_cluster_is_skipped():
    """If a cluster's variance is degenerate enough for SciPy to return
    NaN, the row should be skipped (with alternative='greater' the test
    typically just yields p=1 or NaN — either way the row must not break
    the BH call)."""
    rng = np.random.default_rng(0)
    # 50 cells in cluster A all identical; 200 cells of background
    scores = np.concatenate([
        np.full(50, 0.1),
        rng.normal(0.0, 0.1, 200),
    ])
    labels = np.array(["A"] * 50 + ["bg"] * 200)
    df = compute_cluster_enrichment_stats(scores, labels, min_cells=10)
    # Test should complete; cluster A is either present (with finite p) or
    # skipped; in either case BH shouldn't crash.
    assert "qvalue" in df.columns
    assert df["qvalue"].notna().any() or df.empty


def test_input_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_cluster_enrichment_stats(
            np.array([0.1, 0.2, 0.3]),
            np.array(["A", "B"]),
        )
