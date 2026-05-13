"""Synthetic-data unit tests for the core AUCell scorer and helpers.

Run with::

    python -m pytest tests/test_aucell_scoring.py -q
"""

import sys
from pathlib import Path

# Allow running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pytest

from analysis import (
    compute_aucell_scores,
    compute_aucell_scores_multi,
    compute_empirical_null_aucell,
    compute_cluster_enrichment_stats,
)
from data_loading import _build_adata_gene_lookup, lookup_gene_in_atlas


def _make_planted_atlas(
    n_cells: int = 200,
    n_genes: int = 50,
    n_planted_cells: int = 50,
    n_signature_genes: int = 5,
    background_max: int = 10,
    boost: int = 50,
    seed: int = 0,
) -> ad.AnnData:
    """Build a small AnnData with a clean planted signal.

    Cells 0..n_planted_cells-1 strongly express genes 0..n_signature_genes-1
    on top of a uniform integer-count background.
    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, background_max, size=(n_cells, n_genes)).astype(np.float32)
    counts[:n_planted_cells, :n_signature_genes] += boost
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.raw = adata
    return adata


# ---------------------------------------------------------------------------
# T1: AUCell scoring against a planted signature
# ---------------------------------------------------------------------------

def test_aucell_planted_signature_scores_high_in_planted_cells():
    adata = _make_planted_atlas()
    sig = [f"g{i}" for i in range(5)]
    scores = compute_aucell_scores(adata, sig, top_fraction=0.05, seed=0)
    assert scores.shape == (adata.n_obs,)
    # Planted cells should approach the max-AUC of 1.0; background ~ baseline
    assert scores[:50].mean() > 0.9, f"planted mean too low: {scores[:50].mean():.3f}"
    assert scores[50:].mean() < 0.3, f"background mean too high: {scores[50:].mean():.3f}"
    assert np.isfinite(scores).all()


def test_aucell_deterministic_under_same_seed():
    adata = _make_planted_atlas()
    sig = [f"g{i}" for i in range(5)]
    s1 = compute_aucell_scores(adata, sig, top_fraction=0.05, seed=42)
    s2 = compute_aucell_scores(adata, sig, top_fraction=0.05, seed=42)
    assert np.array_equal(s1, s2)


def test_aucell_empty_signature_returns_zeros():
    adata = _make_planted_atlas()
    scores = compute_aucell_scores(adata, [], top_fraction=0.05, seed=0)
    assert scores.shape == (adata.n_obs,)
    assert np.all(scores == 0.0)


def test_aucell_unmatched_genes_reported():
    adata = _make_planted_atlas()
    info: dict = {}
    scores = compute_aucell_scores(
        adata, ["g0", "g1", "no_such_gene"],
        top_fraction=0.1, seed=0, info_out=info,
    )
    assert info["n_query_matched"] == 2
    assert info["n_query_requested"] == 3
    assert "no_such_gene" in info["unmatched"]
    assert scores.shape == (adata.n_obs,)


# ---------------------------------------------------------------------------
# T2: multi vs single-sig equivalence (within the un-bumped regime)
# ---------------------------------------------------------------------------

def test_aucell_multi_matches_single_when_no_bump():
    adata = _make_planted_atlas()
    sig_a = [f"g{i}" for i in range(5)]
    sig_b = [f"g{i}" for i in range(5, 10)]
    # top_fraction=0.5 -> shared n_top=25, both signatures size 5 -> no bump
    multi = compute_aucell_scores_multi(
        adata, [sig_a, sig_b], top_fraction=0.5, seed=0,
    )
    single_a = compute_aucell_scores(adata, sig_a, top_fraction=0.5, seed=0)
    single_b = compute_aucell_scores(adata, sig_b, top_fraction=0.5, seed=0)
    assert np.allclose(multi[:, 0], single_a, atol=1e-6)
    assert np.allclose(multi[:, 1], single_b, atol=1e-6)


# ---------------------------------------------------------------------------
# T3: empirical null
# ---------------------------------------------------------------------------

def test_empirical_null_planted_cluster_ranks_above_background():
    adata = _make_planted_atlas()
    sig = [f"g{i}" for i in range(5)]
    labels = np.array(["A"] * 50 + ["B"] * 150)
    df = compute_empirical_null_aucell(
        adata, sig, labels,
        compute_aucell_fn=compute_aucell_scores,
        n_control_sets=30, n_bins=3, seed=0, top_fraction=0.5,
        min_cluster_size=10,
    )
    assert set(df.index) == {"A", "B"}
    # Planted cluster gets the higher z and the smaller q
    assert df.loc["A", "z_empirical"] > df.loc["B", "z_empirical"]
    assert df.loc["A", "qvalue_empirical"] <= df.loc["B", "qvalue_empirical"]
    # No NaNs from the non-degenerate clusters
    assert np.isfinite(df["null_mean"]).all()


def test_empirical_null_degenerate_cluster_emits_nan_pvalue():
    """When all controls give the same score in a cluster (null_sd == 0),
    z and p should be NaN so BH skips the row instead of awarding ~0.5."""
    # Tiny atlas where one cluster has only 1 cell — variance collapses
    adata = _make_planted_atlas(n_cells=60, n_planted_cells=10)
    sig = [f"g{i}" for i in range(5)]
    # Cluster B has only 1 cell, won't be eligible; force eligibility small
    labels = np.array(["A"] * 10 + ["B"] * 50)
    df = compute_empirical_null_aucell(
        adata, sig, labels,
        compute_aucell_fn=compute_aucell_scores,
        n_control_sets=1, n_bins=2, seed=0, top_fraction=0.5,
        min_cluster_size=5,
    )
    # With only 1 control set, null_sd is by definition zero everywhere
    assert df["null_sd"].fillna(0).eq(0).all()
    assert df["pvalue_empirical"].isna().all()
    assert df["z_empirical"].isna().all()


# ---------------------------------------------------------------------------
# T4: case-sensitive gene lookup
# ---------------------------------------------------------------------------

def _atlas_with_names(var_names, n_cells: int = 20) -> ad.AnnData:
    """Atlas with the given var names — set before .raw snapshot so the raw
    layer reflects the names the test cares about."""
    n_genes = len(var_names)
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 10, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var_names = list(var_names)
    adata.raw = adata
    return adata


def test_gene_lookup_distinguishes_case_distinct_paralogs():
    adata = _atlas_with_names(["Foo", "FOO", "bar", "Bar", "baz", "BAZ", "qux", "QUX"])
    lookup, _, _ = _build_adata_gene_lookup(adata)
    hit_foo = lookup_gene_in_atlas(lookup, "Foo")
    hit_foo_upper = lookup_gene_in_atlas(lookup, "FOO")
    assert hit_foo is not None and hit_foo_upper is not None
    assert hit_foo[0] == "Foo" and hit_foo_upper[0] == "FOO"
    assert hit_foo[1] != hit_foo_upper[1]


def test_gene_lookup_case_insensitive_fallback():
    adata = _atlas_with_names(["Foo", "Bar", "Baz", "Qux"])
    lookup, _, _ = _build_adata_gene_lookup(adata)
    # Exact case
    assert lookup_gene_in_atlas(lookup, "Foo") is not None
    # Lowercase falls back
    hit_lower = lookup_gene_in_atlas(lookup, "foo")
    assert hit_lower is not None and hit_lower[0] == "Foo"
    # Uppercase falls back too
    hit_upper = lookup_gene_in_atlas(lookup, "FOO")
    assert hit_upper is not None and hit_upper[0] == "Foo"


def test_gene_lookup_missing_returns_none():
    adata = _atlas_with_names(["Foo", "Bar", "Baz", "Qux"])
    lookup, _, _ = _build_adata_gene_lookup(adata)
    assert lookup_gene_in_atlas(lookup, "xyz") is None
    assert lookup_gene_in_atlas(lookup, "") is None


# ---------------------------------------------------------------------------
# Welch t-test sanity: depleted cluster gets p ≈ 1 (was 0.5 before C1 fix)
# ---------------------------------------------------------------------------

def test_welch_one_sided_p_correct_for_depleted_cluster():
    rng = np.random.default_rng(0)
    scores = np.concatenate([
        rng.normal(1.0, 0.1, 100),
        rng.normal(0.0, 0.1, 300),
    ])
    labels = np.array(["A"] * 100 + ["B"] * 300)
    df = compute_cluster_enrichment_stats(scores, labels, min_cells=10)
    p_a = float(df.loc[df["cluster"] == "A", "pvalue"].iloc[0])
    p_b = float(df.loc[df["cluster"] == "B", "pvalue"].iloc[0])
    # Cluster A is enriched -> p ≈ 0; cluster B is depleted -> p ≈ 1
    assert p_a < 1e-30
    assert p_b > 0.99
