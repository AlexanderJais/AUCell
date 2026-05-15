"""Synthetic test that an all-True POA mask produces identical AUCell
scores to scoring the full atlas — verifying that `adata[mask].copy()`
and downstream caches don't silently introduce a difference for the
no-op case.

Cheap regression net for the audit's H10 / C5 concern about view-vs-copy
semantics for `.raw`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd

from analysis import compute_aucell_scores


def _make_atlas(n_cells: int = 300, n_genes: int = 80, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 10, size=(n_cells, n_genes)).astype(np.float32)
    counts[:60, :8] += 30  # planted signature in first cluster
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["cluster"] = (["A"] * 60 + ["B"] * (n_cells - 60))
    adata.raw = adata
    return adata


def test_aucell_scores_match_full_atlas_under_all_true_mask():
    """An all-True POA mask is a no-op restriction; AUCell scores against
    `adata[mask].copy()` must match those against the unrestricted atlas."""
    adata = _make_atlas()
    sig = [f"g{i}" for i in range(8)]

    scores_full = compute_aucell_scores(adata, sig, top_fraction=0.1, seed=0)

    mask = np.ones(adata.n_obs, dtype=bool)
    adata_view = adata[mask].copy()
    scores_view = compute_aucell_scores(adata_view, sig, top_fraction=0.1, seed=0)

    assert scores_full.shape == scores_view.shape
    assert np.allclose(scores_full, scores_view, atol=1e-6), (
        "AUCell scores diverged between full atlas and adata[True].copy()"
    )


def test_aucell_scores_match_partial_mask_against_manual_subset():
    """A partial POA mask should give the same scores as if the input had
    been the subset to begin with — sanity-check that .copy() preserves
    raw layer alignment."""
    adata = _make_atlas()
    sig = [f"g{i}" for i in range(8)]

    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[::2] = True  # every other cell
    sub_a = adata[mask].copy()
    # Build the same subset from scratch (independent path)
    sub_b = ad.AnnData(X=adata.raw.X[mask].copy())
    sub_b.var_names = adata.var_names
    sub_b.raw = sub_b

    scores_a = compute_aucell_scores(sub_a, sig, top_fraction=0.1, seed=0)
    scores_b = compute_aucell_scores(sub_b, sig, top_fraction=0.1, seed=0)
    assert np.allclose(scores_a, scores_b, atol=1e-6)
