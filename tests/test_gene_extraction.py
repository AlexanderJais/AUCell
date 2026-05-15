"""Tests for `_extract_gene_submatrix` and `_map_var_indices_to_raw`.

The pipeline often resolves a gene index in `adata.var` (the highly-
variable subset) but the actual expression matrix it scores against
lives in `adata.raw.X` (the full gene panel). The mapping between the
two spaces is non-trivial when var is a strict subset of raw, and silent
fallback paths can produce mismatched gene-name labels (audit H12).
These tests pin the contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import pytest

from data_loading import _extract_gene_submatrix, _map_var_indices_to_raw


def _atlas_with_raw_superset(var_names, raw_var_names, n_cells: int = 20):
    """Build an AnnData where adata.raw contains MORE genes than adata.var.

    This is the realistic scenario after Scanpy's HVG filter: var is a
    subset of raw, and var positions ≠ raw positions for the same symbol.
    """
    n_raw = len(raw_var_names)
    rng = np.random.default_rng(0)
    raw_counts = rng.integers(0, 10, size=(n_cells, n_raw)).astype(np.float32)
    raw_ad = ad.AnnData(X=sp.csr_matrix(raw_counts))
    raw_ad.var_names = list(raw_var_names)

    # var: a subset of raw genes
    var_idx_in_raw = [raw_var_names.index(g) for g in var_names]
    sub = raw_ad[:, var_idx_in_raw].copy()
    sub.raw = raw_ad
    return sub, var_idx_in_raw


def test_indices_in_raw_returns_correct_columns():
    raw_names = [f"g{i}" for i in range(10)]
    var_names = ["g1", "g3", "g7"]
    adata, _ = _atlas_with_raw_superset(var_names, raw_names)

    # Indices that already point into raw space
    raw_idx = np.array([0, 5, 9], dtype=np.int64)
    X, survived = _extract_gene_submatrix(
        adata, raw_idx, use_raw=True, indices_in_raw=True,
    )
    assert survived.all()
    assert X.shape == (adata.n_obs, 3)
    # Each column should equal the corresponding column of adata.raw.X
    expected = adata.raw.X[:, raw_idx].toarray() if sp.issparse(adata.raw.X) else adata.raw.X[:, raw_idx]
    assert np.allclose(np.asarray(X), expected)


def test_var_indices_remapped_to_raw_when_use_raw():
    raw_names = [f"g{i}" for i in range(10)]
    var_names = ["g1", "g3", "g7"]
    adata, var_in_raw = _atlas_with_raw_superset(var_names, raw_names)

    # Indices in var space — _extract should remap to raw and return
    # columns equivalent to raw_X[:, var_in_raw[var_idx]].
    var_idx = np.array([0, 2], dtype=np.int64)  # picks g1 and g7
    X, survived = _extract_gene_submatrix(
        adata, var_idx, use_raw=True, indices_in_raw=False,
    )
    assert survived.all()
    assert X.shape == (adata.n_obs, 2)
    expected = adata.raw.X[:, [var_in_raw[i] for i in var_idx]].toarray()
    assert np.allclose(np.asarray(X), expected)


def test_map_var_indices_to_raw_round_trips_by_name():
    raw_names = [f"g{i}" for i in range(10)]
    var_names = ["g2", "g5", "g8"]
    adata, var_in_raw = _atlas_with_raw_superset(var_names, raw_names)

    var_idx = np.array([0, 1, 2])
    raw_idx, survived = _map_var_indices_to_raw(adata, var_idx)
    assert survived.all()
    assert list(raw_idx) == var_in_raw


def test_out_of_bounds_var_index_marked_unsurvived():
    raw_names = [f"g{i}" for i in range(10)]
    var_names = ["g0", "g1", "g2"]
    adata, _ = _atlas_with_raw_superset(var_names, raw_names)

    # Index 999 is out of bounds for var (only 3 var entries)
    var_idx = np.array([0, 999, 2])
    X, survived = _extract_gene_submatrix(
        adata, var_idx, use_raw=True, indices_in_raw=False,
    )
    # The out-of-bounds index gets marked as not-survived
    assert survived[0] and not survived[1] and survived[2]
    assert X.shape[1] == 2  # only two survived


def test_empty_gene_indices_returns_empty_matrix():
    raw_names = [f"g{i}" for i in range(5)]
    var_names = ["g0", "g1"]
    adata, _ = _atlas_with_raw_superset(var_names, raw_names)
    X, survived = _extract_gene_submatrix(
        adata, np.array([], dtype=np.int64), use_raw=True,
    )
    assert X.shape[1] == 0
    assert survived.size == 0
