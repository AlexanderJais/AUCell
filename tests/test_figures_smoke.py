"""Smoke tests for the figure_* functions.

These don't assert visual correctness — they just confirm each figure
function builds a matplotlib Figure from minimal synthetic inputs
without raising. The cheapest possible regression net against the kind
of audit findings the round-1/round-2 commits hit (e.g. H8 colour-scale
edge case, M6 zero-mass spike, L3 clip annotation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from figures import (
    figure_aucell_cluster_barplot,
    figure_aucell_histogram,
    figure_aucell_umap,
    figure_aucell_violins,
    figure_aucell_zscore_violins,
    figure_bactrap_volcano,
    figure_celltype_umap,
    figure_marker_gene_diagnostic,
    figure_umap_enrichment,
)


@pytest.fixture
def cell_data():
    rng = np.random.default_rng(0)
    n = 800
    labels = rng.choice([f"C{i}" for i in range(20)], size=n)
    scores = rng.uniform(0, 0.4, size=n)
    # Plant some signal so one cluster has higher scores
    scores[labels == "C0"] += 0.2
    umap = rng.normal(size=(n, 2))
    return labels, scores, umap


@pytest.fixture
def bactrap_df():
    rng = np.random.default_rng(0)
    n = 500
    # `_hypomap_gene_name` is added by match_genes in the real pipeline; the
    # volcano renderer uses it to label top points.
    names = [f"g{i}" for i in range(n)]
    return pd.DataFrame({
        "gene_name": names,
        "_hypomap_gene_name": names,
        "log2FoldChange": rng.normal(0, 2, n),
        "padj": rng.uniform(1e-10, 1, n),
    })


def _close(fig):
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_volcano_smoke(bactrap_df):
    _close(figure_bactrap_volcano(bactrap_df, highlight_genes=["g0", "g1"]))


def test_volcano_handles_no_highlight(bactrap_df):
    _close(figure_bactrap_volcano(bactrap_df, highlight_genes=[]))


def test_volcano_clips_high_significance():
    """Regression for L3: when padj is microscopic, the clip line should
    render without raising."""
    names = [f"g{i}" for i in range(20)]
    df = pd.DataFrame({
        "gene_name": names,
        "_hypomap_gene_name": names,
        "log2FoldChange": np.linspace(-3, 5, 20),
        "padj": np.full(20, 1e-200),  # all clipped
    })
    _close(figure_bactrap_volcano(df))


def test_cluster_barplot_smoke(cell_data):
    labels, scores, _ = cell_data
    _close(figure_aucell_cluster_barplot(scores, labels, top_n=10, min_cluster_cells=5))


def test_violins_smoke(cell_data):
    labels, scores, _ = cell_data
    _close(figure_aucell_violins(scores, labels, top_n=8, min_cluster_cells=5))


def test_zscore_violins_smoke(cell_data):
    labels, scores, _ = cell_data
    z = pd.Series(
        np.random.default_rng(0).normal(2, 1, 20),
        index=[f"C{i}" for i in range(20)],
    )
    _close(figure_aucell_zscore_violins(
        scores, labels, z_by_cluster=z, top_n=8, min_cluster_cells=5,
    ))


def test_zscore_violins_all_z_tied_doesnt_crash(cell_data):
    """Regression for H8 (symmetric expansion when zmin == zmax)."""
    labels, scores, _ = cell_data
    z = pd.Series(np.full(20, 2.5), index=[f"C{i}" for i in range(20)])
    _close(figure_aucell_zscore_violins(
        scores, labels, z_by_cluster=z, top_n=5, min_cluster_cells=5,
    ))


def test_histogram_smoke(cell_data):
    _, scores, _ = cell_data
    _close(figure_aucell_histogram(scores))


def test_histogram_all_zero():
    """Regression for M6 — make sure the nonzero-subset plot path handles
    a degenerate all-zero input without raising."""
    _close(figure_aucell_histogram(np.zeros(500)))


def test_celltype_umap_smoke(cell_data):
    labels, _, umap = cell_data
    _close(figure_celltype_umap(
        umap, labels, highlight_clusters=[f"C{i}" for i in range(5)],
    ))


def test_aucell_umap_smoke(cell_data):
    _, scores, umap = cell_data
    _close(figure_aucell_umap(umap, scores))


def test_umap_enrichment_smoke(cell_data):
    labels, scores, umap = cell_data
    _close(figure_umap_enrichment(umap, labels, scores))


def test_marker_gene_diagnostic_smoke(cell_data):
    labels, _, _ = cell_data
    clusters = [f"C{i}" for i in range(8)]
    gene_stats = pd.DataFrame({
        "mean_expr": np.random.default_rng(0).uniform(0, 2, len(clusters)),
        "fraction_expressing": np.random.default_rng(0).uniform(0, 0.5, len(clusters)),
    }, index=clusters)
    _close(figure_marker_gene_diagnostic(gene_stats, clusters))
