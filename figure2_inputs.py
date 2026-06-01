"""Shared inputs for Figure 2 (gene expression within a region).

Pure numpy/pandas/scipy — no Streamlit, no seaborn — so the standalone
Figure 2 script and the Streamlit app compute *identical* per-cell
expression, expressing masks and cluster rankings, and therefore render
identical UMAPs via figures.figure_gene_poa_umap.

Normalisation matches the approved standalone figure: CP10k + log1p using
the precomputed per-cell total `nCount_RNA`, on the matrix where the gene
symbol resolves (main `.X` first, then `.raw`, mirroring how the original
figure was produced).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def region_cell_mask(adata, keywords=("preoptic",),
                     region_col: str = "Region_summarized") -> np.ndarray:
    """Per-cell region mask, identical to data_loading.get_poa_cell_mask in its
    per-cell mode (annotation_col=None, include_na=False): a cell is True iff
    any keyword is a case-insensitive substring of its region label, treating
    NaN / "nan"/"none"/"na"/"" as missing (and therefore not a hit).

    Kept here (pure, no Streamlit) so the standalone script and the app share
    the exact same mask without importing the Streamlit-backed module.
    """
    kws = tuple(str(k).strip().lower() for k in keywords if str(k).strip())
    raw = adata.obs[region_col]
    text = raw.astype(str)
    na = raw.isna() | text.str.strip().str.lower().isin({"nan", "none", "na", ""})
    lowered = text.str.lower()
    hit = np.zeros(len(text), dtype=bool)
    for kw in kws:
        hit |= lowered.str.contains(kw, regex=False, na=False).to_numpy()
    hit &= ~na.to_numpy()
    return hit


def resolve_gene_index(adata, symbol: str) -> Optional[Tuple[int, str, bool]]:
    """Resolve a gene symbol to (column_index, display_name, use_raw).

    Searches the main matrix first (var_names then a feature_name-style symbol
    column), then `.raw`. CELLxGENE atlases keep Ensembl IDs in var_names and
    symbols in `feature_name`. Returns None if not found.
    """
    sym = symbol.strip().lower()
    targets = [(adata.var_names, adata.var, False)]
    if adata.raw is not None:
        targets.append((adata.raw.var_names, adata.raw.var, True))
    for var_names, var_df, use_raw in targets:
        names = np.array([str(v).lower() for v in var_names])
        hit = np.where(names == sym)[0]
        if len(hit):
            return int(hit[0]), str(np.asarray(var_names)[hit[0]]), use_raw
        for col in ("feature_name", "gene_name", "gene_symbol", "symbol",
                    "external_gene_name"):
            if col in var_df.columns:
                vals = var_df[col].astype(str).str.lower().values
                hit = np.where(vals == sym)[0]
                if len(hit):
                    return int(hit[0]), str(var_df[col].iloc[hit[0]]), use_raw
    return None


def _dense_column(adata, gene_idx: int, use_raw: bool) -> np.ndarray:
    src = adata.raw if (use_raw and adata.raw is not None) else adata
    col = src[:, gene_idx].X
    if sparse.issparse(col):
        col = col.toarray()
    return np.asarray(col, dtype=float).ravel()


def gene_expressing_masks(adata, symbols, threshold: float = 0.0):
    """Resolve a list of gene symbols to per-cell *expressing* masks.

    Used by the multi-gene composite dot plot. For each symbol we resolve its
    column (main matrix then ``.raw``, like ``resolve_gene_index``) and mark a
    cell as expressing when its raw count > ``threshold`` — the same definition
    ``gene_poa_inputs`` uses for the single-gene panels.

    Returns
    -------
    names : list[str]
        The requested symbols that resolved, in input order (used as the dot
        plot's column labels). Unresolved symbols are dropped.
    expressing : bool array (n_cells, len(names))
        Column ``j`` is the expressing mask for ``names[j]``. Shape
        ``(n_cells, 0)`` if nothing resolved.
    missing : list[str]
        Symbols that did not resolve in the atlas.
    """
    names, cols, missing = [], [], []
    for sym in symbols:
        hit = resolve_gene_index(adata, sym)
        if hit is None:
            missing.append(str(sym))
            continue
        idx, _disp, use_raw = hit
        col = _dense_column(adata, idx, use_raw)
        cols.append(col > threshold)
        names.append(str(sym))
    if cols:
        expressing = np.column_stack(cols)
    else:
        expressing = np.zeros((adata.n_obs, 0), dtype=bool)
    return names, expressing, missing


def gene_poa_inputs(
    adata,
    gene_idx: int,
    use_raw: bool,
    region_mask: np.ndarray,
    cell_labels: np.ndarray,
    min_cells: int = 20,
    top_n: int = 20,
    threshold: float = 0.0,
):
    """Compute the per-cell expression, expressing mask, ranking and highlight
    list used by Figure 2.

    Returns
    -------
    expr : float array (n_cells,)
        CP10k + log1p expression per cell.
    expressing : bool array (n_cells,)
        Raw count > ``threshold``.
    ranking : DataFrame indexed by cluster (region cells, >= ``min_cells``),
        columns n_cells / mean_lognorm / frac_expressing, sorted descending.
    highlight : list[str]
        All region clusters with **detectable** expression (frac_expressing >
        0), capped at ``top_n``. This is the set of POA-resident Pnoc-positive
        clusters shown in the paper (Table S1) — every preoptic sub-region
        (MPA, LPO, periventricular), not just the top few by mean.
    """
    col = _dense_column(adata, gene_idx, use_raw)
    depth = np.asarray(adata.obs["nCount_RNA"], dtype=float)
    depth[depth == 0] = 1.0
    expr = np.log1p(col / depth * 1e4)
    expressing = col > threshold

    region_mask = np.asarray(region_mask, dtype=bool)
    cell_labels = np.asarray(cell_labels)
    df = pd.DataFrame({
        "cluster": cell_labels[region_mask],
        "expr": expr[region_mask],
        "frac": expressing[region_mask].astype(float),
    })
    ranking = (df.groupby("cluster")
                 .agg(n_cells=("expr", "size"),
                      mean_lognorm=("expr", "mean"),
                      frac_expressing=("frac", "mean")))
    ranking = ranking[ranking["n_cells"] >= min_cells].sort_values(
        "mean_lognorm", ascending=False)
    detectable = ranking[ranking["frac_expressing"] > 0]
    highlight = detectable.head(top_n).index.tolist()
    return expr, expressing, ranking, highlight
