"""
Core analysis functions for AUCell-based mapping of a bulk-RNA-seq DE
signature onto a single-cell atlas.

Provides signature extraction and refinement, per-cell AUCell scoring,
expression-bin matched empirical-null AUCell, and cluster-level
enrichment statistics.
"""

import logging

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats, sparse
from statsmodels.stats.multitest import multipletests
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

from data_loading import _build_adata_gene_lookup, lookup_gene_in_atlas


def get_enriched_genes(
    bactrap_df: pd.DataFrame,
    padj_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    min_ip_expression: float = 0.0,
    ip_col: str = "IP",
) -> pd.DataFrame:
    """
    Filter bacTRAP data for significantly enriched genes.

    Returns subset of bactrap_df passing padj, log2FC, and (optionally)
    minimum IP expression thresholds.

    The *min_ip_expression* filter suppresses the DESeq2 low-count
    artefact in which a gene with near-zero Input counts gets an
    inflated log₂FC from pseudocount division (e.g. 28 IP reads vs
    0 Input → log₂FC ~7).  These genes would dominate any
    log₂FC-ranked top-N list despite being statistically weak.  When
    ``ip_col`` is not present in the DataFrame the filter is skipped
    with a warning.
    """
    for required_col in ("padj", "log2FoldChange"):
        if required_col not in bactrap_df.columns:
            raise ValueError(
                f"Required column '{required_col}' not found in bacTRAP data. "
                f"Available columns: {list(bactrap_df.columns)}"
            )
    n_total = len(bactrap_df)
    has_padj = bactrap_df["padj"].notna().sum()
    passes_padj = (bactrap_df["padj"] < padj_cutoff).sum()
    passes_fc = (bactrap_df["log2FoldChange"] > log2fc_cutoff).sum()
    # Sanity check: if the bacTRAP table is dominated by negative fold
    # changes, the user has likely uploaded an Input-vs-IP table (sign
    # inverted) and the positive-side filter will return ~nothing. Warn
    # loudly so they get a useful breadcrumb instead of a silent zero.
    sig_padj = (bactrap_df["padj"].notna()) & (bactrap_df["padj"] < padj_cutoff)
    n_sig_up = int(((bactrap_df["log2FoldChange"] > 0) & sig_padj).sum())
    n_sig_down = int(((bactrap_df["log2FoldChange"] < 0) & sig_padj).sum())
    if (n_sig_up + n_sig_down) >= 100 and n_sig_up < 0.25 * (n_sig_up + n_sig_down):
        logger.warning(
            "get_enriched_genes: only %d of %d significant DE genes are "
            "UP-regulated (%.1f%%). This is unusual for a bacTRAP IP-vs-Input "
            "contrast — if you uploaded an Input-vs-IP table the sign is "
            "flipped and the positive-side filter will keep almost nothing. "
            "Verify the comparison direction in your DESeq2 output.",
            n_sig_up, n_sig_up + n_sig_down,
            100.0 * n_sig_up / max(n_sig_up + n_sig_down, 1),
        )
    mask = (
        (bactrap_df["padj"].notna())
        & (bactrap_df["padj"] < padj_cutoff)
        & (bactrap_df["log2FoldChange"] > log2fc_cutoff)
    )
    ip_filter_applied = False
    if min_ip_expression > 0.0:
        if ip_col in bactrap_df.columns:
            ip_values = pd.to_numeric(bactrap_df[ip_col], errors="coerce")
            ip_mask = ip_values.fillna(0) >= min_ip_expression
            n_before_ip = mask.sum()
            mask = mask & ip_mask
            ip_filter_applied = True
            logger.info(
                "get_enriched_genes: IP filter '%s' >= %.3g removed %d/%d genes "
                "(%d → %d after filter)",
                ip_col, min_ip_expression,
                n_before_ip - mask.sum(), n_before_ip,
                n_before_ip, mask.sum(),
            )
        else:
            logger.warning(
                "get_enriched_genes: min_ip_expression=%.3g requested but "
                "column '%s' not in bacTRAP data — filter SKIPPED. "
                "Available columns: %s",
                min_ip_expression, ip_col, list(bactrap_df.columns),
            )
    n_enriched = mask.sum()
    logger.info("get_enriched_genes: %d total, %d with padj, %d pass padj<%.3f, "
                "%d pass log2FC>%.2f, %d pass both%s",
                n_total, has_padj, passes_padj, padj_cutoff, passes_fc, log2fc_cutoff, n_enriched,
                f" (after min_IP>={min_ip_expression:.3g} filter)" if ip_filter_applied else "")
    return bactrap_df[mask].copy()


_SIGNATURE_DROPLOG_COLUMNS = [
    "gene", "status", "detection_rate", "max_cluster_mean",
    "frac_clusters_above_thresh", "reason",
]


def filter_signature_genes_by_atlas(
    candidate_genes: List[str],
    cluster_mean_expr: pd.DataFrame,        # rows = genes, cols = clusters, log-norm
    cell_detection_rate: pd.Series,         # index = genes, values = fraction of cells with count > 0
    *,
    apply_detectability: bool = True,
    min_detection_rate: float = 0.02,
    min_max_cluster_mean: float = 0.05,
    apply_specificity: bool = True,
    specificity_cluster_mean_thresh: float = 0.5,
    specificity_max_cluster_fraction: float = 0.5,
    logger=None,
) -> Tuple[List[str], pd.DataFrame]:
    """Apply detectability and specificity filters to a candidate signature.

    Two operationally-defined pre-filters, intended for the bacTRAP candidate
    signature *after* the DESeq2 padj / log₂FC / min-IP filters and *before*
    π-score ranking / top-N selection:

    * **Filter A — HypoMap detectability.** Drop gene ``g`` if either the
      fraction of atlas cells with detected counts of ``g`` is below
      ``min_detection_rate``, or the maximum per-cluster mean log-norm
      expression of ``g`` across all clusters is below ``min_max_cluster_mean``.
      A gene that's at zero in a cell's top-τ pool can never appear in the
      AUCell window and contributes only noise.
    * **Filter B — Cluster specificity.** Drop gene ``g`` if its per-cluster
      mean log-norm expression exceeds ``specificity_cluster_mean_thresh`` in
      more than ``specificity_max_cluster_fraction`` of clusters.  Broadly
      expressed housekeeping / pan-neuronal genes carry no cell-type
      specificity and pull non-target clusters into the AUCell top.

    Filters are applied in order (detectability first, then specificity).  A
    gene that would fail both is reported against whichever filter caught it
    first; it is not double-counted.  When **both** filters are disabled the
    candidate list is returned unchanged with every drop-log row marked
    ``'kept'``.

    Returns
    -------
    kept : list[str]
        Genes that passed both filters, in input order.
    drop_log : pd.DataFrame
        One row per candidate gene, columns
        ``gene, status, detection_rate, max_cluster_mean,
        frac_clusters_above_thresh, reason`` — ``status`` is one of
        ``'kept'``, ``'dropped_detectability'``, ``'dropped_specificity'``,
        ``'dropped_not_in_atlas'``.  Statistics are NaN where they weren't
        computed because the gene isn't in the atlas.
    """
    log = logger or logging.getLogger(__name__)
    cand = [str(g) for g in candidate_genes]
    n_total = len(cand)

    log.info("Signature refinement: detectability=%s, specificity=%s",
             "on" if apply_detectability else "off",
             "on" if apply_specificity else "off")
    log.info("Candidate genes: %d", n_total)

    # ---- per-gene atlas statistics (vectorised) ----
    cme = cluster_mean_expr if cluster_mean_expr is not None else pd.DataFrame()
    # transpose if the candidate genes appear to live on the columns
    if len(cme.index) and len(cme.columns):
        cand_set = set(cand)
        if (len(cand_set & set(map(str, cme.columns)))
                > len(cand_set & set(map(str, cme.index)))):
            cme = cme.T
    if len(cme.index) and cme.index.duplicated().any():
        cme = cme.groupby(level=0).mean()

    det = cell_detection_rate if cell_detection_rate is not None else pd.Series(dtype=float)
    det = pd.to_numeric(pd.Series(det), errors="coerce")
    if det.index.duplicated().any():
        det = det.groupby(level=0).mean()

    cand_idx = pd.Index(cand)
    in_cme = cand_idx.isin(cme.index)
    in_det = cand_idx.isin(det.index)
    in_atlas = in_cme & in_det

    sub = cme.reindex(cand)
    if len(sub.columns):
        sub = sub.apply(pd.to_numeric, errors="coerce")
        max_cluster_mean = sub.max(axis=1, skipna=True).to_numpy(dtype=float)
        n_present = sub.notna().sum(axis=1).to_numpy()
        n_above = (sub > specificity_cluster_mean_thresh).sum(axis=1).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_above = np.where(n_present > 0, n_above / np.maximum(n_present, 1), np.nan)
    else:
        max_cluster_mean = np.full(n_total, np.nan)
        frac_above = np.full(n_total, np.nan)
    detection_rate = det.reindex(cand).to_numpy(dtype=float)

    # NaN-out statistics for genes that aren't fully in the atlas
    detection_rate = np.where(in_atlas, detection_rate, np.nan)
    max_cluster_mean = np.where(in_atlas, max_cluster_mean, np.nan)
    frac_above = np.where(in_atlas, frac_above, np.nan)

    status = np.array(["kept"] * n_total, dtype=object)
    reason = np.array([""] * n_total, dtype=object)

    both_off = (not apply_detectability) and (not apply_specificity)
    if not both_off:
        not_in_atlas = ~in_atlas
        status[not_in_atlas] = "dropped_not_in_atlas"
        reason[not_in_atlas] = "not in atlas"

        if apply_detectability:
            open_mask = (status == "kept")
            dr = np.where(np.isnan(detection_rate), 0.0, detection_rate)
            mcm = np.where(np.isnan(max_cluster_mean), 0.0, max_cluster_mean)
            fail_rate = open_mask & (dr < min_detection_rate)
            fail_mean = open_mask & (mcm < min_max_cluster_mean)
            fail_det = fail_rate | fail_mean
            for i in np.flatnonzero(fail_det):
                rs = []
                if fail_rate[i]:
                    rs.append(f"detection_rate={detection_rate[i]:.4g} < {min_detection_rate:g}")
                if fail_mean[i]:
                    rs.append(f"max_cluster_mean={max_cluster_mean[i]:.4g} < {min_max_cluster_mean:g}")
                reason[i] = "; ".join(rs)
            status[fail_det] = "dropped_detectability"

        if apply_specificity:
            open_mask = (status == "kept")
            fa = np.where(np.isnan(frac_above), 0.0, frac_above)
            fail_spec = open_mask & (fa > specificity_max_cluster_fraction)
            for i in np.flatnonzero(fail_spec):
                reason[i] = (
                    f"frac of clusters with mean > {specificity_cluster_mean_thresh:g} "
                    f"= {frac_above[i]:.1%} > {specificity_max_cluster_fraction:.0%}"
                )
            status[fail_spec] = "dropped_specificity"

    kept = [g for g, s in zip(cand, status) if s == "kept"]

    drop_log = pd.DataFrame(
        {
            "gene": cand,
            "status": status,
            "detection_rate": detection_rate,
            "max_cluster_mean": max_cluster_mean,
            "frac_clusters_above_thresh": frac_above,
            "reason": reason,
        },
        columns=_SIGNATURE_DROPLOG_COLUMNS,
    )

    # ---- logging ----
    n_not_atlas = int((status == "dropped_not_in_atlas").sum())
    n_det = int((status == "dropped_detectability").sum())
    n_spec = int((status == "dropped_specificity").sum())
    n_kept = len(kept)
    # "after detectability" = candidates minus (not-in-atlas + detectability drops)
    log.info(
        "After detectability filter: %d kept, %d dropped (min_detection=%g, min_max_cluster_mean=%g)",
        n_total - n_not_atlas - n_det, n_not_atlas + n_det,
        min_detection_rate, min_max_cluster_mean,
    )
    log.info(
        "After specificity filter: %d kept, %d dropped (cluster_mean_thresh=%g, max_cluster_frac=%g)",
        n_kept, n_spec, specificity_cluster_mean_thresh, specificity_max_cluster_fraction,
    )
    log.info("Final refined signature: %d genes", n_kept)

    dropped_rows = drop_log[drop_log["status"] != "kept"]
    if 0 < len(dropped_rows) < 50:
        for _, r in dropped_rows.iterrows():
            log.info("  dropped %s (%s): %s", r["gene"], r["status"], r["reason"])
    elif len(dropped_rows) >= 50:
        log.info("  %d genes dropped — per-gene reasons at DEBUG level", len(dropped_rows))
        for _, r in dropped_rows.iterrows():
            log.debug("  dropped %s (%s): %s", r["gene"], r["status"], r["reason"])

    # Tripwire: Pnoc must survive (sanity check, not a hard error).
    if ("Pnoc" in set(cand)) and ("Pnoc" not in set(kept)):
        _pnoc_row = drop_log.loc[drop_log["gene"] == "Pnoc"].iloc[0]
        log.warning(
            "Signature refinement DROPPED 'Pnoc' — status=%s, reason='%s', "
            "detection_rate=%s, max_cluster_mean=%s, frac_clusters_above_thresh=%s. "
            "This is unexpected for a Pnoc-Cre signature; review the refinement thresholds.",
            _pnoc_row["status"], _pnoc_row["reason"],
            _pnoc_row["detection_rate"], _pnoc_row["max_cluster_mean"],
            _pnoc_row["frac_clusters_above_thresh"],
        )

    return kept, drop_log


def rank_enriched_genes(
    df: pd.DataFrame,
    metric: str = "pi_score",
) -> pd.DataFrame:
    """
    Sort an enriched-gene DataFrame by the chosen ranking metric,
    most-enriched first.

    Metrics
    -------
    ``"pi_score"`` (recommended default)
        ``|log₂FC| × -log₁₀(padj)`` — the π-score of Xiao et al.
        (Bioinformatics 2014). Combines effect size and significance so
        a gene with a modest but highly significant fold-change (e.g.
        Pnoc in a Pnoc-Cre line: FC=2, padj=1e-17) outranks a low-count
        zero-Input artefact (e.g. FC=7, padj=1e-4).
    ``"log2fc"``
        Raw ``log₂FoldChange``. Historical default; vulnerable to
        low-count inflation.
    ``"padj"``
        ``-log₁₀(padj)``. Ranks by statistical significance only; ignores
        effect size.

    Ties are broken by padj (ascending) then by log₂FC (descending).
    """
    if len(df) == 0:
        return df.copy()
    df = df.copy()
    eps = 1e-300  # avoid log10(0) for genes with padj==0

    if metric == "pi_score":
        score = np.abs(df["log2FoldChange"]) * -np.log10(df["padj"].clip(lower=eps))
    elif metric == "log2fc":
        score = df["log2FoldChange"].astype(float)
    elif metric == "padj":
        score = -np.log10(df["padj"].clip(lower=eps))
    else:
        raise ValueError(
            f"Unknown ranking metric '{metric}'. "
            f"Valid options: 'pi_score', 'log2fc', 'padj'."
        )

    df = df.assign(_rank_score=score)
    df = df.sort_values(
        ["_rank_score", "padj", "log2FoldChange"],
        ascending=[False, True, False],
        kind="mergesort",
    ).drop(columns="_rank_score").reset_index(drop=True)
    logger.info(
        "rank_enriched_genes: metric='%s', %d genes ranked. "
        "Top 3: %s",
        metric, len(df),
        df["_hypomap_gene_name"].head(3).tolist() if "_hypomap_gene_name" in df.columns else df.head(3).index.tolist(),
    )
    return df


def compute_aucell_scores(
    adata,
    gene_names: List[str],
    use_raw: bool = True,
    top_fraction: float = 0.05,
    seed: int = 0,
    info_out: Optional[Dict] = None,
    progress_callback=None,
) -> np.ndarray:
    """
    Compute AUCell scores for each cell.

    AUCell (Aibar et al., Nature Methods 2017) ranks genes by expression
    within each cell, then computes the Area Under the recovery Curve (AUC)
    for the gene set of interest within the top-ranked genes.

    This is more robust than simple mean expression because:
    - It's rank-based (insensitive to normalization differences)
    - It focuses on highly expressed genes per cell
    - It's threshold-free

    Tie-breaking: in sparse single-cell data thousands of genes per cell are
    tied at low integer counts (especially 0). ``np.argpartition`` and
    ``np.argsort`` break ties by memory layout, so signature genes at low
    matrix indices would systematically win their ties (or lose, depending on
    layout) — biasing AUCell scores. We add per-cell uniform jitter smaller
    than the smallest gap between distinct expression values, which preserves
    the order of *distinct* values but randomises the order of ties. This
    matches R AUCell's ``ties.method = "random"`` (Aibar et al. 2017,
    Methods §"Building the rankings").

    Args:
        adata: AnnData object
        gene_names: list of bacTRAP-enriched gene names
        use_raw: whether to use adata.raw for expression
        top_fraction: fraction of ranked genes to consider (default 5%)
        seed: RNG seed for the per-cell tie-breaking jitter (reproducible)
        info_out: optional dict; if provided, augmented with scoring diagnostics
            (``n_query_matched``, ``unmatched``, ``n_top``, ``n_top_bumped``,
            ``effective_top_fraction``) so the caller can surface them to users.

    Returns:
        Array of AUCell scores, one per cell.
    """
    # Use the shared gene-name lookup built by data_loading — it resolves
    # both symbols and Ensembl IDs, matching the lookup used everywhere else
    # in the pipeline (fix #5: previously this function used a simpler
    # lowercase-only lookup that silently dropped genes when the matrix
    # layer used a different namespace than the signature).
    lookup, source_gene_names, is_raw_lookup = _build_adata_gene_lookup(
        adata, use_raw=use_raw,
    )
    if is_raw_lookup and adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X

    query_idx: List[int] = []
    unmatched: List[str] = []
    for g in gene_names:
        hit = lookup_gene_in_atlas(lookup, g)
        if hit is not None:
            query_idx.append(hit[1])
        else:
            unmatched.append(str(g))

    n_cells = X.shape[0]
    n_total_genes = X.shape[1]
    n_query = len(query_idx)

    logger.info("compute_aucell_scores: %d/%d genes found, %d cells, top_fraction=%.2f, seed=%d",
                n_query, len(gene_names), n_cells, top_fraction, seed)
    if unmatched:
        _sample = unmatched[:20]
        logger.warning(
            "  %d/%d signature genes did not match the %s layer lookup "
            "(first 20: %s). Check gene-namespace / alias resolution between "
            "the bacTRAP DE table and the atlas.",
            len(unmatched), len(gene_names),
            "raw" if is_raw_lookup else "X",
            _sample,
        )
    if n_query == 0:
        logger.warning("  no genes found — returning zero AUCell scores")
        if info_out is not None:
            info_out.update({
                "n_query_matched": 0,
                "unmatched": unmatched,
                "n_top": 0,
                "n_top_bumped": False,
                "effective_top_fraction": 0.0,
            })
        return np.zeros(n_cells)

    # Boolean mask for query genes (vectorized membership test)
    query_mask = np.zeros(n_total_genes, dtype=bool)
    query_mask[query_idx] = True

    # Number of top genes to consider per cell. When the signature is larger
    # than top_fraction * n_total_genes, n_top is bumped to the signature
    # size — the canonical Aibar formula requires n_top >= n_query so every
    # hit can be recovered. Warn loudly in that case (fix #6): the user
    # thinks they're at "top 5%" but the window is actually wider, which
    # makes scores less stringent than the slider suggests.
    requested_n_top = int(n_total_genes * top_fraction)
    n_top = max(requested_n_top, n_query)
    n_top = min(n_top, n_total_genes)
    n_top_bumped = n_top > requested_n_top
    effective_top_fraction = float(n_top) / max(n_total_genes, 1)
    if n_top_bumped:
        logger.warning(
            "  n_top bumped from %d (%.2f%% of genes) to %d (%.2f%%) to fit "
            "the signature of %d genes. The AUCell window is wider than the "
            "requested top_fraction — effective threshold is %.2f%%.",
            requested_n_top, top_fraction * 100,
            n_top, effective_top_fraction * 100,
            n_query, effective_top_fraction * 100,
        )
    else:
        logger.info("  n_top=%d (%.2f%% of %d genes)",
                    n_top, effective_top_fraction * 100, n_total_genes)
    # Theoretical maximum of sum(cumsum(is_hit)) when all n_query query genes
    # occupy the top n_query positions:
    #   cumsum = [1, 2, ..., n_query, n_query, ..., n_query]  (n_top entries)
    #   sum    = n_query*(n_query+1)/2 + n_query*(n_top - n_query)
    #          = n_query * (n_top - (n_query - 1)/2)
    # Note: n_top >= n_query is guaranteed above, so this is always positive.
    max_auc = n_query * (n_top - (n_query - 1) / 2)

    # Two independent RNG streams so that the per-cell jitter is purely a
    # function of `seed` and not of how many cells were sampled for the
    # jitter-scale heuristic. Without the split, rng.choice(n_cells, ...)
    # consumed a variable amount of state before the jitter draws, so
    # otherwise-identical signature runs on atlases with different
    # n_cells would produce different AUCell numbers at the same seed.
    sampling_rng, jitter_rng = np.random.default_rng(seed).spawn(2)

    # Determine a safe jitter scale: must be smaller than the smallest gap
    # between distinct expression values, otherwise jitter could reorder
    # genuinely distinct values. For raw integer counts the smallest gap is 1
    # (so jitter < 0.5 is safe). For non-integer (e.g. log-normalised) data we
    # use HALF the smallest *stored* nonzero value, computed over the full
    # nnz array (O(nnz) single pass) — sampling 500 cells gave an unstable
    # estimate that varied across atlases and POA restrictions.
    if sparse.issparse(X):
        nz_data = X.data
        global_max = float(nz_data.max()) if nz_data.size else 0.0
        global_min_nz = float(nz_data.min()) if nz_data.size else 0.0
        is_integer = (
            nz_data.size > 0 and bool(np.all(nz_data == np.round(nz_data)))
        )
    else:
        X_dense = np.asarray(X)
        global_max = float(X_dense.max()) if X_dense.size else 0.0
        nz_vals = X_dense[X_dense > 0]
        global_min_nz = float(nz_vals.min()) if nz_vals.size else 0.0
        is_integer = (
            X_dense.size > 0 and bool(np.all(X_dense == np.round(X_dense)))
        )
    if is_integer and global_max > 5:
        jitter_scale = np.float32(0.49)
        logger.info("  input looks like raw integer counts; jitter_scale=0.49")
    else:
        if global_min_nz > 0:
            jitter_scale = np.float32(0.49 * global_min_nz)
        else:
            jitter_scale = np.float32(1e-6)
        logger.warning(
            "  input does not look like raw integer counts (max=%.2f, integer=%s); "
            "using scaled jitter (%.2e, ½ of smallest stored nonzero value). "
            "AUCell is designed for raw counts (Aibar et al. 2017) — set "
            "use_raw=True against an integer-count layer for the cleanest behaviour.",
            global_max, is_integer, float(jitter_scale),
        )
    # sampling_rng is no longer used for jitter-scale derivation, but the
    # split keeps `jitter_rng` decoupled from any future use of `sampling_rng`.
    del sampling_rng

    # Process in cell chunks — vectorized within each chunk.
    #
    # Pre-allocate the dense + jitter buffers once and reuse across chunks.
    # The old loop body produced ~2.7 GB of transient allocation per HypoMap
    # chunk (float64 toarray + float32 astype copy + jitter alloc + jitter*scale
    # alloc); preallocation keeps the chunk buffers stable across iterations.
    # Chunk size is reduced from 5000 → 2000 to offset the buffers staying
    # resident: argpartition() internally allocates a (chunk_n, n_genes)
    # int64 scratch (~1 GB at 5000×28000) per call, so a smaller chunk_n
    # caps the peak alongside the preallocated buffers.
    #
    # All RNG draws / arithmetic are bit-identical to the previous code path
    # (numpy's Generator.random emits values in flat row-major order regardless
    # of chunk size, so per-cell jitter is unchanged) — scores are identical.
    chunk_size = 2000
    scores = np.zeros(n_cells, dtype=np.float32)
    eff_chunk = min(chunk_size, n_cells) if n_cells else 1
    dense_buf = np.empty((eff_chunk, n_total_genes), dtype=np.float32)
    jitter_buf = np.empty((eff_chunk, n_total_genes), dtype=np.float32)
    js = np.float32(jitter_scale)
    n_chunks = (n_cells + chunk_size - 1) // chunk_size if n_cells else 0

    for ci, start in enumerate(range(0, n_cells, chunk_size)):
        end = min(start + chunk_size, n_cells)
        chunk_n = end - start
        dense_view = dense_buf[:chunk_n]
        jitter_view = jitter_buf[:chunk_n]

        X_chunk = X[start:end, :]
        if sparse.issparse(X_chunk):
            # scipy.sparse.csr_matrix.toarray accepts an `out=` buffer (≥ scipy
            # 1.0); using it avoids materialising a separate float64 copy then
            # casting to float32.
            try:
                X_chunk.toarray(out=dense_view)
            except TypeError:
                np.copyto(dense_view, X_chunk.toarray(), casting="unsafe")
        else:
            np.copyto(dense_view, np.asarray(X_chunk), casting="unsafe")

        # Per-cell uniform jitter ∈ [0, jitter_scale). Smaller than the
        # smallest distinct gap, so distinct values keep their order while
        # ties are randomised — equivalent to ties.method="random" in R.
        jitter_rng.random(dtype=np.float32, out=jitter_view)
        np.multiply(jitter_view, js, out=jitter_view)
        np.add(dense_view, jitter_view, out=dense_view)

        # For each cell, get top-n gene indices via argpartition (O(n) per cell).
        # Then sort the top-n by jittered expression descending — fully
        # vectorised across cells — and reduce to AUC.
        top_idx = np.argpartition(dense_view, -n_top, axis=1)[:, -n_top:]
        top_vals = np.take_along_axis(dense_view, top_idx, axis=1)
        order = np.argsort(-top_vals, axis=1)
        sorted_top = np.take_along_axis(top_idx, order, axis=1)
        is_hit = query_mask[sorted_top]  # (chunk_n, n_top) bool
        cumhits = np.cumsum(is_hit, axis=1)
        chunk_auc = cumhits.sum(axis=1)
        if max_auc > 0:
            scores[start:end] = chunk_auc / max_auc
        # else: scores already zero-initialised
        if progress_callback is not None:
            try:
                progress_callback(ci + 1, n_chunks)
            except Exception:
                logger.debug("progress_callback raised; continuing", exc_info=True)
    del dense_buf, jitter_buf

    logger.info("  AUCell scores: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                float(scores.mean()), float(scores.std()), float(scores.min()), float(scores.max()))

    if info_out is not None:
        info_out.update({
            "n_query_matched": int(n_query),
            "n_query_requested": int(len(gene_names)),
            "unmatched": list(unmatched),
            "n_top": int(n_top),
            "n_top_bumped": bool(n_top_bumped),
            "requested_top_fraction": float(top_fraction),
            "effective_top_fraction": float(effective_top_fraction),
            "source_layer": "raw" if is_raw_lookup else "X",
        })

    return scores


def compute_aucell_scores_multi(
    adata,
    gene_name_lists: List[List[str]],
    *,
    use_raw: bool = True,
    top_fraction: float = 0.05,
    seed: int = 0,
    progress_callback=None,
) -> np.ndarray:
    """Score several gene-set signatures with AUCell in a single pass.

    Equivalent to calling :func:`compute_aucell_scores` once per signature, but
    the expensive per-cell ranking — random-jitter tie-breaking + ``argpartition``
    + ``argsort`` — is done **once** and shared across all signatures; only the
    cheap per-signature recovery-curve AUC differs.  Used to score the empirical-
    null control sets without N separate passes over the (≈385k-cell) atlas.

    All signatures share the same top-*k* window ``k = max(⌈top_fraction·G⌉,
    max_s |S_s|)``.  This is identical to what the single-signature scorer would
    pick for each signature unless some signature is larger than the τ window
    (the "bumped" regime), in which case the wider shared window is a minor
    approximation; for the empirical null (control sets the same size as the
    signature, both ≪ the τ window in practice) the result is bit-identical to
    looping :func:`compute_aucell_scores`.

    ``progress_callback`` (optional) is called ``fn(chunks_done, n_chunks)``.

    Returns
    -------
    np.ndarray of shape ``(n_cells, len(gene_name_lists))``, dtype float32.
    """
    lookup, _source_gene_names, is_raw_lookup = _build_adata_gene_lookup(adata, use_raw=use_raw)
    X = adata.raw.X if (is_raw_lookup and adata.raw is not None) else adata.X
    n_cells, n_genes = X.shape
    n_sigs = len(gene_name_lists)
    if n_sigs == 0:
        return np.zeros((n_cells, 0), dtype=np.float32)

    # Resolve each signature to its (sorted, de-duplicated) gene-column indices.
    query_idx_per_sig: List[np.ndarray] = []
    n_query_per_sig = np.zeros(n_sigs, dtype=np.int64)
    for s, names in enumerate(gene_name_lists):
        idxs = []
        for g in names:
            hit = lookup_gene_in_atlas(lookup, g)
            if hit is not None:
                idxs.append(hit[1])
        arr = np.array(sorted(set(idxs)), dtype=np.int64)
        query_idx_per_sig.append(arr)
        n_query_per_sig[s] = arr.size

    max_nq = int(n_query_per_sig.max()) if n_sigs else 0
    n_top_requested = max(int(n_genes * top_fraction), 1)
    n_top = max(n_top_requested, max(max_nq, 1))
    n_top = min(n_top, n_genes)
    if n_top > n_top_requested:
        logger.warning(
            "compute_aucell_scores_multi: bumping shared n_top from %d "
            "(%.2f%% of %d genes) to %d so it covers the largest signature "
            "(%d genes). This widens the ranking window for every signature; "
            "if you didn't intend this, reduce signature size or raise the "
            "top_fraction slider.",
            n_top_requested, 100 * top_fraction, n_genes, n_top, max_nq,
        )

    # query_masks[s, g] = 1.0 iff gene g ∈ signature s
    query_masks = np.zeros((n_sigs, n_genes), dtype=np.float32)
    for s, arr in enumerate(query_idx_per_sig):
        if arr.size:
            query_masks[s, arr] = 1.0
    query_masks_T = np.ascontiguousarray(query_masks.T)  # (n_genes, n_sigs)

    # theoretical max discrete-AUC per signature within the shared top-k window
    nq = n_query_per_sig.astype(np.float64)
    max_auc = np.where(nq > 0, nq * (n_top - (nq - 1.0) / 2.0), 1.0).astype(np.float64)
    # recovery-curve weights: a hit at rank j (0-indexed) within the top-k window
    # contributes (n_top - j) to  sum_r C(r) == sum(cumsum(is_hit)).
    w = np.arange(n_top, 0, -1, dtype=np.float64)

    # ---- jitter scale: same heuristic as compute_aucell_scores ----
    sampling_rng, jitter_rng = np.random.default_rng(int(seed)).spawn(2)
    sample_n = min(500, n_cells)
    sample_pick = (
        sampling_rng.choice(n_cells, sample_n, replace=False)
        if n_cells > sample_n else np.arange(n_cells)
    )
    sample_X = X[sample_pick, :]
    sample_X = np.asarray(sample_X.toarray()) if sparse.issparse(sample_X) else np.asarray(sample_X)
    sample_max = float(sample_X.max()) if sample_X.size else 0.0
    sample_is_integer = sample_X.size > 0 and bool(np.all(sample_X == np.round(sample_X)))
    if sample_is_integer and sample_max > 5:
        jitter_scale = np.float32(0.49)
    else:
        nz = sample_X[sample_X > 0]
        jitter_scale = np.float32(0.49 * float(nz.min())) if nz.size else np.float32(1e-6)

    logger.info(
        "compute_aucell_scores_multi: %d signatures, %d cells, n_top=%d, "
        "sig sizes %d..%d, seed=%d",
        n_sigs, n_cells, n_top,
        int(n_query_per_sig.min()) if n_sigs else 0, max_nq, int(seed),
    )

    chunk_size = 2000  # see compute_aucell_scores for rationale on size choice
    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    scores = np.zeros((n_cells, n_sigs), dtype=np.float32)
    # Preallocate dense + jitter buffers once and reuse — same memory
    # optimisation as compute_aucell_scores. Arithmetic is bit-identical to
    # the prior path so multi-vs-single equivalence (test T2) is preserved.
    eff_chunk = min(chunk_size, n_cells) if n_cells else 1
    dense_buf = np.empty((eff_chunk, n_genes), dtype=np.float32)
    jitter_buf = np.empty((eff_chunk, n_genes), dtype=np.float32)
    js = np.float32(jitter_scale)
    for ci, start in enumerate(range(0, n_cells, chunk_size)):
        end = min(start + chunk_size, n_cells)
        chunk_n = end - start
        dense_view = dense_buf[:chunk_n]
        jitter_view = jitter_buf[:chunk_n]

        X_chunk = X[start:end, :]
        if sparse.issparse(X_chunk):
            try:
                X_chunk.toarray(out=dense_view)
            except TypeError:
                np.copyto(dense_view, X_chunk.toarray(), casting="unsafe")
        else:
            np.copyto(dense_view, np.asarray(X_chunk), casting="unsafe")
        jitter_rng.random(dtype=np.float32, out=jitter_view)
        np.multiply(jitter_view, js, out=jitter_view)
        np.add(dense_view, jitter_view, out=dense_view)

        top_idx = np.argpartition(dense_view, -n_top, axis=1)[:, -n_top:]    # (chunk_n, n_top)
        vals = np.take_along_axis(dense_view, top_idx, axis=1)
        order = np.argsort(-vals, axis=1)                                    # descending by value
        sorted_top = np.take_along_axis(top_idx, order, axis=1)              # (chunk_n, n_top)

        # Sparse recovery-weight matrix  W[i, sorted_top[i, j]] = w[j];
        # then  AUC[i, s] = sum_g W[i, g] * query_masks[s, g]  ==  W @ Q.T
        indptr = np.arange(0, chunk_n * n_top + 1, n_top, dtype=np.int64)
        W_sp = sparse.csr_matrix(
            (np.tile(w, chunk_n), sorted_top.ravel(), indptr),
            shape=(chunk_n, n_genes),
        )
        auc_chunk = W_sp.dot(query_masks_T)                                  # (chunk_n, n_sigs)
        scores[start:end, :] = (auc_chunk / max_auc[None, :]).astype(np.float32)
        if progress_callback is not None:
            try:
                progress_callback(ci + 1, n_chunks)
            except Exception:
                logger.debug("progress_callback raised; continuing", exc_info=True)

    del dense_buf, jitter_buf
    logger.info("compute_aucell_scores_multi: done — scores mean=%.4f over %d signatures",
                float(scores.mean()) if scores.size else 0.0, n_sigs)
    return scores


def _atlas_gene_mean_logexpr(adata, use_raw: bool = True, *, cache_suffix: str = "") -> np.ndarray:
    """Atlas-wide mean expression per gene, log1p-scaled (raw-layer order).

    Used purely to bin genes by expression level for control-set matching, so
    a (monotone) raw-mean → log1p transform is sufficient — the bin
    assignments are rank-based and unaffected by the transform.  Cached on
    ``adata.uns['gene_mean_expr[_<cache_suffix>]']`` so it is computed once per
    atlas load (the ``cache_suffix`` distinguishes e.g. a POA-restricted view);
    the sparse column sums are cheap (no dense materialisation).
    """
    # Include `use_raw` in the cache key so callers that pass different layer
    # choices never collide on the same suffix.
    layer_tag = "raw" if (use_raw and adata.raw is not None) else "X"
    cache_key = (
        f"gene_mean_expr_{layer_tag}_{cache_suffix}" if cache_suffix
        else f"gene_mean_expr_{layer_tag}"
    )
    if use_raw and adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X
    n_genes_expected = X.shape[1]
    cached = adata.uns.get(cache_key)
    if cached is not None:
        cached = np.asarray(cached)
        if cached.shape[0] == n_genes_expected:
            return cached
    if sparse.issparse(X):
        gene_sum = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    else:
        gene_sum = np.asarray(X, dtype=np.float64).sum(axis=0)
    gene_mean = np.log1p(gene_sum / max(X.shape[0], 1))
    adata.uns[cache_key] = gene_mean
    logger.info("Built bin lookup over %d atlas genes", len(gene_mean))
    return gene_mean


def _expression_bins(gene_mean_expr: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each gene to an expression quantile bin (0..n_bins-1).

    Ties (e.g. the large mass of never-expressed genes) collapse bins; the
    returned array still gives every gene a finite bin index.  Cached-friendly
    but cheap, so recomputed per call to follow the user's ``n_bins``.
    """
    s = pd.Series(np.asarray(gene_mean_expr, dtype=np.float64))
    try:
        bins = pd.qcut(s.rank(method="first"), q=int(max(n_bins, 1)),
                       labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(s), dtype=int))
    return bins.fillna(0).astype(int).to_numpy()


def compute_empirical_null_aucell(
    adata,
    signature_genes: List[str],
    cluster_labels,
    compute_aucell_fn,
    *,
    n_control_sets: int = 100,
    n_bins: int = 5,
    seed: int = 0,
    top_fraction: float = 0.05,
    min_cluster_size: int = 20,
    use_raw: bool = True,
    mask_signature: str = "",
    logger=None,
    progress_callback=None,
) -> pd.DataFrame:
    """Empirical-null AUCell with expression-matched control gene sets.

    For each cluster, compares the bacTRAP-signature AUCell mean against the
    distribution of AUCell means obtained from ``n_control_sets`` random
    control gene sets matched to the signature in size and atlas-wide
    expression structure (Aibar et al., *Nat. Methods* 2017).  This removes
    the "baseline gene-rank-width" bias that inflates AUCell scores in
    broadly-active neuronal clusters.

    Parameters
    ----------
    adata
        Atlas the AUCell scoring runs against — a restricted view (e.g. the
        POA-cell subset) when an atlas restriction is active.
    signature_genes
        The bacTRAP signature gene symbols (atlas namespace).
    cluster_labels
        Per-cell cluster labels aligned positionally with ``adata.obs_names``.
    compute_aucell_fn
        Callable with the ``compute_aucell_scores`` signature — injected so
        the function is testable and so the same scorer (and tie-breaking
        regime) is used for the signature and the controls.
    n_control_sets, n_bins, seed, top_fraction, min_cluster_size
        See module / sidebar docs.
    mask_signature
        Appended to the ``adata.uns`` cache keys for the gene-mean / quantile-
        bin lookups so a restricted view never reuses full-atlas statistics.
    progress_callback
        Optional ``fn(i, n)`` called during the batched control scoring pass.

    Returns
    -------
    DataFrame indexed by cluster with columns
        ``null_mean``, ``null_sd``, ``z_empirical``, ``pvalue_empirical``,
        ``qvalue_empirical``, ``n_control_sets_used``.
    """
    log = logger or globals()["logger"]
    rng = np.random.default_rng(int(seed))

    # ---- Expression-matched control gene sets ----
    # When an atlas restriction (e.g. POA-only) is active, `adata` here is the
    # restricted view, so gene means / quantile bins are recomputed on the
    # restricted cells; `mask_signature` is appended to the cache keys so the
    # restricted statistics never collide with the full-atlas ones.
    lookup, gene_names, is_raw = _build_adata_gene_lookup(adata, use_raw=use_raw)
    gene_mean_expr = _atlas_gene_mean_logexpr(adata, use_raw=is_raw, cache_suffix=mask_signature)
    gene_bins = _expression_bins(gene_mean_expr, n_bins)
    _bins_key = f"gene_expr_bins_{mask_signature}" if mask_signature else "gene_expr_bins"
    adata.uns[_bins_key] = gene_bins

    sig_idx: List[int] = []
    for g in signature_genes:
        hit = lookup_gene_in_atlas(lookup, g)
        if hit is not None:
            sig_idx.append(hit[1])
    sig_idx = list(dict.fromkeys(sig_idx))  # de-dup, preserve order
    sig_idx_set = set(sig_idx)

    # Bin → candidate indices (excluding signature genes themselves)
    bin_to_candidates: Dict[int, np.ndarray] = {}
    for b in np.unique(gene_bins):
        cand = np.where(gene_bins == b)[0]
        cand = cand[~np.isin(cand, list(sig_idx_set))]
        bin_to_candidates[int(b)] = cand

    matchable_idx: List[int] = []
    dropped_idx: List[int] = []
    for idx in sig_idx:
        b = int(gene_bins[idx])
        if bin_to_candidates.get(b) is not None and bin_to_candidates[b].size > 0:
            matchable_idx.append(idx)
        else:
            dropped_idx.append(idx)
    k_total = len(sig_idx)
    k_matched = len(matchable_idx)
    k_dropped = len(dropped_idx)
    log.info(
        "Empirical null: enabled (N=%d, bins=%d, seed=%d)",
        int(n_control_sets), int(n_bins), int(seed),
    )
    log.info(
        "Signature size: %d, in-bin matchable: %d, dropped: %d",
        k_total, k_matched, k_dropped,
    )
    if k_dropped:
        log.warning(
            "  %d signature gene(s) have no in-bin control candidates and were "
            "excluded from control matching AND signature scoring so the null "
            "and signature share the same gene-set size: %s",
            k_dropped, [gene_names[i] for i in dropped_idx][:20],
        )
    if k_matched == 0:
        log.warning("compute_empirical_null_aucell: no matchable signature genes — skipping")
        return pd.DataFrame(columns=[
            "null_mean", "null_sd", "z_empirical", "pvalue_empirical",
            "qvalue_empirical", "n_control_sets_used",
        ])

    # ---- Per-cell signature AUCell (scored on the *matchable* subset so the
    # signature and the controls share the same gene-set size — otherwise
    # k_dropped signature genes would inflate the test statistic relative to
    # the null purely by gene-count). ----
    matchable_gene_names = [str(gene_names[i]) for i in matchable_idx]
    sig_scores = np.asarray(
        compute_aucell_fn(adata, matchable_gene_names,
                          top_fraction=top_fraction, seed=int(seed)),
        dtype=np.float64,
    )
    labels = pd.Series(np.asarray(cluster_labels).astype(str))
    if len(labels) != len(sig_scores):
        raise ValueError(
            f"cluster_labels ({len(labels)}) and AUCell scores ({len(sig_scores)}) "
            f"length mismatch"
        )
    cluster_sizes = labels.value_counts()
    eligible_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
    if not eligible_clusters:
        log.warning("compute_empirical_null_aucell: no cluster >= %d cells", min_cluster_size)
        return pd.DataFrame(columns=[
            "null_mean", "null_sd", "z_empirical", "pvalue_empirical",
            "qvalue_empirical", "n_control_sets_used",
        ])

    def _per_cluster_means(scores: np.ndarray) -> pd.Series:
        return pd.Series(scores).groupby(labels.values).mean().reindex(eligible_clusters)

    sig_means = _per_cluster_means(sig_scores)

    import time as _time
    # Draw all N control gene-name lists up front (deterministic given `seed`),
    # then score them in a single batched pass over the atlas. Within each
    # control set, genes drawn from the same bin are sampled WITHOUT
    # replacement so a small bin can't double-count one control gene and
    # mechanically inflate that bin's contribution to the null. If a bin has
    # fewer candidates than required, fall back to with-replacement (rare).
    from collections import Counter as _Counter
    bin_demand: "_Counter[int]" = _Counter(int(gene_bins[idx]) for idx in matchable_idx)
    control_name_lists: List[List[str]] = []
    for _c in range(int(n_control_sets)):
        ctrl_idx: List[int] = []
        for b, k in bin_demand.items():
            cand = bin_to_candidates[b]
            replace = cand.size < k
            picks = rng.choice(cand, size=k, replace=replace)
            ctrl_idx.extend(int(x) for x in np.atleast_1d(picks))
        control_name_lists.append([str(gene_names[i]) for i in ctrl_idx])

    _t0 = _time.time()
    ctrl_scores_mat = compute_aucell_scores_multi(
        adata, control_name_lists,
        use_raw=use_raw, top_fraction=top_fraction, seed=int(seed),
        progress_callback=progress_callback,
    )  # (n_cells, N)
    _t_total = _time.time() - _t0
    n_used = int(n_control_sets)
    # Per-cluster mean of every control set at once → (n_eligible, N) → (N, n_eligible)
    control_cluster_means = (
        pd.DataFrame(np.asarray(ctrl_scores_mat, dtype=np.float64))
        .groupby(labels.values).mean()
        .reindex(eligible_clusters)
        .to_numpy()
        .T
    )
    log.info(
        "Empirical null: scored %d control sets in one batched pass — "
        "%.1fs total (%.3fs/set amortised)",
        n_used, _t_total, _t_total / max(n_used, 1),
    )

    null_mean = control_cluster_means.mean(axis=0)
    null_sd = control_cluster_means.std(axis=0, ddof=1) if n_used > 1 else np.zeros(len(eligible_clusters))
    sig_vec = sig_means.to_numpy(dtype=np.float64)
    degenerate = ~np.isfinite(null_sd) | (null_sd == 0)
    z = np.where(degenerate, np.nan, (sig_vec - null_mean) / np.where(degenerate, 1.0, null_sd))
    # one-sided empirical p (add-1 smoothing): controls >= signature. For
    # degenerate clusters (null variance ≈ 0, all controls identical) the
    # empirical p is meaningless — emit NaN so they're excluded from BH.
    ge_counts = (control_cluster_means >= sig_vec[None, :]).sum(axis=0)
    pvals = (1.0 + ge_counts) / (n_used + 1.0)
    pvals = np.where(degenerate, np.nan, pvals)

    if degenerate.any():
        log.warning(
            "Clusters with degenerate null_sd (set to NaN z): %s",
            [eligible_clusters[i] for i in np.where(degenerate)[0]],
        )

    out = pd.DataFrame({
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z_empirical": z,
        "pvalue_empirical": pvals,
        "n_control_sets_used": n_used,
    }, index=pd.Index(eligible_clusters, name="cluster"))
    valid_p = out["pvalue_empirical"].notna().to_numpy()
    qvals = np.full(len(out), np.nan)
    if valid_p.any():
        _, q, _, _ = multipletests(out.loc[valid_p, "pvalue_empirical"].values, method="fdr_bh")
        qvals[valid_p] = q
    out["qvalue_empirical"] = qvals
    out = out[[
        "null_mean", "null_sd", "z_empirical", "pvalue_empirical",
        "qvalue_empirical", "n_control_sets_used",
    ]]

    # Sanity check: rank correlation between signature mean and z (should be
    # positive but < ~0.95 — the null must re-order at least some clusters).
    try:
        valid = np.isfinite(z) & np.isfinite(sig_vec)
        if valid.sum() >= 3:
            rho = float(stats.spearmanr(sig_vec[valid], z[valid])[0])
            log.info("Empirical null sanity: Spearman(mean, z_empirical) = %.3f over %d clusters",
                     float(rho), int(valid.sum()))
            ranked_by_mean = sig_means[valid].sort_values(ascending=False).index.tolist()
            z_series = pd.Series(z, index=eligible_clusters)
            ranked_by_z = z_series[valid].sort_values(ascending=False).index.tolist()
            for cl in ranked_by_mean[:10]:
                drop = ranked_by_z.index(cl) - ranked_by_mean.index(cl)
                if drop > 0:
                    log.info("  %s: rank by mean=%d -> by z=%d (drops %d)",
                             cl, ranked_by_mean.index(cl) + 1, ranked_by_z.index(cl) + 1, drop)
    except Exception:
        log.debug("Empirical null sanity check failed", exc_info=True)

    return out


def validate_aucell_input(
    adata, use_raw: bool = True, sample_size: int = 500, seed: int = 0,
) -> Dict:
    """Sanity-check the expression layer that will be fed into AUCell.

    AUCell is defined on gene-expression *ranks* per cell; the paper (Aibar
    et al. 2017) and the reference implementations build rankings from raw
    UMI counts. Feeding in pre-log-normalised data is not strictly wrong —
    the rank order of *distinct* values is preserved by any monotonic
    transform — but ties collapse very differently (most log-normalised
    matrices lose the fine-grained tie structure of low counts), so scores
    and their interpretation shift. This function inspects a random sample
    of the chosen layer and returns a machine-readable QC report that the
    caller can surface to the user.

    Returns:
        dict with keys
            layer: "raw" or "X"
            n_cells, n_genes, n_cells_sampled
            max_value, min_nonzero
            is_integer: bool — all sampled values are integers
            looks_like_counts: bool — is_integer AND max_value > 50
            warnings: list[str] — user-facing strings (empty when clean)
            info: list[str] — informational lines
    """
    report: Dict = {
        "layer": "raw" if (use_raw and adata.raw is not None) else "X",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_cells_sampled": 0,
        "max_value": 0.0,
        "min_nonzero": float("nan"),
        "is_integer": False,
        "looks_like_counts": False,
        "warnings": [],
        "info": [],
    }
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        report["n_genes"] = int(adata.raw.n_vars)
    else:
        X = adata.X
        if use_raw and adata.raw is None:
            report["warnings"].append(
                "use_raw=True was requested but adata.raw is None; falling "
                "back to adata.X. If adata.X is log-normalised, AUCell tie "
                "structure will differ from the canonical raw-count version."
            )

    n_cells = X.shape[0]
    sample_n = min(sample_size, n_cells)
    if n_cells > sample_n:
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(n_cells, sample_n, replace=False)
        sample = X[pick, :]
    else:
        sample = X
    if sparse.issparse(sample):
        sample = np.asarray(sample.toarray())
    else:
        sample = np.asarray(sample)
    report["n_cells_sampled"] = int(sample.shape[0])

    if sample.size == 0:
        report["warnings"].append("Selected layer is empty.")
        return report

    sample_max = float(sample.max())
    nonzero = sample[sample > 0]
    sample_min_nz = float(nonzero.min()) if nonzero.size > 0 else float("nan")
    is_int = bool(np.all(sample == np.round(sample)))
    looks_like_counts = is_int and sample_max > 50

    report["max_value"] = sample_max
    report["min_nonzero"] = sample_min_nz
    report["is_integer"] = is_int
    report["looks_like_counts"] = looks_like_counts
    report["info"].append(
        f"AUCell input layer: {report['layer']} ({sample.shape[0]} cells x "
        f"{sample.shape[1]} genes sampled, max={sample_max:.2f}, "
        f"min_nonzero={sample_min_nz:.3g}, integer={is_int})"
    )

    if not looks_like_counts:
        if is_int and sample_max <= 50:
            report["warnings"].append(
                f"AUCell input has integer values but max = {sample_max:.0f} "
                f"(<= 50) — unusually low for raw UMI counts. Verify that "
                f"adata.raw contains counts and not e.g. a binarised layer."
            )
        else:
            report["warnings"].append(
                f"AUCell input does NOT look like raw integer counts "
                f"(max={sample_max:.2f}, integer={is_int}). AUCell "
                f"(Aibar et al. 2017) is defined on raw counts; log-"
                f"normalised or scaled input will preserve distinct-value "
                f"rank order but collapses tie structure differently. "
                f"Consider pointing adata.raw at a raw-count layer before "
                f"scoring — compute_aucell_scores will still run and will "
                f"scale its tie-breaking jitter accordingly."
            )

    logger.info("validate_aucell_input: layer=%s, looks_like_counts=%s, "
                "max=%.2f, integer=%s, n_warnings=%d",
                report["layer"], looks_like_counts, sample_max, is_int,
                len(report["warnings"]))
    return report


def compute_cluster_enrichment_stats(
    aucell_scores: np.ndarray,
    cell_labels: np.ndarray,
    min_cells: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-cluster enrichment significance from AUCell scores.

    For each cluster we test the one-sided null that its AUCell scores are
    drawn from the same distribution as the rest of the atlas, using
    Welch's t-test (unequal variance). Multiple testing across clusters is
    corrected with Benjamini-Hochberg to give a per-cluster q-value.

    Important caveat: the in-cluster vs out-of-cluster AUCell scores are NOT
    strictly independent (every cell's score is computed from the same
    per-cell gene rankings against a fixed signature). The Welch test
    treats them as if they were, so the p-values it returns are
    anti-conservative. Use them as a quick descriptive ranking, not as a
    rigorous significance call — for the latter, prefer the matched-
    expression empirical null implemented in
    :func:`compute_empirical_null_aucell`, which compares each cluster's
    score against its own permutation distribution and is robust to the
    rank-coupling.

    This fills the gap the previous pipeline had: ranking clusters by mean
    AUCell alone cannot distinguish "strongly enriched" from "a small
    cluster that happens to have a slightly above-average mean" — the
    q-value addresses exactly that. Clusters with fewer than *min_cells*
    cells are dropped (Welch's t breaks down at very small n).

    Returns a DataFrame (sorted by qvalue, ascending) with columns:
        cluster, n_cells, mean, sem, std, t_stat, pvalue, qvalue, significant
    """
    scores = np.asarray(aucell_scores, dtype=np.float64)
    labels = np.asarray(cell_labels)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"aucell_scores ({scores.shape[0]}) and cell_labels "
            f"({labels.shape[0]}) length mismatch"
        )

    unique_labels = pd.unique(labels)
    rows = []
    for cl in unique_labels:
        mask = labels == cl
        n = int(mask.sum())
        if n < min_cells:
            continue
        in_scores = scores[mask]
        out_scores = scores[~mask]
        if in_scores.size < 2 or out_scores.size < 2:
            continue
        # Welch's one-sided t-test: cluster > rest. SciPy's alternative=
        # parameter (>=1.6) gives the correct upper-tail p directly, avoiding
        # the broken `p_two/2 if t>0` halving heuristic at t≈0 / degenerate
        # variance.
        t_stat, p_one = stats.ttest_ind(
            in_scores, out_scores,
            equal_var=False, nan_policy="omit", alternative="greater",
        )
        if np.isnan(t_stat) or np.isnan(p_one):
            continue
        rows.append({
            "cluster": str(cl),
            "n_cells": n,
            "mean": float(in_scores.mean()),
            "std": float(in_scores.std(ddof=1)),
            "sem": float(in_scores.std(ddof=1) / np.sqrt(n)),
            "t_stat": float(t_stat),
            "pvalue": float(p_one),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("compute_cluster_enrichment_stats: no clusters passed "
                       "the min_cells=%d filter", min_cells)
        df["qvalue"] = []
        df["significant"] = []
        return df

    _, qvals, _, _ = multipletests(df["pvalue"].values, method="fdr_bh")
    df["qvalue"] = qvals
    df["significant"] = df["qvalue"] < alpha

    df = df.sort_values(["qvalue", "pvalue"], ascending=True).reset_index(drop=True)

    n_sig = int(df["significant"].sum())
    # BH-FDR controls the false-discovery rate proportional to the number of
    # tests; surface that count explicitly so a reader of the log can sanity-
    # check the q-value scale against the chosen annotation level (more
    # clusters -> more tests -> more expected discoveries at the same alpha).
    logger.info(
        "compute_cluster_enrichment_stats: %d/%d clusters tested (>= %d "
        "cells), %d significant at BH-FDR q < %.3f (expected ~%.1f false "
        "positives at this alpha if the global null held)",
        len(df), len(unique_labels), int(min_cells), n_sig, alpha,
        float(alpha) * len(df),
    )
    return df


# =========================================================================
# Composite Ranking
# =========================================================================

