"""
Data loading and gene matching utilities for AUCell-based mapping of a
bulk-RNA-seq DE signature onto a single-cell atlas.
"""

import logging
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import streamlit as st
from scipy import sparse
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading HypoMap atlas (this may take a few minutes)...")
def load_hypomap(file_path: str) -> ad.AnnData:
    """Load the HypoMap h5ad atlas with memory-efficient settings."""
    logger.info("Loading HypoMap from %s", file_path)
    adata = sc.read_h5ad(file_path, backed=None)
    logger.info("HypoMap loaded: %d cells x %d vars", adata.n_obs, adata.n_vars)
    logger.info("  var_names (first 10): %s", list(adata.var_names[:10]))
    logger.info("  var.columns: %s", list(adata.var.columns))
    if adata.raw is not None:
        logger.info("  raw present: %d vars", adata.raw.n_vars)
        logger.info("  raw.var_names (first 10): %s", list(adata.raw.var_names[:10]))
        logger.info("  raw.var.columns: %s", list(adata.raw.var.columns))
        # Log sample values from each raw.var column
        for col in adata.raw.var.columns:
            sample = adata.raw.var[col].dropna().head(5).tolist()
            logger.info("  raw.var['%s'] sample: %s", col, sample)
    else:
        logger.info("  raw: None")
    logger.info("  obs.columns: %s", list(adata.obs.columns))
    # Ensure expression matrix is sparse
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    if adata.raw is not None and not sparse.issparse(adata.raw.X):
        # Rebuild raw layer with sparse matrix via the public API
        import anndata
        raw_adata = anndata.AnnData(
            X=sparse.csr_matrix(adata.raw.X),
            var=adata.raw.var,
        )
        adata.raw = raw_adata
    return adata


@st.cache_data(show_spinner="Loading bacTRAP data...")
def load_bactrap(file_path: str) -> pd.DataFrame:
    """Load the bacTRAP FPKM/DESeq2 results from an Excel file."""
    logger.info("Loading bacTRAP from %s", file_path)
    df = pd.read_excel(file_path, engine="openpyxl")
    logger.info("bacTRAP loaded: %d rows x %d columns", len(df), len(df.columns))
    logger.info("  columns: %s", list(df.columns))
    logger.info("  dtypes:\n%s", df.dtypes.to_string())
    logger.info("  index dtype: %s, sample: %s", df.index.dtype, list(df.index[:5]))
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name in ("string", "category"):
            sample = df[col].dropna().head(5).tolist()
            logger.info("  column '%s' sample: %s", col, sample)
    return df


def get_annotation_columns(adata: ad.AnnData) -> List[str]:
    """Return candidate annotation columns from .obs (categorical or string dtypes)."""
    candidates = []
    for col in adata.obs.columns:
        dtype = adata.obs[col].dtype
        if dtype.name == "category" or dtype == object:
            nunique = adata.obs[col].nunique()
            # Likely an annotation if it has a reasonable number of unique values
            if 2 <= nunique <= 5000:
                candidates.append(col)
    return sorted(candidates)


def _looks_like_ensembl(gene_names: np.ndarray, sample_size: int = 20) -> bool:
    """Check whether gene names look like Ensembl IDs by sampling."""
    if len(gene_names) == 0:
        return False
    sample = gene_names[:sample_size]
    n_ens = sum(1 for g in sample if str(g).startswith(("ENSMUSG", "ENSG")))
    return n_ens > len(sample) * 0.5


def _find_symbol_column(var_df: pd.DataFrame) -> Optional[str]:
    """Find a gene-symbol column in a var DataFrame."""
    checked = []
    for col in ["gene_name", "gene_symbol", "symbol", "Gene", "gene_short_name",
                "external_gene_name", "mgi_symbol", "feature_name"]:
        if col in var_df.columns:
            # Verify this column actually has non-Ensembl values
            sample = var_df[col].dropna().head(20)
            if len(sample) > 0 and not _looks_like_ensembl(sample.values):
                logger.info("_find_symbol_column: found '%s', sample: %s", col, sample.tolist()[:5])
                return col
            checked.append((col, sample.tolist()[:5]))
    logger.info("_find_symbol_column: no symbol column found. Checked: %s. Available: %s",
                checked, list(var_df.columns))
    return None


def get_gene_names_from_adata(adata: ad.AnnData, use_raw: bool = False) -> np.ndarray:
    """Extract gene names from the AnnData object, trying multiple locations.

    When *use_raw* is True and ``adata.raw`` exists, gene names are taken from
    the raw layer (which typically contains the full pre-HVG-filtering gene set).

    If var_names look like Ensembl IDs, a gene-symbol column is searched in
    both the target var DataFrame AND ``adata.var`` as fallback (since
    ``adata.raw.var`` may lack annotation columns).
    """
    if use_raw and adata.raw is not None:
        gene_names = np.array(adata.raw.var_names)
        var_df = adata.raw.var
        logger.info("get_gene_names_from_adata: using raw layer (%d genes)", len(gene_names))
    else:
        gene_names = adata.var_names.values.copy()
        var_df = adata.var
        logger.info("get_gene_names_from_adata: using var layer (%d genes)", len(gene_names))

    logger.info("  var_names sample (first 10): %s", [str(g) for g in gene_names[:10]])
    is_ensembl = _looks_like_ensembl(gene_names)
    logger.info("  looks_like_ensembl: %s", is_ensembl)

    # Check if var_names look like Ensembl IDs; if so, look for a symbol column
    if is_ensembl:
        logger.info("  searching for symbol column in target var_df (columns: %s)", list(var_df.columns))
        sym_col = _find_symbol_column(var_df)
        if sym_col is not None:
            gene_names = var_df[sym_col].values.copy()
            logger.info("  resolved via column '%s', sample: %s", sym_col, [str(g) for g in gene_names[:10]])
        elif use_raw and adata.raw is not None:
            # Fallback: adata.var may have the symbol column even if raw.var doesn't.
            logger.info("  no symbol column in raw.var, trying adata.var fallback (columns: %s)",
                        list(adata.var.columns))
            sym_col_main = _find_symbol_column(adata.var)
            if sym_col_main is not None:
                ens_to_sym = {}
                for ens_id, sym in zip(adata.var.index, adata.var[sym_col_main]):
                    sym_str = str(sym).strip()
                    if sym_str and sym_str.lower() != "nan":
                        ens_to_sym[str(ens_id)] = sym_str
                logger.info("  built Ensembl→symbol map from adata.var['%s']: %d entries", sym_col_main, len(ens_to_sym))
                if ens_to_sym:
                    gene_names = np.array([
                        ens_to_sym.get(str(g), str(g)) for g in gene_names
                    ])
                    n_resolved = sum(1 for g in gene_names if not str(g).startswith(("ENSMUSG", "ENSG")))
                    logger.info("  after map: %d/%d resolved to symbols", n_resolved, len(gene_names))
                    logger.info("  sample after resolution: %s", [str(g) for g in gene_names[:10]])
            else:
                logger.warning("  NO symbol column found anywhere — gene names remain as Ensembl IDs!")
    else:
        logger.info("  var_names are not Ensembl IDs, using as-is")

    return gene_names


def _detect_gene_column(bactrap_df: pd.DataFrame) -> str:
    """Auto-detect the column in bacTRAP data that contains gene identifiers.

    Checks common column names and the DataFrame index.  Returns the best
    candidate column name (or ``"_index"`` if the index should be used).
    """
    # Priority-ordered list of likely gene-name column names
    candidates = [
        "gene_name", "gene_symbol", "symbol", "Gene", "GeneSymbol",
        "gene_id", "GeneID", "external_gene_name", "mgi_symbol",
        "SYMBOL", "gene_short_name", "feature_name", "Name",
    ]
    for col in candidates:
        if col in bactrap_df.columns:
            return col

    # Check if first column looks like gene names (common in DESeq2 output)
    first_col = bactrap_df.columns[0]
    sample_vals = bactrap_df[first_col].dropna().head(20).astype(str)
    if len(sample_vals) > 0:
        # If most values are non-numeric strings, likely gene names
        n_alpha = sum(1 for v in sample_vals if v and not v.replace(".", "").replace("-", "").replace("_", "").isdigit())
        if n_alpha > len(sample_vals) * 0.5:
            return first_col

    # Check the index
    if bactrap_df.index.dtype == object or bactrap_df.index.dtype.name == "string":
        sample_idx = [str(x) for x in bactrap_df.index[:20]]
        n_alpha = sum(1 for v in sample_idx if v and not v.replace(".", "").replace("-", "").replace("_", "").isdigit())
        if n_alpha > len(sample_idx) * 0.5:
            return "_index"

    # Last resort: return first column
    return first_col


def lookup_gene_in_atlas(
    lookup: Dict[str, Tuple[str, int]],
    name,
) -> Optional[Tuple[str, int]]:
    """Resolve a gene symbol against a lookup built by `_build_adata_gene_lookup`.

    Tries the exact-case spelling first so case-distinct paralogs (e.g. mixed
    mouse / human symbols) are not silently merged, then falls back to a
    case-insensitive match.
    """
    name_str = str(name).strip()
    if not name_str:
        return None
    hit = lookup.get(name_str)
    if hit is not None:
        return hit
    return lookup.get(name_str.lower())


def _atlas_identity(adata: ad.AnnData) -> Tuple[int, int, int]:
    """Cheap, hashable atlas fingerprint used as a cache key supplement.

    AnnData isn't itself hashable for Streamlit's caching, but the shape +
    first/last var-name hashes are stable across reruns and distinguish two
    different atlases loaded in the same session (catching the case where
    `_build_adata_gene_lookup` would otherwise return the first atlas's
    lookup for the second).
    """
    n_obs = int(adata.n_obs)
    n_vars = int(adata.n_vars)
    names = adata.var_names
    fingerprint = hash((
        str(names[0]) if len(names) else "",
        str(names[-1]) if len(names) else "",
    )) & 0xFFFFFFFF
    return (n_obs, n_vars, fingerprint)


def _build_adata_gene_lookup(
    adata: ad.AnnData, use_raw: bool = True,
) -> Tuple[Dict[str, Tuple[str, int]], np.ndarray, bool]:
    """Public lookup builder. Computes an atlas-identity fingerprint and
    delegates to the cached implementation so two different atlases loaded in
    the same Streamlit session can't silently share each other's lookup."""
    return _build_adata_gene_lookup_cached(
        adata, use_raw=use_raw, atlas_id=_atlas_identity(adata),
    )


@st.cache_resource(show_spinner=False)
def _build_adata_gene_lookup_cached(
    _adata: ad.AnnData,
    use_raw: bool = True,
    atlas_id: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Dict[str, Tuple[str, int]], np.ndarray, bool]:
    """Build a comprehensive gene lookup from an AnnData object.

    The leading underscore on ``_adata`` tells Streamlit not to attempt to
    hash the AnnData argument (AnnData objects are not hashable by
    Streamlit's caching machinery). ``atlas_id`` (produced by
    `_atlas_identity`) is a hashable fingerprint that DOES participate in
    the cache key so a different atlas loaded in the same session is not
    silently served the first atlas's lookup.

    Returns:
        lookup: gene name -> (display_name, column_index). Keys include both
                exact-case symbols and their lowercase form; exact case wins
                when both are present, so case-distinct paralogs are
                preserved while lookups remain case-insensitive by default.
        gene_names: array of resolved gene names
        is_raw: whether indices point into adata.raw.var
    """
    del atlas_id  # only used to influence the Streamlit cache key
    adata = _adata
    has_raw = adata.raw is not None and use_raw
    logger.info("_build_adata_gene_lookup: has_raw=%s", has_raw)
    gene_names = get_gene_names_from_adata(adata, use_raw=has_raw)

    lookup: Dict[str, Tuple[str, int]] = {}
    n_case_collisions = 0
    for idx, name in enumerate(gene_names):
        name_str = str(name).strip()
        key = name_str.lower()
        if not key or key == "nan":
            continue
        # Exact-case key, only set if not already taken by an earlier gene with
        # the same exact spelling — preserves case-distinct paralogs.
        if name_str not in lookup:
            lookup[name_str] = (name_str, idx)
        # Lowercase fallback. Last-write-wins on case-folded collisions, but
        # only used when exact case lookup misses.
        if key in lookup and lookup[key][1] != idx:
            n_case_collisions += 1
        lookup[key] = (name_str, idx)
    if n_case_collisions:
        logger.warning(
            "  %d case-folded gene-name collisions in lookup (exact-case "
            "match still preferred at query time)", n_case_collisions,
        )

    logger.info("  primary lookup size: %d (from resolved gene names)", len(lookup))

    # Also add Ensembl IDs as lookup keys (pointing to the same indices)
    # so that bacTRAP data with Ensembl IDs can still match.
    if has_raw and adata.raw is not None:
        raw_var_names = np.array(adata.raw.var_names)
    else:
        raw_var_names = adata.var_names.values

    raw_is_ensembl = _looks_like_ensembl(raw_var_names)
    names_are_ensembl = _looks_like_ensembl(gene_names)
    logger.info("  raw_var_names looks_like_ensembl: %s", raw_is_ensembl)
    logger.info("  resolved gene_names looks_like_ensembl: %s", names_are_ensembl)

    if raw_is_ensembl:
        n_added = 0
        for idx, ens_id in enumerate(raw_var_names):
            ens_str = str(ens_id).strip().lower()
            if ens_str and ens_str != "nan" and ens_str not in lookup:
                display = str(gene_names[idx]).strip()
                if display and display.lower() != "nan":
                    lookup[ens_str] = (display, idx)
                    n_added += 1
        logger.info("  added %d Ensembl ID keys to lookup", n_added)
    elif not names_are_ensembl:
        var_df = adata.raw.var if has_raw and adata.raw is not None else adata.var
        for col in ["gene_ids", "gene_id", "ensembl_id", "Ensembl", "ensembl"]:
            if col in var_df.columns:
                n_added = 0
                for idx, ens_id in enumerate(var_df[col]):
                    ens_str = str(ens_id).strip().lower()
                    if ens_str and ens_str != "nan" and ens_str not in lookup:
                        display = str(gene_names[idx]).strip()
                        if display and display.lower() != "nan":
                            lookup[ens_str] = (display, idx)
                            n_added += 1
                logger.info("  added %d Ensembl keys from var_df['%s']", n_added, col)
                break

    logger.info("  final lookup size: %d", len(lookup))
    # Log a few sample entries from the lookup
    sample_keys = list(lookup.keys())[:10]
    logger.info("  lookup sample: %s", {k: lookup[k][0] for k in sample_keys})

    return lookup, gene_names, has_raw


def match_genes(
    bactrap_df: pd.DataFrame,
    adata: ad.AnnData,
    gene_col: Optional[str] = None,
    _prebuilt_lookup: Optional[Tuple[Dict[str, Tuple[str, int]], np.ndarray, bool]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int], bool]:
    """
    Match bacTRAP gene identifiers to HypoMap genes.

    When *gene_col* is None the column is auto-detected by trying common
    column names and selecting the one that yields the most matches.

    When ``adata.raw`` exists the lookup is built from the **raw** layer so
    that *all* genes are available for matching (not just the highly-variable
    subset stored in ``adata.var``).

    Parameters:
        _prebuilt_lookup: Optional pre-built lookup tuple from
            ``_build_adata_gene_lookup`` to avoid redundant recomputation.

    Returns:
        bactrap_matched: subset of bactrap_df with matched genes
        matched_gene_names: list of matched gene symbols (as they appear in HypoMap)
        gene_to_adata_idx: mapping from gene name to column index.
            Indices are in **raw** space when ``adata.raw`` exists,
            otherwise in ``adata.var`` space.
        matched_in_raw: True when indices refer to adata.raw.var space.
    """
    logger.info("match_genes called with gene_col=%s", gene_col)

    # Use pre-built lookup if provided, otherwise build it (cached)
    if _prebuilt_lookup is not None:
        adata_gene_lookup, adata_gene_names, has_raw = _prebuilt_lookup
    else:
        adata_gene_lookup, adata_gene_names, has_raw = _build_adata_gene_lookup(adata)

    # Determine which bacTRAP column to use for gene matching
    if gene_col is None:
        gene_col = _auto_select_gene_col(bactrap_df, adata_gene_lookup)

    logger.info("match_genes: using gene_col='%s'", gene_col)

    if gene_col == "_index":
        bt_gene_values = bactrap_df.index.astype(str)
    elif gene_col in bactrap_df.columns:
        bt_gene_values = bactrap_df[gene_col].astype(str)
    else:
        raise ValueError(
            f"Column '{gene_col}' not found in bacTRAP data. "
            f"Available columns: {list(bactrap_df.columns)}"
        )

    bt_sample = [str(v) for v in bt_gene_values[:10]]
    logger.info("  bacTRAP gene values sample (first 10): %s", bt_sample)

    # Match bacTRAP genes
    matched_rows = []
    matched_gene_names = []
    gene_to_adata_idx = {}
    unmatched_sample = []

    for i, bt_gene_raw in enumerate(bt_gene_values):
        bt_gene = bt_gene_raw.strip()
        hit = lookup_gene_in_atlas(adata_gene_lookup, bt_gene)
        if hit is not None:
            original_name, adata_idx = hit
            matched_rows.append(bactrap_df.iloc[i])
            matched_gene_names.append(original_name)
            gene_to_adata_idx[original_name] = adata_idx
        elif len(unmatched_sample) < 20:
            unmatched_sample.append(bt_gene)

    logger.info("match_genes result: %d/%d matched (%.1f%%)",
                len(matched_rows), len(bt_gene_values),
                100 * len(matched_rows) / max(len(bt_gene_values), 1))
    if matched_gene_names:
        logger.info("  matched sample: %s", matched_gene_names[:10])
    logger.info("  unmatched sample: %s", unmatched_sample)

    if len(matched_rows) == 0:
        # Loud, actionable failure — otherwise downstream just sees an
        # empty DataFrame and surfaces "no genes pass enrichment thresholds"
        # much later, without a breadcrumb back to the matching step.
        _bt_sample = [str(v) for v in bt_gene_values[:5]]
        _atlas_sample = list(adata_gene_lookup.keys())[:5]
        logger.error(
            "match_genes: 0 / %d bacTRAP genes matched the atlas lookup. "
            "gene_col='%s', atlas lookup size=%d. "
            "Sample bacTRAP values: %s. Sample atlas keys: %s. "
            "The selected gene column probably does not contain symbols / "
            "Ensembl IDs recognised by HypoMap — check the 'Gene Matching "
            "Diagnostics' panel in the Data Overview tab and try another "
            "column from the bacTRAP file.",
            len(bt_gene_values), gene_col, len(adata_gene_lookup),
            _bt_sample, _atlas_sample,
        )
        empty_df = bactrap_df.iloc[:0].copy()
        empty_df["_hypomap_gene_name"] = pd.Series(dtype=str)
        return empty_df, [], {}, has_raw

    bactrap_matched = pd.DataFrame(matched_rows)
    bactrap_matched = bactrap_matched.reset_index(drop=True)
    bactrap_matched["_hypomap_gene_name"] = matched_gene_names

    # Deduplicate: when multiple bacTRAP rows map to the same HypoMap gene
    # (e.g. multiple Ensembl IDs → same symbol), keep the row with the
    # strongest signal rather than the arbitrary first occurrence.  Sort
    # order: smallest padj, then largest |log2FC|, with NaN padj last.
    n_before = len(bactrap_matched)
    _sort_cols, _sort_asc = [], []
    if "padj" in bactrap_matched.columns:
        _sort_cols.append("padj")
        _sort_asc.append(True)          # smaller padj first; NaN goes last
    if "log2FoldChange" in bactrap_matched.columns:
        bactrap_matched["_abs_l2fc"] = bactrap_matched["log2FoldChange"].abs()
        _sort_cols.append("_abs_l2fc")
        _sort_asc.append(False)         # larger |log2FC| first
    if _sort_cols:
        bactrap_matched = bactrap_matched.sort_values(
            _sort_cols, ascending=_sort_asc, na_position="last", kind="mergesort",
        )
    bactrap_matched = bactrap_matched.drop_duplicates(subset="_hypomap_gene_name", keep="first")
    if "_abs_l2fc" in bactrap_matched.columns:
        bactrap_matched = bactrap_matched.drop(columns="_abs_l2fc")
    bactrap_matched = bactrap_matched.reset_index(drop=True)
    matched_gene_names = bactrap_matched["_hypomap_gene_name"].tolist()
    gene_to_adata_idx = {g: gene_to_adata_idx[g] for g in matched_gene_names}
    if n_before != len(bactrap_matched):
        logger.info("  deduplicated %d → %d genes (removed %d duplicate HypoMap mappings; "
                     "kept row with smallest padj / largest |log2FC|)",
                     n_before, len(bactrap_matched), n_before - len(bactrap_matched))

    return bactrap_matched, matched_gene_names, gene_to_adata_idx, has_raw


def _auto_select_gene_col(
    bactrap_df: pd.DataFrame,
    adata_gene_lookup: Dict[str, Tuple[str, int]],
) -> str:
    """Try multiple candidate columns and pick the one with the most matches."""
    candidates = []

    # Try known column names
    for col in ["gene_name", "gene_symbol", "symbol", "Gene", "GeneSymbol",
                "gene_id", "GeneID", "external_gene_name", "mgi_symbol",
                "SYMBOL", "gene_short_name", "feature_name", "Name"]:
        if col in bactrap_df.columns:
            candidates.append(col)

    # Also try the first column and the index
    first_col = bactrap_df.columns[0]
    if first_col not in candidates:
        candidates.append(first_col)
    candidates.append("_index")

    best_col = candidates[0] if candidates else "_index"
    best_count = 0

    logger.info("_auto_select_gene_col: testing %d candidates: %s", len(candidates), candidates)

    for col in candidates:
        if col == "_index":
            values = bactrap_df.index.astype(str)
        else:
            values = bactrap_df[col].astype(str)

        n_matched = sum(
            1 for v in values
            if str(v).strip().lower() in adata_gene_lookup
        )
        sample = [str(v) for v in values[:5]]
        logger.info("  col='%s': %d matches, sample=%s", col, n_matched, sample)
        if n_matched > best_count:
            best_count = n_matched
            best_col = col

    logger.info("_auto_select_gene_col: selected '%s' with %d matches", best_col, best_count)
    return best_col


def _resolve_gene_names(
    adata: ad.AnnData,
    gene_indices: np.ndarray,
    from_raw: bool = False,
) -> List[str]:
    """Resolve gene indices to display names.

    When *from_raw* is True, indices are looked up in ``adata.raw.var``.
    """
    adata_gene_names = get_gene_names_from_adata(adata, use_raw=from_raw)
    return [str(adata_gene_names[i]) for i in gene_indices]


def _map_var_indices_to_raw(
    adata: ad.AnnData,
    gene_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map gene indices from adata.var space to adata.raw.var space.

    When adata.raw has more genes than adata (common after gene filtering),
    indices into adata.var do NOT correspond to the same columns in
    adata.raw.X. This function translates them via gene name lookup.

    Returns:
        raw_indices: array of indices into adata.raw.var
        survived_mask: boolean mask over gene_indices indicating which
            original indices were successfully mapped (for correct label alignment)
    """
    if adata.raw is None:
        return gene_indices, np.ones(len(gene_indices), dtype=bool)

    # Get gene names for the requested indices in adata.var. Indices that
    # fall outside adata.n_vars are out-of-bounds (an upstream miscount)
    # and are silently marked as not-survived so the caller drops them,
    # rather than tripping an IndexError deep inside this helper.
    var_gene_names = get_gene_names_from_adata(adata)
    n_var = len(var_gene_names)
    oob_mask = (np.asarray(gene_indices) < 0) | (np.asarray(gene_indices) >= n_var)
    query_names = [
        str(var_gene_names[int(i)]).lower() if not oob else None
        for i, oob in zip(gene_indices, oob_mask)
    ]

    # Build lookup for raw var names — prefer gene symbols over Ensembl IDs
    # so that the lookup matches how match_genes() found these genes.
    raw_var_names = adata.raw.var_names
    raw_lookup = {}

    # First pass: raw var_names (may be Ensembl IDs or symbols)
    for i, name in enumerate(raw_var_names):
        raw_lookup[str(name).lower()] = i

    # Second pass: if raw var_names are Ensembl IDs, add gene symbol entries.
    # Gene symbols take precedence (overwrite) since queries use symbols.
    if _looks_like_ensembl(np.array(raw_var_names)):
        for col in ["gene_name", "gene_symbol", "symbol", "Gene", "gene_short_name"]:
            if col in adata.raw.var.columns:
                for i, name in enumerate(adata.raw.var[col]):
                    name_lower = str(name).strip().lower()
                    if name_lower and name_lower != "nan":
                        raw_lookup[name_lower] = i
                break

    raw_indices = []
    survived = []
    for j, name in enumerate(query_names):
        if name is not None and name in raw_lookup:
            raw_indices.append(raw_lookup[name])
            survived.append(True)
        else:
            survived.append(False)

    return (
        np.array(raw_indices, dtype=int),
        np.array(survived, dtype=bool),
    )


def _extract_gene_submatrix(
    adata: ad.AnnData,
    gene_indices: np.ndarray,
    use_raw: bool = True,
    indices_in_raw: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a dense (n_cells, n_genes) submatrix for the given gene indices.

    When *indices_in_raw* is False (legacy behaviour), gene_indices are in
    adata.var space and are remapped to adata.raw.var when *use_raw* is True.

    When *indices_in_raw* is True, gene_indices are already in adata.raw.var
    space and no remapping is performed.

    Returns:
        X_sub: dense array of shape (n_cells, n_survived_genes)
        survived_mask: boolean mask over gene_indices indicating which
            genes were successfully extracted (True = present in output)
    """
    survived_mask = np.ones(len(gene_indices), dtype=bool)

    if indices_in_raw and adata.raw is not None:
        # Indices already point into adata.raw.var — use directly
        X = adata.raw.X
    elif use_raw and adata.raw is not None:
        X = adata.raw.X
        raw_indices, survived_mask = _map_var_indices_to_raw(adata, gene_indices)
        gene_indices = raw_indices
    else:
        X = adata.X

    n_cells = X.shape[0]
    n_genes = len(gene_indices)

    if n_genes == 0:
        return np.empty((n_cells, 0), dtype=np.float32), survived_mask

    # Validate indices are within bounds
    max_idx = X.shape[1]
    valid = gene_indices < max_idx
    if not np.all(valid):
        # survived_positions maps each entry in gene_indices (post-mapping)
        # back to its position in the original gene_indices array
        survived_positions = np.where(survived_mask)[0]
        for i, is_valid in enumerate(valid):
            if not is_valid:
                survived_mask[survived_positions[i]] = False
        gene_indices = gene_indices[valid]
        n_genes = len(gene_indices)

    if n_genes == 0:
        return np.empty((n_cells, 0), dtype=np.float32), survived_mask

    # Sparse column access on CSR is expensive because fancy indexing
    # converts the whole matrix to CSC internally. The large-set chunked
    # path below used to pay that conversion cost once per chunk (≈150
    # full passes over nnz on HypoMap). Convert to CSC once up-front so
    # subsequent column slicing is O(nnz of the selected columns).
    if sparse.issparse(X) and X.getformat() != "csc":
        # Cache the CSC view on adata so subsequent calls don't repay the
        # tocsc() cost — gene-submatrix extraction is called repeatedly
        # (per cluster mean, per detection rate, etc.) and each tocsc()
        # is O(nnz) on HypoMap's ~1 billion stored values.
        cache_key = "_csc_raw" if (use_raw and adata.raw is not None) else "_csc_X"
        cached = adata.uns.get(cache_key)
        if (
            cached is not None
            and sparse.issparse(cached)
            and cached.shape == X.shape
            and cached.dtype == X.dtype
        ):
            X = cached
        else:
            X = X.tocsc()
            try:
                adata.uns[cache_key] = X
            except (TypeError, ValueError):
                # Some AnnData versions reject sparse matrices in uns —
                # fall through, we still saved this call's conversion.
                pass

    # For small gene sets, a single fancy-indexed slice is fine.
    if n_genes <= 500:
        X_sub = X[:, gene_indices]
        if sparse.issparse(X_sub):
            result = np.asarray(X_sub.toarray())
        else:
            result = np.asarray(X_sub)
        return result, survived_mask

    # For larger sets, process in chunks to limit peak memory. X is now
    # CSC (if it was sparse to begin with) so each chunk's column slice
    # is cheap; peak memory is bounded by (n_cells x chunk_size) floats.
    chunk_size = 200
    out = np.empty((n_cells, n_genes), dtype=np.float32)
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        chunk_idx = gene_indices[start:end]
        X_chunk = X[:, chunk_idx]
        if sparse.issparse(X_chunk):
            out[:, start:end] = np.asarray(X_chunk.toarray())
        else:
            out[:, start:end] = np.asarray(X_chunk)
    return out, survived_mask


def _get_total_counts_per_cell(
    adata: ad.AnnData, use_raw: bool = True
) -> np.ndarray:
    """Return total counts per cell (n_cells,).

    Prefers a pre-computed ``nCount_RNA`` column in ``adata.obs`` when
    present; otherwise computes the row sums of the full raw (or X) matrix
    once.  Caches the result on ``adata.uns`` to avoid repeated work.
    """
    cache_key = "_total_counts_raw" if use_raw else "_total_counts_X"
    if cache_key in adata.uns:
        cached = np.asarray(adata.uns[cache_key])
        if cached.shape[0] == adata.n_obs:
            return cached

    if "nCount_RNA" in adata.obs.columns:
        totals = adata.obs["nCount_RNA"].values.astype(np.float32)
        adata.uns[cache_key] = totals
        logger.info("_get_total_counts_per_cell: using obs['nCount_RNA']")
        return totals

    if use_raw and adata.raw is not None:
        X_full = adata.raw.X
    else:
        X_full = adata.X
    if sparse.issparse(X_full):
        totals = np.asarray(X_full.sum(axis=1)).ravel().astype(np.float32)
    else:
        totals = np.asarray(X_full.sum(axis=1)).astype(np.float32)
    adata.uns[cache_key] = totals
    logger.info("_get_total_counts_per_cell: computed row sums for %d cells", len(totals))
    return totals


def _looks_like_raw_counts(X_genes: np.ndarray, sample_size: int = 1000) -> bool:
    """Heuristic: raw UMI counts contain integers with max > 50."""
    if X_genes.size == 0:
        return False
    n_cells = X_genes.shape[0]
    if n_cells > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, sample_size, replace=False)
        sample = X_genes[idx, :]
    else:
        sample = X_genes
    return float(np.max(sample)) > 50.0


def compute_cluster_mean_expression(
    adata: ad.AnnData,
    gene_indices: List[int],
    annotation_col: str,
    min_cells: int = 10,
    use_raw: bool = True,
    indices_in_raw: bool = False,
    normalize: Optional[bool] = None,
    target_sum: float = 1e4,
) -> pd.DataFrame:
    """
    Compute mean expression per cluster for a set of genes.

    When *indices_in_raw* is False (legacy), gene_indices refer to
    positions in ``adata.var`` and are remapped to ``adata.raw.var``
    internally.  When True, they already point into ``adata.raw.var``.

    If *normalize* is True (or None with auto-detected raw counts), each
    cell is size-normalized to ``target_sum`` and log1p-transformed before
    the per-cluster mean is computed.  This avoids the library-depth bias
    that otherwise lets high-count clusters (e.g. ependymal, endothelial,
    stromal) dominate downstream correlation and NNLS deconvolution.

    Returns a DataFrame with shape (n_survived_genes, n_clusters).
    """
    if annotation_col not in adata.obs.columns:
        raise ValueError(
            f"Annotation column '{annotation_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    gene_indices_arr = np.array(gene_indices)
    labels = adata.obs[annotation_col].values

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels = unique_labels[counts >= min_cells]

    X_genes, survived_mask = _extract_gene_submatrix(
        adata, gene_indices_arr, use_raw=use_raw, indices_in_raw=indices_in_raw,
    )

    # ---- Per-cell normalization (normalize_total + log1p) ------------------
    # Done on the extracted submatrix (cells x selected_genes) using
    # pre-computed per-cell totals from the FULL matrix so the size factor
    # is correct.  Without this the cluster means reflect raw UMI counts,
    # giving disproportionate weight to high-depth non-neuronal clusters
    # during correlation/NNLS.
    # NOTE: the auto-detect path (normalize=None -> _looks_like_raw_counts)
    # was dropped because the heuristic ran on the submatrix of selected
    # signature genes — already-log-normalised data with a few high-mean
    # genes could falsely "look raw" and get double-log-normalised.  All
    # repo callers pass an explicit True/False; default to True.
    if normalize is None:
        normalize = True
    if normalize and X_genes.size > 0:
        # Use raw totals when the extracted matrix itself came from raw
        totals_use_raw = use_raw or indices_in_raw
        totals = _get_total_counts_per_cell(adata, use_raw=totals_use_raw)
        # Guard against zero totals
        safe_totals = np.where(totals > 0, totals, 1.0).astype(np.float32)
        scale = (target_sum / safe_totals).astype(np.float32)
        # In-place size-factor scaling + log1p
        X_genes = X_genes.astype(np.float32, copy=False) * scale[:, None]
        np.log1p(X_genes, out=X_genes)
        logger.info(
            "compute_cluster_mean_expression: applied normalize_total(target=%.0f) + log1p "
            "(post-norm max=%.3f)",
            target_sum, float(X_genes.max()) if X_genes.size else 0.0,
        )

    result = {}
    for label in valid_labels:
        mask = labels == label
        result[str(label)] = X_genes[mask, :].mean(axis=0)

    # Use survived_mask to pick the correct gene names
    gene_names = _resolve_gene_names(
        adata, gene_indices_arr[survived_mask], from_raw=indices_in_raw,
    )
    df = pd.DataFrame(result, index=gene_names)
    # Deduplicate: multiple indices can resolve to the same gene symbol
    if df.index.duplicated().any():
        logger.info("compute_cluster_mean_expression: dropping %d duplicate gene names",
                     df.index.duplicated().sum())
        df = df.groupby(df.index).mean()
    return df


def compute_fraction_expressing(
    adata: ad.AnnData,
    gene_indices: List[int],
    annotation_col: str,
    min_cells: int = 10,
    use_raw: bool = True,
    threshold: float = 0.0,
    indices_in_raw: bool = False,
) -> pd.DataFrame:
    """
    Compute fraction of cells expressing each gene (>threshold) per cluster.

    Returns a DataFrame with shape (n_survived_genes, n_clusters).
    """
    if annotation_col not in adata.obs.columns:
        raise ValueError(
            f"Annotation column '{annotation_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    gene_indices_arr = np.array(gene_indices)
    labels = adata.obs[annotation_col].values

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels = unique_labels[counts >= min_cells]

    X_genes, survived_mask = _extract_gene_submatrix(
        adata, gene_indices_arr, use_raw=use_raw, indices_in_raw=indices_in_raw,
    )

    result = {}
    for label in valid_labels:
        mask = labels == label
        result[str(label)] = (X_genes[mask, :] > threshold).mean(axis=0)

    gene_names = _resolve_gene_names(
        adata, gene_indices_arr[survived_mask], from_raw=indices_in_raw,
    )
    df = pd.DataFrame(result, index=gene_names)
    if df.index.duplicated().any():
        df = df.groupby(df.index).mean()
    return df


def compute_cell_detection_rate(
    adata: ad.AnnData,
    gene_indices: List[int],
    use_raw: bool = True,
    indices_in_raw: bool = False,
) -> pd.Series:
    """Atlas-wide fraction of cells with raw count > 0, per gene.

    Returns a Series indexed by resolved gene symbol (duplicates averaged).
    Used by the signature-refinement detectability filter — a gene below this
    threshold essentially never appears in any cell's top-τ AUCell window.
    """
    gene_indices_arr = np.array(gene_indices)
    X_genes, survived_mask = _extract_gene_submatrix(
        adata, gene_indices_arr, use_raw=use_raw, indices_in_raw=indices_in_raw,
    )
    if X_genes.size == 0:
        return pd.Series(dtype=float)
    frac = (X_genes > 0).mean(axis=0)
    gene_names = _resolve_gene_names(
        adata, gene_indices_arr[survived_mask], from_raw=indices_in_raw,
    )
    s = pd.Series(np.asarray(frac, dtype=float), index=gene_names)
    if s.index.duplicated().any():
        s = s.groupby(s.index).mean()
    logger.info("compute_cell_detection_rate: %d genes, median rate=%.4f",
                len(s), float(s.median()) if len(s) else float("nan"))
    return s


def _atlas_stat_key(base: str, mask_signature: str = "") -> str:
    """Deterministic ``adata.uns`` key for a cached atlas statistic.

    ``mask_signature`` is appended (with a leading underscore) so toggling an
    atlas restriction (e.g. POA-only) never reuses stale full-atlas data.
    """
    ms = (mask_signature or "").strip("_")
    return f"{base}_{ms}" if ms else base


def get_atlas_cluster_mean_expr(
    adata: ad.AnnData,
    gene_indices: List[int],
    annotation_col: str,
    *,
    mask_signature: str = "",
    min_cells: int = 10,
    indices_in_raw: bool = False,
    normalize: bool = True,
    target_sum: float = 1e4,
) -> pd.DataFrame:
    """Per-cluster mean (log-norm) expression for ``gene_indices``, cached on
    ``adata.uns['cluster_mean_expr_<annotation_col>[_<mask_signature>]']``.

    Thin caching wrapper around :func:`compute_cluster_mean_expression` so the
    matched-gene cluster-mean matrix (used by correlation / NNLS / the
    signature-refinement and Cre-driver filters) is computed once per atlas
    load and per (cluster column, atlas-mask) combination, rather than on every
    parameter tweak.  Cache validity is keyed by the requested gene-set size,
    ``min_cells`` and ``normalize`` in addition to the column/mask in the key.
    """
    key = _atlas_stat_key(f"cluster_mean_expr_{annotation_col}", mask_signature)
    n_genes = len(gene_indices)
    # Sort indices before hashing so two callers with the same gene set in
    # different orders hit the same cache entry (the function's output is
    # order-independent — it's reindexed by gene before return).
    gi_hash = hash(tuple(sorted(int(i) for i in gene_indices)))
    cached = adata.uns.get(key)
    if isinstance(cached, dict):
        df = cached.get("df")
        if (isinstance(df, pd.DataFrame)
                and cached.get("n_genes") == n_genes
                and cached.get("gi_hash") == gi_hash
                and cached.get("min_cells") == int(min_cells)
                and cached.get("normalize") == bool(normalize)):
            logger.info("get_atlas_cluster_mean_expr: cache hit (%s, %d genes)", key, n_genes)
            return df
    df = compute_cluster_mean_expression(
        adata, gene_indices, annotation_col, min_cells=min_cells,
        indices_in_raw=indices_in_raw, normalize=normalize, target_sum=target_sum,
    )
    adata.uns[key] = {
        "df": df, "n_genes": n_genes, "gi_hash": gi_hash,
        "min_cells": int(min_cells), "normalize": bool(normalize),
    }
    logger.info("get_atlas_cluster_mean_expr: computed & cached (%s, %d genes × %d clusters)",
                key, df.shape[0], df.shape[1])
    return df


def get_atlas_gene_detection_rate(
    adata: ad.AnnData,
    *,
    mask_signature: str = "",
    use_raw: bool = True,
) -> pd.Series:
    """Atlas-wide per-gene detection rate (fraction of cells with count > 0),
    cached on ``adata.uns['gene_detection_rate[_<mask_signature>]']``.

    Computed over the raw layer (all genes) from the stored-nonzero counts —
    O(nnz), no dense materialisation.  Indexed by resolved gene symbol; if a
    symbol maps to several raw genes the largest detection rate is kept (the
    detectability filter only cares whether the gene is detectable at all).
    """
    key = _atlas_stat_key("gene_detection_rate", mask_signature)
    cached = adata.uns.get(key)
    if isinstance(cached, pd.Series) and len(cached):
        logger.info("get_atlas_gene_detection_rate: cache hit (%s, %d genes)", key, len(cached))
        return cached
    use_raw_layer = bool(use_raw and adata.raw is not None)
    X = adata.raw.X if use_raw_layer else adata.X
    # AnnData slicing semantics for `.raw` differ across versions: some keep
    # the parent's full-atlas raw rows, others row-slice. If raw doesn't match
    # the (possibly restricted) view's cell count, fall back to .X so the
    # detection rate is computed on the cells we're actually scoring.
    if use_raw_layer and X.shape[0] != adata.n_obs:
        logger.warning(
            "get_atlas_gene_detection_rate: adata.raw.X rows (%d) != adata.n_obs "
            "(%d) — falling back to adata.X so detection rate matches the "
            "active cell subset", X.shape[0], adata.n_obs,
        )
        X = adata.X
        use_raw_layer = False
    n_cells = X.shape[0]
    if sparse.issparse(X):
        # Stored-nonzero count per column — O(nnz) time, O(n_genes) space, no
        # dense / boolean-sparse materialisation.  Count matrices loaded from
        # h5ad effectively never store explicit zeros, so this equals the
        # number of cells with count > 0.
        nnz_per_gene = X.getnnz(axis=0)
    else:
        nnz_per_gene = (np.asarray(X) > 0).sum(axis=0)
    rates = np.asarray(nnz_per_gene, dtype=float) / max(n_cells, 1)
    gene_names = get_gene_names_from_adata(adata, use_raw=use_raw_layer)
    s = pd.Series(rates, index=[str(g) for g in gene_names])
    if s.index.duplicated().any():
        # Use .mean() to match compute_cluster_mean_expression /
        # compute_fraction_expressing — otherwise the refinement filter
        # compares values aggregated by different rules across helpers.
        s = s.groupby(level=0).mean()
    adata.uns[key] = s
    logger.info("get_atlas_gene_detection_rate: computed & cached (%s, %d genes, median=%.4f)",
                key, len(s), float(s.median()) if len(s) else float("nan"))
    return s


def build_mask_signature(poa_keywords, include_na: bool) -> str:
    """Deterministic suffix used in cache keys and CSV filenames for the
    POA-only atlas restriction.

    ``build_mask_signature(("preoptic",), True)`` → ``"poaonly_preoptic_na"``.
    """
    parts = ["poaonly"] + sorted(str(kw).strip().lower() for kw in poa_keywords if str(kw).strip())
    if include_na:
        parts.append("na")
    return "_".join(parts)


def get_poa_cell_mask(
    adata,
    *,
    region_col: str = "Region_summarized",
    poa_keywords=("preoptic",),
    include_na: bool = True,
    logger=None,
) -> pd.Series:
    """Boolean mask of POA-compatible cells.

    A cell is ``True`` iff either its ``region_col`` value is missing (NaN) and
    ``include_na`` is True, or any keyword (case-insensitive) appears as a
    substring of its ``region_col`` value.  Matches the per-cell methodology of
    HypoMap Table S1 ("Pnoc means computed within POA cells"); ``include_na``
    defaults to True so clusters with no regional assignment (e.g. the highest-
    Pnoc S1 cluster `C185-67: Pnoc.Mixed.GABA-2`, 100 % NA) are retained.

    Raises
    ------
    KeyError if ``region_col`` is absent from ``adata.obs`` (the message lists
    the available columns whose name starts with "region", to help the caller).

    Returns
    -------
    pd.Series, dtype=bool, indexed by ``adata.obs_names``.
    """
    log = logger or logging.getLogger(__name__)
    keywords = tuple(str(kw).strip().lower() for kw in poa_keywords if str(kw).strip())
    if region_col not in adata.obs.columns:
        region_like = [c for c in adata.obs.columns if "region" in str(c).lower()]
        raise KeyError(
            f"Region column '{region_col}' not found in adata.obs. "
            f"Columns containing 'region': {region_like or '(none)'}. "
            f"All columns: {list(adata.obs.columns)}"
        )

    raw = adata.obs[region_col]
    na_mask = raw.isna()
    text = raw.astype(str)
    # treat the string forms of missing values as NA too (h5ad round-trips can
    # turn NaN/None into the literal strings "nan"/"None"/"NA"/"")
    na_mask = na_mask | text.str.strip().str.lower().isin({"nan", "none", "na", ""})
    lowered = text.str.lower()
    keyword_hit = pd.Series(False, index=raw.index)
    for kw in keywords:
        keyword_hit = keyword_hit | lowered.str.contains(kw, regex=False, na=False)
    # a string-NA cell shouldn't also count as a keyword hit
    keyword_hit = keyword_hit & ~na_mask

    keep = keyword_hit | (na_mask if include_na else pd.Series(False, index=raw.index))
    keep.index = adata.obs_names
    keep = keep.astype(bool)

    n_total = int(len(keep))
    n_keep = int(keep.sum())
    n_kw = int(keyword_hit.sum())
    n_na = int((na_mask & keep.values).sum())
    retained_breakdown = (
        text[keep.values].value_counts().head(5).to_dict() if n_keep else {}
    )
    excluded_breakdown = (
        text[~keep.values].value_counts().head(5).to_dict() if (n_total - n_keep) else {}
    )
    log.info("POA restriction: keywords=%s, include_na=%s", list(keywords), include_na)
    log.info("POA mask: N_poa = %d of %d cells (%.1f%%) — %d keyword matches + %d NA-included",
             n_keep, n_total, 100.0 * n_keep / max(n_total, 1), n_kw, n_na)
    log.info("Region breakdown of POA-compatible cells (top 5): %s", retained_breakdown)
    log.info("Region breakdown of EXCLUDED cells (top 5): %s", excluded_breakdown)
    return keep


def compute_single_gene_cluster_stats(
    adata: ad.AnnData,
    gene_name: str,
    annotation_col: str,
    adata_gene_lookup: Dict[str, Tuple[str, int]],
    has_raw: bool,
    min_cells: int = 10,
    normalize: bool = True,
    threshold: float = 0.0,
) -> Optional[pd.DataFrame]:
    """
    Sanity-check helper: per-cluster mean expression and fraction expressing
    for a single named gene (e.g. ``"Pnoc"`` for Pnoc-Cre lines).

    Indices in ``adata_gene_lookup`` point into ``adata.raw.var`` when
    ``has_raw`` is True, otherwise into ``adata.var`` — this mirrors
    ``_build_adata_gene_lookup``.

    Returns a DataFrame indexed by cluster with columns:
        - ``mean_expr`` (log-normalized when ``normalize=True``)
        - ``fraction_expressing`` (fraction of cells with raw count > threshold)
        - ``gene`` (the resolved display name)

    Returns ``None`` when the gene is absent from the atlas.
    """
    hit = lookup_gene_in_atlas(adata_gene_lookup, gene_name)
    if hit is None:
        logger.info("compute_single_gene_cluster_stats: '%s' not found in atlas", gene_name)
        return None
    display_name, idx = hit
    logger.info("compute_single_gene_cluster_stats: '%s' -> '%s' (idx=%d, in_raw=%s)",
                gene_name, display_name, idx, has_raw)

    use_raw = has_raw
    indices_in_raw = has_raw

    mean_df = compute_cluster_mean_expression(
        adata, [idx], annotation_col,
        min_cells=min_cells,
        use_raw=use_raw,
        indices_in_raw=indices_in_raw,
        normalize=normalize,
    )
    frac_df = compute_fraction_expressing(
        adata, [idx], annotation_col,
        min_cells=min_cells,
        use_raw=use_raw,
        threshold=threshold,
        indices_in_raw=indices_in_raw,
    )

    if mean_df.empty or frac_df.empty:
        return None

    # Both DataFrames have one row (the gene); take the first row as a Series.
    mean_series = mean_df.iloc[0]
    frac_series = frac_df.iloc[0]

    # Align on cluster name (column names of both source frames).
    common = mean_series.index.intersection(frac_series.index)
    result = pd.DataFrame({
        "mean_expr": mean_series.loc[common].astype(float),
        "fraction_expressing": frac_series.loc[common].astype(float),
    })
    result.index.name = "cluster"
    result["gene"] = display_name
    return result


