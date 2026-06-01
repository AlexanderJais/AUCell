"""
AUCell-based mapping of a bulk-RNA-seq DE signature onto a single-cell atlas.

Streamlit application that scores per-cell enrichment of a bulk-RNA-seq
DE signature with AUCell, computes an expression-matched empirical null,
and surfaces per-cluster enrichment statistics, paired UMAP projections,
and a Cre-driver sanity panel. The worked example is bacTRAP onto
HypoMap (Steuernagel et al., Nature Metabolism 2022); the pipeline is
atlas-agnostic.

Run with: streamlit run app.py
"""

import io
import logging
import zipfile

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — writes to aucell.log alongside app.py
# ---------------------------------------------------------------------------
_LOG_FILE = Path(__file__).parent / "aucell.log"
# Cap on the number of trailing log lines embedded in downloads / ZIPs.
# A long Streamlit session can accumulate MBs of log — embedding the whole
# thing slows the ZIP path noticeably and is rarely useful for debugging,
# where the recent traffic is what matters.
_LOG_EXPORT_MAX_LINES = 5000


def _read_log_for_export(path: Path, max_lines: int = _LOG_EXPORT_MAX_LINES) -> str:
    """Read the tail of the log file (most recent `max_lines`) for export.

    Reads efficiently for typical log sizes; for very large files we still
    pull the whole text once and slice — keeping the implementation simple
    and avoiding seek/rewind edge cases. If the file genuinely grows past
    tens of MB the slice is fast in Python anyway.
    """
    txt = path.read_text(errors="replace")
    lines = txt.splitlines()
    if len(lines) <= max_lines:
        return txt
    truncated = lines[-max_lines:]
    return (
        f"[log truncated: showing last {max_lines} of {len(lines)} lines]\n"
        + "\n".join(truncated)
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE, mode="a"),
        logging.StreamHandler(),          # also print to terminal
    ],
    force=True,
)
logger = logging.getLogger(__name__)
# Suppress noisy third-party loggers
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger.info("="*60)
logger.info("App startup / Streamlit rerun")

st.set_page_config(
    page_title="AUCell signature → atlas mapping",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: file inputs and parameters
# ---------------------------------------------------------------------------

st.sidebar.title("bacTRAP → HypoMap Mapping")
st.sidebar.markdown("---")

st.sidebar.subheader("Input Files")

bactrap_file = st.sidebar.text_input(
    "bacTRAP FPKM table (.xlsx)",
    value="",
    placeholder="/path/to/IPvsInput_deg.xlsx",
    help="Path to the bacTRAP DESeq2 results Excel file.",
)

hypomap_file = st.sidebar.text_input(
    "HypoMap atlas (.h5ad)",
    value="",
    placeholder="/path/to/hypomap.h5ad",
    help="Path to the HypoMap AnnData h5ad file.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

padj_cutoff = st.sidebar.slider(
    "Adjusted p-value cutoff", 0.001, 0.1, 0.05, 0.001, format="%.3f",
    help=(
        "Benjamini–Hochberg-adjusted p-value threshold for a bacTRAP gene to "
        "count as 'enriched'. Works in concert with the log₂FC cutoff — "
        "both must be satisfied. The standard DESeq2 threshold of 0.05 is "
        "a reasonable default; lower it (e.g. 0.01) for a more stringent "
        "signature, raise it toward 0.1 if your bacTRAP is underpowered."
    ),
)
log2fc_cutoff = st.sidebar.slider(
    "log₂FC cutoff", 0.0, 5.0, 1.0, 0.25,
    help=(
        "Minimum log₂ fold change (IP vs Input) for a bacTRAP gene to "
        "count as 'enriched'. 1.0 (2-fold) is the convention in the "
        "translational-profiling literature. Pair with the Min IP "
        "expression filter below — raising log₂FC alone admits "
        "pseudocount artefacts from genes with near-zero Input."
    ),
)
min_ip_expression = st.sidebar.slider(
    "Min IP expression (baseMean)", 0.0, 200.0, 10.0, 5.0,
    help=(
        "Minimum mean IP expression (DESeq2 baseMean-style, from the 'IP' "
        "column) required for a gene to pass the enrichment filter. "
        "DESeq2 inflates log₂FC for low-count genes with near-zero Input "
        "(e.g. 28 IP reads vs 0 Input → log₂FC≈7), so the default ranking "
        "otherwise pulls these artefacts above real Cre-driver signal. "
        "Set to 0 to disable. A typical value of 10 removes zero-Input "
        "low-count noise without dropping genuinely cell-type-specific genes."
    ),
)
ranking_metric_label = st.sidebar.selectbox(
    "Gene ranking metric",
    ["π-score (|log₂FC| × -log₁₀padj)", "log₂FoldChange", "-log₁₀(padj)"],
    index=0,
    help=(
        "How the app picks the top-N genes for AUCell, dot-plot, and "
        "heat-map. π-score (Xiao et al. 2014) balances effect size with "
        "significance so a strongly significant moderate-FC gene (e.g. "
        "Pnoc, FC=2, padj=1e-17) outranks a low-count pseudocount artefact "
        "(FC=7, padj=1e-4). Choose 'log₂FoldChange' for the historical "
        "behaviour."
    ),
)
_ranking_metric_map = {
    "π-score (|log₂FC| × -log₁₀padj)": "pi_score",
    "log₂FoldChange": "log2fc",
    "-log₁₀(padj)": "padj",
}
ranking_metric = _ranking_metric_map[ranking_metric_label]
top_n_genes = st.sidebar.slider(
    "Top N genes for scoring", 10, 500, 50, 10,
    help=(
        "Size of the bacTRAP signature passed to AUCell, the dot plot, and "
        "the heatmap (after the padj / log₂FC / min-IP-expression filters, "
        "ranked by the metric above). 50 is a good balance for rank-based "
        "scoring — small enough to stay inside the AUCell 5 %% top-ranked "
        "window (≈ 1,500 genes on HypoMap), large enough to be robust to "
        "dropout in any individual cell. If this exceeds "
        "~τ × total-genes, the AUCell window is automatically widened and "
        "the tab will warn you (see Methods, *AUCell scoring*)."
    ),
)
aucell_top_fraction = st.sidebar.slider(
    "AUCell top-ranked fraction", 0.01, 0.20, 0.05, 0.01, format="%.2f",
    help=(
        "Fraction of genes (ranked by expression within each cell) in which "
        "the bacTRAP signature must appear to contribute to the AUCell score. "
        "Smaller = more stringent: at 0.01 only cells where the signature "
        "concentrates in the top 1% of expressed genes score highly. "
        "AUCell default is 0.05 (Aibar et al. 2017). Use 0.01–0.03 for a "
        "more conservative call on bacTRAP-target identity."
    ),
)
min_cells_per_cluster = st.sidebar.slider(
    "Min cells per cluster", 1, 100, 10, 1,
    help=(
        "Minimum cell count required for a cluster to enter the per-cluster "
        "mean-expression and per-gene detection-rate computations that drive "
        "signature refinement. Distinct from 'Min cells for AUCell "
        "top-N ranking' below, which only gates AUCell figure rankings."
    ),
)
min_cells_for_rank = st.sidebar.slider(
    "Min cells for AUCell top-N ranking", 1, 200, 20, 1,
    help=(
        "Clusters with fewer than this many cells are excluded from the "
        "top-ranked set shown in figures 1b/1c and Suppl. S2. Mean AUCell "
        "for very small clusters is dominated by shrinkage variance, so a "
        "2-cell cluster with a slightly above-average mean can otherwise "
        "claim a top slot purely by chance. Raw per-cluster CSV is "
        "unaffected and still contains every cluster."
    ),
)
umap_subsample = st.sidebar.slider(
    "UMAP subsample (cells)", 10000, 200000, 50000, 5000,
    help=(
        "How many cells to render in the UMAP panels (Figures 1a/1b and "
        "Suppl. S4). Subsampling only affects rendering speed and PDF "
        "file size — all cells are used for AUCell scoring and every "
        "statistical test. 50 k is a readable balance for HypoMap's "
        "~385 k cells; drop below 20 k for faster previews, raise above "
        "100 k if you need rare clusters to survive the random sample."
    ),
)
# ---- Atlas restriction: optionally restrict the whole pipeline
# to a chosen hypothalamic region. Off by default — reproduces full-atlas
# behaviour. POA (preoptic) and MBH (ARC + VMH + DMH) ship as presets; a
# Custom option exposes the raw keyword list.
_REGION_PRESETS = {
    "POA (preoptic area)": {
        "label": "poa",
        "pretty": "POA",
        "keywords": "preoptic",
    },
    "MBH (ARC + VMH + DMH)": {
        "label": "mbh",
        "pretty": "MBH",
        # Substring-matched case-insensitively against Region_summarized;
        # "arcuate" → ARC, "ventromedial" → VMH, "dorsomedial" → DMH.
        "keywords": "arcuate, ventromedial, dorsomedial",
    },
}
with st.sidebar.expander("Atlas restriction", expanded=False):
    region_choice = st.selectbox(
        "Region restriction",
        ["None (full atlas)", "POA (preoptic area)", "MBH (ARC + VMH + DMH)", "Custom keywords"],
        index=0,
        help=(
            "Run the entire pipeline (per-cluster means, AUCell, empirical "
            "null, and Cre-driver baseline filter) over cells from the "
            "chosen hypothalamic region only. POA is directly comparable to "
            "HypoMap Table S1; MBH covers the arcuate (ARC), ventromedial "
            "(VMH), and dorsomedial (DMH) nuclei. Choose Custom keywords to "
            "edit the substring list manually. Default: full hypothalamic "
            "atlas."
        ),
    )
    poa_only = region_choice != "None (full atlas)"
    if region_choice in _REGION_PRESETS:
        _preset = _REGION_PRESETS[region_choice]
        region_label = _preset["label"]
        region_pretty = _preset["pretty"]
        _default_keywords = _preset["keywords"]
    elif region_choice == "Custom keywords":
        region_label = "custom"
        region_pretty = "Custom-region"
        _default_keywords = "preoptic"
    else:
        region_label = "poa"
        region_pretty = "POA"
        _default_keywords = "preoptic"

    if poa_only:
        if region_choice == "Custom keywords":
            poa_keywords_input = st.text_input(
                "Region keywords (comma-separated)", value=_default_keywords,
                help=(
                    "Case-insensitive substring match against the "
                    "`Region_summarized` column. Example: "
                    "`arcuate, ventromedial, dorsomedial` for MBH; "
                    "`preoptic, paraventricular` to broaden POA."
                ),
            )
        else:
            poa_keywords_input = _default_keywords
            st.caption(
                f"Keywords (substring match against `Region_summarized`): "
                f"`{poa_keywords_input}`."
            )
        poa_include_na = st.checkbox(
            "Include cells with no regional assignment (NA)", value=True,
            help=(
                "Cluster-level NA inclusion: a cluster is admitted if it has "
                "≥1 cell whose `Region_summarized` matches a region keyword, "
                "OR if **every** one of its cells has a missing region "
                "label. This retains clusters that HypoMap left "
                "unannotated (e.g. C185-67 Pnoc.Mixed.GABA-2 for POA, 100 % "
                "NA) while excluding mixed clusters where NA cells coexist "
                "with non-target, non-NA cells (e.g. striatal/cortical "
                "clusters like C185-105 under the POA mask)."
            ),
        )
        poa_min_cells = st.number_input(
            f"Min {region_pretty} cells per cluster",
            min_value=1, max_value=2000, value=20, step=1,
            help=(
                f"Clusters with fewer than this many {region_pretty} cells "
                f"drop out of the analysis entirely."
            ),
        )
    else:
        poa_keywords_input, poa_include_na, poa_min_cells = _default_keywords, True, 20
poa_keywords = tuple(s.strip().lower() for s in str(poa_keywords_input).split(",") if s.strip()) or ("preoptic",)
poa_min_cells = int(poa_min_cells)

# ---- Signature refinement: drop signature genes that are
# undetectable in HypoMap or too broadly expressed to be cell-type-specific.
with st.sidebar.expander("Signature refinement", expanded=False):
    sig_filter_detectability = st.checkbox(
        "Filter by HypoMap detectability",
        value=True,
        help=(
            "Drops signature genes that are essentially undetectable in "
            "HypoMap and can't contribute to AUCell (detection rate below "
            "the floor below, or max per-cluster mean log-norm expression "
            "below the floor below)."
        ),
    )
    sig_min_detection_rate = st.slider(
        "Min cell detection rate", 0.00, 0.20, 0.02, 0.005, format="%.3f",
        help="Minimum fraction of atlas cells with a non-zero count for the gene.",
    )
    sig_min_max_cluster_mean = st.slider(
        "Min max-cluster-mean expression", 0.00, 0.50, 0.05, 0.01, format="%.2f",
        help="Minimum value of the gene's largest per-cluster mean log-norm expression.",
    )
    sig_filter_specificity = st.checkbox(
        "Filter for cluster specificity",
        value=True,
        help=(
            "Drops broadly-expressed signature genes (housekeeping / "
            "pan-neuronal like Snap25, Sst, Polr2h) that carry no cell-type "
            "specificity and pull non-target clusters into the AUCell top."
        ),
    )
    sig_specificity_thresh = st.slider(
        "Specificity: cluster mean threshold", 0.1, 2.0, 0.5, 0.1, format="%.1f",
        help="Per-cluster mean log-norm expression above which a cluster counts as 'expressing' the gene.",
    )
    sig_specificity_max_fraction = st.slider(
        "Specificity: max cluster fraction", 0.1, 1.0, 0.5, 0.05, format="%.2f",
        help="Drop the gene if more than this fraction of clusters exceed the threshold above.",
    )

# ---- Empirical-null AUCell: score expression-matched random
# control gene sets and report per-cluster z-score / empirical p-value.
with st.sidebar.expander("Empirical null", expanded=False):
    empirical_null_enabled = st.checkbox(
        "Compute empirical-null z-scores",
        value=True,
        help=(
            "Score N expression-matched random control gene sets with AUCell "
            "and report, per cluster, a z-score and one-sided empirical "
            "p-value of the bacTRAP-signature mean relative to the control "
            "distribution (Aibar et al. 2017). Removes the 'baseline "
            "gene-rank-width' bias that inflates AUCell in broadly-active "
            "neuronal clusters. Adds columns to aucell_per_cluster.csv."
        ),
    )
    empirical_null_n = st.slider(
        "Control sets (N)", 20, 500, 100, 20,
        help=(
            "More sets → tighter null estimate but slower. 100 is a "
            "reasonable compromise; 500 is the AUCell paper default."
        ),
    )
    empirical_null_bins = st.slider(
        "Expression bins", 3, 10, 5, 1,
        help="Number of atlas-wide expression-quantile bins used to match control genes to the signature.",
    )
    empirical_null_seed = st.number_input(
        "Random seed", min_value=0, max_value=2**31 - 1, value=0, step=1,
        help="Seed for the control-set sampling RNG and the AUCell tie-breaking jitter (reproducible).",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Cre-driver Sanity Check")
sanity_gene = st.sidebar.text_input(
    "Cre-driver gene",
    value="Pnoc",
    help=(
        "Marker gene used as a confidence check on the mapping — typically "
        "the Cre-driver of the bacTRAP line (e.g. Pnoc for Pnoc-Cre;NuTRAP). "
        "Top-ranked clusters that also express this gene are high-confidence; "
        "those that don't may reflect developmental Cre lineage tracing, "
        "snRNA-seq dropout, or background. Note: snRNA-seq dropout for "
        "neuropeptides means 'not detected' ≠ 'not expressed'."
    ),
)
sanity_fraction_threshold = st.sidebar.slider(
    "Expression threshold (fraction)",
    0.0, 0.5, 0.05, 0.01, format="%.2f",
    help=(
        "**Display-only.** Minimum fraction of cells expressing the "
        "Cre-driver gene for a cluster to be flagged as 'expressing' in "
        "the Cre-driver Check sanity table. Does NOT change any ranking "
        "— use the baseline slider below for that. 5% is a permissive "
        "default that tolerates dropout."
    ),
)
_baseline_help = (
    "**Ranking filter.** When > 0, drop clusters whose mean "
    "(log-normalized) Cre-driver expression falls below this floor "
    "from the AUCell cluster figures (1b, S2, 1c) and the per-cell "
    "AUCell projection (1a) — cells in dropped clusters are excluded. "
    "Filtered CSV downloads are suffixed with the filter signature "
    "(e.g. `aucell_per_cluster_pnoc_ge0p05.csv`). Set to 0 to disable. "
)
if poa_only:
    _baseline_help += (
        f"**{region_pretty}-only mode is on**, so this cutoff compares against the "
        f"Cre-driver gene's mean log-norm expression *in {region_pretty} cells only* — "
        f"those values are typically larger than the full-atlas values for "
        f"genes enriched in {region_pretty} cells, so the same numeric cutoff is "
        f"stricter here. For POA-restricted Pnoc-Cre: default 0.15 excludes "
        f"the warm-sensitive Pnoc minority (POA-restricted Pnoc < 0.08); use "
        f"0.03 when that subset should be retained."
    )
else:
    _baseline_help += (
        "(Full-atlas mode — the cutoff compares against full-atlas mean "
        "log-norm expression.) Caveat: snRNA-seq dropout for neuropeptides "
        "means 'not detected' ≠ 'not expressed'. Start at ~0.05 and inspect "
        "the sanity-check table to tune."
    )
sanity_baseline_mean_expr = st.sidebar.slider(
    "Baseline Cre-driver mean expression (log-norm)",
    0.0, 1.5, 0.0, 0.01, format="%.2f",
    help=_baseline_help,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Figure Settings")
fig_width_mode = st.sidebar.radio(
    "Figure width", ["Single column (89mm)", "Double column (183mm)"],
    index=0,
    help=(
        "Target print width for the exported PDF / SVG figures. Nature's "
        "column widths are 89 mm (single) and 183 mm (double). Choose "
        "single for individual panels that will be placed in a one-column "
        "slot; double for figures that will span the page. Some functions "
        "force a layout regardless (e.g. two-panel UMAPs always render at "
        "double-column width)."
    ),
)
double_column = "Double" in fig_width_mode

# Annotation column selector — populated after data load
annotation_col_key = "annotation_col"

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("AUCell signature → single-cell atlas mapping")
st.caption(
    "Score per-cell enrichment of a bulk-RNA-seq DE signature on a "
    "single-cell atlas with AUCell, against an expression-matched "
    "empirical null. Worked example: bacTRAP → HypoMap."
)

# Check file inputs — must be existing files, not directories
bt_path = Path(bactrap_file.strip()) if bactrap_file.strip() else None
hm_path = Path(hypomap_file.strip()) if hypomap_file.strip() else None

files_ready = (
    bt_path is not None
    and hm_path is not None
    and bt_path.is_file()
    and hm_path.is_file()
)

if not files_ready and (bt_path or hm_path):
    if bt_path and not bt_path.exists():
        st.warning(f"bacTRAP file not found: `{bt_path}`")
    elif bt_path and bt_path.is_dir():
        st.warning(f"bacTRAP path is a directory, not a file: `{bt_path}`")
    if hm_path and not hm_path.exists():
        st.warning(f"HypoMap file not found: `{hm_path}`")
    elif hm_path and hm_path.is_dir():
        st.warning(f"HypoMap path is a directory, not a file: `{hm_path}`")

if not files_ready:
    st.info("Enter file paths in the sidebar and click **Run Analysis** to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Data Loading (always loads when files are ready, uses caching)
# ---------------------------------------------------------------------------

from data_loading import (
    load_hypomap,
    load_bactrap,
    get_annotation_columns,
    match_genes,
    get_atlas_cluster_mean_expr,
    get_atlas_gene_detection_rate,
    compute_single_gene_cluster_stats,
    get_poa_cell_mask,
    build_mask_signature,
    _detect_gene_column,
    _build_adata_gene_lookup,
)
from analysis import (
    get_enriched_genes,
    filter_signature_genes_by_atlas,
    rank_enriched_genes,
    compute_aucell_scores,
    compute_empirical_null_aucell,
    validate_aucell_input,
    compute_cluster_enrichment_stats,
)
from figure2_inputs import (
    resolve_gene_index,
    gene_poa_inputs,
    gene_expressing_masks,
    region_cell_mask,
)
from figures import (
    setup_nature_style,
    figure_bactrap_volcano,
    figure_umap_enrichment,
    figure_aucell_umap,
    figure_celltype_umap,
    figure_gene_poa_umap,
    figure_pnoc_overview,
    figure_aucell_cluster_barplot,
    figure_aucell_violins,
    figure_aucell_zscore_violins,
    figure_aucell_histogram,
    figure_marker_gene_diagnostic,
    fig_to_bytes,
)

# Load data with caching
with st.spinner("Loading data..."):
    bactrap_df = load_bactrap(bactrap_file.strip())
    adata = load_hypomap(hypomap_file.strip())

# Early UMAP sanity check — fail now rather than after minutes of analysis
if not any(key in adata.obsm for key in ("X_umap", "X_UMAP")):
    st.error(
        "No UMAP coordinates found in HypoMap `.obsm` (expected `X_umap`). "
        "UMAP projection is required for Figures 1a and S2. "
        "Please provide an atlas that includes precomputed UMAP coordinates."
    )
    st.stop()

# Early sanity check on bacTRAP required columns
for _req_col in ("padj", "log2FoldChange"):
    if _req_col not in bactrap_df.columns:
        st.error(
            f"Required column **`{_req_col}`** not found in the bacTRAP file. "
            f"Available columns: `{list(bactrap_df.columns)}`. "
            f"Expected DESeq2-style output with `padj` and `log2FoldChange` columns."
        )
        st.stop()

# Annotation column selection
ann_cols = get_annotation_columns(adata)
if len(ann_cols) == 0:
    st.error("No suitable annotation columns found in HypoMap .obs.")
    st.stop()

default_idx = 0
# Prefer C185_named (best resolution for bacTRAP mapping), then C66_named
for preferred in ["C185_named", "C66_named"]:
    if preferred in ann_cols:
        default_idx = ann_cols.index(preferred)
        break
else:
    # Fallback: look for any cell_type or cluster column
    for i, col in enumerate(ann_cols):
        col_lower = col.lower()
        if "cell_type" in col_lower or "celltype" in col_lower or "cluster" in col_lower:
            default_idx = i
            break

annotation_col = st.sidebar.selectbox(
    "Annotation column",
    ann_cols,
    index=default_idx,
    key=annotation_col_key,
    help=(
        "HypoMap cell-type annotation level used for every cluster-level "
        "analysis and figure (`C7_named` … `C465_named`). Finer levels "
        "(higher numbers) give more granular clusters but smaller cell "
        "counts per cluster — which may push populations below the "
        "'Min cells for AUCell top-N ranking' threshold. `C185_named` is "
        "the HypoMap default and a reasonable starting point for "
        "hypothalamic neuropeptide signatures."
    ),
)

# ---- POA-only atlas restriction ----
# When active, every cluster-level AND per-cell computation runs against a
# POA-cell subset (adata_view), built once per (atlas file, mask signature)
# and cached in session state. When off, adata_view IS adata (full atlas).
poa_mask = None
mask_signature = ""
if poa_only:
    try:
        poa_mask = get_poa_cell_mask(
            adata, poa_keywords=poa_keywords, include_na=poa_include_na,
            annotation_col=annotation_col, logger=logger,
        ).to_numpy(dtype=bool)
    except KeyError as e:
        st.error(
            f"**{region_pretty} restriction can't be applied:** {e}\n\n"
            f"Set 'Region restriction' to 'None (full atlas)', or tell us "
            f"which `.obs` column holds the regional labels."
        )
        st.stop()
    if not poa_mask.any():
        st.error(
            f"**{region_pretty} restriction selected zero cells.** Check the "
            f"keyword list (`{', '.join(poa_keywords)}`) against the values "
            f"in `Region_summarized`, or set 'Region restriction' to "
            f"'None (full atlas)'."
        )
        st.stop()
    if poa_mask.all():
        poa_mask = None  # nothing excluded — treat as no-op
        st.sidebar.caption(
            f":warning: {region_pretty} keywords matched every cell — no restriction applied."
        )
    else:
        mask_signature = build_mask_signature(
            poa_keywords, poa_include_na, annotation_col,
            region_label=region_label,
        )

if poa_mask is not None:
    _view_key = (hypomap_file.strip(), mask_signature, int(poa_mask.sum()))
    if st.session_state.get("_adata_view_key") != _view_key:
        # Evict the previous view BEFORE materialising the new one so we
        # don't briefly hold two multi-GB AnnData copies in memory at once.
        st.session_state.pop("_adata_view", None)
        with st.spinner(f"Building {region_pretty}-restricted atlas view ({int(poa_mask.sum()):,} cells)..."):
            # `.copy()` is intentional: downstream helpers cache atlas-wide
            # statistics on `adata_view.uns`, which only persists on a
            # materialised AnnData (not a view). A view would otherwise force
            # full re-computation of cluster means, detection rates, and the
            # gene-expression bins on every re-run.
            st.session_state["_adata_view"] = adata[poa_mask].copy()
        st.session_state["_adata_view_key"] = _view_key
    adata_view = st.session_state["_adata_view"]
    _n_poa = int(poa_mask.sum())
    st.sidebar.caption(
        f"{region_pretty}-only: **{_n_poa:,}** / {adata.n_obs:,} cells "
        f"({100.0 * _n_poa / max(adata.n_obs, 1):.1f}%), signature `{mask_signature}`."
    )
else:
    adata_view = adata
poa_active = poa_mask is not None
# Effective per-cluster cell-count floors (POA cells when the restriction is on)
_eff_min_cells_cluster = max(min_cells_per_cluster, poa_min_cells) if poa_active else min_cells_per_cluster
_eff_min_cells_rank = max(min_cells_for_rank, poa_min_cells) if poa_active else min_cells_for_rank

# Gene column selection for bacTRAP data
# Build HypoMap lookup once for auto-detection
_adata_lookup, _adata_gnames, _adata_has_raw = _build_adata_gene_lookup(adata)
auto_gene_col = _detect_gene_column(bactrap_df)

# Build candidate list: auto-detected first, then other string/object columns + index
_gene_col_candidates = []
if auto_gene_col != "_index" and auto_gene_col in bactrap_df.columns:
    _gene_col_candidates.append(auto_gene_col)
for col in bactrap_df.columns:
    if col not in _gene_col_candidates:
        # Include string-like columns and columns with Ensembl-looking values
        dtype = bactrap_df[col].dtype
        if dtype == object or dtype.name in ("string", "category"):
            _gene_col_candidates.append(col)
_gene_col_candidates.append("(use row index)")

_default_gene_idx = 0
# Try auto-selecting the column with the best match count
_best_matches = 0
for i, col in enumerate(_gene_col_candidates):
    if col == "(use row index)":
        _vals = bactrap_df.index.astype(str)
    else:
        _vals = bactrap_df[col].astype(str)
    _n = sum(1 for v in _vals if str(v).strip().lower() in _adata_lookup)
    if _n > _best_matches:
        _best_matches = _n
        _default_gene_idx = i

gene_col_selection = st.sidebar.selectbox(
    "bacTRAP gene column",
    _gene_col_candidates,
    index=_default_gene_idx,
    help=(
        "Column in the bacTRAP file containing gene identifiers. "
        "Auto-detected by testing each candidate column against the "
        "HypoMap symbol + Ensembl lookup and picking the highest match "
        "rate. Both gene symbols and Ensembl IDs work. `(use row index)` "
        "lets you use the DataFrame index when gene IDs live there. "
        "Override only if the auto-pick looks wrong — the Gene Matching "
        "Diagnostics section of the Data Overview tab shows the per-"
        "column match rate that drove the auto-selection."
    ),
)
# Map UI selection to the internal value expected by match_genes
_gene_col_for_matching = "_index" if gene_col_selection == "(use row index)" else gene_col_selection

# Early validation: catch a stale / invalid gene-column selection here, not
# 100 lines later deep inside match_genes. Columns can disappear between
# the initial widget render and a rerun (e.g. after the user edits the
# bacTRAP file). The `_index` sentinel is always valid because every
# DataFrame has an index.
if (
    _gene_col_for_matching != "_index"
    and _gene_col_for_matching not in bactrap_df.columns
):
    st.error(
        f"Selected bacTRAP gene column **`{_gene_col_for_matching}`** is "
        f"not present in the bacTRAP file. Available columns: "
        f"`{list(bactrap_df.columns)}`. Pick a different column in the "
        f"sidebar."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

# Progress bar placeholder — rendered above tabs so it's always visible
progress_placeholder = st.empty()

tab1, tab_aucell, tab_sanity, tab4, tab_export = st.tabs([
    "📊 Data Overview",
    "⭐ AUCell (Main Figure)",
    "🔍 Cre-driver Check",
    "🗺️ UMAP Projection (Suppl.)",
    "📥 Export",
])

# Build a fingerprint of all analysis parameters so we can skip recomputation
# on Streamlit reruns when nothing changed.
_analysis_params = (
    bactrap_file.strip(), hypomap_file.strip(),
    _gene_col_for_matching, annotation_col,
    padj_cutoff, log2fc_cutoff, min_ip_expression, ranking_metric,
    top_n_genes, aucell_top_fraction,
    min_cells_per_cluster, int(min_cells_for_rank),
    umap_subsample,
    # POA-only atlas restriction
    bool(poa_active), mask_signature, int(poa_min_cells),
    # Signature refinement
    sig_filter_detectability, sig_min_detection_rate, sig_min_max_cluster_mean,
    sig_filter_specificity, sig_specificity_thresh, sig_specificity_max_fraction,
    # Empirical-null AUCell
    bool(empirical_null_enabled), int(empirical_null_n), int(empirical_null_bins),
    int(empirical_null_seed),
)

if run_button or st.session_state.analysis_done:

    # Reuse cached results on rerun if parameters haven't changed
    _cached = st.session_state.get("_analysis_cache")
    _need_recompute = run_button or _cached is None or _cached.get("params") != _analysis_params

    if _need_recompute:
        # ---- Log user inputs (record-keeping) ----
        logger.info("-" * 60)
        logger.info("User inputs for this run:")
        logger.info("  bacTRAP file: %s", bactrap_file.strip())
        logger.info("  HypoMap file: %s", hypomap_file.strip())
        logger.info("  gene column: %s", _gene_col_for_matching)
        logger.info("  annotation column: %s", annotation_col)
        logger.info("  padj cutoff: %.3f", padj_cutoff)
        logger.info("  log2FC cutoff: %.2f", log2fc_cutoff)
        logger.info("  min IP expression: %.1f", min_ip_expression)
        logger.info("  ranking metric: %s", ranking_metric)
        logger.info("  top N genes: %d", top_n_genes)
        logger.info("  AUCell top fraction: %.2f", aucell_top_fraction)
        logger.info("  min cells per cluster: %d", min_cells_per_cluster)
        logger.info("  min cells for AUCell top-N ranking: %d", min_cells_for_rank)
        logger.info("  UMAP subsample: %d", umap_subsample)
        if poa_active:
            logger.info("  %s restriction: ON — keywords=%s, include_na=%s, mask_signature='%s', "
                        "min_region_cells=%d, N_region=%d/%d cells",
                        region_pretty, list(poa_keywords), poa_include_na, mask_signature,
                        poa_min_cells, int(poa_mask.sum()), adata.n_obs)
        else:
            logger.info("  Region restriction: OFF (full hypothalamic atlas)")
        logger.info("  signature refinement: detectability=%s (min_det=%.3f, min_max_mean=%.3f), "
                    "specificity=%s (thresh=%.2f, max_frac=%.2f)",
                    sig_filter_detectability, sig_min_detection_rate, sig_min_max_cluster_mean,
                    sig_filter_specificity, sig_specificity_thresh, sig_specificity_max_fraction)
        logger.info("  empirical null: enabled=%s (N=%d, bins=%d, seed=%d)",
                    empirical_null_enabled, int(empirical_null_n), int(empirical_null_bins),
                    int(empirical_null_seed))
        logger.info("  Cre-driver gene: %s", sanity_gene)
        logger.info("  Cre-driver expression fraction threshold: %.2f", sanity_fraction_threshold)
        logger.info("  Cre-driver baseline mean expression (log-norm): %.2f", sanity_baseline_mean_expr)
        logger.info("  figure width mode: %s", fig_width_mode)
        logger.info("-" * 60)

        # ---- Gene matching ----
        progress = progress_placeholder.progress(0, text="Matching genes...")

        bactrap_matched, matched_genes, gene_to_idx, matched_in_raw = match_genes(
            bactrap_df, adata, gene_col=_gene_col_for_matching,
            _prebuilt_lookup=(_adata_lookup, _adata_gnames, _adata_has_raw),
        )

        if len(matched_genes) == 0:
            progress.empty()
            st.error(
                f"**No genes matched.** The selected gene column "
                f"**`{_gene_col_for_matching}`** produced zero matches against "
                f"the HypoMap atlas lookup (size ≈ {len(_adata_lookup):,}).\n\n"
                f"**How to fix:**\n"
                f"1. Open the **Data Overview** tab and check **Gene Matching "
                f"Diagnostics** — it shows sample gene IDs from both sides so "
                f"you can see whether you're passing symbols where Ensembl IDs "
                f"are expected (or vice versa).\n"
                f"2. Override **bacTRAP gene column** in the sidebar and try "
                f"the column with the highest match rate.\n"
                f"3. If no column works, the atlas may be in a different "
                f"species or use an unusual symbol convention — check the "
                f"sample atlas keys logged to `aucell.log`."
            )
            st.stop()

        progress.progress(10, text="Genes matched. Computing cluster means...")

        # ---- Cluster mean expression ----
        # Explicit normalize=True: the raw layer holds counts, and downstream
        # correlation / NNLS require log-normalized means in comparable space.
        # Leaving it on auto is unsafe when the gene subset is sparse (the
        # heuristic samples only the selected genes and can falsely decide
        # the data is already normalized — see the second call below).
        gene_indices = [gene_to_idx[g] for g in matched_genes]
        # adata_view == adata unless the POA-only restriction is
        # active, in which case it's the POA-cell subset and the WHOLE pipeline
        # (per-cell scoring, UMAP, every cluster statistic) runs against it.
        # Cached on adata_view.uns (keyed by cluster column + mask signature) so
        # cutoff tweaks don't recompute the matched-gene cluster-mean matrix.
        cluster_mean_expr = get_atlas_cluster_mean_expr(
            adata_view, gene_indices, annotation_col,
            mask_signature=mask_signature, min_cells=_eff_min_cells_cluster,
            indices_in_raw=matched_in_raw, normalize=True,
        )
        progress.progress(25, text="Cluster means computed. Identifying enriched genes...")

        # ---- Enriched genes ----
        # Pass the min-IP-expression filter through; it guards against the
        # DESeq2 low-count log₂FC-inflation artefact (see rank_enriched_genes
        # docstring for details).
        enriched_df = get_enriched_genes(
            bactrap_matched, padj_cutoff, log2fc_cutoff,
            min_ip_expression=min_ip_expression, ip_col="IP",
        )

        # ---- Signature refinement ----
        # Drop candidate signature genes that are undetectable in HypoMap or
        # too broadly expressed to be cell-type-specific, BEFORE π-score
        # ranking / top-N selection, so the filtered list flows into AUCell
        # and every other downstream method.
        n_enriched_prefilter = len(enriched_df)
        signature_refinement_active = (
            (sig_filter_detectability or sig_filter_specificity) and len(enriched_df) > 0
        )
        if signature_refinement_active:
            _cand_genes = enriched_df["_hypomap_gene_name"].tolist()
            _cand_detection = get_atlas_gene_detection_rate(adata_view, mask_signature=mask_signature)
            _refined_genes, sig_drop_log = filter_signature_genes_by_atlas(
                _cand_genes, cluster_mean_expr, _cand_detection,
                apply_detectability=sig_filter_detectability,
                min_detection_rate=sig_min_detection_rate,
                min_max_cluster_mean=sig_min_max_cluster_mean,
                apply_specificity=sig_filter_specificity,
                specificity_cluster_mean_thresh=sig_specificity_thresh,
                specificity_max_cluster_fraction=sig_specificity_max_fraction,
                logger=logger,
            )
            enriched_df = enriched_df[
                enriched_df["_hypomap_gene_name"].isin(set(_refined_genes))
            ].copy()
            # Acceptance-criterion sanity checks, logged explicitly.
            _known_contaminants = ["Sst", "Polr2h", "Eid2", "Pdxp", "Ppil1", "Arl6ip4", "Mrpl12", "Emc9"]
            _dl = sig_drop_log.set_index("gene")
            _present = [g for g in _known_contaminants if g in _dl.index]
            _spec_dropped = [g for g in _present if _dl.loc[g, "status"] == "dropped_specificity"]
            if _present:
                logger.info(
                    "Signature refinement contaminant check: %d/%d known broadly-expressed "
                    "genes present in candidates; dropped by specificity: %s; NOT dropped: %s",
                    len(_present), len(_known_contaminants), _spec_dropped,
                    [g for g in _present if g not in _spec_dropped],
                )
            if "Pnoc" in set(_cand_genes):
                if "Pnoc" in set(_refined_genes):
                    logger.info("Signature refinement tripwire: 'Pnoc' SURVIVED refinement (kept).")
                # (a WARNING is already emitted inside the filter if Pnoc was dropped)
        else:
            sig_drop_log = pd.DataFrame(columns=[
                "gene", "status", "detection_rate", "max_cluster_mean",
                "frac_clusters_above_thresh", "reason",
            ])
        st.session_state["signature_drop_log"] = (
            sig_drop_log if signature_refinement_active else None
        )

        # Rank enriched genes by the user-chosen metric (default π-score,
        # which balances effect size with statistical significance so
        # low-count / zero-Input pseudocount artefacts don't dominate).
        enriched_sorted = rank_enriched_genes(enriched_df, metric=ranking_metric)
        top_enriched_genes = enriched_sorted["_hypomap_gene_name"].tolist()[:top_n_genes]

        if len(top_enriched_genes) == 0:
            st.warning(
                f"No genes pass enrichment thresholds (padj < {padj_cutoff}, "
                f"log₂FC > {log2fc_cutoff}). AUCell scores will be zero — try "
                "relaxing the cutoffs."
            )

        progress.progress(78, text="Validating AUCell input layer...")

        # ---- AUCell input-layer validation (fix #2: guard against
        # non-raw-count layers silently being fed into AUCell) ----
        aucell_qc = validate_aucell_input(
            adata_view, use_raw=True, seed=int(empirical_null_seed),
        )

        progress.progress(80, text="Computing AUCell scores...")

        # ---- AUCell scoring ----
        # Use the empirical-null seed so the signature scoring shares the
        # tie-breaking regime with the empirical-null controls; with the
        # default seed=0 this is identical to the previous behaviour.
        # adata_view is the POA subset when the restriction is on, the full
        # atlas otherwise — so per-cell scoring and Fig 1a follow the mode.
        aucell_run_info: dict = {}
        aucell_scores = compute_aucell_scores(
            adata_view, top_enriched_genes, top_fraction=aucell_top_fraction,
            seed=int(empirical_null_seed),
            info_out=aucell_run_info,
        )

        # Merge scoring diagnostics into the QC report so the UI surfaces
        # match rate + n_top bumps (fixes #5 / #6) alongside the input-layer
        # check (fix #2).
        if aucell_run_info.get("unmatched"):
            _u = aucell_run_info["unmatched"]
            aucell_qc.setdefault("warnings", []).append(
                f"{len(_u)} / {aucell_run_info['n_query_requested']} signature "
                f"genes did not match the atlas gene-name lookup — first 20: "
                f"{_u[:20]}. These genes contribute nothing to the score."
            )
        if aucell_run_info.get("n_top_bumped"):
            aucell_qc.setdefault("warnings", []).append(
                f"Signature ({aucell_run_info['n_query_matched']} genes) is larger "
                f"than the requested top_fraction = {aucell_run_info['requested_top_fraction']:.2%}, "
                f"so n_top was bumped to {aucell_run_info['n_top']} "
                f"({aucell_run_info['effective_top_fraction']:.2%} of genes). "
                f"The AUCell window is wider than the slider suggests — "
                f"scores are less stringent than intended. Reduce 'Top N genes "
                f"for scoring' or raise 'AUCell top-ranked fraction' to align "
                f"the two."
            )
        aucell_qc.setdefault("info", []).append(
            f"AUCell window: n_top = {aucell_run_info.get('n_top', '?')} genes "
            f"({aucell_run_info.get('effective_top_fraction', 0) * 100:.2f}% of "
            f"{adata_view.raw.n_vars if adata_view.raw is not None else adata_view.n_vars}); "
            f"signature matched {aucell_run_info.get('n_query_matched', '?')} / "
            f"{aucell_run_info.get('n_query_requested', '?')} genes."
        )

        progress.progress(83, text="Computing per-cluster enrichment significance...")

        # ---- AUCell result tables (raw data underlying figures 1a–1c + S2/S3) ----
        # adata_view is the whole atlas (or the POA subset), so the
        # per-cell scores, the per-cell CSV, Fig 1a and the cluster aggregation
        # are all over the same cell set.
        _cell_labels_arr = adata_view.obs[annotation_col].values.astype(str)
        aucell_per_cell_df = pd.DataFrame({
            "cell_id": adata_view.obs_names.astype(str),
            "cluster": _cell_labels_arr,
            "aucell_score": aucell_scores,
        })
        _agg_scores = aucell_scores
        _agg_labels = _cell_labels_arr

        # Per-cluster significance (fix #3: Welch's one-sided t-test
        # cluster-vs-rest with BH-FDR so users can separate "truly enriched"
        # from "small cluster with a slightly above-average mean")
        aucell_cluster_stats_df = compute_cluster_enrichment_stats(
            _agg_scores, _agg_labels, min_cells=10, alpha=0.05,
        )

        # Preserve the original ordering (sorted by mean descending) for the
        # figures, but merge in the significance columns so the downloadable
        # table is the authoritative reference.
        _grp = (
            pd.DataFrame({"cluster": _agg_labels, "aucell_score": _agg_scores})
            .groupby("cluster")["aucell_score"]
        )
        aucell_per_cluster_df = pd.DataFrame({
            "n_cells": _grp.count(),
            "mean": _grp.mean(),
            "median": _grp.median(),
            "std": _grp.std(ddof=1),
        })
        aucell_per_cluster_df["sem"] = (
            aucell_per_cluster_df["std"] / np.sqrt(aucell_per_cluster_df["n_cells"])
        )
        aucell_per_cluster_df = (
            aucell_per_cluster_df.sort_values("mean", ascending=False)
            .reset_index()
        )
        if not aucell_cluster_stats_df.empty:
            aucell_per_cluster_df = aucell_per_cluster_df.merge(
                aucell_cluster_stats_df[
                    ["cluster", "t_stat", "pvalue", "qvalue", "significant"]
                ],
                on="cluster", how="left",
            )

        # ---- Empirical-null AUCell ----
        # Score N expression-matched random control gene sets and report a
        # per-cluster z-score / empirical p-value against the control
        # distribution. Adds columns to aucell_per_cluster.csv.
        empirical_null_df = pd.DataFrame()
        if empirical_null_enabled and len(top_enriched_genes) > 0:
            progress.progress(83, text=f"Empirical null: scoring {int(empirical_null_n)} control sets...")

            def _null_progress(i, n, _p=progress, _N=int(empirical_null_n)):
                # i/n is a fraction of the batched control-scoring pass; map it
                # onto progress 83..93 so the bar stays monotone with the
                # subsequent figure-generation step (95).
                _p.progress(min(83 + int(10 * i / max(n, 1)), 93),
                            text=f"Empirical null: {_N} control sets ({int(100 * i / max(n, 1))}%)...")

            try:
                empirical_null_df = compute_empirical_null_aucell(
                    adata_view, top_enriched_genes,
                    adata_view.obs[annotation_col].values.astype(str),
                    compute_aucell_scores,
                    n_control_sets=int(empirical_null_n),
                    n_bins=int(empirical_null_bins),
                    seed=int(empirical_null_seed),
                    top_fraction=aucell_top_fraction,
                    min_cluster_size=_eff_min_cells_rank,
                    mask_signature=mask_signature,
                    logger=logger,
                    progress_callback=_null_progress,
                )
            except Exception as e:
                logger.exception("Empirical-null AUCell computation failed")
                st.warning(
                    f"Empirical-null AUCell computation failed: {e}. See "
                    f"`aucell.log` for the traceback. Analysis "
                    f"continues without the empirical-null columns."
                )
                empirical_null_df = pd.DataFrame()
            if not empirical_null_df.empty:
                _null_cols = empirical_null_df.reset_index()
                aucell_per_cluster_df = aucell_per_cluster_df.merge(
                    _null_cols, on="cluster", how="left",
                )
                # Clusters below the size gate get no empirical-null row; the
                # left-merge leaves NaN there. Keep the count an integer (0 =
                # "not evaluated for this cluster") rather than a float-with-NaN.
                if "n_control_sets_used" in aucell_per_cluster_df.columns:
                    aucell_per_cluster_df["n_control_sets_used"] = (
                        aucell_per_cluster_df["n_control_sets_used"].fillna(0).astype(int)
                    )

        progress.progress(95, text="Generating figures...")

        # ---- Subsample for UMAP ----
        sub_indices = None
        if adata_view.n_obs > umap_subsample:
            rng = np.random.default_rng(42)
            sub_indices = np.sort(rng.choice(adata_view.n_obs, size=umap_subsample, replace=False))

        # Get UMAP coordinates (existence already verified at load time)
        umap_key = "X_umap" if "X_umap" in adata_view.obsm else "X_UMAP"
        umap_coords = adata_view.obsm[umap_key]
        cell_labels = adata_view.obs[annotation_col].values.astype(str)

        progress.progress(100, text="Analysis complete!")
        progress_placeholder.empty()

        # ---- Cache all analysis results in session state ----
        st.session_state._analysis_cache = {
            "params": _analysis_params,
            "bactrap_matched": bactrap_matched,
            "matched_genes": matched_genes,
            "enriched_df": enriched_df,
            "sig_drop_log": sig_drop_log,
            "n_enriched_prefilter": n_enriched_prefilter,
            "signature_refinement_active": signature_refinement_active,
            "top_enriched_genes": top_enriched_genes,
            "aucell_scores": aucell_scores,
            "aucell_qc": aucell_qc,
            "aucell_per_cell_df": aucell_per_cell_df,
            "aucell_per_cluster_df": aucell_per_cluster_df,
            "aucell_cluster_stats_df": aucell_cluster_stats_df,
            "empirical_null_df": empirical_null_df,
            "sub_indices": sub_indices,
            "umap_coords": umap_coords,
            "cell_labels": cell_labels,
            "enriched_sorted": enriched_sorted,
        }
    else:
        # ---- Restore cached results (no recomputation needed) ----
        _c = _cached
        bactrap_matched = _c["bactrap_matched"]
        matched_genes = _c["matched_genes"]
        enriched_df = _c["enriched_df"]
        sig_drop_log = _c.get("sig_drop_log", pd.DataFrame())
        n_enriched_prefilter = _c.get("n_enriched_prefilter", len(enriched_df))
        signature_refinement_active = bool(_c.get("signature_refinement_active", False))
        st.session_state["signature_drop_log"] = (
            sig_drop_log if signature_refinement_active else None
        )
        top_enriched_genes = _c["top_enriched_genes"]
        aucell_scores = _c["aucell_scores"]
        aucell_qc = _c.get("aucell_qc", {"warnings": [], "info": []})
        aucell_per_cell_df = _c["aucell_per_cell_df"]
        aucell_per_cluster_df = _c["aucell_per_cluster_df"]
        aucell_cluster_stats_df = _c.get("aucell_cluster_stats_df", pd.DataFrame())
        empirical_null_df = _c.get("empirical_null_df", pd.DataFrame())
        sub_indices = _c["sub_indices"]
        umap_coords = _c["umap_coords"]
        cell_labels = _c["cell_labels"]
        enriched_sorted = _c["enriched_sorted"]
        progress_placeholder.empty()

    st.session_state.analysis_done = True

    # ---- Cre-driver sanity stats (up-front so the baseline filter below
    # can use them, and the sanity tab can reuse the cache) ----
    # Floor at min(cluster-mean gate, AUCell gate) so the baseline filter
    # never drops an AUCell-eligible cluster just because it fell below
    # sanity's own size gate (user raising min_cells_per_cluster above
    # min_cells_for_rank would otherwise silently tighten AUCell too).
    _sanity_min_cells = min(_eff_min_cells_cluster, _eff_min_cells_rank)
    _SANITY_NORMALIZE = True  # if this default changes, the cache key below
    # also has to so the cached log-norm values don't survive a re-tune.
    sanity_cache_key = (
        hypomap_file.strip(), annotation_col,
        _sanity_min_cells, sanity_gene.strip().lower(),
        mask_signature, _SANITY_NORMALIZE,
    )
    _sanity_cached = st.session_state.get("_sanity_cache")
    if _sanity_cached is not None and _sanity_cached.get("key") == sanity_cache_key:
        sanity_stats = _sanity_cached["stats"]
    else:
        with st.spinner(f"Computing per-cluster {sanity_gene} expression..."):
            sanity_stats = compute_single_gene_cluster_stats(
                adata_view, sanity_gene.strip(), annotation_col,
                adata_gene_lookup=_adata_lookup,
                has_raw=_adata_has_raw,
                min_cells=_sanity_min_cells,
                normalize=_SANITY_NORMALIZE,
            )
        st.session_state["_sanity_cache"] = {
            "key": sanity_cache_key, "stats": sanity_stats,
        }

    # ---- Optional baseline Cre-driver expression filter ----
    # When the slider is > 0, drop clusters whose mean Cre-driver expression
    # falls below the floor from the AUCell cluster figures and the per-cell
    # projection. Applied post-cache so toggling the slider doesn't
    # invalidate the expensive parts of the analysis.
    baseline_allowed = None
    baseline_filter_state = "disabled"  # "disabled" | "active" | "broken_missing_gene" | "broken_empty"
    if sanity_baseline_mean_expr > 0:
        if sanity_stats is None or len(sanity_stats) == 0:
            baseline_filter_state = "broken_missing_gene"
        else:
            baseline_allowed = set(
                sanity_stats.index[
                    sanity_stats["mean_expr"] >= sanity_baseline_mean_expr
                ].astype(str)
            )
            if len(baseline_allowed) == 0:
                baseline_allowed = None
                baseline_filter_state = "broken_empty"
            else:
                baseline_filter_state = "active"

    # Clusters eligible for the AUCell top-N figure rankings: the baseline
    # Cre-driver filter survivors when active, otherwise no restriction.  (The
    # POA restriction, when active, already limits the cluster universe at the
    # data level — adata_view holds POA cells only — so no extra set needed.)
    _aucell_allowed = (
        {str(c) for c in baseline_allowed} if baseline_allowed is not None else None
    )

    # Apply the baseline filter to every AUCell output — the user's intent is
    # to remove Cre-driver-negative clusters from the AUCell analysis, so the
    # cluster-level panels (barplot S2, violin 1c, cell-type UMAP highlight
    # 1b, per-cluster CSV) AND the per-cell panels (AUCell UMAP fig 1a,
    # per-cell CSV) all drop cells belonging to filtered-out clusters.
    sub_indices_filtered = sub_indices
    if baseline_allowed is not None:
        aucell_per_cluster_df = aucell_per_cluster_df[
            aucell_per_cluster_df["cluster"].astype(str).isin(baseline_allowed)
        ].reset_index(drop=True)
        _baseline_cell_mask = np.isin(cell_labels, list(baseline_allowed))
        aucell_per_cell_df = aucell_per_cell_df[_baseline_cell_mask].reset_index(drop=True)
        if sub_indices is None:
            sub_indices_filtered = np.where(_baseline_cell_mask)[0]
        else:
            sub_indices_filtered = sub_indices[_baseline_cell_mask[sub_indices]]

    # Sidebar-side mirror of the filter state so the user sees the
    # effective setting even if they've scrolled past the global banner.
    if baseline_filter_state == "active":
        st.sidebar.caption(
            f"Baseline {sanity_gene} filter: **{len(baseline_allowed)}"
            f"/{len(sanity_stats)}** clusters pass mean ≥ "
            f"{sanity_baseline_mean_expr:.2f}."
        )
    elif baseline_filter_state == "broken_missing_gene":
        st.sidebar.caption(
            f":warning: `{sanity_gene}` not in atlas — filter ignored."
        )
    elif baseline_filter_state == "broken_empty":
        st.sidebar.caption(
            f":warning: threshold {sanity_baseline_mean_expr:.2f} rejects "
            f"every cluster — filter ignored."
        )

    # Cache figure bytes so the Export tab doesn't regenerate them.
    # Only reset when a new analysis run is triggered (run_button pressed
    # or parameters changed), not on every Streamlit rerun.
    if _need_recompute:
        st.session_state.fig_bytes = {}
        st.session_state.table_bytes = {}

    if "fig_bytes" not in st.session_state:
        st.session_state.fig_bytes = {}
    if "table_bytes" not in st.session_state:
        st.session_state.table_bytes = {}

    # Pre-encode AUCell result tables.  Both the per-cell (~400k rows) and
    # per-cluster (~185 rows) CSVs change when the baseline filter or the
    # POA restriction change the cell / cluster universe, so re-serialise
    # whenever the filter signature changes.
    _baseline_sig = (
        tuple(sorted(baseline_allowed)) if baseline_allowed is not None else (),
        mask_signature,
    )
    if st.session_state.table_bytes.get("_aucell_per_cell_sig") != _baseline_sig:
        st.session_state.table_bytes["aucell_per_cell"] = (
            aucell_per_cell_df.to_csv(index=False).encode()
        )
        st.session_state.table_bytes["_aucell_per_cell_sig"] = _baseline_sig
    if st.session_state.table_bytes.get("_aucell_per_cluster_sig") != _baseline_sig:
        st.session_state.table_bytes["aucell_per_cluster"] = (
            aucell_per_cluster_df.to_csv(index=False).encode()
        )
        st.session_state.table_bytes["_aucell_per_cluster_sig"] = _baseline_sig

    def _cache_fig(name, fig):
        st.session_state.fig_bytes[name] = {
            "pdf": fig_to_bytes(fig, "pdf"),
            "svg": fig_to_bytes(fig, "svg"),
        }

    # Filename suffix applied to filter-affected CSV downloads so a
    # collaborator who opens a 40-row aucell_per_cluster.csv can tell from
    # the filename alone that it's a filtered subset, not the full atlas.
    # Segments combine in the order
    #   {cre_driver_filter}_{poaonly_suffix}_{refined_suffix}_{null_filter}
    # Dots are replaced with 'p' (safe on every filesystem).
    _empirical_null_active = bool(
        empirical_null_enabled
        and isinstance(empirical_null_df, pd.DataFrame)
        and not empirical_null_df.empty
    )
    # POA segment in download filenames. Cap the joined keyword list so a
    # user supplying many keywords doesn't produce a pathologically long
    # filename, but always carry at least 1-3 keywords + na suffix so the
    # filename remains diagnostic rather than collapsing to a bare
    # "poaonly".
    def _truncated_mask_signature(sig: str, max_chars: int = 60) -> str:
        if len(sig) <= max_chars:
            return sig
        return sig[: max_chars - 4] + "etc"

    _poa_segment = (
        _truncated_mask_signature(mask_signature) if poa_active else None
    )

    def _filter_signature_parts() -> list:
        segs = []
        if baseline_allowed is not None:
            segs.append(
                f"{sanity_gene.lower()}_ge{sanity_baseline_mean_expr:.2f}".replace(".", "p")
            )
        if _poa_segment:
            segs.append(_poa_segment)
        if signature_refinement_active:
            segs.append("refined")
        if _empirical_null_active:
            segs.append(f"null{int(empirical_null_n)}")
        return segs

    def _filtered_name(basename: str) -> str:
        segs = _filter_signature_parts()
        if not segs:
            return basename
        stem, _, ext = basename.rpartition(".")
        return f"{stem}_{'_'.join(segs)}.{ext}"

    # Filter-signature string used in the drop-log filename ("" when no filter
    # touched the cluster universe / signature).
    _filter_signature_str = "_".join(_filter_signature_parts())

    # ---- Global baseline-filter status banner (above the tab group) ----
    # Rendered once so every tab — not just AUCell — makes the filter state
    # obvious.  broken_* states distinguish "slider > 0 but doing nothing"
    # from "slider at 0" so a user whose threshold discards everything
    # doesn't silently see an unfiltered dashboard.
    if baseline_filter_state == "active":
        st.info(
            f"**Baseline {sanity_gene} filter active** — "
            f"{len(baseline_allowed)} / {len(sanity_stats)} clusters pass "
            f"mean {sanity_gene} ≥ {sanity_baseline_mean_expr:.2f}. "
            f"Applies to the AUCell cluster figures (1b / S2 / 1c) **and** "
            f"to the per-cell AUCell panel (fig 1a) and per-cell CSV, "
            f"which drop cells belonging to filtered-out clusters."
        )
    elif baseline_filter_state == "broken_missing_gene":
        st.warning(
            f"**Baseline filter ignored** — `{sanity_gene}` not found in "
            f"the HypoMap atlas (slider at {sanity_baseline_mean_expr:.2f}). "
            f"Tabs show unfiltered data. Check gene-symbol casing "
            f"(e.g. `Pnoc`, not `PNOC`)."
        )
    elif baseline_filter_state == "broken_empty":
        st.error(
            f"**Baseline filter ignored** — threshold "
            f"mean {sanity_gene} ≥ {sanity_baseline_mean_expr:.2f} "
            f"discards every cluster. Tabs show unfiltered data. "
            f"Lower the slider."
        )

    if poa_active:
        _n_poa = int(poa_mask.sum())
        _n_view_clusters = adata_view.obs[annotation_col].nunique()
        st.info(
            f"**{region_pretty}-only mode active** — the entire pipeline "
            f"(cluster means, marker genes, AUCell, empirical null, "
            f"Cre-driver baseline filter, correlation, Fisher, NNLS, GSEA, "
            f"and the per-cell AUCell UMAP) runs over **{_n_poa:,}** "
            f"{region_pretty} cells / {adata.n_obs:,} total "
            f"({100.0 * _n_poa / max(adata.n_obs, 1):.1f}%; "
            f"{_n_view_clusters} clusters at `{annotation_col}`), keywords "
            f"`{', '.join(poa_keywords)}`, NA cells "
            f"{'included' if poa_include_na else 'excluded'}. "
            f"Cre-driver baseline cutoffs now operate on "
            f"{region_pretty}-restricted values (see the slider tooltip). "
            f"Filtered CSV downloads carry a `_{mask_signature}` suffix."
        )

    # ======================================================================
    # TAB 1: Data Overview
    # ======================================================================
    with tab1:
        st.header("Data Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("bacTRAP genes", len(bactrap_df))
        with col2:
            if poa_active:
                st.metric(f"HypoMap cells ({region_pretty} / total)",
                          f"{adata_view.n_obs:,} / {adata.n_obs:,}",
                          help=f"{region_pretty} restriction active — the pipeline runs over the {region_pretty} subset.")
            else:
                st.metric("HypoMap cells", f"{adata.n_obs:,}")
        with col3:
            n_clusters = adata_view.obs[annotation_col].nunique()
            st.metric(f"Clusters ({annotation_col})", n_clusters,
                      help=(f"{region_pretty}-restricted cluster count" if poa_active else None))

        st.markdown("---")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Genes matched", len(matched_genes))
        with col5:
            st.metric("Enriched genes (padj + FC)", len(enriched_df))
        with col6:
            match_pct = (len(matched_genes) / len(bactrap_df) * 100) if len(bactrap_df) > 0 else 0.0
            st.metric("Match rate", f"{match_pct:.1f}%")

        st.subheader("Gene Matching Diagnostics")

        # Show which column was used and sample gene names from each dataset
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown(f"**bacTRAP gene column:** `{gene_col_selection}`")
            if gene_col_selection == "(use row index)":
                _sample_bt = [str(x) for x in bactrap_df.index[:10]]
            else:
                _sample_bt = bactrap_df[gene_col_selection].dropna().head(10).astype(str).tolist()
            st.markdown("Sample bacTRAP gene IDs:")
            st.code("\n".join(_sample_bt))
        with diag_col2:
            _sample_hm = [str(x) for x in _adata_gnames[:10]]
            st.markdown(f"**HypoMap gene names** (resolved, n={len(_adata_gnames)}):")
            st.code("\n".join(_sample_hm))
            # Show raw var_names too if different
            if _adata_has_raw and adata.raw is not None:
                _raw_vn = [str(x) for x in adata.raw.var_names[:5]]
                if _raw_vn != [str(x) for x in _adata_gnames[:5]]:
                    st.markdown("Raw var_names (first 5):")
                    st.code("\n".join(_raw_vn))

        # Bar chart of matched vs unmatched
        match_data = pd.DataFrame({
            "Category": ["Matched", "Unmatched"],
            "Count": [len(matched_genes), len(bactrap_df) - len(matched_genes)],
        })
        st.bar_chart(match_data.set_index("Category"))

        st.subheader("bacTRAP Data Preview")
        st.dataframe(bactrap_df.head(20), use_container_width=True)

        st.subheader("Top Enriched Genes")
        st.caption(
            f"Filters: padj < {padj_cutoff:.3g}, log₂FC > {log2fc_cutoff:.2f}, "
            f"IP ≥ {min_ip_expression:.0f}. "
            f"Ranked by **{ranking_metric_label}**."
        )
        if len(enriched_df) > 0:
            display_cols = ["_hypomap_gene_name", "log2FoldChange", "padj", "IP", "Input"]
            available_cols = [c for c in display_cols if c in enriched_df.columns]
            st.dataframe(
                enriched_sorted[available_cols].head(30).reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.warning("No genes pass the current enrichment thresholds.")

        # ---- Signature refinement diagnostics ----
        # Only rendered when at least one refinement filter actually ran.
        if signature_refinement_active and isinstance(sig_drop_log, pd.DataFrame) and len(sig_drop_log):
            with st.expander("Signature refinement diagnostics", expanded=False):
                _n_total = len(sig_drop_log)
                _n_kept = int((sig_drop_log["status"] == "kept").sum())
                _n_det = int((sig_drop_log["status"] == "dropped_detectability").sum())
                _n_spec = int((sig_drop_log["status"] == "dropped_specificity").sum())
                _n_na = int((sig_drop_log["status"] == "dropped_not_in_atlas").sum())
                _n_dropped = _n_total - _n_kept
                _parts = []
                if _n_det:
                    _parts.append(f"{_n_det} low detectability")
                if _n_spec:
                    _parts.append(f"{_n_spec} low specificity")
                if _n_na:
                    _parts.append(f"{_n_na} not in atlas")
                _breakdown = (": " + ", ".join(_parts)) if _parts else ""
                st.markdown(
                    f"**{_n_total} candidate genes → {_n_kept} kept "
                    f"({_n_dropped} dropped{_breakdown})** — "
                    f"detectability filter: **{'on' if sig_filter_detectability else 'off'}**, "
                    f"specificity filter: **{'on' if sig_filter_specificity else 'off'}**."
                )

                _show = sig_drop_log.copy()
                _show.columns = [
                    "Gene", "Status", "Detection rate",
                    "Max cluster mean (log-norm)", "Frac. clusters > threshold", "Reason",
                ]
                st.dataframe(
                    _show.style.format({
                        "Detection rate": "{:.4f}",
                        "Max cluster mean (log-norm)": "{:.4f}",
                        "Frac. clusters > threshold": "{:.1%}",
                    }, na_rep="—"),
                    use_container_width=True,
                )

                _dropped = sig_drop_log[sig_drop_log["status"] != "kept"].copy()
                if len(_dropped):
                    st.markdown("**Top 20 dropped genes by detection rate** "
                                "(highly-detected-but-dropped genes — typically the "
                                "broadly-expressed contaminants the specificity filter targets):")
                    _top20 = (
                        _dropped.sort_values("detection_rate", ascending=False, na_position="last")
                        .head(20)[["gene", "status", "detection_rate", "max_cluster_mean",
                                   "frac_clusters_above_thresh", "reason"]]
                        .reset_index(drop=True)
                    )
                    _top20.columns = [
                        "Gene", "Status", "Detection rate",
                        "Max cluster mean (log-norm)", "Frac. clusters > threshold", "Reason",
                    ]
                    st.dataframe(
                        _top20.style.format({
                            "Detection rate": "{:.4f}",
                            "Max cluster mean (log-norm)": "{:.4f}",
                            "Frac. clusters > threshold": "{:.1%}",
                        }, na_rep="—"),
                        use_container_width=True,
                    )

                _droplog_name = f"signature_refinement_droplog_{_filter_signature_str}.csv"
                st.download_button(
                    "Download signature refinement drop-log (CSV)",
                    sig_drop_log.to_csv(index=False).encode(),
                    _droplog_name, "text/csv",
                    key="dl_sig_refine_droplog_csv",
                    help="One row per candidate gene: gene, status, detection_rate, "
                         "max_cluster_mean, frac_clusters_above_thresh, reason.",
                )

        # ---- Region restriction diagnostics ----
        if poa_active:
            with st.expander(f"{region_pretty} restriction diagnostics", expanded=False):
                _n_poa = int(poa_mask.sum())
                _n_full = int(adata.n_obs)
                st.markdown(
                    f"**{region_pretty} restriction active:** {_n_poa:,} of {_n_full:,} cells "
                    f"retained ({100.0 * _n_poa / max(_n_full, 1):.1f}%) — keywords "
                    f"`{', '.join(poa_keywords)}`, NA cells "
                    f"{'**included**' if poa_include_na else '**excluded**'}, "
                    f"`min_{region_label}_cells = {poa_min_cells}`."
                )
                try:
                    _region = adata.obs["Region_summarized"].astype(str)
                    _ret = _region[poa_mask].value_counts().head(5)
                    _exc = _region[~poa_mask].value_counts().head(5)
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("**Top retained regions**")
                        st.dataframe(_ret.rename("cells").to_frame(), use_container_width=True)
                    with rc2:
                        st.markdown("**Top excluded regions**")
                        st.dataframe(_exc.rename("cells").to_frame(), use_container_width=True)
                except Exception:
                    st.caption("(Region breakdown unavailable.)")
                # Clusters that drop out (too few POA cells)
                _full_sizes = adata.obs[annotation_col].astype(str).value_counts()
                _poa_sizes = adata_view.obs[annotation_col].astype(str).value_counts()
                _gate = max(int(poa_min_cells), int(min_cells_for_rank))
                _drop_rows = []
                for _c, _nf in _full_sizes.items():
                    _np_ = int(_poa_sizes.get(_c, 0))
                    if _np_ < _gate:
                        _drop_rows.append({
                            "cluster": _c, "n_poa_cells": _np_, "n_full_atlas_cells": int(_nf),
                        })
                if _drop_rows:
                    _drop_df = pd.DataFrame(_drop_rows).sort_values(
                        "n_full_atlas_cells", ascending=False,
                    ).reset_index(drop=True)
                    _drop_df = _drop_df.rename(columns={"n_poa_cells": f"n_{region_label}_cells"})
                    st.markdown(
                        f"**{len(_drop_df)} clusters drop out** under the {region_pretty} mask "
                        f"(fewer than {_gate} {region_pretty} cells) — full-atlas cell counts shown "
                        f"for reference:"
                    )
                    st.dataframe(_drop_df, use_container_width=True)
                else:
                    st.caption(f"No clusters fall below the {region_pretty} cell-count gate.")

        st.subheader("Figure: bacTRAP Volcano Plot")
        _volcano_highlight = [sanity_gene.strip()] if sanity_gene.strip() else []
        fig_volcano = figure_bactrap_volcano(
            bactrap_matched,
            highlight_genes=_volcano_highlight,
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            double_column=double_column,
        )
        st.pyplot(fig_volcano)
        _cache_fig("fig_volcano_bactrap", fig_volcano)

        col_pdf, col_svg = st.columns(2)
        with col_pdf:
            st.download_button(
                "Download PDF",
                st.session_state.fig_bytes["fig_volcano_bactrap"]["pdf"],
                "fig_volcano_bactrap.pdf", "application/pdf",
                key="dl_fig_volcano_bt_pdf",
            )
        with col_svg:
            st.download_button(
                "Download SVG",
                st.session_state.fig_bytes["fig_volcano_bactrap"]["svg"],
                "fig_volcano_bactrap.svg", "image/svg+xml",
                key="dl_fig_volcano_bt_svg",
            )
        plt.close(fig_volcano)

    # ======================================================================
    # TAB: AUCell (MAIN FIGURE)
    # ======================================================================
    with tab_aucell:
        st.header("Main Figure: AUCell Enrichment Analysis")
        st.markdown(
            "AUCell (rank-based Area Under the Curve) is the primary cell-type "
            "mapping method — it is **normalization-insensitive**, **threshold-free**, "
            "and quantifies per-cell enrichment of the bacTRAP gene set. "
            f"Scores computed from the top **{len(top_enriched_genes)}** enriched genes."
        )

        # Input-layer QC (fix #2) — warn loudly when the layer fed into
        # AUCell does not look like raw counts.
        _qc_warnings = aucell_qc.get("warnings", []) if aucell_qc else []
        if _qc_warnings:
            for _w in _qc_warnings:
                st.warning(_w)
        else:
            _info = aucell_qc.get("info", []) if aucell_qc else []
            if _info:
                with st.expander("AUCell input QC", expanded=False):
                    for _line in _info:
                        st.caption(_line)

        # Per-cluster significance summary (fix #3)
        if aucell_cluster_stats_df is not None and not aucell_cluster_stats_df.empty:
            _n_tested = len(aucell_cluster_stats_df)
            _n_sig = int(aucell_cluster_stats_df["significant"].sum())
            st.markdown(
                f"**Enrichment significance (Welch's t, BH-FDR):** "
                f"{_n_sig} / {_n_tested} clusters pass **q < 0.05** "
                f"(cluster AUCell distribution vs. rest of atlas). "
                f"Full per-cluster statistics — `t_stat`, `pvalue`, `qvalue`, "
                f"`significant` — are appended to the downloadable "
                f"`aucell_per_cluster.csv`."
            )
        if _empirical_null_active:
            _n_used = int(empirical_null_df["n_control_sets_used"].iloc[0])
            _n_pos = int((aucell_per_cluster_df.get("qvalue_empirical", pd.Series(dtype=float)) < 0.05).sum())
            st.markdown(
                f"**Empirical null (matched-expression controls):** scored "
                f"**{_n_used}** random control gene sets matched to the "
                f"signature in size and atlas-wide expression bins; "
                f"{_n_pos} cluster(s) pass **q_empirical < 0.05**. The "
                f"`null_mean`, `null_sd`, `z_empirical`, `pvalue_empirical` "
                f"and `qvalue_empirical` columns are appended to "
                f"`aucell_per_cluster.csv`; see the companion violin panel "
                f"below Figure 1c for the top 15 by `z_empirical`."
            )

        # Figure 1a: AUCell UMAP
        st.subheader("Figure 1a: AUCell Enrichment UMAP")
        _fig_1a_n_cells = int(len(aucell_per_cell_df))
        _fig_1a_filter_clause = (
            f" restricted to clusters passing the baseline {sanity_gene} "
            f"≥ {sanity_baseline_mean_expr:.2f} filter"
            if baseline_allowed is not None else ""
        )
        st.markdown(
            f"**Figure 1a.** AUCell enrichment score for the bacTRAP signature "
            f"projected onto the HypoMap UMAP embedding"
            f"{_fig_1a_filter_clause} "
            f"(n = {_fig_1a_n_cells:,} cells). For each cell, the area under the "
            f"recovery curve for the top "
            f"**{len(top_enriched_genes)}** π-score-ranked bacTRAP-enriched "
            f"genes was computed within the top "
            f"**{aucell_top_fraction:.0%}** of genes by expression rank in "
            f"that cell and normalised to its theoretical maximum, so values "
            f"lie on [0, 1]. Colour encodes AUCell score (magma colormap); "
            f"the scale is clipped at the 2nd and 98th percentiles to "
            f"suppress outlier saturation. Bottom-left arrows mark the "
            f"UMAP1 / UMAP2 axes. See Methods (*AUCell scoring*) for the "
            f"full derivation."
        )
        fig_1a = figure_aucell_umap(
            umap_coords, aucell_scores,
            double_column=double_column,
            subsample_idx=sub_indices_filtered,
            seed=int(empirical_null_seed),
        )
        st.pyplot(fig_1a)
        _cache_fig("fig_1a_aucell_umap", fig_1a)

        col_pdf, col_svg, col_csv = st.columns(3)
        with col_pdf:
            st.download_button(
                "Download PDF",
                st.session_state.fig_bytes["fig_1a_aucell_umap"]["pdf"],
                "fig_1a_aucell_umap.pdf", "application/pdf",
                key="dl_fig_1a_pdf",
            )
        with col_svg:
            st.download_button(
                "Download SVG",
                st.session_state.fig_bytes["fig_1a_aucell_umap"]["svg"],
                "fig_1a_aucell_umap.svg", "image/svg+xml",
                key="dl_fig_1a_svg",
            )
        with col_csv:
            st.download_button(
                "Download CSV (per-cell)",
                st.session_state.table_bytes["aucell_per_cell"],
                _filtered_name("aucell_per_cell.csv"), "text/csv",
                key="dl_fig_1a_csv",
                help=(
                    "cell_id, cluster, aucell_score — filtered to "
                    "clusters passing the baseline filter when active."
                ),
            )
        plt.close(fig_1a)

        # Figure 1b: Cell-type annotation UMAP (top-15 AUCell clusters)
        st.subheader("Figure 1b: Cell-type Annotation UMAP (top-15 AUCell clusters)")
        _filter_clause_1b = (
            f" and passing the baseline {sanity_gene} filter "
            f"(mean ≥ {sanity_baseline_mean_expr:.2f})"
            if baseline_allowed is not None else ""
        )
        st.markdown(
            f"**Figure 1b.** Same UMAP layout as (a), coloured by HypoMap "
            f"cell-type annotation (`{annotation_col}`). Only the 15 "
            f"clusters with the highest mean AUCell score (among clusters "
            f"with ≥ **{min_cells_for_rank}** cells{_filter_clause_1b}) "
            f"are drawn in colour and overplotted on a light-grey "
            f"background of all remaining cells; this keeps small but "
            f"highly enriched populations visible while conveying overall "
            f"atlas topology. The legend lists the highlighted clusters "
            f"in rank order (top entry = highest mean AUCell). The 20-cell "
            f"floor excludes small populations whose cluster mean is "
            f"dominated by shrinkage variance and would otherwise claim "
            f"top slots by chance — they remain in `aucell_per_cluster.csv`."
        )
        _size_filtered_ranked = (
            aucell_per_cluster_df[
                aucell_per_cluster_df["n_cells"] >= min_cells_for_rank
            ]
            .sort_values("mean", ascending=False)
        )
        _top15_aucell_clusters_ct = (
            _size_filtered_ranked.head(15)["cluster"].astype(str).tolist()
        )
        fig_1a_ct = figure_celltype_umap(
            umap_coords=umap_coords,
            cell_labels=cell_labels,
            highlight_clusters=_top15_aucell_clusters_ct,
            double_column=double_column,
            subsample_idx=sub_indices,
            seed=int(empirical_null_seed),
        )
        st.pyplot(fig_1a_ct)
        _cache_fig("fig_1a_celltype_umap", fig_1a_ct)

        col_pdf, col_svg, col_csv = st.columns(3)
        with col_pdf:
            st.download_button(
                "Download PDF",
                st.session_state.fig_bytes["fig_1a_celltype_umap"]["pdf"],
                "fig_1a_celltype_umap.pdf", "application/pdf",
                key="dl_fig_1a_ct_pdf",
            )
        with col_svg:
            st.download_button(
                "Download SVG",
                st.session_state.fig_bytes["fig_1a_celltype_umap"]["svg"],
                "fig_1a_celltype_umap.svg", "image/svg+xml",
                key="dl_fig_1a_ct_svg",
            )
        with col_csv:
            st.download_button(
                "Download CSV (per-cluster mean)",
                st.session_state.table_bytes["aucell_per_cluster"],
                _filtered_name("aucell_per_cluster.csv"), "text/csv",
                key="dl_fig_1a_ct_csv",
                help="Top-15 rows are the highlighted clusters in this panel.",
            )
        plt.close(fig_1a_ct)

        # Figure 2: Pnoc expression across the preoptic area
        # Always computed on the FULL atlas (independent of the sidebar region
        # restriction) so the grey backdrop shows whole-atlas topology like
        # fig_1a. Uses the shared figure2_inputs + figures.figure_gene_poa_umap
        # code path, so this panel is identical to the standalone
        # figure2_pnoc_poa.py output.
        FIG2_GENE = "Pnoc"
        FIG2_KEYWORDS = ("preoptic",)
        FIG2_REGION_LABEL = "preoptic"
        FIG2_MIN_CELLS = 20
        # Genes shown as columns in the composite dot plot (Pnoc kept first so
        # the panel leads with the driver gene).
        FIG2_DOTPLOT_GENES = [
            FIG2_GENE, "Gal", "Bdnf", "Adcyap1", "Slc17a6", "Sntg2", "Lepr",
        ]
        st.subheader(f"Figure 2: {FIG2_GENE} expression across the preoptic area")
        _fig2_top_n = st.number_input(
            "Max POA Pnoc-expressing clusters to highlight",
            min_value=1, max_value=25, value=20, step=1, key="fig2_top_n",
            help="All preoptic clusters with detectable Pnoc (Table S1) are "
                 "highlighted, up to this many.",
        )
        _fig2_hit = resolve_gene_index(adata, FIG2_GENE)
        _fig2_ready = False
        if _fig2_hit is None:
            st.warning(
                f"`{FIG2_GENE}` not found in the atlas — Figure 2 skipped."
            )
        else:
            _fig2_idx, _fig2_disp, _fig2_use_raw = _fig2_hit
            _fig2_umap_key = "X_umap" if "X_umap" in adata.obsm else "X_UMAP"
            _fig2_umap = adata.obsm[_fig2_umap_key]
            _fig2_labels = adata.obs[annotation_col].astype(str).values
            _fig2_region = region_cell_mask(adata, keywords=FIG2_KEYWORDS)
            _fig2_expr, _fig2_expressing, _fig2_ranking, _fig2_highlight = (
                gene_poa_inputs(
                    adata, _fig2_idx, _fig2_use_raw, _fig2_region, _fig2_labels,
                    min_cells=FIG2_MIN_CELLS, top_n=int(_fig2_top_n),
                )
            )
            # Per-gene expressing masks for the composite dot plot columns.
            _dot_names, _dot_expressing, _dot_missing = gene_expressing_masks(
                adata, FIG2_DOTPLOT_GENES,
            )
            if _dot_missing:
                st.caption(
                    "Dot plot: genes not found in the atlas and skipped — "
                    + ", ".join(_dot_missing)
                )
            # Subsample the backdrop to the same density as fig_1a (same
            # umap_subsample count and seed=42), so the two figures match.
            _fig2_sub = None
            if adata.n_obs > umap_subsample:
                _fig2_rng = np.random.default_rng(42)
                _fig2_sub = np.sort(_fig2_rng.choice(
                    adata.n_obs, size=umap_subsample, replace=False))
            st.markdown(
                f"**Figure 2.** Full-atlas UMAP (same layout as 1a). "
                f"**Left:** the {len(_fig2_highlight)} preoptic clusters with "
                f"detectable *{FIG2_GENE}* (≥ {FIG2_MIN_CELLS} cells; the "
                f"POA-resident {FIG2_GENE}-positive clusters of Table S1, "
                f"spanning MPA / LPO / periventricular) drawn in colour over a "
                f"light-grey backdrop of all atlas cells. **Right:** "
                f"*{FIG2_GENE}* expression of the preoptic cells (non-expressing "
                f"cells join the grey backdrop). Preoptic = `Region_summarized` "
                f"containing '{FIG2_KEYWORDS[0]}' "
                f"(n = {int(_fig2_region.sum()):,} cells)."
            )
            fig_2 = figure_gene_poa_umap(
                umap_coords=_fig2_umap,
                cell_labels=_fig2_labels,
                region_mask=_fig2_region,
                gene_expr=_fig2_expr,
                expressing_mask=_fig2_expressing,
                highlight_clusters=_fig2_highlight,
                gene_name=FIG2_GENE,
                region_label=FIG2_REGION_LABEL,
                double_column=double_column,
                subsample_idx=_fig2_sub,
            )
            st.pyplot(fig_2)
            _cache_fig("fig_2_pnoc_poa", fig_2)
            st.session_state.table_bytes["fig_2_pnoc_poa_clusters"] = (
                _fig2_ranking.to_csv().encode()
            )

            _f2_pdf, _f2_svg, _f2_csv = st.columns(3)
            with _f2_pdf:
                st.download_button(
                    "Download PDF",
                    st.session_state.fig_bytes["fig_2_pnoc_poa"]["pdf"],
                    "fig2_pnoc_poa.pdf", "application/pdf",
                    key="dl_fig_2_pdf",
                )
            with _f2_svg:
                st.download_button(
                    "Download SVG",
                    st.session_state.fig_bytes["fig_2_pnoc_poa"]["svg"],
                    "fig2_pnoc_poa.svg", "image/svg+xml",
                    key="dl_fig_2_svg",
                )
            with _f2_csv:
                st.download_button(
                    "Download CSV (per-cluster ranking)",
                    st.session_state.table_bytes["fig_2_pnoc_poa_clusters"],
                    "fig2_pnoc_poa_clusters.csv", "text/csv",
                    key="dl_fig_2_csv",
                    help=f"Preoptic clusters ranked by mean {FIG2_GENE}.",
                )
            plt.close(fig_2)
            _fig2_ready = True

        st.markdown("---")
        st.caption(
            "Panels below are supplementary to the main figure (1a–c); they "
            "are published as supplementary figures in the Methods document."
        )

        # Supplementary S2: AUCell Cluster Barplot (was Figure 1b)
        st.subheader("Supplementary Figure S2: Mean AUCell Score per Cluster")
        _filter_clause_s2 = (
            f" and also filtered to clusters with mean {sanity_gene} ≥ "
            f"{sanity_baseline_mean_expr:.2f}"
            if baseline_allowed is not None else ""
        )
        st.markdown(
            f"**Supplementary Figure S2.** Horizontal barplot of mean AUCell "
            f"score per HypoMap cluster (top 25 by mean, clusters with "
            f"< **{min_cells_for_rank}** cells excluded from ranking"
            f"{_filter_clause_s2}). Error bars are standard error of the "
            f"mean. Bar colour encodes the cluster mean (magma colormap). "
            f"Source table: `aucell_per_cluster.csv`."
        )
        fig_1b = figure_aucell_cluster_barplot(
            aucell_scores, cell_labels,
            top_n=25, double_column=double_column,
            min_cluster_cells=min_cells_for_rank,
            allowed_clusters=_aucell_allowed,
        )
        st.pyplot(fig_1b)
        _cache_fig("fig_1b_aucell_barplot", fig_1b)

        col_pdf, col_svg, col_csv = st.columns(3)
        with col_pdf:
            st.download_button(
                "Download PDF",
                st.session_state.fig_bytes["fig_1b_aucell_barplot"]["pdf"],
                "fig_1b_aucell_barplot.pdf", "application/pdf",
                key="dl_fig_1b_pdf",
            )
        with col_svg:
            st.download_button(
                "Download SVG",
                st.session_state.fig_bytes["fig_1b_aucell_barplot"]["svg"],
                "fig_1b_aucell_barplot.svg", "image/svg+xml",
                key="dl_fig_1b_svg",
            )
        with col_csv:
            st.download_button(
                "Download CSV (per-cluster)",
                st.session_state.table_bytes["aucell_per_cluster"],
                _filtered_name("aucell_per_cluster.csv"), "text/csv",
                key="dl_fig_1b_csv",
                help=(
                    "cluster, n_cells, mean, median, std, sem, t_stat, pvalue, "
                    "qvalue, significant — sorted by mean descending. Significance "
                    "is Welch's one-sided t (cluster > rest) with BH-FDR."
                ),
            )
        plt.close(fig_1b)

        # Figure 1c: AUCell Violin Plots
        st.subheader("Figure 1c: AUCell Score Distributions (Top-15 Clusters)")
        _filter_clause_1c = (
            f" Top-15 is taken over clusters with mean {sanity_gene} ≥ "
            f"{sanity_baseline_mean_expr:.2f}."
            if baseline_allowed is not None else ""
        )
        st.markdown(
            f"**Figure 1c.** Violin plots of the full AUCell score "
            f"distribution within each of the 15 top-ranked clusters from "
            f"(b), ordered from highest (top) to lowest (bottom) cluster "
            f"mean. The short **solid black** vertical bar inside each "
            f"violin marks the cluster mean; the **dashed grey** bar marks "
            f"the cluster median (the two nearly coincide when the "
            f"distribution is symmetric, in which case they read as a "
            f"single I-shape — see the on-figure legend). Violin fill "
            f"colour encodes the cluster mean (magma colormap). "
            f"Per-cluster means, medians, SEMs and Welch's one-sided "
            f"*t*-test *p*/*q*-values against the rest of the atlas are "
            f"available in `aucell_per_cluster.csv`.{_filter_clause_1c}"
        )
        fig_1c = figure_aucell_violins(
            aucell_scores, cell_labels,
            top_n=15, double_column=True,
            min_cluster_cells=min_cells_for_rank,
            allowed_clusters=_aucell_allowed,
        )
        st.pyplot(fig_1c)
        _cache_fig("fig_1c_aucell_violins", fig_1c)

        col_pdf, col_svg, col_mean, col_cell = st.columns(4)
        with col_pdf:
            st.download_button(
                "Download PDF",
                st.session_state.fig_bytes["fig_1c_aucell_violins"]["pdf"],
                "fig_1c_aucell_violins.pdf", "application/pdf",
                key="dl_fig_1c_pdf",
            )
        with col_svg:
            st.download_button(
                "Download SVG",
                st.session_state.fig_bytes["fig_1c_aucell_violins"]["svg"],
                "fig_1c_aucell_violins.svg", "image/svg+xml",
                key="dl_fig_1c_svg",
            )
        with col_mean:
            st.download_button(
                "Download CSV (per-cluster mean)",
                st.session_state.table_bytes["aucell_per_cluster"],
                _filtered_name("aucell_per_cluster.csv"), "text/csv",
                key="dl_fig_1c_mean_csv",
                help=(
                    "Plotted quantities: cluster, n_cells, mean (black bar), "
                    "median (grey dashed bar), std, sem, plus Welch's t-test "
                    "significance columns (t_stat, pvalue, qvalue, significant). "
                    "Sorted by mean descending — the top 15 rows are the "
                    "clusters shown in the violin."
                ),
            )
        with col_cell:
            st.download_button(
                "Download CSV (per-cell)",
                st.session_state.table_bytes["aucell_per_cell"],
                "aucell_per_cell.csv", "text/csv",
                key="dl_fig_1c_csv",
                help="Per-cell AUCell scores — use to reconstruct the full violin shape.",
            )
        plt.close(fig_1c)

        # ---- Composite overview: Figure 2 + fig_1a + fig_1c ----
        if _fig2_ready:
            st.subheader(
                "Composite: Pnoc preoptic UMAPs + cell-type UMAP + AUCell violins"
            )
            st.markdown(
                "Five-panel overview. **Top:** the two Figure 2 panels (top "
                "*Pnoc* preoptic clusters and *Pnoc* expression). **Middle-left:** "
                "a dot plot of the fraction of each cluster's preoptic cells "
                "expressing " + ", ".join(f"*{g}*" for g in _dot_names) + " "
                "(dot size = % expressing, colour = cluster). **Bottom:** "
                "`fig_1a_celltype_umap` (top-15 AUCell clusters) and "
                "`fig_1c_aucell_violins`. All panels share the same "
                "rendering and equal dimensions."
            )
            _overview_font_scale = st.slider(
                "Composite font size", min_value=1.0, max_value=2.0,
                value=1.4, step=0.1, key="overview_font_scale",
                help="Uniformly scales all text (labels, ticks, legends, "
                     "colorbars) in the composite. 1.0 = the standalone "
                     "figures' native size.",
            )
            fig_overview = figure_pnoc_overview(
                fig2_umap=_fig2_umap,
                fig2_labels=_fig2_labels,
                fig2_region_mask=_fig2_region,
                fig2_gene_expr=_fig2_expr,
                fig2_expressing_mask=_fig2_expressing,
                fig2_highlight=_fig2_highlight,
                fig2_subsample_idx=_fig2_sub,
                dotplot_gene_names=_dot_names,
                dotplot_expressing=_dot_expressing,
                ct_highlight=_top15_aucell_clusters_ct,
                aucell_scores=aucell_scores,
                view_labels=cell_labels,
                gene_name=FIG2_GENE,
                region_label=FIG2_REGION_LABEL,
                double_column=double_column,
                violin_top_n=15,
                violin_min_cluster_cells=min_cells_for_rank,
                violin_allowed_clusters=_aucell_allowed,
                font_scale=_overview_font_scale,
            )
            st.pyplot(fig_overview)
            _cache_fig("fig_pnoc_overview", fig_overview)
            _ov_pdf, _ov_svg = st.columns(2)
            with _ov_pdf:
                st.download_button(
                    "Download PDF",
                    st.session_state.fig_bytes["fig_pnoc_overview"]["pdf"],
                    "fig_pnoc_overview.pdf", "application/pdf",
                    key="dl_fig_overview_pdf",
                )
            with _ov_svg:
                st.download_button(
                    "Download SVG",
                    st.session_state.fig_bytes["fig_pnoc_overview"]["svg"],
                    "fig_pnoc_overview.svg", "image/svg+xml",
                    key="dl_fig_overview_svg",
                )
            plt.close(fig_overview)

        # ---- Empirical-null companion violins ----
        if _empirical_null_active and "z_empirical" in aucell_per_cluster_df.columns:
            _n_used = int(empirical_null_df["n_control_sets_used"].iloc[0])
            st.subheader("Figure 1c (companion): Top-15 by empirical z-score")
            st.markdown(
                f"**Top 15 by empirical z-score (matched-expression null).** "
                f"Same per-cluster AUCell distributions as above, but ranked "
                f"by `z_empirical` — (cluster mean − mean of {_n_used} "
                f"expression-matched random control gene-set means) / their SD "
                f"(Aibar et al. 2017). Re-orders clusters whose AUCell is "
                f"inflated by a wide gene-rank distribution rather than by true "
                f"bacTRAP-signature enrichment. Annotations show z; colour "
                f"encodes z (viridis). Source: the `z_empirical`, "
                f"`pvalue_empirical`, `qvalue_empirical`, `null_mean`, "
                f"`null_sd` columns in `aucell_per_cluster.csv`."
            )
            _z_series = aucell_per_cluster_df.set_index("cluster")["z_empirical"]
            fig_1c_z = figure_aucell_zscore_violins(
                aucell_scores, cell_labels, _z_series,
                top_n=15, double_column=True,
                min_cluster_cells=min_cells_for_rank,
                allowed_clusters=_aucell_allowed,
            )
            st.pyplot(fig_1c_z)
            _cache_fig("fig_1c_aucell_zscore_violins", fig_1c_z)
            col_pdf, col_svg = st.columns(2)
            with col_pdf:
                st.download_button(
                    "Download PDF",
                    st.session_state.fig_bytes["fig_1c_aucell_zscore_violins"]["pdf"],
                    "fig_1c_aucell_zscore_violins.pdf", "application/pdf",
                    key="dl_fig_1c_z_pdf",
                )
            with col_svg:
                st.download_button(
                    "Download SVG",
                    st.session_state.fig_bytes["fig_1c_aucell_zscore_violins"]["svg"],
                    "fig_1c_aucell_zscore_violins.svg", "image/svg+xml",
                    key="dl_fig_1c_z_svg",
                )
            plt.close(fig_1c_z)

        # Supplementary S3: AUCell Score Histogram (was Figure 1d)
        st.subheader("Supplementary Figure S3: Global AUCell Score Distribution")
        st.markdown(
            "**Supplementary Figure S3.** Global histogram of per-cell "
            "AUCell scores across the atlas. Dashed vertical lines mark "
            "the 90th, 95th and 99th percentiles as well as the mean. "
            "Cells above the 95th percentile form the candidate pool "
            "for belonging to the bacTRAP target population."
        )
        fig_1d = figure_aucell_histogram(
            aucell_scores, double_column=double_column,
        )
        st.pyplot(fig_1d)
        _cache_fig("fig_1d_aucell_histogram", fig_1d)

        col_pdf, col_svg, col_csv = st.columns(3)
        with col_pdf:
            st.download_button(
                "Download PDF",
                st.session_state.fig_bytes["fig_1d_aucell_histogram"]["pdf"],
                "fig_1d_aucell_histogram.pdf", "application/pdf",
                key="dl_fig_1d_pdf",
            )
        with col_svg:
            st.download_button(
                "Download SVG",
                st.session_state.fig_bytes["fig_1d_aucell_histogram"]["svg"],
                "fig_1d_aucell_histogram.svg", "image/svg+xml",
                key="dl_fig_1d_svg",
            )
        with col_csv:
            st.download_button(
                "Download CSV (per-cell)",
                st.session_state.table_bytes["aucell_per_cell"],
                "aucell_per_cell.csv", "text/csv",
                key="dl_fig_1d_csv",
                help="Per-cell AUCell scores — source data for the histogram.",
            )
        plt.close(fig_1d)

    # ======================================================================
    # TAB: Cre-driver Sanity Check
    # ======================================================================
    with tab_sanity:
        st.header(f"Cre-driver Sanity Check: {sanity_gene}")
        st.markdown(
            f"In a **{sanity_gene}-Cre;NuTRAP** experiment we'd expect the "
            f"top-ranked HypoMap clusters to express *{sanity_gene}*. This panel "
            f"shows mean expression and fraction of cells expressing "
            f"*{sanity_gene}* across the top-ranked clusters by AUCell mean."
        )
        st.info(
            "**Caveats — read before interpreting:** "
            "(1) Cre-lox is permanent lineage tracing — any cell that ever "
            f"expressed *{sanity_gene}* during development is labelled, even "
            f"if current mRNA is undetectable. (2) HypoMap is single-nucleus "
            f"data; neuropeptides like *{sanity_gene}* are notoriously prone "
            "to dropout. (3) Atlas expression is a snapshot; *Pnoc* is "
            "state-dependent (feeding, stress, estrous). Use this as a "
            "**confidence weight**, not a hard filter."
        )

        # sanity_stats is computed up-front (see post-cache block earlier).
        # The tab just consumes the already-cached result.
        if sanity_stats is None or len(sanity_stats) == 0:
            st.error(
                f"**`{sanity_gene}`** was not found in the HypoMap atlas. "
                "Check the spelling and capitalization (mouse symbols are "
                "title-cased: `Pnoc`, not `PNOC` or `pnoc`)."
            )
        else:
            # ---- Cluster ordering: AUCell mean, fall back to empirical
            # z-score, then Cre-driver expression ----
            _filter_suffix = (
                f" (after {sanity_gene} ≥ {sanity_baseline_mean_expr:.2f} filter)"
                if baseline_allowed is not None else ""
            )
            _aucell_ranked = (
                aucell_per_cluster_df.sort_values("mean", ascending=False)["cluster"]
                .astype(str).tolist()
                if aucell_per_cluster_df is not None and len(aucell_per_cluster_df) > 0
                else []
            )
            _z_ranked = (
                aucell_per_cluster_df.dropna(subset=["z_empirical"]).sort_values(
                    "z_empirical", ascending=False
                )["cluster"].astype(str).tolist()
                if (
                    aucell_per_cluster_df is not None
                    and "z_empirical" in aucell_per_cluster_df.columns
                ) else []
            )
            if _aucell_ranked:
                ranked_clusters = _aucell_ranked
                rank_source = f"AUCell mean per cluster{_filter_suffix}"
            elif _z_ranked:
                ranked_clusters = _z_ranked
                rank_source = f"AUCell empirical z-score{_filter_suffix}"
            else:
                ranked_clusters = sanity_stats.sort_values(
                    "mean_expr", ascending=False,
                ).index.tolist()
                rank_source = f"{sanity_gene} expression (no AUCell ranking available)"

            top_n_sanity = st.slider(
                "Top N clusters to display", 5, 50, 20, 1,
                key="sanity_top_n",
                help=(
                    "How many of the top AUCell-ranked clusters to show on "
                    "the Cre-driver sanity panel. Widens or narrows the "
                    "barplot; does not change any computation."
                ),
            )
            top_clusters_sanity = ranked_clusters[:top_n_sanity]

            # Build the merged display table
            sanity_table = sanity_stats.loc[
                [c for c in top_clusters_sanity if c in sanity_stats.index]
            ].copy()
            # Add rank column from whichever ranking source we used
            sanity_table["rank"] = [
                ranked_clusters.index(c) + 1 if c in ranked_clusters else np.nan
                for c in sanity_table.index
            ]
            sanity_table = sanity_table.sort_values("rank")
            sanity_table["passes_threshold"] = (
                sanity_table["fraction_expressing"] >= sanity_fraction_threshold
            )

            # ---- Summary metric ----
            n_pass = int(sanity_table["passes_threshold"].sum())
            n_total = len(sanity_table)
            atlas_median_frac = float(sanity_stats["fraction_expressing"].median())

            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric(
                    f"Top-{n_total} clusters expressing {sanity_gene}",
                    f"{n_pass} / {n_total}",
                    help=(
                        f"Clusters where ≥{sanity_fraction_threshold*100:.0f}% "
                        "of cells have non-zero counts for the Cre-driver gene."
                    ),
                )
            with mcol2:
                st.metric(
                    f"Atlas-wide median fraction expressing",
                    f"{atlas_median_frac*100:.1f}%",
                    help=(
                        "Median across ALL clusters in the atlas — useful "
                        "baseline for judging dropout."
                    ),
                )
            with mcol3:
                top_frac = float(sanity_table["fraction_expressing"].max()) if n_total else 0.0
                st.metric(
                    f"Highest fraction in top-{n_total}",
                    f"{top_frac*100:.1f}%",
                )

            st.caption(f"Cluster ordering: **{rank_source}**.")

            # ---- Diagnostic figure ----
            st.subheader(f"Figure: {sanity_gene} expression across top-ranked clusters")
            fig_sanity = figure_marker_gene_diagnostic(
                sanity_stats, top_clusters_sanity,
                gene_name=sanity_gene,
                fraction_threshold=sanity_fraction_threshold,
                double_column=True,
            )
            st.pyplot(fig_sanity)
            _cache_fig("fig_sanity_check", fig_sanity)

            col_pdf, col_svg = st.columns(2)
            with col_pdf:
                st.download_button(
                    "Download PDF",
                    st.session_state.fig_bytes["fig_sanity_check"]["pdf"],
                    f"fig_sanity_{sanity_gene.lower()}.pdf", "application/pdf",
                    key="dl_fig_sanity_pdf",
                )
            with col_svg:
                st.download_button(
                    "Download SVG",
                    st.session_state.fig_bytes["fig_sanity_check"]["svg"],
                    f"fig_sanity_{sanity_gene.lower()}.svg", "image/svg+xml",
                    key="dl_fig_sanity_svg",
                )
            plt.close(fig_sanity)

            # ---- Detailed table ----
            st.subheader(f"Per-cluster {sanity_gene} expression (top {n_total})")

            display_table = sanity_table[[
                "rank", "mean_expr", "fraction_expressing", "passes_threshold",
            ]].copy()
            display_table.columns = [
                "Composite rank", f"Mean {sanity_gene} (log-norm)",
                f"Fraction expressing {sanity_gene}",
                f"≥ {sanity_fraction_threshold*100:.0f}% threshold",
            ]

            def _highlight_fail(row):
                col = f"≥ {sanity_fraction_threshold*100:.0f}% threshold"
                if col in row.index and not row[col]:
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_table.style.apply(_highlight_fail, axis=1).format({
                    "Composite rank": "{:.0f}",
                    f"Mean {sanity_gene} (log-norm)": "{:.3f}",
                    f"Fraction expressing {sanity_gene}": "{:.1%}",
                }),
                use_container_width=True,
            )

            st.caption(
                "Highlighted rows = top-ranked clusters that **do not** pass "
                "the expression threshold. Investigate these: developmental "
                "lineage tracing, dropout, or potential artefact."
            )

            # ---- Download full table ----
            full_table = sanity_stats.copy()
            full_table["rank_in_mapping"] = [
                ranked_clusters.index(c) + 1 if c in ranked_clusters else np.nan
                for c in full_table.index
            ]
            full_table = full_table.reset_index()
            st.download_button(
                f"Download full {sanity_gene} per-cluster table (CSV)",
                full_table.to_csv(index=False).encode(),
                f"sanity_check_{sanity_gene.lower()}.csv", "text/csv",
                key="dl_sanity_csv",
            )


    # ======================================================================
    # TAB 4: UMAP Projection (Supplementary)
    # ======================================================================
    with tab4:
        st.header("Supplementary: UMAP AUCell Projection")
        st.markdown(
            f"AUCell enrichment score projected onto the HypoMap UMAP "
            f"alongside the cell-type annotation for side-by-side comparison. "
            f"The left panel highlights the top-15 AUCell-ranked clusters "
            f"(matching main figures 1b/1c); all other cells are greyed out. "
            f"Score computed from the top **{len(top_enriched_genes)}** enriched "
            f"genes (padj < {padj_cutoff}, log₂FC > {log2fc_cutoff})."
        )

        st.subheader("Supplementary Figure S4: UMAP AUCell Map")
        _top15_aucell_clusters = (
            aucell_per_cluster_df[
                aucell_per_cluster_df["n_cells"] >= min_cells_for_rank
            ]
            .sort_values("mean", ascending=False)
            .head(15)["cluster"].astype(str).tolist()
        )
        fig_b = figure_umap_enrichment(
            umap_coords=umap_coords,
            cell_labels=cell_labels,
            enrichment_scores=aucell_scores,
            double_column=True,
            point_size=0.3,
            subsample_idx=sub_indices,
            score_title="AUCell enrichment score",
            score_label="AUCell score",
            highlight_clusters=_top15_aucell_clusters,
            seed=int(empirical_null_seed),
        )
        st.pyplot(fig_b)
        _cache_fig("fig_s4_umap_enrichment", fig_b)

        col_pdf, col_svg = st.columns(2)
        with col_pdf:
            st.download_button(
                "Download PDF",
                st.session_state.fig_bytes["fig_s4_umap_enrichment"]["pdf"],
                "fig_s4_umap.pdf", "application/pdf",
                key="dl_fig_s4_pdf",
            )
        with col_svg:
            st.download_button(
                "Download SVG",
                st.session_state.fig_bytes["fig_s4_umap_enrichment"]["svg"],
                "fig_s4_umap.svg", "image/svg+xml",
                key="dl_fig_s4_svg",
            )
        plt.close(fig_b)





    # ======================================================================
    # TAB: Export
    # ======================================================================
    with tab_export:
        st.header("Export All Results")

        st.subheader("Figures")
        st.markdown("Download all figures as a ZIP archive (PDF + SVG).")

        # Build ZIP from cached figure bytes (no regeneration needed)
        cached_bytes = st.session_state.get("fig_bytes", {})
        cached_tables = st.session_state.get("table_bytes", {})
        if cached_bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, fmt_dict in cached_bytes.items():
                    zf.writestr(f"{name}.pdf", fmt_dict["pdf"])
                    zf.writestr(f"{name}.svg", fmt_dict["svg"])
                # Include AUCell raw-data tables alongside the figures.
                # Skip private metadata entries (e.g. cache signatures) that
                # share this dict but aren't serialised CSV payloads.
                for tbl_name, tbl_bytes in cached_tables.items():
                    if tbl_name.startswith("_") or not isinstance(
                        tbl_bytes, (bytes, bytearray, memoryview)
                    ):
                        continue
                    zf.writestr(f"{tbl_name}.csv", tbl_bytes)
                # Include log file (best-effort; skip if unreadable).
                if _LOG_FILE.is_file():
                    try:
                        zf.writestr("aucell.log", _read_log_for_export(_LOG_FILE))
                    except OSError:
                        logger.exception(
                            "Failed to include log file in export ZIP",
                        )
            buf.seek(0)

            st.download_button(
                "Download All Figures (ZIP)",
                buf.getvalue(),
                "aucell_figures.zip",
                "application/zip",
                use_container_width=True,
                key="dl_all_figs_zip",
            )
        else:
            st.info("Run the analysis first to generate figures.")

        st.markdown("---")
        st.subheader("Tables")

        col_t3, col_t4 = st.columns(2)
        with col_t3:
            st.download_button(
                "Matched genes (CSV)",
                bactrap_matched.to_csv(index=False).encode(),
                "matched_genes.csv", "text/csv",
                key="dl_matched_csv",
            )
        with col_t4:
            if len(enriched_df) > 0:
                st.download_button(
                    "Enriched genes (CSV)",
                    enriched_df.to_csv(index=False).encode(),
                    "enriched_genes.csv", "text/csv",
                    key="dl_enriched_csv",
                )

        if signature_refinement_active and isinstance(sig_drop_log, pd.DataFrame) and len(sig_drop_log):
            st.download_button(
                "Signature refinement drop-log (CSV)",
                sig_drop_log.to_csv(index=False).encode(),
                f"signature_refinement_droplog_{_filter_signature_str}.csv", "text/csv",
                key="dl_sig_refine_droplog_export",
            )

        # AUCell raw data — pre-encoded in table_bytes (populated during run)
        aucell_table_bytes = st.session_state.get("table_bytes", {})
        if "aucell_per_cell" in aucell_table_bytes:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.download_button(
                    "AUCell per-cell scores (CSV)",
                    aucell_table_bytes["aucell_per_cell"],
                    _filtered_name("aucell_per_cell.csv"), "text/csv",
                    key="dl_aucell_per_cell_export",
                    help=(
                        "cell_id, cluster, aucell_score — raw data for figures 1a, 1c and "
                        "supplementary S3. Restricted to clusters passing the baseline "
                        "filter when active."
                    ),
                )
            with col_a2:
                st.download_button(
                    "AUCell per-cluster summary (CSV)",
                    aucell_table_bytes["aucell_per_cluster"],
                    _filtered_name("aucell_per_cluster.csv"), "text/csv",
                    key="dl_aucell_per_cluster_export",
                    help=(
                        "cluster, n_cells, mean, median, std, sem, t_stat, pvalue, "
                        "qvalue, significant — raw data for figures 1b/1c and supplementary S2."
                    ),
                )

        st.markdown("---")
        st.subheader("Diagnostics")
        if _LOG_FILE.is_file():
            try:
                _log_bytes = _read_log_for_export(_LOG_FILE).encode()
            except OSError as e:
                # File exists but can't be read (permissions, lock, disk
                # fault) — surface the reason rather than handing the user
                # an empty download.
                logger.exception("Failed to read log file %s", _LOG_FILE)
                st.info(
                    f"Log file exists at `{_LOG_FILE}` but could not be "
                    f"read: {e}. Check file permissions."
                )
            else:
                st.download_button(
                    "Download log file",
                    _log_bytes,
                    "aucell.log", "text/plain",
                    use_container_width=True,
                    key="dl_log_file",
                )
        else:
            st.info("No log file generated yet.")

else:
    st.info("Click **Run Analysis** in the sidebar to start.")
