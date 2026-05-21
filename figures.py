"""
Nature-grade publication figure generation for AUCell-based mapping of a
bulk-RNA-seq DE signature onto a single-cell atlas.

All figures follow Nature journal specifications:
- Sans-serif font (Arial/Helvetica), editable text in PDF/SVG
- Axis labels 7pt, tick labels 6pt, panel titles 8pt bold
- Line width 0.5pt axes, 0.75pt plot elements
- Single column 89mm (3.5in) or double column 183mm (7.2in)
- 300 DPI for rasterized elements
- Colorblind-friendly palettes
- White background, no gridlines, no top/right spines
"""

import io
import logging
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import seaborn as sns
from typing import Optional, List, Dict, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------

def setup_nature_style():
    """Configure matplotlib for Nature-grade figures.

    Called at the top of every figure function rather than once at module
    import time. This is deliberate: users (e.g. in a Jupyter notebook)
    may have their own rcParams for other plots interleaved with these,
    and we want each figure function to produce a consistent Nature-
    style output regardless of ambient state. The cost is a handful of
    dict updates per figure — negligible compared to the figure work
    itself.

    Do NOT rely on rcParams persisting between calls; treat each figure
    function as if it owns the rcParams for its duration. If you need
    truly scoped styling, wrap your call in ``plt.rc_context(...)``
    around the figure function — but be aware that calling
    ``setup_nature_style()`` inside the function will override the
    context for its body.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.75,
        "patch.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,  # TrueType fonts in PDF (editable)
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # editable text in SVG
        # Do NOT enable constrained_layout globally — it conflicts with
        # bbox_to_anchor legends and manual colorbar pad placement.
        "figure.constrained_layout.use": False,
    })


def get_figure_width(double_column: bool = False) -> float:
    """Return figure width in inches for Nature format."""
    return 7.2 if double_column else 3.5


def get_qualitative_palette(n: int) -> List[str]:
    """Return a colorblind-friendly qualitative palette."""
    if n <= 10:
        return list(sns.color_palette("tab10", n).as_hex())
    elif n <= 20:
        return list(sns.color_palette("tab20", n).as_hex())
    else:
        base = list(sns.color_palette("tab20", 20).as_hex())
        extra = list(sns.color_palette("Set3", min(n - 20, 12)).as_hex())
        return (base + extra)[:n]


def _add_umap_axis_arrows(
    ax,
    x_label: str = "UMAP1",
    y_label: str = "UMAP2",
    length: float = 0.07,
    origin: tuple = (0.02, 0.02),
    linewidth: float = 0.5,
    fontsize: float = 4.5,
    mutation_scale: Optional[float] = None,
) -> None:
    """Draw two small axis arrows in the bottom-left corner of a UMAP panel.

    Replaces the conventional x/y axes on dimensionality-reduction plots with
    the compact convention common in single-cell publications: two arrows
    anchored at the bottom-left, labelled "UMAP1" / "UMAP2". *length* and
    *origin* are expressed in axes fraction coordinates, so the arrows scale
    with the panel and stay in the corner regardless of data range.

    Call after all data have been plotted so annotations sit on top.
    """
    x0, y0 = origin
    arrow_style = dict(
        arrowstyle="-|>,head_length=1.5,head_width=1.0",
        linewidth=linewidth,
        color="black",
        shrinkA=0, shrinkB=0,
    )
    if mutation_scale is not None:
        arrow_style["mutation_scale"] = mutation_scale
    ax.annotate(
        "", xy=(x0 + length, y0), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=arrow_style,
    )
    ax.annotate(
        "", xy=(x0, y0 + length), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=arrow_style,
    )
    ax.text(
        x0 + length + 0.004, y0, x_label,
        transform=ax.transAxes, ha="left", va="center", fontsize=fontsize,
    )
    ax.text(
        x0, y0 + length + 0.004, y_label,
        transform=ax.transAxes, ha="center", va="bottom", fontsize=fontsize,
        rotation=90, rotation_mode="anchor",
    )


# ---------------------------------------------------------------------------
# Supplementary Figure S1: Correlation Barplot
# ---------------------------------------------------------------------------

def figure_umap_enrichment(
    umap_coords: np.ndarray,
    cell_labels: np.ndarray,
    enrichment_scores: np.ndarray,
    double_column: bool = True,
    point_size: float = 0.3,
    subsample_idx: Optional[np.ndarray] = None,
    max_legend_items: int = 20,
    score_title: str = "bacTRAP enrichment score (PoA)",
    score_label: str = "Enrichment score",
    highlight_clusters: Optional[List[str]] = None,
    seed: int = 0,
) -> plt.Figure:
    """
    Two-panel UMAP: left colored by cell-type annotation, right by enrichment score.

    *score_title* / *score_label* let the caller override the right-panel
    title and colorbar text so the same function can render either the
    z-scored mean signature or the AUCell score without duplicating code.

    When *highlight_clusters* is provided, those clusters are coloured in the
    left panel (in the order supplied) and all other cells are greyed out.
    This keeps the annotation panel consistent with per-cluster ranking
    figures (e.g. AUCell 1b/1c) — otherwise the default "top N by cell
    frequency" heuristic hides small, highly-enriched populations.
    """
    logger.info("figure_umap_enrichment: %d cells, %d unique labels, subsample=%s",
                len(umap_coords), len(np.unique(cell_labels)),
                len(subsample_idx) if subsample_idx is not None else "none")
    setup_nature_style()
    width = get_figure_width(double_column=True)  # always double for two panels
    height = width * 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height),
                                    gridspec_kw={"wspace": 0.8})

    if subsample_idx is not None:
        umap_coords = umap_coords[subsample_idx]
        cell_labels = cell_labels[subsample_idx]
        enrichment_scores = enrichment_scores[subsample_idx]

    # Shuffle points for fair overlapping (seed-driven so re-runs are
    # reproducible against the analysis seed, not a hardcoded 42).
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(umap_coords))
    umap_coords = umap_coords[order]
    cell_labels = cell_labels[order]
    enrichment_scores = enrichment_scores[order]

    # --- Left panel: cell-type annotation ---
    unique_labels = np.unique(cell_labels)
    n_labels = len(unique_labels)

    other_label = None
    if highlight_clusters is not None and len(highlight_clusters) > 0:
        # Caller-driven selection: colour the supplied clusters in order,
        # grey out everything else. Used by the AUCell UMAP to match the
        # top-N cluster set shown in the ranking figures.
        top_labels = [c for c in highlight_clusters if c in set(unique_labels)]
        top_set = set(top_labels)
        other_label = "Other"
        plot_labels = np.array([l if l in top_set else other_label for l in cell_labels])
        unique_plot = top_labels + [other_label]
    elif n_labels > max_legend_items:
        # Keep top N by frequency, rest grouped as a catch-all
        counts = Counter(cell_labels)
        top_labels = [label for label, _ in counts.most_common(max_legend_items)]
        top_set = set(top_labels)
        other_label = "Other (grouped)"
        plot_labels = np.array([l if l in top_set else other_label for l in cell_labels])
        unique_plot = sorted(set(plot_labels) - {other_label}) + [other_label]
    else:
        plot_labels = cell_labels
        unique_plot = sorted(unique_labels)

    palette = get_qualitative_palette(max(len(unique_plot) - (1 if other_label else 0), 1))
    # Preserve caller-supplied ordering when highlight_clusters is used so
    # the legend matches the 1b/1c rank order; otherwise zip normally.
    color_map = {}
    for i, l in enumerate(unique_plot):
        if l == other_label:
            continue
        color_map[l] = palette[i % len(palette)]
    if other_label is not None:
        color_map[other_label] = "#d3d3d3"
    point_colors = np.array([color_map[l] for l in plot_labels])

    if highlight_clusters is not None and other_label is not None:
        # Draw the grey "Other" layer first, then each highlighted cluster
        # on top — otherwise small populations (e.g. Chat.GABA-7) get buried
        # under ~380k grey cells from the rng-shuffled concatenation.
        is_other = plot_labels == other_label
        ax1.scatter(
            umap_coords[is_other, 0], umap_coords[is_other, 1],
            c=point_colors[is_other], s=point_size, alpha=0.4,
            edgecolors="none", rasterized=True,
        )
        for cl in top_labels:
            mask = plot_labels == cl
            if not mask.any():
                continue
            ax1.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                c=[color_map[cl]], s=point_size * 2.5, alpha=0.95,
                edgecolors="none", rasterized=True,
            )
    else:
        ax1.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=point_colors, s=point_size, alpha=0.6,
            edgecolors="none", rasterized=True,
        )
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    ax1.set_title("Cell-type annotation")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Legend below the left panel — avoids overlapping with right panel
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=color_map[l], markersize=3, label=l)
        for l in unique_plot
    ]
    ncol = max(2, -(-len(unique_plot) // 10))  # spread across columns
    ax1.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
        fontsize=3.5, frameon=False, ncol=ncol,
        handletextpad=0.1, columnspacing=0.3, labelspacing=0.2,
    )

    # --- Right panel: enrichment score ---
    # Guard against all-NaN input — np.nanpercentile returns NaN which
    # would leave the colorbar undefined and the scatter silently blank.
    if np.all(np.isnan(enrichment_scores)):
        logger.warning("figure_umap_enrichment: all enrichment scores are NaN; "
                       "falling back to vmin=0, vmax=1 for an empty colormap")
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(enrichment_scores, 2))
        vmax = float(np.nanpercentile(enrichment_scores, 98))

    sc = ax2.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=enrichment_scores, cmap="magma", s=point_size, alpha=0.7,
        edgecolors="none", rasterized=True,
        vmin=vmin, vmax=vmax,
    )
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    ax2.set_title(score_title)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(sc, ax=ax2, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label(score_label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    return fig


# ---------------------------------------------------------------------------
# Supplementary Figure S4: Marker Overlap Dot Plot
# ---------------------------------------------------------------------------

def figure_bactrap_volcano(
    bactrap_matched: pd.DataFrame,
    highlight_genes: Optional[List[str]] = None,
    padj_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    top_n_labels: int = 15,
    double_column: bool = False,
    title: Optional[str] = "bacTRAP translational profiling",
) -> plt.Figure:
    """
    Classic volcano plot of bacTRAP DESeq2 results.
    x = log2FoldChange, y = -log10(padj).
    Highlights significantly enriched genes and optionally labels specific genes.
    """
    setup_nature_style()
    width = get_figure_width(double_column)
    height = width * 0.8

    fig, ax = plt.subplots(figsize=(width, height))

    if len(bactrap_matched) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    df = bactrap_matched.copy()
    n_before = len(df)
    df = df.dropna(subset=["log2FoldChange", "padj"])
    logger.info("figure_bactrap_volcano: %d genes (%d dropped for NaN), highlight=%s",
                len(df), n_before - len(df), highlight_genes)
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    _Y_CLIP = 50.0
    _n_clipped = int((df["neg_log10_padj"] > _Y_CLIP).sum())
    df["neg_log10_padj"] = df["neg_log10_padj"].clip(upper=_Y_CLIP)

    # Classify points
    sig_up = (df["padj"] < padj_cutoff) & (df["log2FoldChange"] > log2fc_cutoff)
    sig_down = (df["padj"] < padj_cutoff) & (df["log2FoldChange"] < -log2fc_cutoff)
    nonsig = ~sig_up & ~sig_down
    logger.info("  sig_up=%d, sig_down=%d, nonsig=%d", sig_up.sum(), sig_down.sum(), nonsig.sum())

    # Plot non-significant
    ax.scatter(
        df.loc[nonsig, "log2FoldChange"], df.loc[nonsig, "neg_log10_padj"],
        c="#bbbbbb", s=8, alpha=0.4, edgecolors="none", zorder=1,
    )
    # Plot significant down
    ax.scatter(
        df.loc[sig_down, "log2FoldChange"], df.loc[sig_down, "neg_log10_padj"],
        c="#4575b4", s=12, alpha=0.7, edgecolors="none", zorder=2,
        label="Down-regulated",
    )
    # Plot significant up (enriched in IP)
    ax.scatter(
        df.loc[sig_up, "log2FoldChange"], df.loc[sig_up, "neg_log10_padj"],
        c="#d62728", s=12, alpha=0.7, edgecolors="none", zorder=2,
        label="Enriched in IP",
    )

    # Threshold lines
    thresh_y = -np.log10(padj_cutoff)
    ax.axhline(y=thresh_y, color="black", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.axvline(x=log2fc_cutoff, color="black", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.axvline(x=-log2fc_cutoff, color="black", linestyle="--", linewidth=0.4, alpha=0.4)

    # Label highlight genes (e.g. Pnoc) — always label these regardless of significance
    if highlight_genes is None:
        highlight_genes = []
    highlight_set = set(g.lower() for g in highlight_genes)

    # Auto-label top enriched genes + forced highlights
    top_up = df[sig_up].nlargest(top_n_labels, "neg_log10_padj")
    genes_to_label = set(top_up["_hypomap_gene_name"].tolist())

    # Add highlight genes
    for _, row in df.iterrows():
        gname = str(row.get("_hypomap_gene_name", ""))
        if gname.lower() in highlight_set:
            genes_to_label.add(gname)

    texts = []
    for _, row in df.iterrows():
        gname = str(row.get("_hypomap_gene_name", ""))
        if gname in genes_to_label:
            is_highlight = gname.lower() in highlight_set
            texts.append(
                ax.text(
                    row["log2FoldChange"], row["neg_log10_padj"],
                    gname, fontsize=5 if is_highlight else 4.5,
                    fontweight="bold" if is_highlight else "normal",
                    color="#d62728" if is_highlight else "black",
                )
            )
            # Mark highlight genes with a ring
            if is_highlight:
                ax.scatter(
                    [row["log2FoldChange"]], [row["neg_log10_padj"]],
                    s=50, facecolors="none", edgecolors="#d62728",
                    linewidths=1.0, zorder=5,
                )

    if len(texts) > 0:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.3),
        )

    ax.set_xlabel(r"$\log_2$(Fold Change)")
    ax.set_ylabel(r"$-\log_{10}$(adjusted p-value)")
    if _n_clipped:
        # Surface the clipping so a stack of points at the top of the plot
        # isn't read as "tied at exactly −log10(padj)=50".
        ax.axhline(_Y_CLIP, color="grey", linestyle=":", linewidth=0.4, alpha=0.7)
        ax.text(
            ax.get_xlim()[1], _Y_CLIP,
            f"  clipped at {_Y_CLIP:.0f} (n={_n_clipped})",
            fontsize=5, color="grey", va="center", ha="left",
        )
    if title:
        ax.set_title(title)
    ax.legend(fontsize=5, frameon=False, loc="upper left")

    return fig


# ---------------------------------------------------------------------------
# Supplementary Figure S5: Heatmap
# ---------------------------------------------------------------------------

def figure_aucell_umap(
    umap_coords: np.ndarray,
    aucell_scores: np.ndarray,
    double_column: bool = False,
    point_size: float = 0.3,
    subsample_idx: Optional[np.ndarray] = None,
    seed: int = 0,
) -> plt.Figure:
    """Publication-ready UMAP coloured by AUCell enrichment scores.

    Title and axis labels are omitted (belong in the figure caption) and the
    conventional x/y axes are replaced with two bottom-left arrows via
    `_add_umap_axis_arrows`, matching the single-cell publication convention.
    """
    setup_nature_style()
    width = get_figure_width(double_column)
    height = width * 0.9

    fig, ax = plt.subplots(figsize=(width, height))

    if subsample_idx is not None:
        umap_coords = umap_coords[subsample_idx]
        aucell_scores = aucell_scores[subsample_idx]

    # Shuffle for fair overlap
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(umap_coords))
    umap_coords = umap_coords[order]
    aucell_scores = aucell_scores[order]

    # Guard against all-NaN input — np.nanpercentile would return NaN
    # and leave the colorbar undefined.
    if np.all(np.isnan(aucell_scores)):
        logger.warning("figure_aucell_umap: all AUCell scores are NaN; "
                       "falling back to vmin=0, vmax=1 for an empty colormap")
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(aucell_scores, 2))
        vmax = float(np.nanpercentile(aucell_scores, 98))

    sc = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=aucell_scores, cmap="magma", s=point_size, alpha=0.7,
        edgecolors="none", rasterized=True,
        vmin=vmin, vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _add_umap_axis_arrows(ax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("AUCell score", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)

    return fig


# ---------------------------------------------------------------------------
# Main Figure: Cell-type annotation UMAP (top-N highlighted)
# ---------------------------------------------------------------------------

# Shared UMAP scatter constants so every UMAP panel (fig_1a, Figure 2, ...)
# renders with identical dot sizes and colours. Changing these changes all
# UMAP panels at once — that is the point.
UMAP_OTHER_COLOR = "#d9d9d9"
UMAP_HIGHLIGHT_SCALE = 2.5  # highlighted/coloured dots are this * point_size


def _style_umap_ax(ax) -> None:
    """Strip ticks and spines — the bare UMAP panel look used everywhere."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_cluster_scatter(
    ax, umap_coords, cell_labels, highlight, color_map, point_size,
    other_color=UMAP_OTHER_COLOR, select_mask=None,
) -> None:
    """Grey 'Other' backdrop + coloured highlight clusters.

    This is the single scatter primitive behind every cluster-coloured UMAP
    (fig_1a and Figure 2's left panel), so they look identical. *select_mask*,
    when given, restricts which cells are eligible to be coloured (Figure 2
    colours only the region cells of each cluster); everything else is grey.
    """
    top_set = set(highlight)
    colored = np.isin(cell_labels, list(top_set))
    if select_mask is not None:
        colored = colored & np.asarray(select_mask, dtype=bool)
    is_other = ~colored
    ax.scatter(
        umap_coords[is_other, 0], umap_coords[is_other, 1],
        c=other_color, s=point_size, alpha=0.4,
        edgecolors="none", rasterized=True,
    )
    for cl in highlight:
        mask = cell_labels == cl
        if select_mask is not None:
            mask = mask & np.asarray(select_mask, dtype=bool)
        if not mask.any():
            continue
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=[color_map[cl]], s=point_size * UMAP_HIGHLIGHT_SCALE, alpha=0.95,
            edgecolors="none", rasterized=True,
        )
    _style_umap_ax(ax)


def _draw_expression_scatter(
    ax, umap_coords, values, select_mask, point_size,
    cmap="magma", other_color=UMAP_OTHER_COLOR,
):
    """Grey backdrop of all cells + selected cells coloured by a continuous
    value, using the same dot sizes as `_draw_cluster_scatter` so expression
    and cluster panels match. Returns the mappable for a colorbar."""
    ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=other_color, s=point_size, alpha=0.4,
        edgecolors="none", rasterized=True,
    )
    sel = np.asarray(select_mask, dtype=bool)
    vals = np.asarray(values, dtype=float)[sel]
    vmax = (float(np.nanpercentile(vals, 98)) if vals.size else 1.0) or 1.0
    order = np.argsort(vals)  # brightest drawn last
    sc = ax.scatter(
        umap_coords[sel][order, 0], umap_coords[sel][order, 1],
        c=vals[order], cmap=cmap, s=point_size * UMAP_HIGHLIGHT_SCALE,
        alpha=0.95, edgecolors="none", rasterized=True, vmin=0.0, vmax=vmax,
    )
    _style_umap_ax(ax)
    return sc


def figure_celltype_umap(
    umap_coords: np.ndarray,
    cell_labels: np.ndarray,
    highlight_clusters: List[str],
    double_column: bool = False,
    point_size: float = 0.3,
    subsample_idx: Optional[np.ndarray] = None,
    seed: int = 0,
) -> plt.Figure:
    """Publication-ready UMAP coloured by cell-type annotation.

    The clusters in *highlight_clusters* (in the supplied order — typically
    top-N by AUCell mean) are drawn in colour on top of a grey "Other" layer,
    so small but highly enriched populations remain visible. Axis labels and
    title are omitted; a legend sits to the right of the panel.
    """
    setup_nature_style()
    width = get_figure_width(double_column)
    height = width * 0.9

    fig, ax = plt.subplots(figsize=(width, height))

    if subsample_idx is not None:
        umap_coords = umap_coords[subsample_idx]
        cell_labels = cell_labels[subsample_idx]

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(umap_coords))
    umap_coords = umap_coords[order]
    cell_labels = cell_labels[order]

    unique = set(np.unique(cell_labels).tolist())
    top_labels = [c for c in highlight_clusters if c in unique]

    palette = get_qualitative_palette(max(len(top_labels), 1))
    color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(top_labels)}

    _draw_cluster_scatter(ax, umap_coords, cell_labels, top_labels,
                          color_map, point_size)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[cl], markersize=3.5, label=cl)
        for cl in top_labels
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=UMAP_OTHER_COLOR, markersize=3.5, label="Other")
    )
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=5, frameon=False, handletextpad=0.3,
        labelspacing=0.35, borderaxespad=0,
    )

    logger.info("figure_celltype_umap: %d highlighted clusters",
                len(top_labels))

    return fig


# ---------------------------------------------------------------------------
# Figure 2: gene expression across a region (e.g. Pnoc in the preoptic area)
# ---------------------------------------------------------------------------

def _draw_gene_poa_panels(
    ax_clusters, ax_expr, cax, umap_coords, cell_labels, region_mask,
    gene_expr, expressing_mask, highlight, color_map, point_size,
    gene_name, region_label, legend_loc="below",
):
    """Draw the two Figure-2 panels (cluster identity + expression) onto the
    supplied axes, with the colorbar in *cax*. Shared by figure_gene_poa_umap
    and figure_pnoc_overview so the panels are identical. *legend_loc* is
    "below" (single figure) or "right" (composite, legend goes into the gap
    column to the right of the cluster panel)."""
    region_mask = np.asarray(region_mask, dtype=bool)
    expressing_mask = np.asarray(expressing_mask, dtype=bool)

    _draw_cluster_scatter(ax_clusters, umap_coords, cell_labels, highlight,
                          color_map, point_size, select_mask=region_mask)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[cl], markersize=4, label=cl)
        for cl in highlight
    ]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=UMAP_OTHER_COLOR, markersize=4,
                              label="Other"))
    if legend_loc == "right":
        ax_clusters.legend(handles=handles, loc="center left",
                           bbox_to_anchor=(1.02, 0.5), fontsize=6,
                           frameon=False, ncol=1, handletextpad=0.3,
                           labelspacing=0.3, borderaxespad=0)
    else:
        ncol = 2 if len(handles) <= 8 else 3
        ax_clusters.legend(handles=handles, loc="upper center",
                           bbox_to_anchor=(0.5, -0.02), fontsize=6,
                           frameon=False, ncol=ncol, handletextpad=0.3,
                           columnspacing=0.5, labelspacing=0.35)

    sc = _draw_expression_scatter(
        ax_expr, umap_coords, gene_expr, region_mask & expressing_mask,
        point_size,
    )
    cbar = ax_expr.figure.colorbar(sc, cax=cax)
    cbar.set_label(f"{gene_name} (log-norm)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    return sc


def figure_gene_poa_umap(
    umap_coords: np.ndarray,
    cell_labels: np.ndarray,
    region_mask: np.ndarray,
    gene_expr: np.ndarray,
    expressing_mask: np.ndarray,
    highlight_clusters: List[str],
    gene_name: str = "Pnoc",
    region_label: str = "preoptic",
    double_column: bool = False,
    point_size: float = 0.3,
    subsample_idx: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Two-panel UMAP of a gene's expression within a region, on the full atlas.

    Renders with the **exact same scatter primitives, dot sizes, colours and
    per-panel geometry as `figure_celltype_umap` (fig_1a)** — each panel uses
    the shared `_draw_cluster_scatter` / `_draw_expression_scatter` helpers at
    `point_size` (backdrop) and `point_size * UMAP_HIGHLIGHT_SCALE` (coloured),
    so the two figures match in appearance.

    The two panels are equal-sized: the colorbar lives in its own thin
    gridspec column so it does not shrink the expression panel.

    * **Left** — the *region* cells of each cluster in *highlight_clusters*
      are coloured over a grey "Other" backdrop of all atlas cells.
    * **Right** — the *expressing* region cells are coloured by *gene_expr*
      (magma) over the same grey backdrop; non-expressing cells stay grey.

    All array inputs are over the **full atlas** (same length / order). Pass
    *subsample_idx* (as fig_1a does) to thin the backdrop to the same density.

    Shared by the standalone Figure 2 script and the Streamlit app.
    """
    setup_nature_style()
    region_mask = np.asarray(region_mask, dtype=bool)
    expressing_mask = np.asarray(expressing_mask, dtype=bool)
    gene_expr = np.asarray(gene_expr, dtype=float)
    cell_labels = np.asarray(cell_labels)

    if subsample_idx is not None:
        umap_coords = umap_coords[subsample_idx]
        cell_labels = cell_labels[subsample_idx]
        region_mask = region_mask[subsample_idx]
        expressing_mask = expressing_mask[subsample_idx]
        gene_expr = gene_expr[subsample_idx]

    # Per-panel geometry equals a fig_1a panel, so the UMAP spans the same
    # number of inches and the (absolute-sized) dots look identical. The two
    # panels get equal width (1:1); the colorbar sits in a thin third column.
    panel_w = get_figure_width(double_column)
    fig = plt.figure(figsize=(2 * panel_w + 0.6, panel_w * 0.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    highlight = [c for c in highlight_clusters
                 if c in set(np.unique(cell_labels).tolist())]
    palette = get_qualitative_palette(max(len(highlight), 1))
    color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(highlight)}

    _draw_gene_poa_panels(
        ax1, ax2, cax, umap_coords, cell_labels, region_mask, gene_expr,
        expressing_mask, highlight, color_map, point_size, gene_name,
        region_label,
    )

    logger.info("figure_gene_poa_umap: gene=%s region=%s, %d region cells, "
                "%d highlighted clusters", gene_name, region_label,
                int(region_mask.sum()), len(highlight))
    return fig


# ---------------------------------------------------------------------------
# Composite overview: Figure 2 (top) + fig_1a celltype UMAP + fig_1c violins
# ---------------------------------------------------------------------------

def figure_pnoc_overview(
    fig2_umap: np.ndarray,
    fig2_labels: np.ndarray,
    fig2_region_mask: np.ndarray,
    fig2_gene_expr: np.ndarray,
    fig2_expressing_mask: np.ndarray,
    fig2_highlight: List[str],
    ct_highlight: List[str],
    aucell_scores: np.ndarray,
    view_labels: np.ndarray,
    gene_name: str = "Pnoc",
    region_label: str = "preoptic",
    double_column: bool = True,
    point_size: float = 0.3,
    fig2_subsample_idx: Optional[np.ndarray] = None,
    violin_top_n: int = 15,
    violin_min_cluster_cells: int = 20,
    violin_allowed_clusters: Optional[Iterable[str]] = None,
) -> plt.Figure:
    """Four-panel composite, all panels equal-sized:

        top-left   Figure 2 left  — top `gene_name` clusters in the region
        top-right  Figure 2 right — `gene_name` expression in the region
        bottom-left  fig_1a       — cell-type UMAP, top AUCell clusters
        bottom-right fig_1c       — AUCell-score violins for those clusters

    Every panel is drawn with the shared helpers used by the standalone
    figures, so the panels are identical to the individual figures.

    **All three UMAP panels share one identical grey backdrop**: the full
    atlas (the Figure 2 arrays), subsampled by *fig2_subsample_idx*. The
    bottom-left additionally colours the top-AUCell clusters' cells on that
    same backdrop. (We deliberately do NOT use the analysis view here — when
    a region restriction is active the view drops atlas cells, which made the
    bottom backdrop differ from the top.) The violins use all view cells
    (`aucell_scores` / `view_labels`), matching fig_1c.

    Legends sit in their own columns and the violin labels on the right, so
    nothing overlaps across rows.
    """
    setup_nature_style()

    # Shared full-atlas backdrop arrays (subsample once, reuse for every panel).
    umap = fig2_umap
    labels = np.asarray(fig2_labels)
    region = np.asarray(fig2_region_mask, dtype=bool)
    expr = np.asarray(fig2_gene_expr, dtype=float)
    expressing = np.asarray(fig2_expressing_mask, dtype=bool)
    if fig2_subsample_idx is not None:
        umap = umap[fig2_subsample_idx]
        labels = labels[fig2_subsample_idx]
        region = region[fig2_subsample_idx]
        expr = expr[fig2_subsample_idx]
        expressing = expressing[fig2_subsample_idx]

    panel_w = get_figure_width(double_column)
    # cols: [panel][legend gap][panel][colorbar / right-label gap]
    fig = plt.figure(figsize=(2.7 * panel_w, 2 * panel_w * 0.95))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1, 0.5, 1, 0.06], height_ratios=[1, 1],
        wspace=0.08, hspace=0.3,
    )
    ax_f2c = fig.add_subplot(gs[0, 0])
    ax_f2e = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    ax_ct = fig.add_subplot(gs[1, 0])
    ax_vio = fig.add_subplot(gs[1, 2])

    present = set(np.unique(labels).tolist())
    f2_high = [c for c in fig2_highlight if c in present]
    ct_high = [c for c in ct_highlight if c in present]

    # Shared colour map over the UNION of both cluster panels, so a cluster
    # that appears in both the Pnoc panel and the AUCell panel gets the SAME
    # colour. Order: Pnoc clusters first, then any AUCell-only clusters.
    union = list(f2_high) + [c for c in ct_high if c not in set(f2_high)]
    palette = get_qualitative_palette(max(len(union), 1))
    shared_cmap = {cl: palette[i % len(palette)] for i, cl in enumerate(union)}

    # --- top row: Figure 2 panels (cluster legend to the RIGHT, into col 1) ---
    _draw_gene_poa_panels(
        ax_f2c, ax_f2e, cax, umap, labels, region, expr, expressing,
        f2_high, shared_cmap, point_size, gene_name, region_label,
        legend_loc="right",
    )

    # --- bottom-left: same backdrop, top AUCell clusters coloured ---
    _draw_cluster_scatter(ax_ct, umap, labels, ct_high, shared_cmap, point_size)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=shared_cmap[cl], markersize=4, label=cl)
        for cl in ct_high
    ]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=UMAP_OTHER_COLOR, markersize=4,
                              label="Other"))
    ax_ct.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                 fontsize=6, frameon=False, ncol=1, handletextpad=0.3,
                 labelspacing=0.3, borderaxespad=0)

    # --- bottom-right: fig_1c AUCell violins (labels on the right) ---
    _draw_aucell_violins(ax_vio, aucell_scores, view_labels, top_n=violin_top_n,
                         min_cluster_cells=violin_min_cluster_cells,
                         allowed_clusters=violin_allowed_clusters)
    ax_vio.yaxis.tick_right()
    ax_vio.yaxis.set_label_position("right")

    logger.info("figure_pnoc_overview: gene=%s, %d Fig2 clusters, %d AUCell clusters",
                gene_name, len(f2_high), len(ct_high))
    return fig


# ---------------------------------------------------------------------------
# Main Figure 1b: AUCell Cluster Barplot
# ---------------------------------------------------------------------------

def figure_aucell_cluster_barplot(
    aucell_scores: np.ndarray,
    cell_labels: np.ndarray,
    top_n: int = 25,
    double_column: bool = False,
    min_cluster_cells: int = 20,
    allowed_clusters: Optional[Iterable[str]] = None,
) -> plt.Figure:
    """Horizontal barplot of mean AUCell score per cluster, ranked.

    Clusters with fewer than *min_cluster_cells* cells are excluded from the
    ranking (fix #4: small clusters with a slightly above-average mean
    otherwise dominate the top of the list purely due to shrinkage variance).

    If *allowed_clusters* is provided, only clusters in that set are eligible
    for the ranking — used to hide clusters that fail a Cre-driver baseline
    expression threshold.
    """
    setup_nature_style()
    width = get_figure_width(double_column)

    # Compute mean AUCell score per cluster (size-filtered)
    df = pd.DataFrame({"score": aucell_scores, "cluster": cell_labels})
    if allowed_clusters is not None:
        allowed_set = {str(c) for c in allowed_clusters}
        df = df[df["cluster"].astype(str).isin(allowed_set)]
    cluster_stats = df.groupby("cluster")["score"].agg(["mean", "std", "count"])
    cluster_stats = cluster_stats[cluster_stats["count"] >= min_cluster_cells]
    cluster_stats = cluster_stats.sort_values("mean", ascending=False)
    cluster_stats = cluster_stats.head(top_n).iloc[::-1]  # reverse for bottom-to-top

    if len(cluster_stats) == 0:
        fig, ax = plt.subplots(figsize=(width, 2))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n_bars = len(cluster_stats)
    height = max(width * 0.6, n_bars * 0.18 + 0.8)
    fig, ax = plt.subplots(figsize=(width, height))

    # Color by score. Use the actual min as vmin so dynamic range across the
    # top-N stretches the full colormap rather than washing out top bars.
    _vmin = float(cluster_stats["mean"].min())
    _vmax = float(cluster_stats["mean"].max())
    if _vmax <= _vmin:
        _vmax = _vmin + 1e-9
    norm = Normalize(vmin=_vmin, vmax=_vmax)
    cmap = plt.colormaps["magma"]
    colors = [cmap(norm(v)) for v in cluster_stats["mean"]]

    ax.barh(
        range(n_bars),
        cluster_stats["mean"],
        xerr=cluster_stats["std"] / np.sqrt(cluster_stats["count"]),  # SEM
        color=colors,
        edgecolor="none",
        height=0.7,
        capsize=1.5,
        error_kw={"linewidth": 0.5},
    )

    for i, (_, row) in enumerate(cluster_stats.iterrows()):
        ax.text(row["mean"] + cluster_stats["mean"].max() * 0.02, i,
                f"{row['mean']:.4f}", va="center", ha="left", fontsize=5)

    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(cluster_stats.index, fontsize=6)
    ax.set_xlabel("Mean AUCell score")
    ax.set_title("AUCell enrichment per cluster (top %d)" % top_n)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("AUCell score", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    logger.info("figure_aucell_cluster_barplot: %d clusters, top=%s (%.4f)",
                n_bars, cluster_stats.index[-1], cluster_stats["mean"].iloc[-1])

    return fig


# ---------------------------------------------------------------------------
# Main Figure 1c: AUCell Violin Plot (top clusters)
# ---------------------------------------------------------------------------

def figure_aucell_violins(
    aucell_scores: np.ndarray,
    cell_labels: np.ndarray,
    top_n: int = 15,
    double_column: bool = True,
    min_cluster_cells: int = 20,
    allowed_clusters: Optional[Iterable[str]] = None,
) -> plt.Figure:
    """Violin plots of AUCell score distributions for top clusters.

    Clusters with fewer than *min_cluster_cells* cells are excluded from the
    ranking (fix #4) so the highest-mean slots are not claimed by small
    populations whose means are unstable purely due to low cell count.

    If *allowed_clusters* is provided, only clusters in that set are eligible
    for the ranking — used to hide clusters that fail a Cre-driver baseline
    expression threshold.
    """
    setup_nature_style()
    width = get_figure_width(double_column)

    _, _, top_clusters = _rank_aucell_clusters(
        aucell_scores, cell_labels, top_n, min_cluster_cells, allowed_clusters)

    # Scale height with cluster count. The old `max(width*0.5, 3.5)` floor
    # produced an awkward near-square panel for 1-2 clusters (each violin
    # ~0.2" tall in 3.5" of vertical space). With ≤3 clusters we drop the
    # floor and use a tight per-row height instead.
    n_rows = len(top_clusters)
    if n_rows <= 3:
        height = max(width * 0.25, 0.5 * max(n_rows, 1) + 0.8)
    else:
        height = max(width * 0.5, 3.5)
    fig, ax = plt.subplots(figsize=(width, height))

    _draw_aucell_violins(ax, aucell_scores, cell_labels, top_n=top_n,
                         min_cluster_cells=min_cluster_cells,
                         allowed_clusters=allowed_clusters)

    logger.info("figure_aucell_violins: %d clusters shown", len(top_clusters))

    return fig


def _rank_aucell_clusters(aucell_scores, cell_labels, top_n,
                          min_cluster_cells, allowed_clusters):
    """Return (df, cluster_means, top_clusters) for the AUCell violin ranking.
    Clusters below *min_cluster_cells* are excluded; ranked by mean score."""
    df = pd.DataFrame({"score": aucell_scores, "cluster": cell_labels})
    if allowed_clusters is not None:
        allowed_set = {str(c) for c in allowed_clusters}
        df = df[df["cluster"].astype(str).isin(allowed_set)]
    cluster_counts = df.groupby("cluster")["score"].count()
    eligible = cluster_counts[cluster_counts >= min_cluster_cells].index
    cluster_means = (
        df[df["cluster"].isin(eligible)]
        .groupby("cluster")["score"].mean()
        .sort_values(ascending=False)
    )
    top_clusters = cluster_means.head(top_n).index.tolist()
    return df, cluster_means, top_clusters


def _draw_aucell_violins(ax, aucell_scores, cell_labels, top_n=15,
                         min_cluster_cells=20, allowed_clusters=None):
    """Horizontal AUCell-score violins for the top clusters, drawn onto *ax*.
    Magma-coloured by cluster mean. Shared by figure_aucell_violins and the
    composite overview so the panel is identical. Returns the top clusters."""
    df, cluster_means, top_clusters = _rank_aucell_clusters(
        aucell_scores, cell_labels, top_n, min_cluster_cells, allowed_clusters)
    df_top = df[df["cluster"].isin(top_clusters)].copy()
    if len(df_top) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes)
        return top_clusters

    df_top["cluster"] = pd.Categorical(
        df_top["cluster"], categories=top_clusters, ordered=True)
    parts = ax.violinplot(
        [df_top.loc[df_top["cluster"] == c, "score"].values for c in top_clusters],
        positions=range(len(top_clusters)),
        vert=False, showmeans=False, showmedians=False, showextrema=False,
    )
    cmap = plt.colormaps["magma"]
    norm = Normalize(vmin=0, vmax=cluster_means.iloc[0])
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(norm(cluster_means.iloc[i])))
        body.set_alpha(0.7)
        body.set_edgecolor("grey")
        body.set_linewidth(0.5)
    ax.set_yticks(range(len(top_clusters)))
    ax.set_yticklabels(top_clusters, fontsize=6)
    ax.set_xlabel("AUCell score")
    ax.tick_params(axis="x", labelsize=6)
    ax.invert_yaxis()  # highest-mean cluster at the top
    return top_clusters


def figure_aucell_zscore_violins(
    aucell_scores: np.ndarray,
    cell_labels: np.ndarray,
    z_by_cluster: pd.Series,
    top_n: int = 15,
    double_column: bool = True,
    min_cluster_cells: int = 20,
    allowed_clusters: Optional[Iterable[str]] = None,
) -> plt.Figure:
    """Violin plots of AUCell score distributions for the top clusters ranked
    by the **empirical z-score** (matched-expression null), the supplementary
    companion to ``figure_aucell_violins`` (which ranks by mean).

    ``z_by_cluster`` maps cluster name → ``z_empirical``.  Clusters absent
    from it, with NaN z, or with fewer than ``min_cluster_cells`` cells are
    not eligible for the ranking.
    """
    setup_nature_style()
    width = get_figure_width(double_column)

    df = pd.DataFrame({"score": aucell_scores, "cluster": np.asarray(cell_labels).astype(str)})
    if allowed_clusters is not None:
        allowed_set = {str(c) for c in allowed_clusters}
        df = df[df["cluster"].isin(allowed_set)]
    cluster_counts = df.groupby("cluster")["score"].count()
    eligible = set(cluster_counts[cluster_counts >= min_cluster_cells].index)

    z = pd.Series(z_by_cluster).copy()
    z.index = z.index.astype(str)
    z = z[z.index.isin(eligible)].dropna().sort_values(ascending=False)
    top_clusters = z.head(top_n).index.tolist()
    df_top = df[df["cluster"].isin(top_clusters)].copy()

    if len(df_top) == 0 or len(top_clusters) == 0:
        fig, ax = plt.subplots(figsize=(width, 2))
        ax.text(0.5, 0.5, "No empirical-null data available", ha="center",
                va="center", transform=ax.transAxes)
        return fig

    df_top["cluster"] = pd.Categorical(df_top["cluster"], categories=top_clusters, ordered=True)
    # Same small-N branch as figure_aucell_violins — avoid an empty square
    # panel for 1-3 clusters.
    n_rows = len(top_clusters)
    if n_rows <= 3:
        height = max(width * 0.25, 0.5 * n_rows + 0.8)
    else:
        height = max(width * 0.5, 3.5)
    fig, ax = plt.subplots(figsize=(width, height))

    parts = ax.violinplot(
        [df_top.loc[df_top["cluster"] == c, "score"].values for c in top_clusters],
        positions=range(len(top_clusters)),
        vert=False, showmeans=False, showmedians=False, showextrema=False,
    )
    cmap = plt.colormaps["viridis"]
    zvals = z.loc[top_clusters].to_numpy()
    _zmin = float(np.nanmin(zvals))
    _zmax = float(np.nanmax(zvals))
    if not np.isfinite(_zmin) or not np.isfinite(_zmax) or _zmax <= _zmin:
        # All z's tied (or non-finite) — expand symmetrically around the value
        # so the colour scale isn't visually biased upward.
        _centre = _zmin if np.isfinite(_zmin) else 0.0
        _zmin = _centre - 0.5
        _zmax = _centre + 0.5
    norm = Normalize(vmin=_zmin, vmax=_zmax)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(norm(zvals[i])))
        body.set_alpha(0.7)
        body.set_edgecolor("grey")
        body.set_linewidth(0.5)

    ax.set_yticks(range(len(top_clusters)))
    ax.set_yticklabels(top_clusters, fontsize=6)
    ax.set_xlabel("AUCell score")
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title("Top %d by empirical z-score (matched-expression null)" % len(top_clusters))
    ax.invert_yaxis()

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("empirical z-score", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    logger.info("figure_aucell_zscore_violins: %d clusters shown", len(top_clusters))
    return fig


# ---------------------------------------------------------------------------
# Main Figure 1d: AUCell Score Histogram
# ---------------------------------------------------------------------------

def figure_aucell_histogram(
    aucell_scores: np.ndarray,
    double_column: bool = False,
) -> plt.Figure:
    """Histogram of AUCell scores across all cells, with percentile markers."""
    setup_nature_style()
    width = get_figure_width(double_column)
    fig, ax = plt.subplots(figsize=(width, width * 0.6))

    # Plot the histogram on the nonzero subset only — equal-width bins over
    # the full range would otherwise concentrate the AUCell zero-mass into a
    # single bin that dwarfs the right tail and visually hides percentile
    # lines. The exact zero count is reported in the corner annotation
    # below.
    all_scores = aucell_scores
    nonzero = all_scores[all_scores > 0]
    plot_data = nonzero if nonzero.size else all_scores
    ax.hist(plot_data, bins=100, color="steelblue", edgecolor="none",
            alpha=0.8, density=True)

    # Mark percentiles. Computed over ALL scores (including zeros) so the
    # quoted percentile matches "what fraction of cells fall below this".
    # Label alignment flips to the LEFT of the line when the line sits near
    # the right edge so the text doesn't overflow the axes.
    _xmin, _xmax = ax.get_xlim()
    def _label_for(val):
        # Within ~25% of the right edge → right-align so text grows leftward
        return ("right", f"{{lbl}}\n {val:.4f}") if val > _xmin + 0.75 * (_xmax - _xmin) \
            else ("left",  f" {{lbl}}\n {val:.4f}")
    for pct, ls, lbl in [(90, "--", "90th"), (95, "-.", "95th"), (99, ":", "99th")]:
        val = np.percentile(all_scores, pct)
        ax.axvline(val, color="firebrick", linestyle=ls, linewidth=0.8, alpha=0.8)
        ha, tmpl = _label_for(val)
        ax.text(val, ax.get_ylim()[1] * 0.95, tmpl.format(lbl=lbl).rstrip(),
                fontsize=5, color="firebrick", va="top", ha=ha)

    mean_val = np.mean(all_scores)
    ax.axvline(mean_val, color="black", linestyle="-", linewidth=0.8)
    ha, tmpl = _label_for(mean_val)
    ax.text(mean_val, ax.get_ylim()[1] * 0.80, tmpl.format(lbl="mean").rstrip(),
            fontsize=5, color="black", va="top", ha=ha)

    ax.set_xlabel("AUCell score (nonzero cells)")
    ax.set_ylabel("Density")
    ax.set_title("AUCell score distribution")

    n_zero = int((all_scores == 0).sum())
    n_total = len(all_scores)
    ax.text(0.98, 0.98,
            f"n = {n_total:,}\nzero = {n_zero:,} ({100*n_zero/n_total:.1f}%)\n"
            f"(zeros hidden from histogram)",
            transform=ax.transAxes, fontsize=5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    logger.info("figure_aucell_histogram: %d cells, mean=%.4f, 95th=%.4f",
                n_total, mean_val, np.percentile(all_scores, 95))

    return fig


# ---------------------------------------------------------------------------
# Cre-driver / marker-gene diagnostic
# ---------------------------------------------------------------------------

def figure_marker_gene_diagnostic(
    gene_stats: pd.DataFrame,
    cluster_order: List[str],
    gene_name: str = "Pnoc",
    fraction_threshold: float = 0.05,
    double_column: bool = True,
) -> plt.Figure:
    """
    Two-panel sanity-check plot for a Cre-driver / marker gene.

    For each cluster in *cluster_order* (typically the top AUCell-ranked
    clusters) two horizontal bars are drawn:

      * Left  — mean log-normalised expression of *gene_name*
      * Right — fraction of cells with non-zero counts for *gene_name*

    A vertical reference line on the fraction panel marks
    *fraction_threshold*; bars meeting both "ranked top" AND
    "fraction ≥ threshold" are filled in saturated colour, the rest are
    greyed-out — making it easy to spot top-ranked clusters that fail the
    Pnoc check (likely lineage-tracing artefacts, dropout, or background).

    Parameters
    ----------
    gene_stats : DataFrame
        Output of ``compute_single_gene_cluster_stats`` — index = cluster,
        columns include ``mean_expr`` and ``fraction_expressing``.
    cluster_order : list[str]
        Clusters to display (top-down, e.g. the top 20 from composite
        ranking).  Missing clusters are skipped silently.
    """
    setup_nature_style()
    width = get_figure_width(double_column)

    available = [c for c in cluster_order if c in gene_stats.index]
    if len(available) == 0:
        fig, ax = plt.subplots(figsize=(width, 2))
        ax.text(0.5, 0.5, f"{gene_name} not detected in selected clusters",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Reverse so the top-ranked cluster sits at the top of the plot
    df = gene_stats.loc[available, ["mean_expr", "fraction_expressing"]].iloc[::-1]
    n_bars = len(df)

    height = max(2.0, n_bars * 0.22 + 1.0)
    fig, (ax_mean, ax_frac) = plt.subplots(
        1, 2, figsize=(width, height), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    pass_mask = (df["fraction_expressing"] >= fraction_threshold).values
    color_pass = "#762a83"   # saturated purple, colorblind-safe
    color_fail = "#bdbdbd"   # neutral grey
    bar_colors = np.where(pass_mask, color_pass, color_fail)

    # --- Panel 1: mean expression ----------------------------------------
    ax_mean.barh(
        range(n_bars), df["mean_expr"].values,
        color=bar_colors, edgecolor="none", height=0.7,
    )
    ax_mean.set_yticks(range(n_bars))
    ax_mean.set_yticklabels(df.index, fontsize=5)
    ax_mean.invert_xaxis()                       # bars grow leftward
    ax_mean.yaxis.tick_right()                   # labels live in the gutter
    ax_mean.tick_params(axis="y", which="both", length=0, pad=2)
    ax_mean.set_xlabel(f"Mean {gene_name} expression\n(log-norm)")
    # Tighten x-limit so the label row reads as 0 → max
    mean_max = float(df["mean_expr"].max()) if n_bars else 0.0
    if mean_max > 0:
        ax_mean.set_xlim(mean_max * 1.05, 0)

    # --- Panel 2: fraction expressing ------------------------------------
    ax_frac.barh(
        range(n_bars), df["fraction_expressing"].values,
        color=bar_colors, edgecolor="none", height=0.7,
    )
    ax_frac.axvline(
        fraction_threshold, color="black", linewidth=0.5,
        linestyle="--", zorder=4,
    )
    ax_frac.set_xlabel(f"Fraction of cells\nexpressing {gene_name}")
    ax_frac.set_xlim(0, max(1.0, float(df["fraction_expressing"].max()) * 1.1))
    # Hide redundant y tick labels on the right panel — they live with
    # the left axis (rotated to the right side via yaxis.tick_right).
    ax_frac.tick_params(axis="y", which="both", length=0, labelleft=False)

    # Annotate fraction values
    for i, frac in enumerate(df["fraction_expressing"].values):
        ax_frac.text(
            frac + 0.01, i, f"{frac*100:.0f}%",
            va="center", ha="left", fontsize=5,
        )

    fig.suptitle(
        f"{gene_name} expression across top-ranked clusters "
        f"(threshold = {fraction_threshold*100:.0f}%)",
        fontsize=8, fontweight="bold", y=0.995,
    )

    # Legend: colour-coded pass / fail
    pass_patch = plt.Rectangle((0, 0), 1, 1, color=color_pass)
    fail_patch = plt.Rectangle((0, 0), 1, 1, color=color_fail)
    ax_frac.legend(
        [pass_patch, fail_patch],
        [f"≥ {fraction_threshold*100:.0f}% expressing", "below threshold"],
        loc="lower right", fontsize=5, frameon=False, handlelength=1.2,
    )

    return fig


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

_VALID_EXPORT_FORMATS = frozenset({"pdf", "svg", "png"})


def fig_to_bytes(fig: plt.Figure, fmt: str = "pdf") -> bytes:
    """Convert a matplotlib figure to bytes in the specified format.

    Args:
        fig: matplotlib figure to serialise.
        fmt: one of ``"pdf"``, ``"svg"``, ``"png"``. A typo surfaces
            here with a clear ValueError rather than inside matplotlib's
            savefig dispatch (which tends to produce opaque backend
            registration errors).
    """
    if fmt not in _VALID_EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported figure format {fmt!r}; expected one of "
            f"{sorted(_VALID_EXPORT_FORMATS)}"
        )
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


