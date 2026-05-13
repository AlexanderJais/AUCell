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
import seaborn as sns
from adjustText import adjust_text
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

    # Shuffle points for fair overlapping
    rng = np.random.default_rng(42)
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
    df["neg_log10_padj"] = df["neg_log10_padj"].clip(upper=50)

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
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.3),
        )

    ax.set_xlabel(r"$\log_2$(Fold Change)")
    ax.set_ylabel(r"$-\log_{10}$(adjusted p-value)")
    ax.set_title("bacTRAP translational profiling (PoA IP vs Input)")
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
    rng = np.random.default_rng(42)
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

def figure_celltype_umap(
    umap_coords: np.ndarray,
    cell_labels: np.ndarray,
    highlight_clusters: List[str],
    double_column: bool = False,
    point_size: float = 0.3,
    subsample_idx: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Publication-ready UMAP coloured by cell-type annotation.

    The clusters in *highlight_clusters* (in the supplied order — typically
    top-N by AUCell mean) are drawn in colour on top of a grey "Other" layer,
    so small but highly enriched populations remain visible. Axis labels and
    title are omitted; a legend sits to the right of the panel and the two
    bottom-left arrows mark UMAP1 / UMAP2.
    """
    setup_nature_style()
    width = get_figure_width(double_column)
    height = width * 0.9

    fig, ax = plt.subplots(figsize=(width, height))

    if subsample_idx is not None:
        umap_coords = umap_coords[subsample_idx]
        cell_labels = cell_labels[subsample_idx]

    rng = np.random.default_rng(42)
    order = rng.permutation(len(umap_coords))
    umap_coords = umap_coords[order]
    cell_labels = cell_labels[order]

    unique = set(np.unique(cell_labels).tolist())
    top_labels = [c for c in highlight_clusters if c in unique]
    top_set = set(top_labels)

    palette = get_qualitative_palette(max(len(top_labels), 1))
    color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(top_labels)}
    other_color = "#d9d9d9"

    is_other = np.array([lbl not in top_set for lbl in cell_labels])
    ax.scatter(
        umap_coords[is_other, 0], umap_coords[is_other, 1],
        c=other_color, s=point_size, alpha=0.4,
        edgecolors="none", rasterized=True,
    )
    for cl in top_labels:
        mask = cell_labels == cl
        if not mask.any():
            continue
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=[color_map[cl]], s=point_size * 2.5, alpha=0.95,
            edgecolors="none", rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _add_umap_axis_arrows(ax)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[cl], markersize=3.5, label=cl)
        for cl in top_labels
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=other_color, markersize=3.5, label="Other")
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

    # Color by score
    norm = Normalize(vmin=0, vmax=cluster_stats["mean"].max())
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
    df_top = df[df["cluster"].isin(top_clusters)].copy()

    if len(df_top) == 0:
        fig, ax = plt.subplots(figsize=(width, 2))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # Order by mean score (descending)
    df_top["cluster"] = pd.Categorical(df_top["cluster"], categories=top_clusters, ordered=True)

    height = max(width * 0.5, 3.5)
    fig, ax = plt.subplots(figsize=(width, height))

    parts = ax.violinplot(
        [df_top.loc[df_top["cluster"] == c, "score"].values for c in top_clusters],
        positions=range(len(top_clusters)),
        vert=False,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )

    # Style violins
    cmap = plt.colormaps["magma"]
    norm = Normalize(vmin=0, vmax=cluster_means.iloc[0])
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(norm(cluster_means.iloc[i])))
        body.set_alpha(0.7)
        body.set_edgecolor("grey")
        body.set_linewidth(0.5)
    if "cmeans" in parts:
        parts["cmeans"].set_linewidth(0.8)
        parts["cmeans"].set_color("black")
    if "cmedians" in parts:
        parts["cmedians"].set_linewidth(0.5)
        parts["cmedians"].set_color("grey")
        parts["cmedians"].set_linestyle("--")

    ax.set_yticks(range(len(top_clusters)))
    ax.set_yticklabels(top_clusters, fontsize=6)
    ax.set_xlabel("AUCell score")
    ax.tick_params(axis="x", labelsize=6)

    # Highest-mean cluster at the top of the plot (top_clusters[0]) rather
    # than at y=0 which matplotlib renders at the bottom.
    ax.invert_yaxis()

    # Explicit legend for mean/median — without it the two vertical ticks
    # inside each horizontal violin read as an ambiguous "I"-shape.
    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=0.8, label="mean"),
        plt.Line2D([0], [0], color="grey", linewidth=0.5, linestyle="--", label="median"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=5,
        frameon=False, handlelength=1.5, handletextpad=0.4,
    )

    logger.info("figure_aucell_violins: %d clusters shown", len(top_clusters))

    return fig


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
    height = max(width * 0.5, 3.5)
    fig, ax = plt.subplots(figsize=(width, height))

    parts = ax.violinplot(
        [df_top.loc[df_top["cluster"] == c, "score"].values for c in top_clusters],
        positions=range(len(top_clusters)),
        vert=False, showmeans=True, showmedians=False, showextrema=False,
    )
    cmap = plt.colormaps["viridis"]
    zvals = z.loc[top_clusters].to_numpy()
    _zmin = float(np.nanmin(zvals))
    _zmax = float(np.nanmax(zvals))
    if not np.isfinite(_zmin) or not np.isfinite(_zmax) or _zmax <= _zmin:
        _zmax = _zmin + 1.0
    norm = Normalize(vmin=_zmin, vmax=_zmax)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(norm(zvals[i])))
        body.set_alpha(0.7)
        body.set_edgecolor("grey")
        body.set_linewidth(0.5)
    if "cmeans" in parts:
        parts["cmeans"].set_linewidth(0.8)
        parts["cmeans"].set_color("black")

    # Annotate each violin with its z-score
    for i, c in enumerate(top_clusters):
        ax.text(
            df_top.loc[df_top["cluster"] == c, "score"].max(), i,
            f"  z={zvals[i]:.1f}", va="center", ha="left", fontsize=5,
        )

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

    # Remove zero scores for cleaner visualization
    nonzero = aucell_scores[aucell_scores > 0]
    all_scores = aucell_scores

    ax.hist(all_scores, bins=100, color="steelblue", edgecolor="none",
            alpha=0.8, density=True)

    # Mark percentiles
    for pct, ls, lbl in [(90, "--", "90th"), (95, "-.", "95th"), (99, ":", "99th")]:
        val = np.percentile(all_scores, pct)
        ax.axvline(val, color="firebrick", linestyle=ls, linewidth=0.8, alpha=0.8)
        ax.text(val, ax.get_ylim()[1] * 0.95, f" {lbl}\n {val:.4f}",
                fontsize=5, color="firebrick", va="top")

    mean_val = np.mean(all_scores)
    ax.axvline(mean_val, color="black", linestyle="-", linewidth=0.8)
    ax.text(mean_val, ax.get_ylim()[1] * 0.80, f" mean\n {mean_val:.4f}",
            fontsize=5, color="black", va="top")

    ax.set_xlabel("AUCell score")
    ax.set_ylabel("Density")
    ax.set_title("AUCell score distribution (all cells)")

    n_zero = int((all_scores == 0).sum())
    n_total = len(all_scores)
    ax.text(0.98, 0.98,
            f"n = {n_total:,}\nzero = {n_zero:,} ({100*n_zero/n_total:.1f}%)",
            transform=ax.transAxes, fontsize=5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    logger.info("figure_aucell_histogram: %d cells, mean=%.4f, 95th=%.4f",
                n_total, mean_val, np.percentile(all_scores, 95))

    return fig


# ---------------------------------------------------------------------------
# Main Figure 1e: Composite Consensus Ranking Heatmap
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

    For each cluster in *cluster_order* (typically the top hits from the
    composite ranking) two horizontal bars are drawn:

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


