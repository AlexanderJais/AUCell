"""Figure 2 — Pnoc expression across the preoptic area in HypoMap.

Two-panel UMAP, restricted to all preoptic cells (Medial + Lateral preoptic
+ (Anterior/Preoptic)Periventricular region):

  left  : cluster identity, fig_1a style — the top Pnoc-expressing clusters
          drawn in colour over a grey background of the other preoptic cells.
  right : Pnoc expression (CP10k log1p), magma colormap + colorbar.

Together they answer "in which clusters is Pnoc expressed in the preoptic
area". Styling matches figures.figure_celltype_umap / figure_umap_enrichment
(Nature spec), but this script is self-contained — only needs
scanpy / anndata / numpy / pandas / matplotlib.

    python figure2_pnoc_poa.py /path/to/hypomap.h5ad [output_dir] [top_n]

Writes fig2_pnoc_poa.{pdf,svg,png} and fig2_pnoc_poa_clusters.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

GENE = "Pnoc"
ANNOTATION_COL = "C185_named"
REGION_COL = "Region_summarized"
POA_KEYWORD = "preoptic"
MIN_CELLS = 20          # cluster floor, matches the rest of the analysis
DEFAULT_TOP_N = 6       # how many clusters to highlight / colour


# --- styling (mirrors figures.setup_nature_style) -------------------------
def setup_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "figure.facecolor": "white",
        "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def qualitative_palette(n):
    import seaborn as sns
    if n <= 10:
        return list(sns.color_palette("tab10", n).as_hex())
    return list(sns.color_palette("tab20", min(n, 20)).as_hex())


def add_umap_axis_arrows(ax, length=0.07, origin=(0.02, 0.02)):
    x0, y0 = origin
    style = dict(arrowstyle="-|>,head_length=1.5,head_width=1.0",
                 linewidth=0.5, color="black", shrinkA=0, shrinkB=0,
                 mutation_scale=5)
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=style)
    ax.annotate("", xy=(x0, y0 + length), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=style)
    ax.text(x0 + length + 0.004, y0, "UMAP1", transform=ax.transAxes,
            ha="left", va="center", fontsize=4.5)
    ax.text(x0, y0 + length + 0.004, "UMAP2", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=4.5, rotation=90,
            rotation_mode="anchor")


def find_gene_index(adata, symbol):
    """Return (column_index, displayed_name). Handles Ensembl var_names with a
    feature_name symbol column (CELLxGENE format)."""
    sym = symbol.lower()
    names = np.char.lower(np.array([str(v) for v in adata.var_names]))
    hit = np.where(names == sym)[0]
    if len(hit):
        return int(hit[0]), str(adata.var_names[hit[0]])
    for col in ("feature_name", "gene_name", "gene_symbol", "symbol"):
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).str.lower().values
            hit = np.where(vals == sym)[0]
            if len(hit):
                return int(hit[0]), str(adata.var[col].iloc[hit[0]])
    raise SystemExit(f"Gene '{symbol}' not found in atlas.")


def main(h5ad_path, out_dir=".", top_n=DEFAULT_TOP_N):
    setup_nature_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path} (backed) ...")
    adata = sc.read_h5ad(h5ad_path, backed="r")

    gidx, disp = find_gene_index(adata, GENE)
    print(f"{GENE} -> column {gidx} ({disp})")

    # Pull the single Pnoc column, CP10k + log1p per cell.
    col = adata[:, gidx].X
    col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    depth = np.asarray(adata.obs["nCount_RNA"], dtype=float)
    depth[depth == 0] = 1.0
    expr = np.log1p(col / depth * 1e4)

    labels = adata.obs[ANNOTATION_COL].astype(str).values
    regions = adata.obs[REGION_COL].astype(str)
    umap = np.asarray(adata.obsm["X_umap"])

    poa = regions.str.contains(POA_KEYWORD, case=False, na=False).values
    print(f"Preoptic cells: {int(poa.sum()):,} / {len(poa):,} total")

    # Rank clusters by mean Pnoc within the preoptic area (>=MIN_CELLS).
    df = pd.DataFrame({"cluster": labels[poa], "expr": expr[poa],
                       "frac": (col[poa] > 0).astype(float)})
    g = (df.groupby("cluster")
           .agg(n_cells=("expr", "size"), mean_lognorm=("expr", "mean"),
                frac_expressing=("frac", "mean")))
    g = g[g["n_cells"] >= MIN_CELLS].sort_values("mean_lognorm", ascending=False)
    g.to_csv(out_dir / "fig2_pnoc_poa_clusters.csv")
    highlight = g.head(top_n).index.tolist()
    print(f"Highlighting top {top_n} clusters:\n", g.head(top_n).round(3))

    # --- figure: two panels, full-atlas grey backdrop (fig_1a style) -------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                   gridspec_kw={"wspace": 0.6})
    other = "#d9d9d9"

    # left: full atlas grey, top Pnoc clusters (preoptic subset) coloured
    ax1.scatter(umap[:, 0], umap[:, 1], c=other, s=1.5, alpha=0.4,
                edgecolors="none", rasterized=True)
    palette = qualitative_palette(len(highlight))
    cmap_cl = {cl: palette[i] for i, cl in enumerate(highlight)}
    for cl in highlight:
        m = poa & (labels == cl)          # only the preoptic cells of the cluster
        ax1.scatter(umap[m, 0], umap[m, 1], c=[cmap_cl[cl]], s=5.0, alpha=0.95,
                    edgecolors="none", rasterized=True)
    ax1.set_title("Top Pnoc clusters (preoptic)")
    ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_visible(False)
    add_umap_axis_arrows(ax1)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap_cl[cl], markersize=3.5, label=cl)
               for cl in highlight]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=other, markersize=3.5, label="Other"))
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               fontsize=4.5, frameon=False, ncol=2, handletextpad=0.2,
               columnspacing=0.5, labelspacing=0.3)

    # right: full atlas grey, Pnoc expression overlaid for expressing
    # preoptic cells only (non-expressing preoptic cells join the grey
    # backdrop so the panel shows where Pnoc actually is, not a field of
    # black zeros from magma's low end).
    ax2.scatter(umap[:, 0], umap[:, 1], c=other, s=1.5, alpha=0.4,
                edgecolors="none", rasterized=True)
    expressing = poa & (col > 0)
    ex_e = expr[expressing]
    vmax = float(np.nanpercentile(ex_e, 98)) or 1.0
    o = np.argsort(ex_e)               # draw brightest last
    sc_h = ax2.scatter(umap[expressing][o, 0], umap[expressing][o, 1],
                       c=ex_e[o], cmap="magma", s=4.0, alpha=0.9,
                       edgecolors="none", rasterized=True,
                       vmin=0.0, vmax=vmax)
    ax2.set_title(f"{GENE} expression (preoptic)")
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    add_umap_axis_arrows(ax2)
    cbar = fig.colorbar(sc_h, ax=ax2, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label(f"{GENE} (log-norm)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"fig2_pnoc_poa.{ext}"
        fig.savefig(path)
        print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else ".",
         int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TOP_N)
