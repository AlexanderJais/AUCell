"""Figure 2 — Pnoc expression across the preoptic area in HypoMap.

Standalone driver. Loads the atlas, computes the Figure 2 inputs via
``figure2_inputs`` and renders via ``figures.figure_gene_poa_umap`` — the
exact same code path the Streamlit app uses, so the panels are identical.

Two-panel UMAP on the full atlas (grey backdrop), restricted overlays to all
preoptic cells (Medial + Lateral preoptic + (Anterior/Preoptic)Periventricular
region):

  left  : cluster identity — the top Pnoc-expressing preoptic clusters in
          colour over the grey atlas.
  right : Pnoc expression (CP10k log1p), magma; non-expressing preoptic cells
          join the grey backdrop.

    python figure2_pnoc_poa.py /path/to/hypomap.h5ad [output_dir] [top_n]

Writes fig2_pnoc_poa.{pdf,svg,png} and fig2_pnoc_poa_clusters.csv.
"""

import sys
from pathlib import Path

import numpy as np
import scanpy as sc

from figure2_inputs import (
    resolve_gene_index, gene_poa_inputs, region_cell_mask,
)
from figures import figure_gene_poa_umap

GENE = "Pnoc"
ANNOTATION_COL = "C185_named"
POA_KEYWORDS = ("preoptic",)
REGION_LABEL = "preoptic"
MIN_CELLS = 20
DEFAULT_TOP_N = 6
SUBSAMPLE = 50000  # backdrop density; matches the app's umap_subsample default


def main(h5ad_path, out_dir=".", top_n=DEFAULT_TOP_N):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path, backed="r")

    hit = resolve_gene_index(adata, GENE)
    if hit is None:
        raise SystemExit(f"Gene '{GENE}' not found in atlas.")
    gene_idx, disp, use_raw = hit
    print(f"{GENE} -> column {gene_idx} ({disp}, use_raw={use_raw})")

    labels = adata.obs[ANNOTATION_COL].astype(str).values
    umap = np.asarray(adata.obsm["X_umap"])
    region_mask = region_cell_mask(adata, keywords=POA_KEYWORDS)
    print(f"Preoptic cells: {int(region_mask.sum()):,} / {len(region_mask):,}")

    expr, expressing, ranking, highlight = gene_poa_inputs(
        adata, gene_idx, use_raw, region_mask, labels,
        min_cells=MIN_CELLS, top_n=top_n,
    )
    ranking.to_csv(out_dir / "fig2_pnoc_poa_clusters.csv")
    print(f"Highlighting top {top_n} clusters:\n", ranking.head(top_n).round(3))

    subsample = None
    if adata.n_obs > SUBSAMPLE:
        subsample = np.sort(np.random.default_rng(42).choice(
            adata.n_obs, size=SUBSAMPLE, replace=False))

    fig = figure_gene_poa_umap(
        umap_coords=umap, cell_labels=labels, region_mask=region_mask,
        gene_expr=expr, expressing_mask=expressing,
        highlight_clusters=highlight, gene_name=GENE, region_label=REGION_LABEL,
        subsample_idx=subsample,
    )
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
