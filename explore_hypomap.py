"""Standalone exploration of the HypoMap atlas.

Run on the machine that has the .h5ad file. No Streamlit needed — just
scanpy / anndata / pandas / numpy.

    python explore_hypomap.py /path/to/hypomap.h5ad

It prints:
  - basic shape, obs columns, obsm keys (UMAP)
  - the annotation columns (C7/C66/C185_named) and their cardinality
  - the unique values of Region_summarized (so we can see exactly how the
    preoptic area / any nucleus is labelled)
  - whether Pnoc is in the gene set (var / raw.var, symbol or Ensembl)
  - every cluster whose name mentions "Pnoc", with its POA cell count and
    the regions its cells fall in  -> this is how we pin down "the MPN cluster"
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)


def find_umap_key(adata):
    for k in ("X_umap", "X_UMAP", "umap", "X_umap_aligned"):
        if k in adata.obsm:
            return k
    return None


def find_gene(adata, symbol):
    """Return where `symbol` lives: ('var'|'raw', index, displayed_name) or None.

    Checks var_names and raw.var_names directly, then any var column that
    looks like a gene-symbol column (in case var_names are Ensembl IDs).
    """
    sym = symbol.lower()

    def _search(var_names, var_df, where):
        names = np.array([str(v) for v in var_names])
        # direct symbol match on the index
        hit = np.where(np.char.lower(names) == sym)[0]
        if len(hit):
            return (where, int(hit[0]), names[hit[0]], "var_names")
        # fall back to symbol columns (CELLxGENE uses 'feature_name')
        for col in ("feature_name", "gene_name", "gene_symbol", "symbol",
                    "external_gene_name", "features", "Gene", "feature_id"):
            if var_df is not None and col in var_df.columns:
                colvals = var_df[col].astype(str).str.lower().values
                hit = np.where(colvals == sym)[0]
                if len(hit):
                    return (where, int(hit[0]), var_df[col].iloc[hit[0]], col)
        return None

    res = _search(adata.var_names, adata.var, "var")
    if res:
        return res
    if adata.raw is not None:
        return _search(adata.raw.var_names, adata.raw.var, "raw")
    return None


def main(path):
    print(f"Loading {path} ...")
    adata = sc.read_h5ad(path, backed="r")  # backed='r' keeps memory low
    print(f"\n=== shape: {adata.n_obs:,} cells x {adata.n_vars:,} vars ===")

    print("\n--- obs columns ---")
    print(list(adata.obs.columns))

    print("\n--- obsm keys ---")
    print(list(adata.obsm.keys()), " | UMAP key ->", find_umap_key(adata))

    print("\n--- raw present? ---")
    print("raw:", "yes" if adata.raw is not None else "no",
          f"({adata.raw.n_vars:,} vars)" if adata.raw is not None else "")

    # Annotation columns -------------------------------------------------
    for col in ("C7_named", "C66_named", "C185_named"):
        if col in adata.obs.columns:
            n = adata.obs[col].astype(str).nunique()
            print(f"\n--- {col}: {n} unique clusters ---")

    # Region labels ------------------------------------------------------
    region_col = "Region_summarized"
    if region_col in adata.obs.columns:
        print(f"\n--- unique values of {region_col} ---")
        vc = adata.obs[region_col].astype(str).value_counts(dropna=False)
        print(vc)
    else:
        print(f"\n!! '{region_col}' not in obs. Region-like columns:")
        print([c for c in adata.obs.columns if "region" in c.lower()
               or "area" in c.lower() or "nucleus" in c.lower()])

    # var columns (where do gene symbols live?) --------------------------
    print("\n--- var.columns ---")
    print(list(adata.var.columns))
    print("var_names sample:", list(adata.var_names[:5]))
    if adata.raw is not None:
        print("--- raw.var.columns ---")
        print(list(adata.raw.var.columns))
        print("raw.var_names sample:", list(adata.raw.var_names[:5]))

    # Pnoc gene ----------------------------------------------------------
    print("\n--- Pnoc gene lookup ---")
    hit = find_gene(adata, "Pnoc")
    print("Pnoc ->", hit if hit else "NOT FOUND")
    if hit is None:
        # brute-force substring scan across index + every var column
        print("Substring scan for 'pnoc' across var index and columns:")
        for where, vn, vdf in (("var", adata.var_names, adata.var),
                               ("raw", adata.raw.var_names if adata.raw is not None else [],
                                adata.raw.var if adata.raw is not None else None)):
            idx_hits = [str(v) for v in vn if "pnoc" in str(v).lower()]
            if idx_hits:
                print(f"  {where}.var_names contains: {idx_hits[:10]}")
            if vdf is not None:
                for col in vdf.columns:
                    try:
                        m = vdf[col].astype(str).str.lower().str.contains("pnoc", na=False)
                        if m.any():
                            print(f"  {where}.var['{col}']: {vdf[col][m].head(5).tolist()}")
                    except Exception:
                        pass

    # Pnoc-named clusters + their regions --------------------------------
    ann = "C185_named" if "C185_named" in adata.obs.columns else (
        "C66_named" if "C66_named" in adata.obs.columns else None)
    if ann and region_col in adata.obs.columns:
        labels = adata.obs[ann].astype(str)
        regions = adata.obs[region_col].astype(str)
        pnoc_clusters = sorted(labels[labels.str.contains("Pnoc", case=False)].unique())
        print(f"\n--- clusters in '{ann}' mentioning 'Pnoc' ({len(pnoc_clusters)}) ---")
        for cl in pnoc_clusters:
            mask = labels == cl
            reg_counts = regions[mask].value_counts().head(5).to_dict()
            print(f"\n  {cl}  (n={int(mask.sum()):,})")
            print(f"    regions: {reg_counts}")

        # POA cell count per cluster (Region_summarized contains 'preoptic')
        poa = regions.str.contains("preoptic", case=False, na=False)
        print(f"\n--- POA cells total (Region_summarized ~ 'preoptic'): {int(poa.sum()):,} ---")
        print("Top clusters within POA:")
        print(labels[poa].value_counts().head(20))

    print("\nDone. backed mode used; nothing was modified.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
