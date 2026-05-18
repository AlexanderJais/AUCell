# AUCell

Standalone Streamlit app for **AUCell-based mapping of a bulk-RNA-seq DE
signature onto a single-cell atlas**, with an expression-matched empirical
null, optional signature refinement, optional region restriction, and a
Cre-driver sanity panel.

The worked example bundled here is **bacTRAP → HypoMap** (Steuernagel et al.,
*Nature Metabolism* 2022): a hypothalamic bacTRAP IP-vs-Input differential
expression result is mapped onto the murine hypothalamic single-cell atlas.
The pipeline itself is **atlas-agnostic** — it consumes any DE table with
`log2FoldChange` / `padj` and any AnnData atlas with a `raw` counts layer
and cluster annotations.

## What the app does

1. **Signature extraction.** From the DE table, keep genes with
   `padj < cutoff` and `log2FC > cutoff` (default 0.05 / 1.0), then rank by
   π-score (or another user-chosen metric) and take the top *N*.
2. **Signature refinement (optional).** Drop candidate signature genes that
   are either undetectable in the atlas or too broadly expressed to be
   cell-type-specific. Every drop is logged with a reason
   (`signature_refinement_droplog.csv`).
3. **AUCell scoring.** For every cell, compute the area under the recovery
   curve of the signature within the top-fraction of expressed genes (Aibar
   et al. 2017). Output: one score per cell on [0, 1].
4. **Empirical-null AUCell.** Score *N* expression-bin-matched random
   control gene sets and report per-cluster `null_mean`, `null_sd`,
   `z_empirical`, `pvalue_empirical`, `qvalue_empirical`.
5. **Cluster-level statistics.** Welch's one-sided *t*-test
   (cluster-vs-rest) with BH-FDR.
6. **Region restriction (optional).** Subset the atlas to a region
   (e.g. preoptic area in HypoMap) before running the entire pipeline.
   Keyword-driven over a region column in `obs`; NA-labelled cells are
   admitted at the **cluster** level (a cluster is kept iff it has ≥1
   keyword-matching cell or every one of its cells is NA), so clusters
   that are merely missing regional annotation in the atlas don't leak
   into the region-restricted analysis. See METHODS §8.
7. **Cre-driver sanity panel.** For a user-chosen gene (default `Pnoc`),
   report per-cluster mean expression and fraction expressing across the
   top AUCell-ranked clusters — useful for validating Cre-line-derived
   bacTRAP experiments.

## Figures the app produces

Tabs in the app:

| Tab | Figures |
| --- | --- |
| Data Overview | bacTRAP volcano (S1) |
| AUCell (Main) | AUCell UMAP (1a), cell-type UMAP highlight (1b), per-cluster bar (S2), violin (1c), companion z-violin (1c′), global histogram (S3) |
| Cre-driver Check | Per-cluster expression / fraction-expressing diagnostic |
| UMAP Projection | Paired AUCell + cell-type UMAP (S4) |
| Export | All figures (PDF + SVG) + all tables (CSV) in one ZIP |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Provide:

- a bacTRAP-style DE CSV/TSV with `log2FoldChange`, `padj`, and a gene-symbol
  or Ensembl-ID column (gene column is auto-detected, override in the
  sidebar);
- an AnnData `.h5ad` atlas with a `raw` counts layer, `X_umap` in `obsm`,
  and at least one cluster annotation column in `obs`.

## Repository layout

```
analysis.py                          # signature extraction, AUCell, empirical null, cluster stats
figures.py                           # Nature-style figure generators
data_loading.py                      # AnnData / DE table I/O, gene matching, POA masks
app.py                               # Streamlit UI
requirements.txt
tests/
  test_aucell_scoring.py             # AUCell rank/recovery semantics, raw-layer guard
  test_cache_fingerprint.py          # atlas-stat cache key invariants
  test_cluster_enrichment.py         # Welch t-test / FDR cluster stats
  test_figures_smoke.py              # figure generators run end-to-end (needs adjustText)
  test_gene_extraction.py            # gene-column auto-detection, Ensembl→symbol fallback
  test_poa_restriction.py            # POA mask (per-cell + cluster-level NA admission)
  test_poa_view_equivalence.py       # adata[mask].copy() ↔ full-atlas AUCell parity
  test_signature_refinement.py       # detectability + specificity filter, drop log
scripts/
  validate_acceptance_criteria.py    # live-atlas acceptance criteria for the full pipeline
```

`scripts/validate_acceptance_criteria.py` runs the AUCell pipeline against
a real HypoMap-like atlas and asserts the empirical-null acceptance
criteria documented in `METHODS.md`.

## Tests

```bash
python -m pytest tests/ -q
```

## Citing

If you use this tool, please cite Aibar et al. 2017 (AUCell) and the atlas
you mapped onto.
