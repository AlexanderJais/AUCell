# Methods — AUCell-based DE-signature → single-cell-atlas mapping

This document describes the AUCell-only pipeline implemented in this
repository. The worked example is bacTRAP → HypoMap; the pipeline accepts
any DE table + AnnData atlas pair.

The multi-method companion (correlation / Fisher / NNLS / GSEA / composite)
lives at
[AlexanderJais/bacTRAP-to-HypoMapMapping](https://github.com/AlexanderJais/bacTRAP-to-HypoMapMapping).

## 1. Inputs

- **DE table** (e.g. DESeq2 output for an IP-vs-Input contrast). Required
  columns: `log2FoldChange`, `padj`. A gene-symbol or Ensembl-ID column is
  auto-detected from the candidate set; if both are present, symbols are
  preferred and Ensembl IDs are used as the fallback. An optional `IP` mean
  column is used to guard against the DESeq2 low-count log₂FC-inflation
  artefact (see *Signature extraction* below).
- **Atlas** (`.h5ad`). Must carry a `raw` counts layer (AUCell is a rank
  method and needs raw counts to be normalisation-insensitive), a UMAP in
  `obsm["X_umap"]`, and at least one cluster annotation column in `obs`.

## 2. Gene matching

Atlas var-names and DE-table gene IDs are matched via a lookup table
populated from `var_names`, the raw layer, and any obvious symbol column
on `var`. The candidate column on the DE side is auto-selected by trying
each plausible column and picking the one with the highest match rate
against the atlas lookup; the user can override the selection in the
sidebar. Match-rate diagnostics are surfaced in the **Data Overview** tab
and logged.

## 3. Signature extraction (`get_enriched_genes`, `rank_enriched_genes`)

Retain rows with `padj < padj_cutoff` (default 0.05), `log2FoldChange >
log2fc_cutoff` (default 1.0), and `IP ≥ min_ip_expression` (default 1.0,
to suppress low-count log₂FC inflation). Rank the survivors by the
user-chosen metric — π-score (default), log₂FC, `-log10(padj)`, or
`log2FC × -log10(padj)` — and take the top *N* (default 50).

## 4. Signature refinement (`filter_signature_genes_by_atlas`)

Two optional filters run **before** ranking and top-*N* selection:

- **Detectability.** Drop candidate genes whose atlas-wide detection rate
  is below `min_detection_rate` (default 0.01) or whose maximum
  per-cluster log-norm mean is below `min_max_cluster_mean` (default 0.5).
  Rationale: a gene that is undetectable in the atlas contributes
  uniformly to AUCell ranks and only adds noise.
- **Specificity.** Drop candidate genes that are expressed in more than
  `specificity_max_cluster_fraction` (default 0.5) of clusters above
  `specificity_cluster_mean_thresh` (default 1.0). Rationale: globally
  highly-expressed genes (housekeeping, ribosomal, well-known broad-
  contaminants like `Sst`, `Polr2h`, `Eid2`, `Pdxp`, `Ppil1`, `Arl6ip4`,
  `Mrpl12`, `Emc9`) drown out cell-type-specific signal in rank-based
  scoring.

Every candidate gene is recorded in a drop-log with its status, detection
rate, max cluster mean, fraction of clusters above threshold, and the
reason it was kept or dropped (`signature_refinement_droplog_*.csv`).

## 5. AUCell scoring (`compute_aucell_scores`)

Per Aibar et al. 2017: for each cell, compute the area under the recovery
curve of the signature within the top-fraction of genes by expression rank
in that cell. The area is normalised to its theoretical maximum so scores
lie on [0, 1]. Implementation details:

- The score is computed against the **raw counts layer** (`adata.raw`) —
  log-norm layers compress the rank distribution and bias AUCell.
- The top-fraction is `aucell_top_fraction` (default 0.05). If the
  signature is larger than that window, `n_top` is bumped to
  `len(signature)` so the recovery curve is well-defined; the UI surfaces
  the effective top-fraction in that case.
- Tie-breaking uses the user-specified seed (default 0) so the signature
  scoring shares the tie-breaking regime with the empirical-null control
  sets (see below).

`validate_aucell_input` runs a sanity check on the atlas layer (max
sampled value, sparsity, non-integer fraction) and emits warnings if the
input looks log-normalised rather than raw.

## 6. Empirical-null AUCell (`compute_empirical_null_aucell`)

Score `N` (default 100) **expression-bin-matched random control gene
sets** — same size as the signature, drawn from the same atlas-wide
mean-log-expression bins as the signature genes (default 10 equal-frequency
bins). For each cluster, compute:

- `null_mean`, `null_sd` — control-set mean AUCell distribution
  parameters,
- `z_empirical = (cluster_mean_observed − null_mean) / null_sd`,
- `pvalue_empirical` — two-sided empirical *p*-value against the
  bin-matched null,
- `qvalue_empirical` — BH-FDR-adjusted *p*-value across the clusters that
  pass the cell-count gate.

Clusters smaller than `min_cluster_size` (default = the per-cluster cell-
count floor used for ranking) get no null row; their entries are left
NaN. The `n_control_sets_used` integer column makes "didn't pass the
gate" distinguishable from "no control sets sampled".

The empirical null is the load-bearing significance test of the
pipeline: a cluster with a high raw AUCell mean but a small empirical
z-score is being driven by gene-rank-distribution width rather than by
signature enrichment.

### Acceptance criteria

`scripts/validate_acceptance_criteria.py` runs the empirical null on a
real HypoMap-like atlas and checks:

1. The signature passes empirical-null significance in at least one
   biologically-plausible cluster.
2. Random gene sets of the same size do **not** pass empirical-null
   significance (Type-I error control).
3. The Pnoc / well-known contaminant tripwires fire as expected when
   refinement is enabled.

Run it against your atlas:

```bash
python scripts/validate_acceptance_criteria.py /path/to/atlas.h5ad
```

## 7. Cluster-level statistics (`compute_cluster_enrichment_stats`)

In parallel with the empirical null, we run a **Welch's one-sided
*t*-test** of each cluster's AUCell distribution against the rest of the
atlas, with BH-FDR across clusters. This gives a fast, classical
significance call (columns `t_stat`, `pvalue`, `qvalue`, `significant`)
that's useful as a sanity check on the empirical-null call — the two
should largely agree, but the empirical null is the authoritative test
because it controls for gene-rank-distribution effects that the
*t*-test does not.

## 8. Region restriction (optional)

When **POA-only mode** is enabled (HypoMap-specific, but adaptable), a
boolean mask over cells is built from a keyword search over a region
column (default `Region_summarized`, default keywords `Preoptic`, `POA`).
The mask is named `mask_signature` (e.g. `poaonly_preoptic_poa`) so
downstream caches and CSV filenames stay distinct from full-atlas runs.

Once active, every downstream computation — cluster means, AUCell, the
empirical null, the Cre-driver sanity panel, the UMAPs — runs against
the POA cell subset (`adata_view`). Clusters that fall below the
`min_poa_cells` floor under the mask are excluded from ranking but
remain visible in `aucell_per_cluster.csv` for transparency.

## 9. Cre-driver sanity panel

For a user-chosen gene (default `Pnoc`), compute per-cluster mean
expression (log-normalised) and fraction of cells expressing the gene
above zero. The panel orders clusters by AUCell mean (or, when the
empirical null is enabled and AUCell is unavailable, by `z_empirical`)
and highlights clusters that fail the `fraction_expressing ≥ threshold`
test — useful for assessing whether the top AUCell-ranked clusters could
plausibly host a Cre-driver in a bacTRAP experiment.

Caveats are surfaced in-app: Cre-lox is permanent lineage tracing
(any cell that *ever* expressed the gene is labelled, even if current
mRNA is undetectable); HypoMap is single-nucleus and is prone to
dropout for neuropeptides; expression is state-dependent. The panel is
a **confidence weight**, not a hard filter.

An optional **baseline Cre-driver expression filter** on the cluster
universe is exposed in the sidebar: with a non-zero threshold, only
clusters whose mean expression for the Cre-driver gene meets the floor
are retained for the AUCell cluster panels and per-cell projections.

## 10. Outputs

### Tables (CSV)

| File | Contents |
| --- | --- |
| `aucell_per_cell.csv` | `cell_id, cluster, aucell_score` |
| `aucell_per_cluster.csv` | `cluster, n_cells, mean, median, std, sem, t_stat, pvalue, qvalue, significant` plus empirical-null columns when active |
| `matched_genes.csv` | DE rows with the resolved gene-name column |
| `enriched_genes.csv` | Post-cutoff signature with rank metric values |
| `signature_refinement_droplog_*.csv` | Per-gene refinement status, detection rate, max cluster mean, frac clusters above thresh, reason |

CSV filenames carry suffixes encoding active filters (`refined`,
`null{N}`, the POA mask signature, the Cre-driver threshold) so a
downloaded subset is identifiable from the filename alone.

### Figures (PDF + SVG, Nature style)

- **1a** AUCell enrichment UMAP (per-cell).
- **1b** Cell-type annotation UMAP with the top-15 AUCell clusters
  highlighted.
- **1c** Per-cluster AUCell violin plots (top-15 by mean).
- **1c′** Companion z-violins (top-15 by empirical z-score) — appears
  when the empirical null is enabled.
- **S1** bacTRAP-style DE volcano (Data Overview).
- **S2** Mean AUCell per cluster bar plot.
- **S3** Global per-cell AUCell histogram.
- **S4** Paired AUCell-vs-cell-type UMAP (UMAP Projection tab).
- **Cre-driver diagnostic** — per-cluster expression / fraction.

All figures follow Nature style: editable text in PDF/SVG, 7 pt axis
labels, 6 pt ticks, 0.5 pt axes, single (89 mm) or double (183 mm)
column widths, colour-blind-safe palettes.

## References

- Aibar, S., González-Blas, C.B., Moerman, T. *et al.* SCENIC: single-cell
  regulatory network inference and clustering. *Nat. Methods* **14**,
  1083–1086 (2017). AUCell algorithm.
- Steuernagel, L., Lam, B.Y.H., Klemm, P. *et al.* HypoMap — a
  unified single-cell gene-expression atlas of the murine hypothalamus.
  *Nat. Metab.* **4**, 1402–1419 (2022). Example atlas.
- Multi-method companion repository:
  https://github.com/AlexanderJais/bacTRAP-to-HypoMapMapping
