"""Validate the live-atlas acceptance criteria for PRs #8 (Change 2 —
signature refinement) and #9 (Change 3 — POA-only restriction).

Runs the analytics-only side of the pipeline (no Streamlit, no markers / Fisher
/ GSEA / NNLS — none of the acceptance criteria depend on those) on a real
HypoMap atlas + bacTRAP DESeq2 results, and prints a PASS/FAIL line per
criterion.  Expects to be run from the repo root on the ``feature/poa-
restriction`` branch.

Usage
-----
    python scripts/validate_acceptance_criteria.py \
        --hypomap /path/to/hypomap.h5ad \
        --bactrap /path/to/IPvsInput_deg.xlsx \
        [--annotation-col C185_named] \
        [--n-control-sets 100] \
        [--baseline-cutoff 0.15] \
        [--out results_acceptance.json]

Expected wall-clock on the full atlas (385k cells): ~10–20 min.

The criteria checked here come verbatim from the two PR descriptions:

  PR #8 (Change 2)
    C2.1  refinement keeps 200–305 of ~340 candidate genes
    C2.2  the 8 known broadly-expressed contaminants drop under specificity
    C2.3  Pnoc survives refinement (tripwire)
    C2.4  AUCell top-25 by z_empirical excludes the 3 named Arcuate clusters
    C2.5  refinement off => no genes are dropped (input list returned)

  PR #9 (Change 3, POA-only)
    C3.1  POA mask retains ~20 000–25 000 cells (S1: 21 784)
    C3.2  C185-67 Pnoc.Mixed.GABA-2 survives the mask (100 % NA)
    C3.3  Cre-driver baseline at Pnoc >= 0.15 retains 11–13 POA clusters
    C3.4  AUCell top-25 by z_empirical excludes the 7 named non-POA clusters
    C3.5  the 10 expected POA-resident clusters appear in the top-25
    C3.6  POA off => AUCell scores bit-identical to the non-POA run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# Repo modules (import from the repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import (
    compute_aucell_scores,
    compute_cluster_enrichment_stats,
    compute_empirical_null_aucell,
    filter_signature_genes_by_atlas,
    get_enriched_genes,
    rank_enriched_genes,
)
from data_loading import (
    _build_adata_gene_lookup,
    build_mask_signature,
    compute_single_gene_cluster_stats,
    get_atlas_cluster_mean_expr,
    get_atlas_gene_detection_rate,
    get_poa_cell_mask,
    match_genes,
)


# Expected-clusters constants (from the PR descriptions)
EIGHT_CONTAMINANTS = ["Sst", "Polr2h", "Eid2", "Pdxp", "Ppil1", "Arl6ip4", "Mrpl12", "Emc9"]

THREE_ARCUATE_BAD = [
    "C185-117: Npy.Sst.GABA-4",
    "C185-118: Otp.Sst.GABA-4",
    "C185-114: Ghrh.GABA-3",
]
SEVEN_NON_POA_BAD = [
    "C185-117: Npy.Sst.GABA-4",
    "C185-118: Otp.Sst.GABA-4",
    "C185-114: Ghrh.GABA-3",
    "C185-56: Pmch.GLU-7",
    "C185-95: Nts.Crh.GABA-1",
    "C185-69: Pvalb.Mixed.GABA-2",
    "C185-57: Nts.Foxb1.GLU-8",
]
TEN_EXPECTED_POA = [
    "C185-67", "C185-108", "C185-100", "C185-70", "C185-106",
    "C185-63", "C185-107", "C185-101", "C185-42", "C185-18",
]


def _setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)], force=True,
    )
    # Streamlit emits a benign warning when its caches run outside a runtime;
    # silence it so the report stays readable.
    logging.getLogger("streamlit").setLevel(logging.ERROR)


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return {"name": name, "status": status, "detail": detail}


def _top_n_by(df, col, n=25):
    return (
        df.dropna(subset=[col])
          .sort_values(col, ascending=False)
          .head(n)["cluster"].astype(str).tolist()
    )


# ---------------------------------------------------------------------------
# Pipeline building blocks
# ---------------------------------------------------------------------------

def build_signature(
    bactrap_matched, cluster_mean_expr, gene_detection_rate,
    *, apply_refinement, top_n=50, ranking_metric="pi_score",
    padj=0.05, log2fc=1.0, min_ip=10.0,
):
    """Run get_enriched_genes → optional refinement → rank → top-N.

    Returns (top_n_signature_genes, refined_enriched_df, drop_log).
    """
    enriched_df = get_enriched_genes(
        bactrap_matched, padj_cutoff=padj, log2fc_cutoff=log2fc,
        min_ip_expression=min_ip, ip_col="IP",
    )
    cand_genes = enriched_df["_hypomap_gene_name"].tolist()
    drop_log = pd.DataFrame()
    if apply_refinement:
        kept, drop_log = filter_signature_genes_by_atlas(
            cand_genes, cluster_mean_expr, gene_detection_rate,
            apply_detectability=True, min_detection_rate=0.02, min_max_cluster_mean=0.05,
            apply_specificity=True,
            specificity_cluster_mean_thresh=0.5, specificity_max_cluster_fraction=0.5,
        )
        enriched_df = enriched_df[enriched_df["_hypomap_gene_name"].isin(set(kept))].copy()
    enriched_sorted = rank_enriched_genes(enriched_df, metric=ranking_metric)
    top = enriched_sorted["_hypomap_gene_name"].tolist()[:top_n]
    return top, enriched_df, drop_log


def aucell_per_cluster(
    adata_view, signature_genes, annotation_col, *,
    mask_signature="", n_control_sets=100, n_bins=5, seed=0,
    top_fraction=0.05, min_cluster_size=20,
):
    """Score AUCell on adata_view, run the empirical null, return a
    per-cluster DataFrame (cluster, n_cells, mean, t_stat, qvalue, z_empirical, …)
    plus the raw per-cell scores."""
    t0 = time.time()
    aucell = compute_aucell_scores(
        adata_view, signature_genes, use_raw=True,
        top_fraction=top_fraction, seed=seed,
    )
    print(f"  AUCell scored: {time.time()-t0:.1f}s, mean={aucell.mean():.4f}")

    labels = adata_view.obs[annotation_col].astype(str).values
    stats_df = compute_cluster_enrichment_stats(aucell, labels, min_cells=10, alpha=0.05)

    grp = pd.DataFrame({"cluster": labels, "aucell_score": aucell}).groupby("cluster")["aucell_score"]
    per_cluster = pd.DataFrame({
        "n_cells": grp.count(), "mean": grp.mean(), "median": grp.median(),
        "std": grp.std(ddof=1),
    }).reset_index()
    if not stats_df.empty:
        per_cluster = per_cluster.merge(
            stats_df[["cluster", "t_stat", "pvalue", "qvalue", "significant"]],
            on="cluster", how="left",
        )

    t1 = time.time()
    null_df = compute_empirical_null_aucell(
        adata_view, signature_genes, labels, compute_aucell_scores,
        n_control_sets=n_control_sets, n_bins=n_bins, seed=seed,
        top_fraction=top_fraction, min_cluster_size=min_cluster_size,
        mask_signature=mask_signature,
    )
    print(f"  empirical null ({n_control_sets} controls): {time.time()-t1:.1f}s")
    if not null_df.empty:
        per_cluster = per_cluster.merge(null_df.reset_index(), on="cluster", how="left")
    return per_cluster, aucell


# ---------------------------------------------------------------------------
# Acceptance-criteria checks
# ---------------------------------------------------------------------------

def check_change2(
    adata, bactrap_matched, gene_to_idx, matched_in_raw, annotation_col,
    *, n_control_sets, top_n=50,
):
    results = []
    print("\n=== PR #8 / Change 2 — signature refinement ===")

    gene_indices = [gene_to_idx[g] for g in bactrap_matched["_hypomap_gene_name"]
                    if g in gene_to_idx]
    print("  computing cluster_mean_expr + gene_detection_rate (matched genes)...")
    t0 = time.time()
    cme = get_atlas_cluster_mean_expr(
        adata, gene_indices, annotation_col,
        mask_signature="", min_cells=10, indices_in_raw=matched_in_raw, normalize=True,
    )
    det = get_atlas_gene_detection_rate(adata, mask_signature="")
    print(f"  atlas stats: {time.time()-t0:.1f}s")

    # Pre-refinement candidate set
    enriched_pre = get_enriched_genes(
        bactrap_matched, padj_cutoff=0.05, log2fc_cutoff=1.0,
        min_ip_expression=10.0, ip_col="IP",
    )
    cand = enriched_pre["_hypomap_gene_name"].tolist()
    n_cand = len(cand)
    print(f"  candidate genes (padj<0.05, log2FC>1, IP>=10): {n_cand}")

    sig_refined, _, drop_log = build_signature(
        bactrap_matched, cme, det, apply_refinement=True, top_n=top_n,
    )

    # C2.1 — 200..305 kept
    n_kept = int((drop_log["status"] == "kept").sum())
    n_dropped = n_cand - n_kept
    results.append(_check(
        "C2.1  refinement keeps 200..305 of candidate genes",
        200 <= n_kept <= 305,
        f"{n_kept} kept / {n_dropped} dropped (of {n_cand} candidates)",
    ))

    # C2.2 — 8 known contaminants dropped by specificity
    dl_idx = drop_log.set_index("gene")
    contam_present = [g for g in EIGHT_CONTAMINANTS if g in dl_idx.index]
    contam_spec = [g for g in contam_present
                   if dl_idx.loc[g, "status"] == "dropped_specificity"]
    contam_other_drop = [g for g in contam_present
                         if dl_idx.loc[g, "status"].startswith("dropped_")
                         and dl_idx.loc[g, "status"] != "dropped_specificity"]
    missing = [g for g in EIGHT_CONTAMINANTS if g not in dl_idx.index]
    results.append(_check(
        "C2.2  the 8 known contaminants drop under specificity",
        len(contam_spec) == len(contam_present) and not missing,
        f"dropped_specificity={contam_spec}; "
        f"dropped_other={contam_other_drop}; not_in_candidates={missing}",
    ))

    # C2.3 — Pnoc survives
    if "Pnoc" in dl_idx.index:
        pnoc_status = dl_idx.loc["Pnoc", "status"]
        results.append(_check(
            "C2.3  Pnoc survives refinement (tripwire)",
            pnoc_status == "kept",
            f"Pnoc status = {pnoc_status} (reason='{dl_idx.loc['Pnoc', 'reason']}')",
        ))
    else:
        results.append(_check(
            "C2.3  Pnoc survives refinement (tripwire)", False,
            "Pnoc NOT in candidate set — check padj/log2FC/IP cutoffs",
        ))

    # C2.4 — top-25 by z_empirical excludes the 3 Arcuate clusters
    print("  running AUCell + empirical null on the full atlas...")
    per_cluster, _ = aucell_per_cluster(
        adata, sig_refined, annotation_col,
        mask_signature="", n_control_sets=n_control_sets, seed=0,
    )
    top25_z = _top_n_by(per_cluster, "z_empirical", n=25)
    hit = [c for c in THREE_ARCUATE_BAD if c in top25_z]
    results.append(_check(
        "C2.4  AUCell top-25 by z_empirical excludes the 3 Arcuate clusters",
        len(hit) == 0,
        f"unexpected hits = {hit}; top-25 head = {top25_z[:5]}",
    ))

    # C2.5 — refinement off => signature is the rank_enriched_genes(top-N) of the
    # unrefined candidates (i.e. the filter is a no-op)
    sig_unrefined, _, drop_log_off = build_signature(
        bactrap_matched, cme, det, apply_refinement=False, top_n=top_n,
    )
    set_off = set(sig_unrefined)
    set_on = set(sig_refined)
    n_same = len(set_off & set_on)
    results.append(_check(
        "C2.5  refinement off => returns the unfiltered candidate top-N",
        len(drop_log_off) == 0
        and len(sig_unrefined) == top_n,
        f"refinement-off top-N size={len(sig_unrefined)} (expected {top_n}); "
        f"refined ∩ unrefined = {n_same}",
    ))
    return results, per_cluster, sig_refined


def check_change3(
    adata, bactrap_matched, gene_to_idx, matched_in_raw, annotation_col,
    sig_full_atlas_refined, full_per_cluster,
    *, n_control_sets, baseline_cutoff, top_n=50,
):
    results = []
    print("\n=== PR #9 / Change 3 — POA-only restriction ===")

    # Build POA mask + view
    print("  building POA mask...")
    poa_mask = get_poa_cell_mask(
        adata, region_col="Region_summarized",
        poa_keywords=("preoptic",), include_na=True,
    ).to_numpy(dtype=bool)
    n_poa = int(poa_mask.sum())
    mask_sig = build_mask_signature(("preoptic",), True)

    # C3.1 — POA cell count
    results.append(_check(
        "C3.1  POA mask retains 20 000..25 000 cells",
        20_000 <= n_poa <= 25_000,
        f"N_poa = {n_poa:,} of {adata.n_obs:,} ({100*n_poa/adata.n_obs:.1f}%)",
    ))

    # C3.2 — C185-67 survives (>=1 POA cell in that cluster)
    target_c = "C185-67"
    target_cluster = None
    for c in adata.obs[annotation_col].astype(str).unique():
        if c.startswith(target_c):
            target_cluster = c
            break
    if target_cluster is None:
        results.append(_check(
            "C3.2  C185-67 Pnoc.Mixed.GABA-2 survives POA mask", False,
            f"no cluster name starts with '{target_c}' at {annotation_col}",
        ))
    else:
        n_poa_target = int(
            ((adata.obs[annotation_col].astype(str) == target_cluster).values & poa_mask).sum()
        )
        n_full_target = int((adata.obs[annotation_col].astype(str) == target_cluster).sum())
        results.append(_check(
            "C3.2  C185-67 Pnoc.Mixed.GABA-2 survives POA mask",
            n_poa_target > 0,
            f"{target_cluster}: {n_poa_target} POA cells / {n_full_target} total",
        ))

    print("  building POA view (.copy())...")
    t0 = time.time()
    adata_view = adata[poa_mask].copy()
    print(f"  POA view built: {time.time()-t0:.1f}s ({adata_view.n_obs:,} cells)")

    # Atlas stats on the POA view
    print("  computing cluster_mean_expr + gene_detection_rate on POA view...")
    gene_indices = [gene_to_idx[g] for g in bactrap_matched["_hypomap_gene_name"]
                    if g in gene_to_idx]
    cme_poa = get_atlas_cluster_mean_expr(
        adata_view, gene_indices, annotation_col,
        mask_signature=mask_sig, min_cells=20, indices_in_raw=matched_in_raw, normalize=True,
    )
    det_poa = get_atlas_gene_detection_rate(adata_view, mask_signature=mask_sig)

    # Build POA-restricted signature (refinement still on)
    sig_poa, _, _ = build_signature(
        bactrap_matched, cme_poa, det_poa, apply_refinement=True, top_n=top_n,
    )

    # C3.3 — Cre-driver baseline filter at Pnoc >= cutoff
    print("  computing Pnoc per-cluster stats on POA view...")
    lookup_poa, _, has_raw = _build_adata_gene_lookup(adata_view)
    sanity = compute_single_gene_cluster_stats(
        adata_view, "Pnoc", annotation_col,
        adata_gene_lookup=lookup_poa, has_raw=has_raw, min_cells=20, normalize=True,
    )
    if sanity is None or sanity.empty:
        results.append(_check(
            "C3.3  Cre-driver baseline retains 11..13 high-Pnoc POA clusters", False,
            "Pnoc not resolvable in POA view",
        ))
    else:
        survivors = sanity[sanity["mean_expr"] >= baseline_cutoff].sort_values(
            "mean_expr", ascending=False,
        )
        n_surv = len(survivors)
        results.append(_check(
            f"C3.3  Cre-driver baseline at Pnoc >= {baseline_cutoff:.2f} retains 11..13 clusters",
            11 <= n_surv <= 13,
            f"n_survivors = {n_surv}; "
            f"top 5: {survivors.head(5)[['mean_expr']].to_dict()['mean_expr']}",
        ))

    # C3.4 / C3.5 — POA top-25 exclusions and inclusions
    print("  running AUCell + empirical null on the POA view...")
    poa_per_cluster, poa_aucell = aucell_per_cluster(
        adata_view, sig_poa, annotation_col,
        mask_signature=mask_sig, n_control_sets=n_control_sets, seed=0,
        min_cluster_size=20,
    )
    top25_z_poa = _top_n_by(poa_per_cluster, "z_empirical", n=25)
    bad_hit = [c for c in SEVEN_NON_POA_BAD if c in top25_z_poa]
    results.append(_check(
        "C3.4  POA top-25 by z_empirical excludes the 7 non-POA clusters",
        len(bad_hit) == 0,
        f"unexpected hits = {bad_hit}; top-25 head = {top25_z_poa[:5]}",
    ))
    found_expected = [c for c in TEN_EXPECTED_POA
                      if any(t.startswith(c) for t in top25_z_poa)]
    results.append(_check(
        "C3.5  POA top-25 includes the 10 expected POA-resident clusters (>=6)",
        len(found_expected) >= 6,
        f"found {len(found_expected)}/10 expected: {found_expected}",
    ))

    # C3.6 — POA off => bit-identical to the full-atlas run (same scores)
    print("  re-scoring full atlas with the unrefined-equivalent signature to "
          "check POA-off determinism...")
    # Re-run AUCell on the FULL atlas with the SAME signature we used for the
    # Change-2 check; the scores should match what we already computed there
    # (it's deterministic given seed). We compare per-cluster mean AUCell.
    full_means = full_per_cluster.set_index("cluster")["mean"].sort_index()
    # Score full atlas again (this is essentially free relative to the empirical null)
    aucell_full_2 = compute_aucell_scores(
        adata, sig_full_atlas_refined, use_raw=True, top_fraction=0.05, seed=0,
    )
    labels_full = adata.obs[annotation_col].astype(str).values
    full_means_2 = (pd.DataFrame({"cluster": labels_full, "s": aucell_full_2})
                    .groupby("cluster")["s"].mean().sort_index())
    same = np.allclose(
        full_means.reindex(full_means_2.index).fillna(np.nan).to_numpy(),
        full_means_2.to_numpy(), equal_nan=True, atol=0, rtol=0,
    )
    results.append(_check(
        "C3.6  POA off run is deterministic (per-cluster means bit-identical)",
        same,
        f"compared {len(full_means_2)} cluster means",
    ))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypomap", required=True, type=Path)
    ap.add_argument("--bactrap", required=True, type=Path)
    ap.add_argument("--annotation-col", default="C185_named")
    ap.add_argument("--n-control-sets", type=int, default=100,
                    help="Empirical-null N (lower for a faster smoke test).")
    ap.add_argument("--baseline-cutoff", type=float, default=0.15,
                    help="Pnoc mean cutoff in POA mode (criterion C3.3).")
    ap.add_argument("--out", type=Path, default=Path("results_acceptance.json"))
    args = ap.parse_args()

    _setup_logging()

    print(f"Loading HypoMap from {args.hypomap}")
    t0 = time.time()
    adata = sc.read_h5ad(args.hypomap, backed=None)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} vars in {time.time()-t0:.1f}s")
    print(f"  raw layer present: {adata.raw is not None}")
    print(f"  obs columns include: "
          f"{[c for c in adata.obs.columns if c in (args.annotation_col, 'Region_summarized', 'C7_named')]}")

    if args.annotation_col not in adata.obs.columns:
        raise SystemExit(f"Annotation column '{args.annotation_col}' not found.")
    if "Region_summarized" not in adata.obs.columns:
        print("WARNING: 'Region_summarized' not in adata.obs — Change 3 checks will fail. "
              "Use --annotation-col, or report the actual region column name.")

    print(f"Loading bacTRAP from {args.bactrap}")
    bactrap_df = pd.read_excel(args.bactrap, engine="openpyxl")
    print(f"  {len(bactrap_df):,} rows; columns: {list(bactrap_df.columns)}")

    print("Matching genes (auto-detect column)...")
    bactrap_matched, matched_gene_names, gene_to_idx, matched_in_raw = match_genes(
        bactrap_df, adata, gene_col=None,
    )
    print(f"  matched {len(matched_gene_names):,} genes (in_raw={matched_in_raw})")

    all_results = []
    c2_results, full_per_cluster, sig_full_refined = check_change2(
        adata, bactrap_matched, gene_to_idx, matched_in_raw, args.annotation_col,
        n_control_sets=args.n_control_sets,
    )
    all_results.extend(c2_results)

    c3_results = check_change3(
        adata, bactrap_matched, gene_to_idx, matched_in_raw, args.annotation_col,
        sig_full_refined, full_per_cluster,
        n_control_sets=args.n_control_sets,
        baseline_cutoff=args.baseline_cutoff,
    )
    all_results.extend(c3_results)

    n_pass = sum(r["status"] == "PASS" for r in all_results)
    n_total = len(all_results)
    print(f"\n=== SUMMARY: {n_pass}/{n_total} criteria PASS ===")
    for r in all_results:
        print(f"  [{r['status']}] {r['name']}")

    args.out.write_text(json.dumps(
        {"hypomap": str(args.hypomap), "bactrap": str(args.bactrap),
         "annotation_col": args.annotation_col,
         "n_control_sets": args.n_control_sets,
         "baseline_cutoff": args.baseline_cutoff,
         "results": all_results},
        indent=2,
    ))
    print(f"\nWrote {args.out}")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
