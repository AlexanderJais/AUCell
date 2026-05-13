"""Synthetic-data unit tests for the signature-refinement filters.

Fast (< 5 s), self-contained — no HypoMap atlas required.  Run with::

    python -m pytest tests/test_signature_refinement.py -q
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from analysis import filter_signature_genes_by_atlas


# ---------------------------------------------------------------------------
# 1. Each engineered gene lands in the correct bucket
# ---------------------------------------------------------------------------

def _basic_atlas():
    clusters = ["c1", "c2", "c3", "c4", "c5"]
    # genes × clusters, log-norm mean expression
    cme = pd.DataFrame.from_dict(
        {
            # kept: detectable, cluster-specific
            "KEEP_1":            [2.0, 0.01, 0.0, 0.0, 0.0],
            "KEEP_2":            [1.5, 0.6, 0.0, 0.0, 0.0],     # 1/5 above 0.5
            "KEEP_3":            [3.0, 0.0, 0.0, 0.0, 0.0],
            "KEEP_2of5":         [0.8, 0.6, 0.1, 0.0, 0.0],     # 2/5 above 0.5 -> kept
            # dropped by detectability — low cell detection rate
            "DROP_DET_LOWRATE":  [1.0, 0.0, 0.0, 0.0, 0.0],
            # dropped by detectability — low max cluster mean
            "DROP_DET_LOWMEAN":  [0.03, 0.02, 0.01, 0.0, 0.0],
            "DROP_DET_LOWMEAN_2":[0.04, 0.0, 0.0, 0.0, 0.0],
            # dropped by specificity — broadly expressed
            "DROP_SPEC_BROAD":   [0.8, 0.7, 0.9, 0.6, 0.55],   # 5/5 above 0.5
            "DROP_SPEC_3of5":    [0.6, 0.7, 0.55, 0.1, 0.0],   # 3/5 above 0.5 -> dropped
            # would fail BOTH (low rate + broad) -> logged against detectability (runs first)
            "DROP_BOTH":         [0.6, 0.7, 0.8, 0.6, 0.0],    # 4/5 above 0.5
        },
        orient="index", columns=clusters,
    )
    det = pd.Series({
        "KEEP_1": 0.3, "KEEP_2": 0.5, "KEEP_3": 0.1, "KEEP_2of5": 0.5,
        "DROP_DET_LOWRATE": 0.01,            # < 0.02
        "DROP_DET_LOWMEAN": 0.3, "DROP_DET_LOWMEAN_2": 0.05,
        "DROP_SPEC_BROAD": 0.8, "DROP_SPEC_3of5": 0.7,
        "DROP_BOTH": 0.005,                  # < 0.02
    })
    return cme, det


def test_basic_filter_buckets():
    cme, det = _basic_atlas()
    cand = list(cme.index)
    kept, drop_log = filter_signature_genes_by_atlas(cand, cme, det)

    status = dict(zip(drop_log["gene"], drop_log["status"]))
    assert status == {
        "KEEP_1": "kept", "KEEP_2": "kept", "KEEP_3": "kept", "KEEP_2of5": "kept",
        "DROP_DET_LOWRATE": "dropped_detectability",
        "DROP_DET_LOWMEAN": "dropped_detectability",
        "DROP_DET_LOWMEAN_2": "dropped_detectability",
        "DROP_SPEC_BROAD": "dropped_specificity",
        "DROP_SPEC_3of5": "dropped_specificity",
        "DROP_BOTH": "dropped_detectability",   # detectability runs first
    }
    assert kept == ["KEEP_1", "KEEP_2", "KEEP_3", "KEEP_2of5"]   # input order preserved

    # drop_log schema + stats populated for every row
    assert list(drop_log.columns) == [
        "gene", "status", "detection_rate", "max_cluster_mean",
        "frac_clusters_above_thresh", "reason",
    ]
    dl = drop_log.set_index("gene")
    assert dl.loc["KEEP_1", "detection_rate"] == pytest.approx(0.3)
    assert dl.loc["KEEP_1", "max_cluster_mean"] == pytest.approx(2.0)
    assert dl.loc["DROP_SPEC_3of5", "frac_clusters_above_thresh"] == pytest.approx(0.6)
    # reason strings name the offending statistic
    assert "detection_rate" in dl.loc["DROP_DET_LOWRATE", "reason"]
    assert "max_cluster_mean" in dl.loc["DROP_DET_LOWMEAN", "reason"]
    assert "frac" in dl.loc["DROP_SPEC_BROAD", "reason"].lower()
    # the "fails both" gene's reason is the detectability one, not specificity
    assert "detection_rate" in dl.loc["DROP_BOTH", "reason"]


# ---------------------------------------------------------------------------
# 2. Disabling both filters returns the input unchanged
# ---------------------------------------------------------------------------

def test_disabling_filters_returns_input():
    cme, det = _basic_atlas()
    cand = list(cme.index) + ["GHOST_NOT_IN_ATLAS"]
    kept, drop_log = filter_signature_genes_by_atlas(
        cand, cme, det, apply_detectability=False, apply_specificity=False,
    )
    assert kept == cand                       # exactly the input, in order
    assert (drop_log["status"] == "kept").all()
    assert list(drop_log["gene"]) == cand


# ---------------------------------------------------------------------------
# 3. Pnoc tripwire
# ---------------------------------------------------------------------------

def test_pnoc_tripwire_survives_when_specific(caplog):
    clusters = ["c1", "c2", "c3", "c4", "c5"]
    cme = pd.DataFrame.from_dict(
        {"Pnoc": [2.0, 0.0, 0.0, 0.0, 0.0], "Other": [0.0, 1.0, 0.0, 0.0, 0.0]},
        orient="index", columns=clusters,
    )
    det = pd.Series({"Pnoc": 0.1, "Other": 0.2})   # both above 0.02
    caplog.set_level(logging.WARNING)
    kept, drop_log = filter_signature_genes_by_atlas(["Pnoc", "Other"], cme, det)
    assert "Pnoc" in kept
    assert "Pnoc" not in caplog.text          # no "Pnoc dropped" warning


def test_pnoc_tripwire_warns_when_broadly_expressed(caplog):
    clusters = ["c1", "c2", "c3", "c4", "c5"]
    cme = pd.DataFrame.from_dict(
        {"Pnoc": [0.8, 0.7, 0.9, 0.6, 0.55], "Other": [2.0, 0.0, 0.0, 0.0, 0.0]},
        orient="index", columns=clusters,
    )
    det = pd.Series({"Pnoc": 0.8, "Other": 0.2})
    caplog.set_level(logging.WARNING)
    kept, drop_log = filter_signature_genes_by_atlas(["Pnoc", "Other"], cme, det)
    assert "Pnoc" not in kept
    assert drop_log.set_index("gene").loc["Pnoc", "status"] == "dropped_specificity"
    assert "Pnoc" in caplog.text              # tripwire warning emitted
    assert any(
        r.levelno >= logging.WARNING and "Pnoc" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# 4. Gene absent from the atlas
# ---------------------------------------------------------------------------

def test_gene_not_in_atlas():
    cme = pd.DataFrame.from_dict(
        {"A": [2.0, 0.0], "B": [0.0, 1.5]}, orient="index", columns=["c1", "c2"],
    )
    det = pd.Series({"A": 0.3, "B": 0.4})
    kept, drop_log = filter_signature_genes_by_atlas(["A", "GHOST"], cme, det)
    assert "GHOST" not in kept
    row = drop_log.set_index("gene").loc["GHOST"]
    assert row["status"] == "dropped_not_in_atlas"
    assert row["reason"] == "not in atlas"
    assert math.isnan(row["detection_rate"])
    assert math.isnan(row["max_cluster_mean"])
    assert math.isnan(row["frac_clusters_above_thresh"])
    # a gene present in the detection series but missing from cluster_mean_expr
    # is also "not in atlas"
    det2 = pd.Series({"A": 0.3, "B": 0.4, "C": 0.5})
    kept2, drop_log2 = filter_signature_genes_by_atlas(["A", "C"], cme, det2)
    assert drop_log2.set_index("gene").loc["C", "status"] == "dropped_not_in_atlas"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
