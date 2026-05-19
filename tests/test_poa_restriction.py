"""Synthetic-data unit tests for the POA-only atlas restriction.

Fast (< 10 s), self-contained — no HypoMap atlas required.  Run with::

    python -m pytest tests/test_poa_restriction.py -q
"""

import logging

import numpy as np
import pandas as pd
import pytest

from data_loading import get_poa_cell_mask, build_mask_signature


class _FakeAdata:
    """Minimal stand-in exposing only what get_poa_cell_mask reads."""

    def __init__(self, obs: pd.DataFrame):
        self.obs = obs
        self.obs_names = obs.index


def _obs(regions, col="Region_summarized", extra_cols=None):
    df = pd.DataFrame({col: regions}, index=[f"cell{i}" for i in range(len(regions))])
    for c in (extra_cols or []):
        df[c] = "x"
    return _FakeAdata(df)


# ---------------------------------------------------------------------------
# 1. basic substring matching + NA inclusion
# ---------------------------------------------------------------------------

def test_poa_mask_basic_matching():
    adata = _obs([
        "Medial preoptic area",
        "Lateral preoptic area",
        "Lateral hypothalamic area",
        "Arcuate hypothalamic nucleus",
        np.nan,
    ])
    mask = get_poa_cell_mask(adata)              # defaults: ("preoptic",), include_na=True
    assert mask.dtype == bool
    assert list(mask) == [True, True, False, False, True]
    assert list(mask.index) == list(adata.obs_names)


# ---------------------------------------------------------------------------
# 2. include_na=False drops the unassigned cell
# ---------------------------------------------------------------------------

def test_poa_mask_excludes_na():
    adata = _obs([
        "Medial preoptic area",
        "Lateral preoptic area",
        "Lateral hypothalamic area",
        "Arcuate hypothalamic nucleus",
        np.nan,
    ])
    mask = get_poa_cell_mask(adata, include_na=False)
    assert list(mask) == [True, True, False, False, False]


# ---------------------------------------------------------------------------
# 3. multiple keywords broaden the definition
# ---------------------------------------------------------------------------

def test_poa_mask_multiple_keywords():
    adata = _obs([
        "Medial preoptic area",
        "Paraventricular hypothalamic nucleus",
        "Arcuate hypothalamic nucleus",
    ])
    base = get_poa_cell_mask(adata)
    assert list(base) == [True, False, False]
    broadened = get_poa_cell_mask(adata, poa_keywords=("preoptic", "paraventricular"))
    assert list(broadened) == [True, True, False]


# ---------------------------------------------------------------------------
# 4. case-insensitive matching
# ---------------------------------------------------------------------------

def test_poa_mask_case_insensitive():
    adata = _obs([
        "MEDIAL PREOPTIC AREA",
        "medial preoptic area",
        "Median PreOptic Nucleus",
        "ARCUATE HYPOTHALAMIC NUCLEUS",
    ])
    mask = get_poa_cell_mask(adata)
    assert list(mask) == [True, True, True, False]


# ---------------------------------------------------------------------------
# 5. missing region column → informative KeyError
# ---------------------------------------------------------------------------

def test_poa_mask_missing_region_col():
    adata = _obs(["Medial preoptic area", "Arcuate hypothalamic nucleus"])
    with pytest.raises(KeyError) as exc:
        get_poa_cell_mask(adata, region_col="DoesNotExist")
    msg = str(exc.value)
    assert "DoesNotExist" in msg
    # the message lists columns whose name contains "region"
    assert "Region_summarized" in msg


# ---------------------------------------------------------------------------
# 6. build_mask_signature is deterministic
# ---------------------------------------------------------------------------

def test_build_mask_signature_deterministic():
    assert build_mask_signature(("preoptic",), True) == "poaonly_preoptic_na"
    assert build_mask_signature(("preoptic",), False) == "poaonly_preoptic"
    # order-independent
    assert (build_mask_signature(("preoptic", "paraventricular"), True)
            == build_mask_signature(("paraventricular", "preoptic"), True)
            == "poaonly_paraventricular_preoptic_na")
    # case-folded, deduped of empties
    assert build_mask_signature(("PreOptic", "", "  "), True) == "poaonly_preoptic_na"
    # different inputs → different signature
    assert (build_mask_signature(("preoptic",), True)
            != build_mask_signature(("preoptic",), False))
    assert (build_mask_signature(("preoptic",), True)
            != build_mask_signature(("preoptic", "lateral"), True))


# ---------------------------------------------------------------------------
# 7. logging breakdown
# ---------------------------------------------------------------------------

def test_poa_mask_logging(caplog):
    adata = _obs([
        "Medial preoptic area", "Medial preoptic area",
        "Lateral preoptic area", "Arcuate hypothalamic nucleus", np.nan,
    ])
    caplog.set_level(logging.INFO)
    get_poa_cell_mask(adata, logger=logging.getLogger("test_poa"))
    text = caplog.text
    assert "POA restriction" in text
    assert "POA mask: N_poa" in text
    assert "Region breakdown" in text


# ---------------------------------------------------------------------------
# 8. cluster-level NA inclusion (Option B fix)
# ---------------------------------------------------------------------------

def _cluster_obs(rows, *, region_col="Region_summarized", cluster_col="C185_named"):
    """rows: list of (cluster_name, region_value)."""
    df = pd.DataFrame(
        {
            cluster_col: [r[0] for r in rows],
            region_col: [r[1] for r in rows],
        },
        index=[f"cell{i}" for i in range(len(rows))],
    )
    return _FakeAdata(df)


def test_cluster_level_admits_all_na_cluster():
    # C185-67-style: all cells in a cluster are NA → the whole cluster is
    # admitted, even though no cell matches the keyword
    adata = _cluster_obs([
        ("C185-67", np.nan),
        ("C185-67", np.nan),
        ("C185-67", np.nan),
    ])
    mask = get_poa_cell_mask(adata, annotation_col="C185_named")
    assert list(mask) == [True, True, True]


def test_cluster_level_rejects_mixed_na_non_poa_cluster():
    # C185-105-style: some cells NA, some non-POA non-NA, no keyword match →
    # cluster is NOT admitted, all cells excluded (this is the bug the fix
    # is targeting: under per-cell semantics those NA cells would be kept)
    adata = _cluster_obs([
        ("C185-105", np.nan),
        ("C185-105", np.nan),
        ("C185-105", "Striatum"),
        ("C185-105", "Cortex"),
    ])
    mask = get_poa_cell_mask(adata, annotation_col="C185_named")
    assert list(mask) == [False, False, False, False]


def test_cluster_level_admits_keyword_cluster_keeps_all_cells():
    # Genuinely POA-resident cluster: some cells match "preoptic" → the whole
    # cluster is admitted, including its non-preoptic non-NA cells. The
    # admission rule operates at the cluster level: once admitted, every cell
    # of that cluster is scored regardless of its individual region label.
    adata = _cluster_obs([
        ("C185-66", "Medial preoptic area"),
        ("C185-66", "Lateral preoptic area"),
        ("C185-66", "Arcuate hypothalamic nucleus"),
        ("C185-66", np.nan),
    ])
    mask = get_poa_cell_mask(adata, annotation_col="C185_named")
    assert list(mask) == [True, True, True, True]


def test_cluster_level_mixed_clusters_independent():
    # Several clusters in one frame: verify per-cluster admission rule
    # operates independently. The expected outcome:
    #   POA-cluster (keyword) → admitted (all cells kept)
    #   ALL-NA-cluster (the C185-67 case) → admitted
    #   MIXED-cluster (NA + non-POA, no keyword) → rejected (the bug fix)
    #   NON-POA-cluster (no NA, no keyword) → rejected
    adata = _cluster_obs([
        ("POA",     "Medial preoptic area"),
        ("POA",     "Arcuate hypothalamic nucleus"),  # kept (admitted cluster)
        ("ALLNA",   np.nan),
        ("ALLNA",   np.nan),
        ("MIXED",   np.nan),                          # the spurious case
        ("MIXED",   "Striatum"),
        ("NONPOA",  "Cortex"),
        ("NONPOA",  "Hippocampus"),
    ])
    mask = get_poa_cell_mask(adata, annotation_col="C185_named")
    assert list(mask) == [True, True, True, True, False, False, False, False]


def test_cluster_level_include_na_false_falls_back_to_per_cell():
    # When include_na is False the cluster-level admission rule is moot
    # (no NA inclusion clause to evaluate) and the function reduces to
    # plain per-cell keyword matching, regardless of annotation_col.
    adata = _cluster_obs([
        ("C185-67",  np.nan),
        ("C185-67",  np.nan),
        ("C185-105", np.nan),
        ("C185-105", "Striatum"),
        ("C185-X",   "Medial preoptic area"),
    ])
    mask = get_poa_cell_mask(adata, annotation_col="C185_named", include_na=False)
    assert list(mask) == [False, False, False, False, True]


def test_cluster_level_missing_annotation_col_errors():
    adata = _cluster_obs([("C185-67", np.nan)])
    with pytest.raises(KeyError) as exc:
        get_poa_cell_mask(adata, annotation_col="DoesNotExist")
    assert "DoesNotExist" in str(exc.value)


def test_cluster_level_legacy_default_unchanged():
    # The per-cell default path (annotation_col=None) still admits NA cells
    # from any cluster — preserves backward compatibility for callers that
    # don't opt into the cluster-level rule.
    adata = _cluster_obs([
        ("C185-105", np.nan),                # would be admitted (legacy bug)
        ("C185-105", "Striatum"),
    ])
    mask = get_poa_cell_mask(adata)          # annotation_col=None
    assert list(mask) == [True, False]


def test_build_mask_signature_with_annotation_col():
    # The corrected (cluster-level) mask must produce a different signature
    # so cache keys and CSV filenames never collide with the legacy output.
    legacy = build_mask_signature(("preoptic",), True)
    fixed = build_mask_signature(("preoptic",), True, "C185_named")
    assert legacy == "poaonly_preoptic_na"
    assert fixed == "poaonly_preoptic_na_byc185named"
    assert legacy != fixed
    # annotation_col is ignored when include_na is False (no NA clause)
    assert (build_mask_signature(("preoptic",), False, "C185_named")
            == build_mask_signature(("preoptic",), False))


def test_build_mask_signature_region_label_mbh():
    # MBH (mediobasal hypothalamus = ARC + VMH + DMH) signature carries an
    # "mbhonly" prefix so its CSV/cache outputs never collide with POA's.
    sig = build_mask_signature(
        ("arcuate", "ventromedial", "dorsomedial"), True,
        region_label="mbh",
    )
    assert sig == "mbhonly_arcuate_dorsomedial_ventromedial_na"
    # default (no region_label) stays POA → preserves existing on-disk caches
    poa = build_mask_signature(("preoptic",), True)
    assert poa == "poaonly_preoptic_na"
    assert poa != sig
    # region_label is case-folded and trimmed
    assert build_mask_signature(("preoptic",), False, region_label="  MBH  ") \
        == "mbhonly_preoptic"


def test_get_poa_cell_mask_with_mbh_keywords():
    # The mask function itself is region-agnostic — feeding MBH keywords
    # selects ARC/VMH/DMH cells exactly the same way POA keywords select POA.
    adata = _obs([
        "Medial preoptic area",
        "Arcuate hypothalamic nucleus",
        "Ventromedial hypothalamic nucleus",
        "Dorsomedial hypothalamic nucleus",
        "Striatum",
    ])
    mask = get_poa_cell_mask(
        adata,
        poa_keywords=("arcuate", "ventromedial", "dorsomedial"),
        include_na=False,
    )
    assert list(mask) == [False, True, True, True, False]


def test_cluster_level_logging(caplog):
    adata = _cluster_obs([
        ("POA",   "Medial preoptic area"),
        ("POA",   "Arcuate hypothalamic nucleus"),
        ("ALLNA", np.nan),
        ("MIXED", np.nan),
        ("MIXED", "Striatum"),
    ])
    caplog.set_level(logging.INFO)
    get_poa_cell_mask(adata, annotation_col="C185_named",
                     logger=logging.getLogger("test_poa_cluster"))
    text = caplog.text
    assert "cluster-level" in text
    assert "clusters admitted" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
