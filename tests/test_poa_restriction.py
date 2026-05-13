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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
