"""Regression test for the `_analysis_params` cache fingerprint in app.py.

The fingerprint is a tuple built in app.py at the top of the analysis
section; if any sidebar input that affects the empirical-null or ranked
figures is omitted, Streamlit's recompute gate keeps returning stale
cached results.

Rather than booting Streamlit, this test asserts (by string-grepping
app.py) that every parameter we depend on is wired in. This is a coarse
guard but a much faster regression net than a full UI test, and it caught
the original C2 bug (min_cells_for_rank missing).
"""

import re
from pathlib import Path


APP_PY = (Path(__file__).resolve().parents[1] / "app.py").read_text()


def _extract_analysis_params() -> str:
    """Return the raw text of the `_analysis_params = (...)` tuple."""
    m = re.search(r"_analysis_params\s*=\s*\((.*?)\n\)", APP_PY, flags=re.DOTALL)
    assert m, "_analysis_params tuple not found in app.py"
    return m.group(1)


_PARAMS_SRC = _extract_analysis_params()


def _assert_present(needle: str, hint: str = "") -> None:
    assert needle in _PARAMS_SRC, (
        f"`{needle}` not referenced in _analysis_params — analysis results "
        f"may be served stale when this input changes. {hint}"
    )


def test_fingerprint_includes_data_paths():
    _assert_present("bactrap_file")
    _assert_present("hypomap_file")


def test_fingerprint_includes_filter_cutoffs():
    _assert_present("padj_cutoff")
    _assert_present("log2fc_cutoff")
    _assert_present("min_ip_expression")


def test_fingerprint_includes_ranking_metric_and_top_n():
    _assert_present("ranking_metric")
    _assert_present("top_n_genes")
    _assert_present("aucell_top_fraction")


def test_fingerprint_includes_min_cell_thresholds():
    # min_cells_per_cluster controls cluster eligibility (figures, t-test);
    # min_cells_for_rank gates the empirical null + ranked figures. Both
    # must invalidate the cache.
    _assert_present("min_cells_per_cluster")
    _assert_present(
        "min_cells_for_rank",
        hint="C2 regression — without this the empirical-null cache returns "
        "stale results when the user tweaks the AUCell ranking min-cells slider.",
    )


def test_fingerprint_includes_poa_signature():
    _assert_present("poa_active")
    _assert_present("mask_signature")
    _assert_present("poa_min_cells")


def test_fingerprint_includes_signature_refinement_params():
    _assert_present("sig_filter_detectability")
    _assert_present("sig_min_detection_rate")
    _assert_present("sig_min_max_cluster_mean")
    _assert_present("sig_filter_specificity")
    _assert_present("sig_specificity_thresh")
    _assert_present("sig_specificity_max_fraction")


def test_fingerprint_includes_empirical_null_params():
    _assert_present("empirical_null_enabled")
    _assert_present("empirical_null_n")
    _assert_present("empirical_null_bins")
    _assert_present("empirical_null_seed")
