"""Tests for ml_stats.group_comparison.apply_df_multi_tests.

Split from test_group_comparison.py because this one function needs statsmodels (the
ml_stats extra); the rest of the module only needs numpy/pandas/scipy (core deps).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from pylbsr.ml_stats.group_comparison import apply_df_multi_tests  # noqa: E402

RNG = np.random.default_rng(42)


def test_apply_df_multi_tests_detects_a_clearly_different_continuous_feature() -> None:
    """A continuous feature with well-separated group means gets a small (significant) p-value."""
    n = 30
    x = pd.DataFrame(
        {
            "separated": np.concatenate([RNG.normal(0, 1, n), RNG.normal(10, 1, n)]),
            "identical": RNG.normal(0, 1, 2 * n),
        }
    )
    bool_selected = pd.Series([True] * n + [False] * n)
    cols_type = {"separated": "continuous", "identical": "continuous"}

    pvals = apply_df_multi_tests(x, bool_selected, cols_type, adjust=False)

    assert pvals["separated"] < 0.01
    assert pvals["identical"] > 0.05


def test_apply_df_multi_tests_binary_column_uses_chi2() -> None:
    """A binary feature strongly associated with group membership gets a small p-value."""
    x = pd.DataFrame({"flag": [1] * 20 + [0] * 20})
    bool_selected = pd.Series([True] * 20 + [False] * 20)

    pvals = apply_df_multi_tests(x, bool_selected, {"flag": "binary"}, adjust=False)

    assert pvals["flag"] < 0.001


def test_apply_df_multi_tests_skips_columns_typed_other_or_untyped() -> None:
    """Columns typed "other", or missing from cols_type entirely, are not tested."""
    x = pd.DataFrame(
        {
            "tested": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "skipped_other": [1, 2, 3, 4, 5, 6],
            "skipped_untyped": [1, 2, 3, 4, 5, 6],
        }
    )
    bool_selected = pd.Series([True, True, True, False, False, False])
    cols_type = {"tested": "continuous", "skipped_other": "other"}

    pvals = apply_df_multi_tests(x, bool_selected, cols_type, adjust=False)

    assert list(pvals.index) == ["tested"]


def test_apply_df_multi_tests_adjust_true_does_not_decrease_any_pvalue() -> None:
    """Holm-Bonferroni correction only ever raises (or keeps equal) p-values, never lowers them."""
    x = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "b": [5.0, 4.0, 6.0, 4.0, 5.0, 6.0],
        }
    )
    bool_selected = pd.Series([True, True, True, False, False, False])
    cols_type = {"a": "continuous", "b": "continuous"}

    raw = apply_df_multi_tests(x, bool_selected, cols_type, adjust=False)
    adjusted = apply_df_multi_tests(x, bool_selected, cols_type, adjust=True)

    assert (adjusted >= raw - 1e-12).all()
