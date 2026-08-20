"""Tests for ml_stats.group_comparison -- per-feature two-group statistical comparison.

relative_change/cohen_phi/calculate_pooled_var/calculate_effect_size only need numpy/pandas
(already core pylbsr dependencies); apply_df_multi_tests additionally needs scipy.stats (core)
and statsmodels (the ml_stats extra) -- skipped entirely if statsmodels isn't installed.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pylbsr.ml_stats.group_comparison import (
    calculate_effect_size,
    calculate_pooled_var,
    cohen_phi,
    relative_change,
)


def test_relative_change_default_method_is_ref() -> None:
    """No method given -> divide by b (the reference value)."""
    assert relative_change(12.0, 10.0) == pytest.approx((12.0 - 10.0) / 10.0)


def test_relative_change_mean_method() -> None:
    """method='mean' divides by mean(a, b)."""
    assert relative_change(12.0, 10.0, method="mean") == pytest.approx((12.0 - 10.0) / 11.0)


def test_relative_change_invalid_method_raises_with_clear_message() -> None:
    """An unrecognized method raises KeyError listing the valid options."""
    with pytest.raises(KeyError, match="not available"):
        relative_change(1.0, 2.0, method="not-a-real-method")


def test_relative_change_abs_exp_damps_toward_zero_denominator() -> None:
    """abs_exp shouldn't blow up as b approaches 0, unlike the plain 'ref' method."""
    result = relative_change(5.0, 0.001, method="abs_exp")
    assert math.isfinite(result)


def test_cohen_phi_known_value() -> None:
    """cohen_phi(0.5) == 2*arcsin(sqrt(0.5)) == pi/2."""
    assert cohen_phi(0.5) == pytest.approx(math.pi / 2)


def test_calculate_pooled_var_matches_manual_formula() -> None:
    """Pooled variance matches the textbook (n1-1)*v1 + (n2-1)*v2 / (n1+n2-2) formula."""
    x = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0]})
    bool_select = pd.Series([True, True, True, False, False, False])

    pooled = calculate_pooled_var(x, bool_select)

    v1 = x.loc[bool_select, "feat"].var()
    v2 = x.loc[~bool_select, "feat"].var()
    expected = ((3 - 1) * v1 + (3 - 1) * v2) / (3 + 3 - 2)
    assert pooled["feat"] == pytest.approx(expected)


def test_calculate_pooled_var_treats_single_sample_variance_as_zero() -> None:
    """A group with <2 samples has NaN variance from .var(); pooled_var must fillna(0), not NaN."""
    x = pd.DataFrame({"feat": [5.0, 1.0, 2.0]})
    bool_select = pd.Series([True, False, False])  # "selected" group has only 1 sample

    pooled = calculate_pooled_var(x, bool_select)

    assert not pd.isna(pooled["feat"])


def test_calculate_effect_size_requires_selected_others_columns() -> None:
    """A mean_values_df without exactly ['selected', 'others'] columns is rejected."""
    bad_df = pd.DataFrame({"wrong": [1.0], "columns": [2.0]}, index=["feat1"])
    with pytest.raises(AssertionError, match=r"selected.*others"):
        calculate_effect_size(bad_df, {"feat1": "continuous"}, pd.Series({"feat1": 1.0}))


def test_calculate_effect_size_continuous_feature_is_pooled_var_normalized_diff() -> None:
    """For a continuous feature, effect size is (selected - others) / sqrt(pooled_var)."""
    mean_values = pd.DataFrame({"selected": [10.0], "others": [5.0]}, index=["feat1"])
    pooled_var = pd.Series({"feat1": 4.0})

    effect = calculate_effect_size(mean_values, {"feat1": "continuous"}, pooled_var)

    assert effect["feat1"] == pytest.approx((10.0 - 5.0) / np.sqrt(4.0))


def test_calculate_effect_size_binary_feature_uses_cohen_phi() -> None:
    """For a binary feature, effect size is Cohen's phi difference, not the raw proportion diff."""
    mean_values = pd.DataFrame({"selected": [0.8], "others": [0.2]}, index=["feat1"])
    pooled_var = pd.Series({"feat1": 1.0})  # unused for binary features

    effect = calculate_effect_size(mean_values, {"feat1": "binary"}, pooled_var)

    assert effect["feat1"] == pytest.approx(cohen_phi(0.8) - cohen_phi(0.2))


def test_calculate_effect_size_preserves_original_feature_order() -> None:
    """Mixed binary/continuous features come back in the same row order as the input."""
    mean_values = pd.DataFrame(
        {"selected": [0.8, 10.0, 0.3], "others": [0.2, 5.0, 0.9]},
        index=["bin_feat", "cont_feat", "bin_feat2"],
    )
    features_type = {"bin_feat": "binary", "cont_feat": "continuous", "bin_feat2": "binary"}
    pooled_var = pd.Series({"cont_feat": 4.0})

    effect = calculate_effect_size(mean_values, features_type, pooled_var)

    assert list(effect.index) == ["bin_feat", "cont_feat", "bin_feat2"]
