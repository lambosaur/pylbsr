"""Per-feature statistical comparison between two groups of samples."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_RELATIVE_CHANGE_METHODS: dict[str, Callable[[float, float], float]] = {
    "ref": lambda a, b: b,
    "mean": lambda a, b: float(np.mean((a, b))),
    "abs_mean": lambda a, b: float(np.mean((abs(a), abs(b)))),
    "abs_ref": lambda a, b: abs(b),
    "abs_alt": lambda a, b: abs(a),
    "min": lambda a, b: min(a, b),
    "max": lambda a, b: max(a, b),
    "abs_exp": lambda a, b: b,
}


def relative_change(a: float, b: float, method: str | None = None) -> float:
    """Relative difference of `a` against `b`.

    References:
        https://en.wikipedia.org/wiki/Relative_change_and_difference
        https://stats.stackexchange.com/questions/86708

    Args:
        a: New/alternative value.
        b: Reference value.
        method: How to normalize the difference; one of "ref" (default, divide
            by `b`), "mean", "abs_mean", "abs_ref", "abs_alt", "min", "max", or
            "abs_exp" (a smoothed variant that damps large relative changes
            when `b` is near zero -- see the math.stackexchange link below).

    Returns:
        The relative change.

    Raises:
        KeyError: If `method` isn't one of the options above.
    """
    method = method or "ref"
    try:
        denominator = _RELATIVE_CHANGE_METHODS[method](a, b)
    except KeyError:
        raise KeyError(
            f"Method {method!r} not available; choose among: {list(_RELATIVE_CHANGE_METHODS)}"
        ) from None

    if method == "abs_exp":
        # https://math.stackexchange.com/questions/500723 -- damps the relative
        # change as |a - b| grows, so a near-zero denominator doesn't blow up.
        r = 100
        return float(((a - b) / denominator) * (1 - np.exp(-np.abs(a - b) / r)))
    return (a - b) / denominator


def cohen_phi(prop: float) -> float:
    """Cohen's arcsine-transform effect size for one proportion: 2*arcsin(sqrt(prop))."""
    return float(2 * np.arcsin(np.sqrt(prop)))


def calculate_pooled_var(x: pd.DataFrame, bool_select: pd.Series) -> pd.Series:
    """Pooled per-column variance between the rows selected by `bool_select` and the rest.

    Args:
        x: Table of numeric feature columns.
        bool_select: Boolean row mask; True selects the first group.

    Returns:
        Per-column pooled variance.
    """
    var_selected = x.loc[bool_select, :].var().fillna(0)
    var_others = x.loc[~bool_select, :].var().fillna(0)
    n_selected = bool_select.sum()
    n_others = (~bool_select).sum()
    return ((n_selected - 1) * var_selected + (n_others - 1) * var_others) / (
        n_selected + n_others - 2
    )


def calculate_effect_size(
    mean_values_df: pd.DataFrame, features_type: dict[str, str], pooled_var: pd.Series
) -> pd.Series:
    """Per-feature effect size between a "selected" group and "others".

    For each feature (row in `mean_values_df`), computes a normalized
    difference between the "selected" and "others" column values, using
    Cohen's phi (arcsine-transform difference) for binary features, or a
    pooled-variance-normalized mean difference (Cohen's d-like) otherwise.

    Args:
        mean_values_df: Per-feature "selected"/"others" summary values; must
            have exactly those two columns.
        features_type: Mapping of feature name to "binary", "discrete", or
            "continuous".
        pooled_var: Per-feature pooled variance, as returned by
            `calculate_pooled_var`; used to normalize non-binary features.

    Returns:
        Per-feature effect size, in `mean_values_df`'s original row order.
    """
    assert list(mean_values_df.columns) == ["selected", "others"], (
        "Columns of mean_values_df should be ['selected', 'others']"
    )

    features = mean_values_df.index.values
    effect_sizes = []

    for feature_type in set(features_type.values()):
        feature_subset = [c for c in features if features_type[c] == feature_type]
        if not feature_subset:
            continue

        if feature_type == "binary":
            # Compare proportions via Cohen's phi.
            effect_sizes.append(
                mean_values_df.loc[feature_subset]
                .map(cohen_phi)
                .apply(lambda row: row["selected"] - row["others"], axis=1)
            )
        else:
            # Compare means, normalized by pooled variance.
            effect_sizes.append(
                mean_values_df.loc[feature_subset].apply(
                    lambda row: (row["selected"] - row["others"]) / np.sqrt(pooled_var[row.name]),
                    axis=1,
                )
            )

    return pd.concat(effect_sizes).loc[features]


def apply_df_multi_tests(
    x: pd.DataFrame,
    bool_selected: pd.Series,
    cols_type: dict[str, str],
    adjust: bool = True,
) -> pd.Series:
    """Per-feature statistical test comparing rows selected by `bool_selected` vs. the rest.

    The test applied depends on each column's type in `cols_type`:
        - continuous / discrete: Mann-Whitney U test
        - binary: chi-square contingency test

    Args:
        x: Table of feature columns to test.
        bool_selected: Boolean row mask; True selects the first group.
        cols_type: Mapping of column name to "binary", "discrete", "continuous",
            or "other" (columns typed "other", or missing from this mapping,
            are skipped).
        adjust: Apply Holm-Bonferroni multiple-testing correction across all
            tested columns. Requires the `ml_stats` extra (statsmodels).

    Returns:
        Per-column p-value (corrected, if `adjust`).
    """
    summarizing_columns = [c for c in x.columns if cols_type.get(c, "other") != "other"]

    all_tests = []
    for test_type, column_type in [
        ("mannwhitneyu", "continuous"),
        ("mannwhitneyu", "discrete"),
        ("chi2_contingency", "binary"),
    ]:
        cols = [c for c in summarizing_columns if cols_type[c] == column_type]
        if not cols:
            continue

        pvals = []
        for c in cols:
            try:
                if test_type == "chi2_contingency":
                    contingency = pd.concat(
                        (
                            x.loc[bool_selected, c].value_counts().rename("selected"),
                            x.loc[~bool_selected, c].value_counts().rename("others"),
                        ),
                        axis=1,
                    ).fillna(0)
                    pval = scipy.stats.chi2_contingency(contingency)[1]
                else:
                    group_a = x.loc[bool_selected, c]
                    group_b = x.loc[~bool_selected, c]
                    pval = scipy.stats.mannwhitneyu(group_a, group_b)[1]
            except ValueError as e:
                logger.warning("Test failed for column %r (marked non-significant): %s", c, e)
                pval = 1.0
            pvals.append(pval)

        all_tests.append(pd.Series(pvals, index=cols))

    pvals = pd.concat(all_tests).loc[summarizing_columns]

    if adjust:
        # Deferred import: statsmodels is only needed for this branch (adjust=True), so
        # the rest of this module stays importable without the ml_stats extra installed.
        from statsmodels.stats.multitest import multipletests

        return pd.Series(multipletests(pvals, method="holm")[1], index=pvals.index)
    return pvals
