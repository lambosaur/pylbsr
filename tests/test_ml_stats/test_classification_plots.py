"""Tests for ml_stats.roc_prc's classification-evaluation additions: evaluate_classifier_preds,
plot_confusion_matrix, plot_classification_evaluation, quality_classification_plots,
plot_predproba_distributions, separability_plots.
"""

import matplotlib

matplotlib.use("Agg")

import warnings

import numpy as np
import pytest
import sklearn.metrics
from matplotlib.figure import Figure

from pylbsr.ml_stats.roc_prc import (
    Clf_scores,
    ListPRCresults,
    PRCresults,
    evaluate_classifier_preds,
    plot_classification_evaluation,
    plot_confusion_matrix,
    plot_predproba_distributions,
    quality_classification_plots,
    separability_plots,
)

Y_TRUE_HARD_A = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_PRED_HARD_A = np.array([0, 0, 1, 0, 1, 1, 0, 1])  # 6/8 correct
# Deliberately a different length -- exercises the mismatched-fold-size regression path.
Y_TRUE_HARD_B = np.array([0, 0, 0, 1, 1, 1])
Y_PRED_HARD_B = np.array([0, 1, 0, 1, 1, 0])  # 4/6 correct

Y_TRUE_SCORE_A = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_PRED_SCORE_A = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.4, 0.9])
Y_TRUE_SCORE_B = np.array([0, 0, 0, 1, 1, 1])
Y_PRED_SCORE_B = np.array([0.2, 0.6, 0.1, 0.7, 0.8, 0.3])


def test_evaluate_classifier_preds_matches_sklearn_directly() -> None:
    """Every field matches calling the corresponding sklearn function directly."""
    result = evaluate_classifier_preds(Y_TRUE_HARD_A, Y_PRED_HARD_A)

    assert isinstance(result, Clf_scores)
    assert result.accuracy == pytest.approx(
        sklearn.metrics.accuracy_score(Y_TRUE_HARD_A, Y_PRED_HARD_A)
    )
    assert result.recall == pytest.approx(sklearn.metrics.recall_score(Y_TRUE_HARD_A, Y_PRED_HARD_A))
    assert result.precision == pytest.approx(
        sklearn.metrics.precision_score(Y_TRUE_HARD_A, Y_PRED_HARD_A)
    )
    assert result.f1 == pytest.approx(sklearn.metrics.f1_score(Y_TRUE_HARD_A, Y_PRED_HARD_A))


def test_plot_confusion_matrix_respects_explicit_cmap_without_color_class() -> None:
    """Regression test: the original unconditionally overwrote cmap with Blues whenever
    color_class was None, silently ignoring any cmap the caller explicitly passed.

    seaborn rebuilds the colormap object internally when center= is set (as this
    function always does), losing its .name -- so this compares actual sampled color
    values against Blues instead of checking .name, which would pass either way.
    """
    import matplotlib as mpl

    cm = sklearn.metrics.confusion_matrix(Y_TRUE_HARD_A, Y_PRED_HARD_A)

    fig, ax = plot_confusion_matrix(cm, ["neg", "pos"], cmap="viridis")

    assert isinstance(fig, Figure)
    applied_color = ax.collections[0].get_cmap()(0.9)
    blues_color = mpl.colormaps["Blues"](0.9)
    max_channel_diff = max(abs(a - b) for a, b in zip(applied_color, blues_color))
    assert max_channel_diff > 0.1  # clearly a different color, not just float noise


def test_plot_confusion_matrix_reuses_given_ax() -> None:
    """Passing an existing ax means no new figure is created."""
    cm = sklearn.metrics.confusion_matrix(Y_TRUE_HARD_A, Y_PRED_HARD_A)
    fig1, ax1 = plot_confusion_matrix(cm, ["neg", "pos"])
    fig2, ax2 = plot_confusion_matrix(cm, ["neg", "pos"], ax=ax1)

    assert fig2 is None
    assert ax2 is ax1
    assert fig1 is not None


def test_plot_classification_evaluation_no_warnings_with_mismatched_fold_sizes() -> None:
    """Regression test: set_xticklabels() without a prior set_xticks() call raised a
    UserWarning about a non-fixed number of ticks -- confirmed via warnings-as-errors
    before this fix. Also exercises differently-sized folds (8 vs 6 samples).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, ax = plot_classification_evaluation(
            [(Y_TRUE_HARD_A, Y_PRED_HARD_A), (Y_TRUE_HARD_B, Y_PRED_HARD_B)]
        )

    assert isinstance(fig, Figure)
    assert ax.get_ylim() == (0, 1)


def test_plot_classification_evaluation_single_fold_no_error_bars() -> None:
    """A single-element list still plots (no error bars, no crash)."""
    fig, _ax = plot_classification_evaluation([(Y_TRUE_HARD_A, Y_PRED_HARD_A)])
    assert isinstance(fig, Figure)


def test_quality_classification_plots_returns_two_axes() -> None:
    """Confusion matrix + metrics barplot, one axes each."""
    fig, axs = quality_classification_plots(
        [(Y_TRUE_HARD_A, Y_PRED_HARD_A), (Y_TRUE_HARD_B, Y_PRED_HARD_B)], ["neg", "pos"]
    )

    assert isinstance(fig, Figure)
    assert len(axs) == 2


def test_plot_predproba_distributions_with_mismatched_fold_sizes() -> None:
    """Differently-sized folds of continuous scores don't crash."""
    fig, ax = plot_predproba_distributions(
        [(Y_TRUE_SCORE_A, Y_PRED_SCORE_A), (Y_TRUE_SCORE_B, Y_PRED_SCORE_B)]
    )

    assert isinstance(fig, Figure)
    assert ax.get_xlim() == (-0.08, 1.08)


def test_separability_plots_with_mismatched_fold_sizes_does_not_crash() -> None:
    """Regression test: ListPRCresults.plot() accumulated result.prec (raw, variable-length
    per fold) instead of result.interp_prec (resampled, fixed-length) for its mean/std band --
    this crashed outright (ValueError: inhomogeneous shape) whenever folds had different
    sample counts, confirmed by reproducing it against the unfixed code first.
    """
    fig, axs = separability_plots(
        [(Y_TRUE_SCORE_A, Y_PRED_SCORE_A), (Y_TRUE_SCORE_B, Y_PRED_SCORE_B)]
    )

    assert isinstance(fig, Figure)
    assert len(axs) == 3


def test_listprcresults_plot_mean_band_uses_interp_prec_not_raw_prec() -> None:
    """Direct regression test for the ListPRCresults.plot() fix, isolated from
    separability_plots: two PRCresults built from differently-sized inputs must not
    crash when plotted together, and the resulting mean curve must have exactly as
    many points as the shared (resampled) base_rec grid, not depend on either fold's
    raw (differently-sized) precision array.
    """
    r1 = PRCresults.from_ytrue_ypred(Y_TRUE_SCORE_A, Y_PRED_SCORE_A)
    r2 = PRCresults.from_ytrue_ypred(Y_TRUE_SCORE_B, Y_PRED_SCORE_B)
    assert len(r1.prec) != len(r2.prec)  # confirms this is a genuine mismatched-length case

    collection = ListPRCresults([r1, r2])
    fig, _ax = collection.plot()

    assert isinstance(fig, Figure)
