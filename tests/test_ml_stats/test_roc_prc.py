"""Tests for ml_stats.roc_prc -- ROC/PRC curve results for binary classifier evaluation."""

import numpy as np
import pytest

from pylbsr.ml_stats.roc_prc import ListPRCresults, ListROCresults, PRCresults, ROCresults

# Mildly imbalanced (4 negatives, 2 positives) so the majority-class baseline in PRCresults
# is unambiguous -- a perfectly balanced split leaves pandas' value_counts().idxmax() tie-break
# implementation-defined.
Y_TRUE = np.array([0, 0, 0, 0, 1, 1])
Y_PRED_SEPARABLE = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
Y_PRED_ANTICORRELATED = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])


def test_rocresults_perfect_separation_has_auc_one(recwarn: pytest.WarningsRecorder) -> None:
    """A perfectly-ranked classifier gets ROC AUC close to 1.0, no warning.

    Not exactly 1.0: auc() is computed on a resampled 101-point grid
    (np.interp), which introduces small numerical deviation for a dataset
    this small (6 samples) -- that resampling-for-averaging is the whole
    point of base_fpr/interp_tpr, not a bug.
    """
    result = ROCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_SEPARABLE)
    assert result.auc == pytest.approx(1.0, abs=0.02)
    assert len(recwarn) == 0


def test_rocresults_anticorrelated_scores_warn_and_invert_fixes_it() -> None:
    """Anti-correlated scores trigger a UserWarning; invert() restores good separation."""
    with pytest.warns(UserWarning, match="anti-correlated"):
        result = ROCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_ANTICORRELATED)
    assert result.auc == pytest.approx(0.0, abs=0.02)

    fixed = result.invert()
    assert fixed.auc == pytest.approx(1.0, abs=0.02)


def test_rocresults_invert_without_underlying_data_raises() -> None:
    """invert() on a manually-constructed instance (no stored y_true/y_pred) raises."""
    result = ROCresults(
        fpr=np.array([0.0, 1.0]),
        tpr=np.array([0.0, 1.0]),
        thresholds=None,
        base_fpr=np.array([0.0, 1.0]),
        interp_tpr=np.array([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="from_ytrue_ypred"):
        result.invert()


def test_prcresults_perfect_separation_has_high_auc(recwarn: pytest.WarningsRecorder) -> None:
    """A perfectly-ranked classifier's PRC AUC is close to 1.0 and above the random baseline.

    Not exactly 1.0, same grid-interpolation reason as the ROC test above.
    """
    result = PRCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_SEPARABLE)
    assert result.auc > result.random_clf
    assert result.auc == pytest.approx(1.0, abs=0.02)
    assert len(recwarn) == 0


def test_prcresults_anticorrelated_scores_warn_and_invert_improves_it() -> None:
    """Anti-correlated scores trigger a UserWarning; invert() improves AUC over the baseline."""
    with pytest.warns(UserWarning, match="anti-correlated"):
        result = PRCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_ANTICORRELATED)
    assert result.auc < result.random_clf

    fixed = result.invert()
    assert fixed.auc > fixed.random_clf


def test_prcresults_invert_without_underlying_data_raises() -> None:
    """invert() on a manually-constructed instance (no stored y_true/y_pred) raises."""
    result = PRCresults(
        prec=np.array([1.0, 0.5]),
        rec=np.array([0.0, 1.0]),
        thresholds=None,
        base_rec=np.array([0.0, 1.0]),
        interp_prec=np.array([1.0, 0.5]),
        random_clf=0.5,
    )
    with pytest.raises(ValueError, match="from_ytrue_ypred"):
        result.invert()


def test_listrocresults_rejects_empty_or_wrong_type() -> None:
    """ListROCresults validates its input list eagerly, not lazily on first use."""
    with pytest.raises(ValueError, match="at least one element"):
        ListROCresults([])
    with pytest.raises(ValueError, match="only ROCresults"):
        ListROCresults([object()])  # type: ignore[list-item]


def test_listrocresults_mean_auc_averages_members() -> None:
    """mean_auc is the mean of each member's own .auc."""
    r1 = ROCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_SEPARABLE)
    r2 = ROCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_ANTICORRELATED)

    collection = ListROCresults([r1, r2])

    assert collection.mean_auc == pytest.approx((r1.auc + r2.auc) / 2)


def test_listprcresults_rejects_empty_or_wrong_type() -> None:
    """ListPRCresults validates its input list eagerly, not lazily on first use."""
    with pytest.raises(ValueError, match="at least one element"):
        ListPRCresults([])
    with pytest.raises(ValueError, match="only PRCresults"):
        ListPRCresults([object()])  # type: ignore[list-item]


def test_listprcresults_mean_auc_averages_members() -> None:
    """mean_auc is the mean of each member's own .auc."""
    r1 = PRCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_SEPARABLE)
    r2 = PRCresults.from_ytrue_ypred(Y_TRUE, Y_PRED_ANTICORRELATED)

    collection = ListPRCresults([r1, r2])

    assert collection.mean_auc == pytest.approx((r1.auc + r2.auc) / 2)
