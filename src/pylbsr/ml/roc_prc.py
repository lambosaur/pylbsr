"""ROC and precision-recall curve results, for binary classifier evaluation."""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Self


class PRCresults:
    """Handle data structures for plotting a Precision-Recall Curve from a binary classification task.

    Instanciation: either build from prediction results with `PRCresults.from_ytrue_ypred`
    or with `PRCresults.from_prec_rec` (useful when building a new structure from the
    average performance of a set of PRCresults in a ListPRCresults instance).

    """

    def __init__(
        self,
        prec: np.ndarray,
        rec: np.ndarray,
        thresholds: np.ndarray | None,
        base_rec: np.ndarray,
        interp_prec: np.ndarray,
        random_clf: float,
    ) -> None:
        """Store precomputed precision/recall curve values; see `from_ytrue_ypred`."""
        self.prec = prec
        self.rec = rec
        self.thresholds = thresholds
        self.base_rec = base_rec
        self.interp_prec = interp_prec
        self.random_clf = random_clf
        self._y_true: np.ndarray | None = None
        self._y_pred: np.ndarray | None = None

    @classmethod
    def from_prec_rec(cls) -> Self:
        """Not implemented yet."""
        raise NotImplementedError("This method is not implemented yet.")

    @classmethod
    def from_ytrue_ypred(
        cls, y_true: np.ndarray, y_pred: np.ndarray, base_rec: np.ndarray | None = None
    ) -> Self:
        """Build a PRCresults from true labels and predicted scores."""
        prec: np.ndarray
        rec: np.ndarray
        precrec_thresholds: np.ndarray

        if base_rec is None:
            base_rec = np.linspace(0, 1, 101)

        try:
            prec, rec, precrec_thresholds = sklearn.metrics.precision_recall_curve(
                y_true, y_pred, drop_intermediate=False
            )
        except TypeError:
            prec, rec, precrec_thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)

        if np.isnan(rec).any():
            np.nan_to_num(rec, copy=False)

        prec, rec = prec[::-1], rec[::-1]

        # Interpolate values of y for each x in base_rec, by guessing the
        # function rec = f(prec)
        interp_prec: np.ndarray = np.interp(base_rec, rec, prec)

        rand_clf: float = (
            1 - (y_true == pd.Series(y_true).value_counts().idxmax()).sum() / y_true.shape[0]
        )

        result = cls(
            prec=prec,
            rec=rec,
            thresholds=precrec_thresholds,
            base_rec=base_rec,
            interp_prec=interp_prec,
            random_clf=rand_clf,
        )
        result._y_true = np.array(y_true, dtype=float)
        result._y_pred = np.array(y_pred, dtype=float)
        if result.auc < rand_clf:
            warnings.warn(
                f"PRCresults AUPRC={result.auc:.3f} < random baseline={rand_clf:.3f} - "
                "scores appear anti-correlated with labels. Call .invert() to flip.",
                UserWarning,
                stacklevel=2,
            )
        return result

    def invert(self) -> "PRCresults":
        """Return a new PRCresults computed with negated scores (flip anti-correlated scorer)."""
        if self._y_true is None or self._y_pred is None:
            raise ValueError("invert() requires an instance built via from_ytrue_ypred.")
        return self.__class__.from_ytrue_ypred(self._y_true, -self._y_pred, self.base_rec)

    @property
    def auc(self) -> float:
        """Area under the precision-recall curve."""
        return float(sklearn.metrics.auc(self.base_rec, self.interp_prec))

    def plot(
        self, ax: Axes | None = None, plot_params: dict[str, Any] | None = None
    ) -> tuple[Figure | None, Axes]:
        """Plot the precision-recall curve, creating a new figure if `ax` is None."""
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        if plot_params is None:
            plot_params = {}

        ax.plot(
            self.rec,
            self.prec,
            **plot_params,
        )
        # Add the random clf constant.
        ax.axhline(self.random_clf, linestyle="--", color="#888888")

        return (fig, ax)


class ROCresults:
    """Handle data structures for plotting a ROC Curve from a binary classification task.

    Instanciation: build from prediction results with `ROCresults.from_ytrue_ypred`.
    """

    def __init__(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray | None,
        base_fpr: np.ndarray,
        interp_tpr: np.ndarray,
    ) -> None:
        """Store precomputed ROC curve values; see `from_ytrue_ypred`."""
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.base_fpr = base_fpr
        self.interp_tpr = interp_tpr
        self._y_true: np.ndarray | None = None
        self._y_pred: np.ndarray | None = None

    @property
    def auc(self) -> float:
        """Area under the ROC curve."""
        return float(sklearn.metrics.auc(self.base_fpr, self.interp_tpr))

    @classmethod
    def from_ytrue_ypred(
        cls, y_true: np.ndarray, y_pred: np.ndarray, base_fpr: np.ndarray | None = None
    ) -> "ROCresults":
        """Build a ROCresults from true labels and predicted scores."""
        fpr: np.ndarray
        tpr: np.ndarray
        roc_thresholds: np.ndarray

        if base_fpr is None:
            base_fpr = np.linspace(0, 1, 101)

        try:
            fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(
                y_true, y_pred, drop_intermediate=False
            )
        except TypeError:
            fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y_true, y_pred)

        # Interpolate values of y for each x in base_fpr, by guessing the function tpr = f(fpr)
        interp_tpr: np.ndarray = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        result = cls(
            fpr=fpr, tpr=tpr, thresholds=roc_thresholds, base_fpr=base_fpr, interp_tpr=interp_tpr
        )
        result._y_true = np.array(y_true, dtype=float)
        result._y_pred = np.array(y_pred, dtype=float)
        if result.auc < 0.5:
            warnings.warn(
                f"ROCresults AUROC={result.auc:.3f} < 0.5 - scores appear "
                "anti-correlated with labels. Call .invert() to flip.",
                UserWarning,
                stacklevel=2,
            )
        return result

    def invert(self) -> "ROCresults":
        """Return a new ROCresults computed with negated scores (flip anti-correlated scorer)."""
        if self._y_true is None or self._y_pred is None:
            raise ValueError("invert() requires an instance built via from_ytrue_ypred.")
        return self.__class__.from_ytrue_ypred(self._y_true, -self._y_pred, self.base_fpr)

    def plot(
        self, ax: Axes | None = None, plot_params: dict[str, Any] | None = None
    ) -> tuple[Figure | None, Axes]:
        """Plot the ROC curve, creating a new figure if `ax` is None."""
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        if plot_params is None:
            plot_params = {}

        params: dict[str, Any] = {
            "alpha": 1.0,
            "color": "#6a0019",
        }
        params.update(plot_params)

        ax.plot(
            self.fpr,
            self.tpr,
            **params,
        )

        return (fig, ax)


class ListROCresults:
    """A collection of ROCresults, e.g. from cross-validation folds, with mean/plotting helpers."""

    def __init__(self, list_results: list[ROCresults]) -> None:
        """Store `list_results`; raises ValueError if empty or not all ROCresults."""
        if not len(list_results) > 0:
            raise ValueError("list_results must have at least one element")
        if not all(isinstance(x, ROCresults) for x in list_results):
            raise ValueError("list_results must contain only ROCresults instances")

        ## Check that all the interpolations bases are the same.
        # if not all(
        #     [np.allclose(list_results[0].base_fpr, other.base_fpr) for other in list_results[1:]]
        # ):
        #    raise ValueError("All ROCresults instances must have the same base_fpr")

        self.list_results = list_results

    @property
    def mean_auc(self) -> float:
        """Mean AUC across all ROCresults in this collection."""
        return float(np.nanmean([result.auc for result in self.list_results]))

    def make_mean_roc_results(self) -> ROCresults:
        """Build a single ROCresults from the mean TPR curve across this collection."""
        mean_tprs = self.calculate_mean_tprs()
        return ROCresults(
            fpr=self.list_results[0].fpr,
            tpr=mean_tprs,
            thresholds=None,
            base_fpr=self.list_results[0].base_fpr,
            interp_tpr=mean_tprs,
        )

    def calculate_mean_tprs(self) -> np.ndarray:
        """Mean interpolated TPR curve across this collection."""
        interp_tprs = np.array([result.interp_tpr for result in self.list_results])
        return np.asarray(np.nanmean(interp_tprs, axis=0))

    def plot(
        self,
        mean_only: bool = False,
        ax: Axes | None = None,
        show_surface: bool = True,
        plot_params: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot the mean ROC curve (and, unless `mean_only`, each individual curve)."""
        if plot_params is None:
            plot_params = {}

        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        default_color = "#b12900"

        base_fpr = self.list_results[0].base_fpr.astype(float)

        interp_tprs = []
        for result in self.list_results:
            if not mean_only:
                params: dict[str, Any] = {
                    "alpha": 0.2,
                    "color": default_color,
                }
                params.update(plot_params)
                ax.plot(result.fpr, result.tpr, **params)

            interp_tprs.append(result.interp_tpr)

        inter_tprs = np.array(interp_tprs, dtype=float)
        mean_tprs = np.nanmean(inter_tprs, axis=0)
        std = np.nanstd(inter_tprs, axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = np.maximum(mean_tprs - std, 0)

        auc_mean = sklearn.metrics.auc(base_fpr, mean_tprs)

        ax.plot(
            base_fpr,
            mean_tprs,
            linewidth=2,
            color=plot_params.get("color", default_color),
            label=plot_params.get("label", f"Mean ROC curve\n(AUC={auc_mean:.4})"),
        )

        if show_surface:
            ax.fill_between(
                base_fpr,
                tprs_lower,
                tprs_upper,
                color=plot_params.get("color", default_color),
                alpha=0.3,
            )

        # Random classifier
        ax.plot([0, 1], [0, 1], "--", color="#777777")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate / Recall")
        ax.set_title("ROC curve")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.legend()

        ax.set_aspect("equal")

        plt.tight_layout()

        return (fig, ax)


class ListPRCresults:
    """A collection of PRCresults, e.g. from cross-validation folds, with mean/plotting helpers."""

    def __init__(self, list_results: list[PRCresults]) -> None:
        """Store `list_results`; raises ValueError if empty or not all PRCresults."""
        if not len(list_results) > 0:
            raise ValueError("list_results must have at least one element")
        if not all(isinstance(x, PRCresults) for x in list_results):
            raise ValueError("list_results must contain only PRCresults instances")

        ## Check that all the interpolations bases are the same.
        # if not all(
        #     [np.allclose(list_results[0].base_rec, other.base_rec) for other in list_results[1:]]
        # ):
        #    raise ValueError("All PRCresults instances must have the same base_rec")

        self.list_results = list_results

    @property
    def mean_auc(self) -> float:
        """Mean AUC across all PRCresults in this collection."""
        return float(np.nanmean([result.auc for result in self.list_results]))

    @property
    def mean_random_clf(self) -> float:
        """Mean random-classifier baseline across all PRCresults in this collection."""
        return float(np.nanmean([result.random_clf for result in self.list_results]))

    def make_mean_prc_results(self) -> PRCresults:
        """Build a single PRCresults from the mean precision curve across this collection."""
        mean_precs = self.calculate_mean_precs()
        return PRCresults(
            prec=mean_precs,
            rec=self.list_results[0].rec,
            thresholds=None,
            base_rec=self.list_results[0].base_rec,
            interp_prec=mean_precs,
            random_clf=self.mean_random_clf,
        )

    def calculate_mean_precs(self) -> np.ndarray:
        """Mean interpolated precision curve across this collection."""
        interp_precs = np.array([result.interp_prec for result in self.list_results])
        return np.asarray(np.nanmean(interp_precs, axis=0))

    def plot(
        self,
        mean_only: bool = False,
        show_surface: bool = True,
        ax: Axes | None = None,
        plot_params: dict[str, Any] | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot the mean precision-recall curve (and, unless `mean_only`, each individual curve)."""
        if plot_params is None:
            plot_params = {}

        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        default_color = "#b12900"

        base_rec = self.list_results[0].base_rec.astype(float)

        interp_precs = []
        for result in self.list_results:
            if not mean_only:
                params: dict[str, Any] = {
                    "alpha": 0.2,
                    "color": default_color,
                }
                params.update(plot_params)
                ax.plot(result.rec, result.prec, **params)

            interp_precs.append(result.prec)

        inter_precs = np.array(interp_precs, dtype=float)
        mean_precs = np.nanmean(inter_precs, axis=0)
        std = np.nanstd(inter_precs, axis=0)

        precs_upper = np.minimum(mean_precs + std, 1)
        precs_lower = np.maximum(mean_precs - std, 0)

        auc_mean = sklearn.metrics.auc(base_rec, mean_precs)

        ax.plot(
            base_rec,
            mean_precs,
            linewidth=2,
            color=plot_params.get("color", default_color),
            label=plot_params.get("label", f"Mean PRC curve\n(AUC={auc_mean:.4})"),
        )
        if show_surface:
            ax.fill_between(
                base_rec,
                precs_lower,
                precs_upper,
                color=plot_params.get("color", default_color),
                alpha=0.3,
            )

        # Random classifier
        ax.axhline(self.mean_random_clf, linestyle="--", color="#777777")

        ax.set_ylabel("Precision")
        ax.set_xlabel("True Positive Rate / Recall")
        ax.set_title("PRC curve")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.legend()

        ax.set_aspect("equal")

        plt.tight_layout()

        return (fig, ax)
