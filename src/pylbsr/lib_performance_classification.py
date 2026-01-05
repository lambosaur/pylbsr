#! /usr/bin/env python


from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from typing_extensions import Self


class PRCresults:
    """Handle data structures for plotting a Precision-Recall Curve from a binary classification task.

    Instanciation: either build from prediction results with `PRCresults.from_ytrue_ypred`
    or with `PRCresults.from_prec_rec` (useful when building a new structure from the average performance
    of a set of PRCresults in a ListPRCresults instance).

    """
    def __init__(
        self,
        prec: np.ndarray,
        rec: np.ndarray,
        thresholds: np.ndarray,
        base_rec: np.ndarray,
        interp_prec: np.ndarray,
        random_clf: float,
    ):
        self.prec = prec
        self.rec = rec
        self.thresholds = thresholds
        self.base_rec = base_rec
        self.interp_prec = interp_prec
        self.random_clf = random_clf

    @classmethod
    def from_prec_rec() -> Self:
        raise NotImplementedError("This method is not implemented yet.")

    @classmethod
    def from_ytrue_ypred(cls, y_true: np.ndarray, y_pred: np.ndarray, base_rec: Optional[np.ndarray] = None) -> Self:
        prec: np.ndarray
        rec: np.ndarray
        precrec_thresholds: np.ndarray

        if base_rec is None:
            base_rec = np.linspace(0, 1, 101)

        try:
            prec, rec, precrec_thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred, drop_intermediate=False)
        except TypeError:
            prec, rec, precrec_thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)


        if np.isnan(rec).any():
            np.nan_to_num(rec, copy=False)

        prec, rec = prec[::-1], rec[::-1]

        # Interpolate values of y for each x in base_rec, by guessing the
        # function rec = f(prec)
        interp_prec: np.ndarray = np.interp(base_rec, rec, prec)

        rand_clf: float = 1 - (y_true == pd.Series(y_true).value_counts().idxmax()).sum() / y_true.shape[0]

        return cls(
            prec=prec,
            rec=rec,
            thresholds=precrec_thresholds,
            base_rec=base_rec,
            interp_prec=interp_prec,
            random_clf=rand_clf,
        )

    @property
    def auc(self) -> float:
        return sklearn.metrics.auc(self.base_rec, self.interp_prec)

    def plot(
        self, ax: Optional[plt.Axes] = None, plot_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[plt.Figure], plt.Axes]:
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
    def __init__(
        self, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, base_fpr: np.ndarray, interp_tpr: np.ndarray
    ):
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.base_fpr = base_fpr
        self.interp_tpr = interp_tpr

    @property
    def auc(self) -> float:
        return sklearn.metrics.auc(self.base_fpr, self.interp_tpr)

    @classmethod
    def from_ytrue_ypred(cls, y_true: np.ndarray, y_pred: np.ndarray, base_fpr: np.ndarray = None) -> "ROCresults":
        fpr: np.ndarray
        tpr: np.ndarray
        roc_thresholds: np.ndarray

        if base_fpr is None:
            base_fpr = np.linspace(0, 1, 101)

        try:
            fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
        except TypeError:
            fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y_true, y_pred)

        # Interpolate values of y for each x in base_fpr, by guessing the function tpr = f(fpr)
        interp_tpr: np.ndarray = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        return cls(fpr=fpr, tpr=tpr, thresholds=roc_thresholds, base_fpr=base_fpr, interp_tpr=interp_tpr)

    def plot(
        self, ax: Optional[plt.Axes] = None, plot_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[plt.Figure], plt.Axes]:
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        if plot_params is None:
            plot_params = {}

        params = {
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
    def __init__(self, list_results: List[ROCresults]):
        if not len(list_results) > 0:
            raise ValueError("list_results must have at least one element")
        if not all(isinstance(x, ROCresults) for x in list_results):
            raise ValueError("list_results must contain only ROCresults instances")

        ## Check that all the interpolations bases are the same.
        # if not all([np.allclose(list_results[0].base_fpr, other.base_fpr) for other in list_results[1:]]):
        #    raise ValueError("All ROCresults instances must have the same base_fpr")

        self.list_results = list_results

    @property
    def mean_auc(self) -> float:
        return np.nanmean([result.auc for result in self.list_results])

    def make_mean_roc_results(self) -> ROCresults:
        mean_tprs = self.calculate_mean_tprs()
        return ROCresults(
            fpr=self.list_results[0].fpr,
            tpr=mean_tprs,
            thresholds=None,
            base_fpr=self.list_results[0].base_fpr,
            interp_tpr=mean_tprs,
        )

    def calculate_mean_tprs(self) -> np.ndarray:
        inter_tprs = np.array([result.interp_tpr for result in self.list_results])
        return np.nanmean(inter_tprs, axis=0)

    def plot(
        self, mean_only: bool = False, ax: Optional[plt.Axes] = None,
        show_surface: bool = True,
        plot_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[plt.Figure], plt.Axes]:
        if plot_params is None:
            plot_params = {}

        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        default_color = "#b12900"

        base_fpr = self.list_results[0].base_fpr.astype(float)

        inter_tprs = []
        for result in self.list_results:
            if not mean_only:
                params = {
                    "alpha": 0.2,
                    "color": default_color,
                }
                params.update(plot_params)
                ax.plot(result.fpr, result.tpr, **params)

            inter_tprs.append(result.interp_tpr)

        inter_tprs = np.array(inter_tprs, dtype=float)
        mean_tprs = np.nanmean(inter_tprs, axis=0)
        std = np.nanstd(inter_tprs, axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = np.maximum(mean_tprs - std, 0)

        auc_mean = sklearn.metrics.auc(base_fpr, mean_tprs)

        ax.plot(
            base_fpr, mean_tprs, linewidth=2, color=plot_params.get("color",default_color), label=plot_params.get("label", "Mean ROC curve\n(AUC={:.4})".format(auc_mean))
        )

        if show_surface:
            ax.fill_between(base_fpr, tprs_lower, tprs_upper, color=plot_params.get("color",default_color), alpha=0.3)

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
    def __init__(self, list_results: List[PRCresults]):
        if not len(list_results) > 0:
            raise ValueError("list_results must have at least one element")
        if not all(isinstance(x, PRCresults) for x in list_results):
            raise ValueError("list_results must contain only PRCresults instances")

        ## Check that all the interpolations bases are the same.
        # if not all([np.allclose(list_results[0].base_rec, other.base_rec) for other in list_results[1:]]):
        #    raise ValueError("All PRCresults instances must have the same base_rec")

        self.list_results = list_results

    @property
    def mean_auc(self) -> float:
        return np.nanmean([result.auc for result in self.list_results])

    @property
    def mean_random_clf(self) -> float:
        return np.nanmean([result.random_clf for result in self.list_results])

    def make_mean_prc_results(self) -> PRCresults:
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
        inter_precs = np.array([result.interp_prec for result in self.list_results])
        return np.nanmean(inter_precs, axis=0)

    def plot(
        self, mean_only: bool = False,
        show_surface: bool = True,
        ax: Optional[plt.Axes] = None,
        plot_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[plt.Figure], plt.Axes]:
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
                params = {
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
            base_rec, mean_precs, linewidth=2, color=plot_params.get("color",default_color), label=plot_params.get("label", "Mean PRC curve\n(AUC={:.4})".format(auc_mean))
        )
        if show_surface:
            ax.fill_between(base_rec, precs_lower, precs_upper, color=plot_params.get("color",default_color), alpha=0.3)

        # Random classifier
        # ax.axhline(self.mean_random_clf, linestyle="--", color="#777777")

        ax.set_ylabel("Precision")
        ax.set_xlabel("True Positive Rate / Recall")
        ax.set_title("PRC curve")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.legend()

        ax.set_aspect("equal")

        plt.tight_layout()

        return (fig, ax)


#
#
# class PairedBinaryClassAndPrediction:
#    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
#        if not len(y_true) == len(y_pred):
#            raise ValueError("y_true and y_pred must have the same length")
#
#        self.y_true = y_true
#        self.y_pred = y_pred
#        self.base_fpr = np.linspace(0, 1, 101)
#        self.base_rec = np.linspace(0, 1, 101)
#
#    def calculate_roc_items(self) -> Dict[str, np.ndarray]:
#        fpr: np.ndarray
#        tpr: np.ndarray
#        roc_thresholds: np.ndarray
#
#        fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(self.y_true, self.y_pred, drop_intermediate=False)
#
#        # Interpolate values of y for each x in base_fpr, by guessing the function tpr = f(fpr)
#        interp_tpr: np.ndarray = np.interp(self.base_fpr, fpr, tpr)
#        interp_tpr[0] = 0.0
#
#        return {
#            "fpr": fpr,
#            "tpr": tpr,
#            "thresholds": roc_thresholds,
#            "interp_tpr": interp_tpr,
#        }
#
#    def calculate_prc_items(self) -> Dict[str, Union[np.ndarray, float]]:
#        prec: np.ndarray
#        rec: np.ndarray
#        precrec_thresholds: np.ndarray
#
#        prec, rec, precrec_thresholds = sklearn.metrics.precision_recall_curve(
#            self.y_true, self.y_pred, drop_intermediate=False
#        )
#
#        if np.isnan(rec).any():
#            np.nan_to_num(rec, copy=False)
#
#        prec, rec = prec[::-1], rec[::-1]
#
#        # Interpolate values of y for each x in base_rec, by guessing the
#        # function rec = f(prec)
#        interp_prec: np.ndarray = np.interp(self.base_rec, rec, prec)
#
#        rand_clf: float = 1 - (y_true == pd.Series(y_true).value_counts().idxmax()).sum() / y_true.shape[0]
#
#
# def plot_multi_roc(
#    list_predictions: List[PairedBinaryClassAndPrediction],
#    color: str = "#cf385b",
#    color_surface: str = "#777777",
#    mean_only: bool = False,
#    ax: Optional[plt.Axes] = None,
#    show_plot: bool = True,
# ) -> Tuple[pd.Series, Tuple[Optional[plt.Figure], plt.Axes]]:
#    """ """
#    if ax is None:
#        fig = plt.figure(figsize=(7, 7))
#        ax = fig.add_subplot(1, 1, 1)
#    else:
#        fig = None
#
#    # This list will contain data for creating the mean curve from the different sets of predictions.
#    tprs = []
#    fprs = []
#    inter_tprs = []
#    base_fpr = np.linspace(0, 1, 101)
#
#    for pred_structure in list_predictions:
#        y_true, y_pred = pred_structure.y_true, pred_structure.y_pred
#
#        fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
#
#        if not mean_only:
#            ax.plot(fpr, tpr, alpha=0.2, color=color)
#
#        tprs.append(tpr)
#        fprs.append(fpr)
#
#        # Interpolate values of y for each x in base_fpr, by guessing the function tpr = f(fpr)
#        tpr = np.interp(base_fpr, fpr, tpr)
#        tpr[0] = 0.0
#        inter_tprs.append(tpr)
#
#    inter_tprs = np.array(inter_tprs)
#    mean_tprs = np.nanmean(inter_tprs, axis=0)
#    std = np.nanstd(inter_tprs, axis=0)
#
#    tprs_upper = np.minimum(mean_tprs + std, 1)
#    tprs_lower = mean_tprs - std
#
#    auc_mean = sklearn.metrics.auc(base_fpr, mean_tprs)
#
#    ax.plot(base_fpr, mean_tprs, linewidth=2, color=color, label="Mean ROC curve\n(AUC={:.4})".format(auc_mean))
#
#    ax.fill_between(base_fpr, tprs_lower, tprs_upper, color=color_surface, alpha=0.3)
#
#    # Random classifier
#    ax.plot([0, 1], [0, 1], "--", color="#777777")
#
#    ax.set_xlabel("False Positive Rate")
#    ax.set_ylabel("True Positive Rate / Recall")
#    ax.set_title("ROC curve")
#    ax.set_xlim(-0.01, 1.01)
#    ax.set_ylim(-0.01, 1.01)
#    ax.legend()
#
#    ax.set_aspect("equal")
#
#    plt.tight_layout()
#
#    if show_plot:
#        plt.show()
#
#    # Let's return AUC values per kfold as well as the
#    # mean AUC from the interpolated data.
#    auc_vals = [sklearn.metrics.auc(a, b) for a, b in zip(fprs, tprs)]
#    auc_vals.append(auc_mean)
#    all_auc = pd.Series(auc_vals, index=list(range(len(list_predictions))) + ["mean_intercept"])
#    return all_auc, (fig, ax)
#
#
# def plot_multi_prc_curve(
#    list_predictions: List[PairedBinaryClassAndPrediction],
#    color="#cf385b",
#    color_surface="#777777",
#    mean_only=False,
#    ax=None,
#    show_plot=True,
# ) -> Tuple[pd.Series, Tuple[Optional[plt.Figure], plt.Axes]]:
#    if ax is None:
#        fig = plt.figure(figsize=(7, 7))
#        ax = fig.add_subplot(1, 1, 1)
#    else:
#        fig = None
#
#    # This list will contain data for creating the mean curve
#    # from the k-fold predictions.
#    random_clf_preds = []
#    recs = []
#    precs = []
#    interp_precs = []
#    base_rec = np.linspace(0, 1, 101)
#
#    for pred_structure in list_predictions:
#        y_true, y_pred = pred_structure.y_true, pred_structure.y_pred
#
#        prec, rec, precrec_thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
#
#        if np.isnan(rec).any():
#            np.nan_to_num(rec, copy=False)
#
#        prec, rec = prec[::-1], rec[::-1]
#        if not mean_only:
#            ax.plot(rec, prec, alpha=0.2, color=color)
#
#        recs.append(rec)
#        precs.append(prec)
#
#        # Interpolate values of y for each x in base_rec, by guessing the
#        # function rec = f(prec)
#        prec = np.interp(base_rec, rec, prec)
#        interp_precs.append(prec)
#
#        rand_clf = 1 - (y_true == pd.Series(y_true).value_counts().idxmax()).sum() / y_true.shape[0]
#        random_clf_preds.append(rand_clf)
#
#    interp_precs = np.array(interp_precs)
#    mean_precs = interp_precs.mean(axis=0)
#    std = interp_precs.std(axis=0)
#
#    precs_upper = np.minimum(mean_precs + std, 1)
#    precs_lower = mean_precs - std
#
#    auc_mean = sklearn.metrics.auc(base_rec, mean_precs)
#
#    ax.plot(base_rec, mean_precs, color=color, label="Mean precision-recall curve\n(AUC={:.4})".format(auc_mean))
#    ax.fill_between(base_rec, precs_lower, precs_upper, color=color_surface, alpha=0.3)
#
#    # Add the random clf constant.
#    ax.axhline(np.mean(random_clf_preds), linestyle="--", color="#888888")
#
#    ax.set_xlabel("True Positive Rate / Recall")
#    ax.set_ylabel("Precision")
#    ax.set_title("Precision-Recall curve")
#    ax.set_xlim(-0.01, 1.01)
#    ax.set_ylim(-0.01, 1.01)
#    ax.legend()
#
#    ax.set_aspect("equal")
#
#    plt.tight_layout()
#
#    if show_plot:
#        plt.show()
#
#    # Let's return AUC values per kfold as well as the
#    # mean AUC from the interpolated data.
#    auc_vals = [sklearn.metrics.auc(a, b) for a, b in zip(recs, precs)]
#    auc_vals.append(auc_mean)
#    all_auc = pd.Series(auc_vals, index=list(range(len(list_predictions))) + ["mean_intercept"])
#    return all_auc, (fig, ax)
