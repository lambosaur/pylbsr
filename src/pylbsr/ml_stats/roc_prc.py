"""ROC and precision-recall curve results, for binary classifier evaluation."""

import warnings
from typing import Any, NamedTuple, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from typing_extensions import Self

from pylbsr.plotting import patch_labelling


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

            interp_precs.append(result.interp_prec)

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


class Clf_scores(NamedTuple):
    """Point (single-threshold) classifier evaluation metrics."""

    accuracy: float
    balanced_accuracy: float
    recall: float
    precision: float
    specificity: float
    f1: float


def evaluate_classifier_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Clf_scores:
    """Compute point classification metrics from hard 0/1 predictions.

    Unlike the rest of this module (which evaluates continuous scores across all
    thresholds via ROC/PRC curves), this operates on already-thresholded predictions,
    for binary classes labeled 0 and 1.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1), already thresholded.

    Returns:
        accuracy/balanced_accuracy/recall/precision/specificity/f1.
    """
    return Clf_scores(
        accuracy=sklearn.metrics.accuracy_score(y_true, y_pred),
        balanced_accuracy=sklearn.metrics.balanced_accuracy_score(y_true, y_pred),
        recall=sklearn.metrics.recall_score(y_true, y_pred),
        precision=sklearn.metrics.precision_score(y_true, y_pred),
        specificity=sklearn.metrics.recall_score(1 - y_true, 1 - y_pred),
        f1=sklearn.metrics.f1_score(y_true, y_pred),
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    cmap: str | list[str] | Colormap | None = None,
    color_class: int | None = None,
    title: str | None = None,
    normalize: bool = False,
    max_count: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Heatmap of a confusion matrix, from `sklearn.metrics.confusion_matrix`.

    Args:
        cm: 2D confusion matrix array.
        classes: Class names, in the same order as `cm`'s rows/columns.
        cmap: Colormap; a diverging map is used by default when `color_class` is given,
            otherwise `"Blues"`.
        color_class: If given (0 or 1), colors predicted-`color_class` cells distinctly
            from predicted-other-class cells (by negating the latter's sign before
            plotting, with a diverging colormap).
        title: Plot title.
        normalize: Show row-normalized percentages instead of raw counts (both are
            still annotated on the cells).
        max_count: Colormap upper bound; defaults to `cm`'s own max (or 1 if `normalize`).
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    cm_counts = cm.copy()
    cm = cm.copy()
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
        max_count = 1
    elif max_count is None:
        max_count = cm.max()

    # Distinguish *predicted* color_class cells from *predicted* other-class cells by
    # negating the latter's sign, so a diverging colormap colors them oppositely.
    if color_class is not None:
        cm[:, abs(color_class - 1)] = -cm[:, abs(color_class - 1)]
        min_count = -max_count
        if cmap is None:
            cmap = sns.diverging_palette(160, 0, n=51)
    else:
        # Bug in the original: this branch unconditionally overwrote cmap with Blues,
        # silently ignoring any cmap the caller explicitly passed. Only default it.
        if cmap is None:
            cmap = mpl.colormaps["Blues"]
        min_count = 0

    if normalize:
        annot_mat = (
            pd.DataFrame(cm.astype(str)) + "\n(N=" + pd.DataFrame(cm_counts).map("{:,}".format) + ")"
        ).to_numpy()
        fmt = ""
    else:
        annot_mat = cm
        fmt = ","

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    sns.heatmap(
        cm,
        vmin=min_count,
        vmax=max_count,
        cmap=cmap,
        center=0,
        annot=annot_mat,
        fmt=fmt,
        annot_kws={"size": 20},
        xticklabels=classes,
        yticklabels=classes,
        linewidth=1.2,
        square=True,
        cbar=False,
        ax=ax,
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if title is not None:
        ax.set_title(title)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.grid(False)
    plt.tight_layout()

    return (fig, ax)


def plot_classification_evaluation(
    list_ytrue_ypred: list[tuple[np.ndarray, np.ndarray]],
    target_class: int = 1,
    title: str = "",
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Barplot of point classification metrics (accuracy/precision/recall/F1/...).

    Args:
        list_ytrue_ypred: One (y_true, y_pred) pair of hard 0/1 labels per fold/run
            (a single-element list for a single evaluation). Error bars (std across
            entries) are shown whenever there's more than one.
        target_class: Which class (0 or 1) recall/precision/f1 are computed against.
        title: Plot title.
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    assert target_class in (0, 1), "target_class must be 0 or 1"

    if target_class == 0:
        list_ytrue_ypred = [(1 - y_true, 1 - y_pred) for y_true, y_pred in list_ytrue_ypred]

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    scores_df = pd.DataFrame(
        [evaluate_classifier_preds(y_true, y_pred) for y_true, y_pred in list_ytrue_ypred]
    )

    has_multiple = len(list_ytrue_ypred) > 1
    errorbar = "sd" if has_multiple else None
    sns.barplot(
        data=scores_df.melt(), x="variable", y="value", color="#BBBBBB", errorbar=errorbar, ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=25)

    mean_scores = {field: f"{value:.3}" for field, value in scores_df.mean().to_dict().items()}
    for tick_label, patch in zip(ax.get_xticklabels(), ax.patches):
        # sns.barplot's patches are always Rectangle at runtime; ax.patches is only
        # statically typed as the more general Patch.
        bar = cast(mpl.patches.Rectangle, patch)
        shift = bar.get_width() / 4 if has_multiple else 0
        patch_labelling(ax, bar, mean_scores[tick_label.get_text()], vertical=True, shift=shift)

    sns.despine(ax=ax)
    ax.grid(False)
    tick_positions = ax.get_xticks()
    new_labels = [t.get_text().replace("_", "\n") for t in ax.get_xticklabels()]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(new_labels, rotation=45, ha="right")

    return (fig, ax)


def quality_classification_plots(
    list_ytrue_ypred: list[tuple[np.ndarray, np.ndarray]],
    class_names: list[str],
    normalize: bool = False,
    cmap: str | list[str] | Colormap | None = None,
    color_class: int | None = None,
    main_title: str = "",
) -> tuple[Figure, list[Axes]]:
    """Confusion matrix and point-metric barplot side by side.

    Args:
        list_ytrue_ypred: One (y_true, y_pred) pair of hard 0/1 labels per fold/run.
        class_names: Class names, in label order (index 0, then 1).
        normalize: Passed through to `plot_confusion_matrix`.
        cmap: Passed through to `plot_confusion_matrix`.
        color_class: Passed through to `plot_confusion_matrix`.
        main_title: Figure title.

    Returns:
        `(fig, [confusion_matrix_ax, metrics_ax])`.
    """
    title_cm = "Confusion matrix" if len(list_ytrue_ypred) == 1 else "Average confusion matrix"

    fig = plt.figure(figsize=(21, 6))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1 / 3, 2 / 3])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    mean_cm = (
        np.array(
            [sklearn.metrics.confusion_matrix(y_true, y_pred) for y_true, y_pred in list_ytrue_ypred]
        )
        .mean(axis=0)
        .astype(int)
    )

    plot_confusion_matrix(
        mean_cm,
        class_names,
        cmap=cmap,
        normalize=normalize,
        color_class=color_class,
        title=title_cm,
        ax=ax1,
    )
    plot_classification_evaluation(list_ytrue_ypred, ax=ax2)

    fig.suptitle(main_title)
    return (fig, [ax1, ax2])


def plot_predproba_distributions(
    list_ytrue_ypred: list[tuple[np.ndarray, np.ndarray]],
    class_names: tuple[str, str] = ("Negatives", "Positives"),
    color_positives: str = "#cf385b",
    color_negatives: str = "#43b0d0",
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Density plot of predicted scores, split by true class.

    A "does my classifier separate the classes at all" diagnostic, complementary to
    ROC/PRC curves (which summarize separation across thresholds into a single AUC).

    Args:
        list_ytrue_ypred: One (y_true, y_pred) pair of scores per fold/run; `y_pred`
            is a continuous score (as in `ROCresults.from_ytrue_ypred`), not a hard label.
        class_names: (negative_class_name, positive_class_name).
        color_positives: Color for the positive-class density.
        color_negatives: Color for the negative-class density.
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    positives_scores = [y_pred[y_true == 1] for y_true, y_pred in list_ytrue_ypred]
    negatives_scores = [y_pred[y_true == 0] for y_true, y_pred in list_ytrue_ypred]

    for scores, color in [(positives_scores, color_positives), (negatives_scores, color_negatives)]:
        for fold_scores in scores:
            sns.kdeplot(fold_scores, alpha=0.15, color=color, fill=True, cut=2, ax=ax)
        sns.kdeplot(np.concatenate(scores), fill=False, alpha=0.25, color=color, cut=2, ax=ax)

    max_ylim = ax.get_ylim()[1] * 1.1
    ax.set_ylim(0, max_ylim)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylabel("Density")
    ax.set_xlabel("Prediction score")

    handles = [
        mpl.patches.Patch(color=color_positives, alpha=0.5, label=class_names[1]),
        mpl.patches.Patch(color=color_negatives, alpha=0.5, label=class_names[0]),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0)
    ax.set_title("Density plot of prediction scores")

    return (fig, ax)


def separability_plots(
    list_ytrue_ypred: list[tuple[np.ndarray, np.ndarray]],
    class_names: tuple[str, str] = ("0", "1"),
    color_positives: str = "#cf385b",
    color_negatives: str = "#43b0d0",
    mean_only: bool = False,
    main_title: str = "",
) -> tuple[Figure, list[Axes]]:
    """ROC curve, PRC curve, and prediction-score density plot, side by side.

    Reuses ListROCresults/ListPRCresults for the first two panels rather than a
    separate curve-averaging implementation.

    Args:
        list_ytrue_ypred: One (y_true, y_pred) pair of continuous scores per fold/run.
        class_names: (negative_class_name, positive_class_name), for the density panel.
        color_positives: Curve/density color for the positive class.
        color_negatives: Density color for the negative class (ROC/PRC panels only
            plot one curve color, matching ListROCresults/ListPRCresults' own default).
        mean_only: Passed through to ListROCresults.plot/ListPRCresults.plot.
        main_title: Figure title.

    Returns:
        `(fig, [roc_ax, prc_ax, density_ax])`.
    """
    fig = plt.figure(figsize=(21, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    roc_results = ListROCresults(
        [ROCresults.from_ytrue_ypred(y_true, y_pred) for y_true, y_pred in list_ytrue_ypred]
    )
    prc_results = ListPRCresults(
        [PRCresults.from_ytrue_ypred(y_true, y_pred) for y_true, y_pred in list_ytrue_ypred]
    )
    roc_results.plot(mean_only=mean_only, ax=ax1, plot_params={"color": color_positives})
    prc_results.plot(mean_only=mean_only, ax=ax2, plot_params={"color": color_positives})
    plot_predproba_distributions(
        list_ytrue_ypred,
        class_names=class_names,
        color_positives=color_positives,
        color_negatives=color_negatives,
        ax=ax3,
    )

    fig.suptitle(main_title, fontsize=18, fontweight="bold", y=1.10)
    return (fig, [ax1, ax2, ax3])
