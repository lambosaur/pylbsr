"""PCA diagnostic plots: explained variance, feature loadings, sample projections."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA


def dataframe_rotations(
    pca: PCA, feature_names: list[str], pc_names: list[str] | None = None
) -> pd.DataFrame:
    """Return the PCA rotation (loading) matrix: each feature's weight on each PC.

    Args:
        pca: A fitted sklearn PCA object.
        feature_names: Column names, in the same order used to fit `pca`.
        pc_names: Row labels; defaults to "PC0", "PC1", ...

    Returns:
        DataFrame indexed by PC name, columns are `feature_names`.
    """
    if pc_names is None:
        pc_names = [f"PC{i}" for i in range(len(pca.components_))]
    return pd.DataFrame(pca.components_, index=pc_names, columns=feature_names)


def plot_cumulative_variance_pca(
    pca: PCA,
    nmax: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot cumulative explained variance ratio vs. number of components.

    Useful for picking how many components to keep while still describing the
    data accurately.

    Args:
        pca: A fitted sklearn PCA object.
        nmax: Only plot the first `nmax` components; defaults to all of them.
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    cumulative = pca.explained_variance_ratio_.cumsum()
    n_components = pd.Series(range(1, len(cumulative) + 1), name="N PC")
    if nmax is None:
        nmax = len(cumulative)

    if ax is None:
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    sns.pointplot(x=n_components[:nmax], y=cumulative[:nmax], ax=ax)
    ax.set_xlabel("N PC")
    ax.set_ylabel("cumulative variance\npercentage")
    ax.set_title("Cumulative distribution of the percentage of explained variance")

    return (fig, ax)


def feature_map_factorplot(
    pca: PCA,
    features: list[str],
    pc_x: str = "PC0",
    pc_y: str = "PC1",
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Biplot: 2D arrow plot of feature loadings on two chosen PCA components.

    Args:
        pca: A fitted sklearn PCA object.
        features: Feature names, in the same order used to fit `pca`.
        pc_x: Which component to plot on the x-axis.
        pc_y: Which component to plot on the y-axis.
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    pc_names = [f"PC{i}" for i in range(len(pca.components_))]
    assert pc_x in pc_names, f"'{pc_x}' not in {pc_names}"
    assert pc_y in pc_names, f"'{pc_y}' not in {pc_names}"

    pca_rotations = dataframe_rotations(pca, features, pc_names)
    pc_variance_ratio = pd.Series(pca.explained_variance_ratio_, index=pc_names)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    pc_axes = [pc_x, pc_y]
    for feature, feature_pc in pca_rotations.loc[pc_axes, :].T.iterrows():
        if any(feature_pc.abs() > 0.05):
            arr = ax.arrow(0, 0, feature_pc[pc_axes[0]], feature_pc[pc_axes[1]], width=0.008)
            ax.add_patch(arr)
            ax.text(feature_pc[pc_axes[0]], feature_pc[pc_axes[1]], feature, ha="center")

    ax.set_xlabel(f"{pc_axes[0]} ({100 * pc_variance_ratio[pc_axes[0]]:.1f}% variance)")
    ax.set_ylabel(f"{pc_axes[1]} ({100 * pc_variance_ratio[pc_axes[1]]:.1f}% variance)")

    ax.axhline(0, color="#AAAAAA", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="#AAAAAA", linestyle="--", linewidth=1.5)
    ax.add_patch(plt.Circle((0, 0), 1, color="#666666", fill=False))
    ax.grid(False)

    return (fig, ax)


def heatmap_pca_features_to_pc(
    pca: PCA,
    features: list[str],
    n_max_pc: int | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Heatmap of feature loadings (rows: PCs, columns: features).

    Args:
        pca: A fitted sklearn PCA object.
        features: Feature names, in the same order used to fit `pca`.
        n_max_pc: Only show the first `n_max_pc` components; defaults to all of them.
        title: Plot title; a default is used if None.
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    pc_names = [f"PC{i}" for i in range(len(pca.components_))]
    pca_rotations = dataframe_rotations(pca, features)

    if n_max_pc is None:
        n_max_pc = len(pca.components_)

    if ax is None:
        side = max(45, 0.8 * n_max_pc)
        fig = plt.figure(figsize=(side, side))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    sns.heatmap(
        pca_rotations.loc[pc_names[:n_max_pc], :],
        cmap="PiYG",
        cbar_kws={"shrink": 0.50},
        center=0,
        linewidth=1.2,
        square=True,
        ax=ax,
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(
        title
        or "Principal axes in feature space, representing\n"
        "the directions of maximum variance in the data"
    )

    return (fig, ax)


def plot_samples_pca_2d(
    pca: PCA,
    x: pd.DataFrame,
    labels: list[Any] | None = None,
    pc_x: str = "PC0",
    pc_y: str = "PC1",
    colors_labels: dict[Any, Any] | None = None,
    prefix_label: str = "",
    title: str = "",
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """2D scatter of samples projected onto two chosen PCA components, colored by label.

    Args:
        pca: A fitted sklearn PCA object.
        x: Samples to project (same features/order used to fit `pca`).
        labels: Per-sample group label for coloring; defaults to a single group.
        pc_x: Which component to plot on the x-axis.
        pc_y: Which component to plot on the y-axis.
        colors_labels: Mapping of label to color; a default palette is used if None.
        prefix_label: Prepended to each legend entry.
        title: Plot title.
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    pc_names = [f"PC{i}" for i in range(len(pca.components_))]
    pc_variance_ratio = pd.Series(pca.explained_variance_ratio_, index=pc_names)
    pca_transformed = pd.DataFrame(pca.transform(x), columns=pc_names)

    if labels is None:
        labels = [0] * pca_transformed.shape[0]

    pc_axes = [pc_x, pc_y]
    pca_df_plot = pca_transformed.loc[:, pc_axes].assign(label=labels)

    if not colors_labels:
        palette = sns.color_palette("Paired", len(labels))
        colors_labels = dict(zip(set(labels), palette))

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    for group_name, group in pca_df_plot.groupby("label"):
        ax.scatter(
            group.loc[:, group.columns[0]],
            group.loc[:, group.columns[1]],
            color=colors_labels[group_name],
            alpha=0.6,
            edgecolor="white",
            label=f"{prefix_label}{group_name}",
        )

    ax.set_xlabel(f"{pc_axes[0]} ({100 * pc_variance_ratio[pc_axes[0]]:.1f}% variance)", labelpad=20)
    ax.set_ylabel(f"{pc_axes[1]} ({100 * pc_variance_ratio[pc_axes[1]]:.1f}% variance)", labelpad=20)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_title(title)

    return (fig, ax)


def plot_samples_pca_3d(
    pca: PCA,
    x: pd.DataFrame,
    labels: list[Any] | None = None,
    pc_x: str = "PC0",
    pc_y: str = "PC1",
    pc_z: str = "PC2",
    colors_labels: dict[Any, Any] | None = None,
    prefix_label: str = "",
    title: str = "",
    ax: axes3d.Axes3D | None = None,
) -> tuple[Figure | None, axes3d.Axes3D]:
    """3D scatter of samples projected onto three chosen PCA components, colored by label.

    Args:
        pca: A fitted sklearn PCA object.
        x: Samples to project (same features/order used to fit `pca`).
        labels: Per-sample group label for coloring; defaults to a single group.
        pc_x: Which component to plot on the x-axis.
        pc_y: Which component to plot on the y-axis.
        pc_z: Which component to plot on the z-axis.
        colors_labels: Mapping of label to color; a default palette is used if None.
        prefix_label: Prepended to each legend entry.
        title: Plot title.
        ax: A 3D Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.

    Raises:
        AssertionError: If `ax` is given but isn't a 3D Axes.
    """
    pc_names = [f"PC{i}" for i in range(len(pca.components_))]
    pc_variance_ratio = pd.Series(pca.explained_variance_ratio_, index=pc_names)
    pca_transformed = pd.DataFrame(pca.transform(x), columns=pc_names)

    if labels is None:
        labels = [0] * pca_transformed.shape[0]

    pc_axes = (pc_x, pc_y, pc_z)
    pca_df_plot = pca_transformed.loc[:, list(pc_axes)].assign(label=labels)

    if not colors_labels:
        palette = sns.color_palette("Paired", len(labels))
        colors_labels = dict(zip(set(labels), palette))

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        assert isinstance(ax, axes3d.Axes3D), "'ax' is not a 3D Axes instance"
        fig = None

    for group_name, group in pca_df_plot.groupby("label"):
        ax.scatter(
            group.loc[:, group.columns[0]],
            group.loc[:, group.columns[1]],
            group.loc[:, group.columns[2]],
            color=colors_labels[group_name],
            alpha=0.6,
            edgecolor="white",
            label=f"{prefix_label}{group_name}",
        )

    ax.set_xlabel(f"{pc_axes[0]} ({100 * pc_variance_ratio[pc_axes[0]]:.1f}% variance)", labelpad=20)
    ax.set_ylabel(f"{pc_axes[1]} ({100 * pc_variance_ratio[pc_axes[1]]:.1f}% variance)", labelpad=20)
    ax.set_zlabel(f"{pc_axes[2]} ({100 * pc_variance_ratio[pc_axes[2]]:.1f}% variance)", labelpad=20)
    ax.legend(bbox_to_anchor=(0, 1))
    ax.set_title(title)

    return (fig, ax)
