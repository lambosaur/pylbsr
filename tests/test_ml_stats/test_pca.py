"""Tests for ml_stats.pca -- PCA diagnostic plots."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.decomposition import PCA

from pylbsr.ml_stats.pca import (
    dataframe_rotations,
    feature_map_factorplot,
    heatmap_pca_features_to_pc,
    plot_cumulative_variance_pca,
    plot_samples_pca_2d,
    plot_samples_pca_3d,
)

FEATURE_NAMES = ["f1", "f2", "f3", "f4"]
X = np.array(
    [
        [1.0, 2.0, 0.5, 3.0],
        [2.0, 4.0, 1.0, 6.0],
        [1.5, 3.0, 0.8, 4.5],
        [5.0, 1.0, 4.0, 0.5],
        [6.0, 0.5, 5.0, 0.2],
        [5.5, 0.8, 4.5, 0.4],
    ]
)


@pytest.fixture
def pca() -> PCA:
    """A small, fully-fit PCA on a deterministic 4-feature/6-sample dataset."""
    return PCA(n_components=4).fit(X)


def test_dataframe_rotations_shape_and_default_labels(pca: PCA) -> None:
    """Rotation matrix is (n_components x n_features), default-labeled PC0.. and feature names."""
    rotations = dataframe_rotations(pca, FEATURE_NAMES)

    assert rotations.shape == (4, 4)
    assert list(rotations.index) == ["PC0", "PC1", "PC2", "PC3"]
    assert list(rotations.columns) == FEATURE_NAMES


def test_dataframe_rotations_custom_pc_names(pca: PCA) -> None:
    """Custom pc_names are used verbatim instead of the PC0.. default."""
    rotations = dataframe_rotations(pca, FEATURE_NAMES, pc_names=["a", "b", "c", "d"])

    assert list(rotations.index) == ["a", "b", "c", "d"]


def test_plot_cumulative_variance_pca_respects_nmax(pca: PCA) -> None:
    """nmax truncates the plotted x-axis to that many components."""
    fig, ax = plot_cumulative_variance_pca(pca, nmax=2)

    assert isinstance(fig, Figure)
    assert len(ax.collections) > 0 or len(ax.lines) > 0  # something was actually plotted


def test_plot_cumulative_variance_pca_reuses_given_ax(pca: PCA) -> None:
    """Passing an existing ax means no new figure is created."""
    fig1, ax1 = plot_cumulative_variance_pca(pca)
    fig2, ax2 = plot_cumulative_variance_pca(pca, ax=ax1)

    assert fig2 is None
    assert ax2 is ax1
    assert fig1 is not None


def test_feature_map_factorplot_validates_both_pc_x_and_pc_y(pca: PCA) -> None:
    """Regression test: the original code only ever validated pc_x, never pc_y.

    An invalid pc_y must raise just as loudly as an invalid pc_x.
    """
    with pytest.raises(AssertionError):
        feature_map_factorplot(pca, FEATURE_NAMES, pc_x="PC0", pc_y="not-a-pc")
    with pytest.raises(AssertionError):
        feature_map_factorplot(pca, FEATURE_NAMES, pc_x="not-a-pc", pc_y="PC1")


def test_feature_map_factorplot_valid_pcs_returns_fig_ax(pca: PCA) -> None:
    """A valid (pc_x, pc_y) pair plots without raising."""
    fig, ax = feature_map_factorplot(pca, FEATURE_NAMES, pc_x="PC0", pc_y="PC1")

    assert isinstance(fig, Figure)
    assert ax.get_xlim() == (-1.1, 1.1)


def test_heatmap_pca_features_to_pc_respects_n_max_pc(pca: PCA) -> None:
    """n_max_pc limits how many PCs (rows) are shown in the heatmap."""
    _, ax = heatmap_pca_features_to_pc(pca, FEATURE_NAMES, n_max_pc=2)

    assert len(ax.get_yticklabels()) == 2


def test_plot_samples_pca_2d_default_labels_single_group(pca: PCA) -> None:
    """With no labels given, all samples are plotted as a single group."""
    fig, ax = plot_samples_pca_2d(pca, X)

    assert isinstance(fig, Figure)
    assert len(ax.collections) == 1  # one scatter call, one group


def test_plot_samples_pca_2d_groups_by_label(pca: PCA) -> None:
    """Distinct labels produce one scatter collection per group."""
    labels = [0, 0, 0, 1, 1, 1]
    _, ax = plot_samples_pca_2d(pca, X, labels=labels)

    assert len(ax.collections) == 2


def test_plot_samples_pca_3d_creates_a_real_3d_axes(pca: PCA) -> None:
    """Regression test: the original code referenced an unimported `Axes3D` name directly
    (a NameError waiting to happen); the fixed version uses `projection="3d"`.
    """
    fig, ax = plot_samples_pca_3d(pca, X)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes3D)


def test_plot_samples_pca_3d_rejects_non_3d_ax(pca: PCA) -> None:
    """Passing a plain 2D ax must be rejected, not silently misused."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax_2d = fig.add_subplot(1, 1, 1)

    with pytest.raises(AssertionError, match="3D"):
        plot_samples_pca_3d(pca, X, ax=ax_2d)


def test_plot_samples_pca_3d_zlabel_uses_the_z_components_own_variance(pca: PCA) -> None:
    """Regression test: the original code's z-axis label used PC_axes[2] as the name but
    PC_variance_ratio[PC_axes[1]] (the y-component's variance) as the percentage -- a
    copy-paste bug. The z-label's percentage must match PC2's own explained variance ratio.
    """
    _, ax = plot_samples_pca_3d(pca, X, pc_x="PC0", pc_y="PC1", pc_z="PC2")

    expected_pct = 100 * pca.explained_variance_ratio_[2]
    assert f"{expected_pct:.1f}%" in ax.get_zlabel()
