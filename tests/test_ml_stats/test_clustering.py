"""Tests for ml_stats.clustering -- hierarchical-clustering model selection."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from scipy.cluster.hierarchy import linkage

from pylbsr.ml_stats.clustering import hierarchical_clustering_cut_tree, plot_silhouettes

# Two well-separated blobs of 3 points each -- should cleanly split into 2 clusters.
X = pd.DataFrame(
    [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
        [10.1, 10.1],
        [10.2, 10.0],
    ],
    columns=["x", "y"],
)


def test_hierarchical_clustering_cut_tree_returns_expected_structure() -> None:
    """silhouettes/cluster_labels have the shapes/index documented for the two candidates.

    Also a regression test in itself: the original used
    `.set_axis(list_nclusters, axis=1, inplace=False)`, and `inplace` was removed from
    `DataFrame.set_axis` in pandas 2.0 -- this would TypeError immediately if reintroduced.
    """
    link = linkage(X, method="average")
    result = hierarchical_clustering_cut_tree(X, link, [2, 3])

    assert list(result["silhouettes"].index) == [2, 3]
    assert list(result["cluster_labels"].columns) == [2, 3]
    assert result["cluster_labels"].shape == (6, 2)


def test_hierarchical_clustering_cut_tree_finds_the_two_obvious_clusters() -> None:
    """2 well-separated blobs of 3 points each should give a high silhouette score at k=2."""
    link = linkage(X, method="average")
    result = hierarchical_clustering_cut_tree(X, link, [2])

    assert result["silhouettes"][2] > 0.9


def test_plot_silhouettes_accepts_an_unnamed_series() -> None:
    """Regression test: the original assumed the input Series was already named "silhouette",
    which plain `pd.Series(...)` construction never does -- reset_index() would then produce a
    column named 0, not "silhouette", and sns.pointplot(y="silhouette", ...) would KeyError.
    """
    silhouettes = pd.Series([0.5, 0.8, 0.6], index=[2, 3, 4])  # deliberately unnamed

    fig, ax = plot_silhouettes(silhouettes)

    assert fig is not None
    assert ax.get_xlabel() == "N clusters"


def test_plot_silhouettes_reuses_given_ax() -> None:
    """Passing an existing ax means no new figure is created."""
    import matplotlib.pyplot as plt

    fig_in = plt.figure()
    ax_in = fig_in.add_subplot(1, 1, 1)

    silhouettes = pd.Series([0.5, 0.8], index=[2, 3])
    fig_out, ax_out = plot_silhouettes(silhouettes, ax=ax_in)

    assert fig_out is None
    assert ax_out is ax_in
