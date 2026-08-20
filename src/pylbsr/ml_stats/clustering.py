"""Hierarchical-clustering model selection helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_score


def hierarchical_clustering_cut_tree(
    x: pd.DataFrame, linkage: np.ndarray, list_nclusters: list[int]
) -> dict[str, pd.Series | pd.DataFrame]:
    """Cut a hierarchical-clustering tree at several candidate cluster counts.

    Args:
        x: The table the `linkage` matrix was computed from.
        linkage: Linkage matrix, as returned by `scipy.cluster.hierarchy.linkage`.
        list_nclusters: Candidate numbers of clusters to try.

    Returns:
        Dict with:
            - "silhouettes": a Series mapping each candidate cluster count to its
              silhouette score.
            - "cluster_labels": a DataFrame, samples as rows, one column per
              candidate cluster count, values are that sample's cluster label.
    """
    silhouettes = []
    all_cluster_labels = []

    for n_clust in list_nclusters:
        clusters = pd.Series(cut_tree(linkage, n_clusters=n_clust).flatten())
        all_cluster_labels.append(clusters)
        silhouettes.append(silhouette_score(x, labels=clusters))

    return {
        "silhouettes": pd.Series(silhouettes, index=list_nclusters),
        "cluster_labels": pd.DataFrame(all_cluster_labels).T.set_axis(list_nclusters, axis=1),
    }


def plot_silhouettes(
    silhouettes: pd.Series, title: str = "", ax: Axes | None = None
) -> tuple[Figure | None, Axes]:
    """Point-plot of silhouette score vs. number of clusters.

    Args:
        silhouettes: Series mapping cluster count to silhouette score, as returned
            by `hierarchical_clustering_cut_tree`'s "silhouettes" entry.
        title: Plot title.
        ax: Axes to plot into; a new figure is created if None.

    Returns:
        `(fig, ax)`; `fig` is None when `ax` was passed in.
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    # Name both the index and the values explicitly rather than relying on the
    # input Series already being named "silhouette" -- reset_index() on an
    # unnamed Series produces a column literally named 0, not "silhouette".
    plot_data = silhouettes.rename_axis("N clusters").reset_index(name="silhouette")
    sns.pointplot(data=plot_data, x="N clusters", y="silhouette", ax=ax)

    ax.set_xlabel("N clusters")
    ax.set_title(title)

    return (fig, ax)
