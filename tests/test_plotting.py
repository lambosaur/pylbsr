"""Tests for the pyutils-ported plotting.py additions.

Only covers what was newly added here; the rest of plotting.py (setup_mpl_everything,
hex_luminance, adjust_lightness, stable_categorical_color) was untested before this and is
out of scope for this change.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pylbsr.plotting import (
    bin_to_labels,
    create_regular_grid_axes,
    make_categorical_palette,
    patch_labelling,
    plot_resizelabel,
    pval_stars,
)


def test_make_categorical_palette_covers_every_id_with_a_hex_color() -> None:
    """Every input ID gets a mapped, valid-looking hex color string."""
    palette = make_categorical_palette(["b", "a", "c"])

    assert set(palette) == {"a", "b", "c"}
    for color in palette.values():
        assert color.startswith("#")
        assert len(color) == 7


def test_make_categorical_palette_is_deterministic() -> None:
    """Same input set (any order) always produces the same mapping."""
    p1 = make_categorical_palette(["cluster_2", "cluster_0", "cluster_1"])
    p2 = make_categorical_palette(["cluster_1", "cluster_2", "cluster_0"])

    assert p1 == p2


def test_make_categorical_palette_no_collisions_within_set() -> None:
    """Assigning from a large-enough palette, distinct IDs get distinct colors."""
    palette = make_categorical_palette(list(range(10)), palette="tab20")

    assert len(set(palette.values())) == 10


def test_plot_resizelabel_wraps_long_multi_token_labels() -> None:
    """A label over max_len with >=4 tokens gets a newline roughly in the middle."""
    assert plot_resizelabel("a b c d e", max_len=3) == "a b c\nd e"


def test_plot_resizelabel_leaves_short_labels_untouched() -> None:
    """A short label is returned as-is."""
    assert plot_resizelabel("short") == "short"


def test_plot_resizelabel_leaves_single_token_labels_untouched() -> None:
    """A long label with too few separator tokens is left untouched, even if it's long."""
    long_single_word = "a" * 30
    assert plot_resizelabel(long_single_word) == long_single_word


def test_create_regular_grid_axes_row_count() -> None:
    """Regression test: the original computed n_tot // n_cols + n_tot % n_cols for the row
    count, which over-allocates rows whenever the remainder is >= 2 (e.g. n_tot=8, n_cols=3
    gave 4 rows instead of the correct ceil(8/3)=3) -- confirmed empirically against the
    original formula before fixing. Uses proper ceiling division instead.
    """
    for n_tot, n_cols in [(8, 3), (11, 4), (10, 3), (9, 3), (7, 3)]:
        _, axs = create_regular_grid_axes(n_tot, n_cols, height_row=2, width=6)
        assert len(axs) == n_tot
        # every subplot's gridspec should agree on the same (correct) row count
        expected_rows = math.ceil(n_tot / n_cols)
        assert axs[0].get_gridspec().nrows == expected_rows


def test_create_regular_grid_axes_returns_axes_in_reading_order() -> None:
    """Axes come back left-to-right, top-to-bottom."""
    _, axs = create_regular_grid_axes(4, 2, height_row=2, width=4)

    assert len(axs) == 4
    rows = [ax.get_subplotspec().rowspan.start for ax in axs]
    assert rows == [0, 0, 1, 1]


def test_bin_to_labels_matches_docstring_example() -> None:
    """The exact example from the docstring, verified directly."""
    result = bin_to_labels(
        pd.Series([0.2, 0.03, 0.005, 0.0005]), [0.05, 0.01, 0.001], ["*", "**", "***"]
    )
    assert result.tolist() == ["", "*", "**", "***"]


def test_bin_to_labels_threshold_order_does_not_matter() -> None:
    """Thresholds/labels can be given in any order -- tightest match still wins."""
    result = bin_to_labels(pd.Series([0.0005]), [0.001, 0.05, 0.01], ["***", "*", "**"])
    assert result.tolist() == ["***"]


def test_pval_stars_default_thresholds() -> None:
    """pval_stars is bin_to_labels pre-bound to the standard */**/*** thresholds."""
    result = pval_stars(pd.Series([0.5, 0.04, 0.005, 0.0001]))
    assert result.tolist() == ["", "*", "**", "***"]


def test_patch_labelling_vertical_positive_bar_places_label_above() -> None:
    """A vertical bar with a positive value gets its label above the bar (va="bottom")."""
    _fig, ax = plt.subplots()
    (bar,) = ax.bar(["a"], [5])

    patch_labelling(ax, bar, "5")

    annotations = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
    assert len(annotations) == 1
    assert annotations[0].get_text() == "5"
    assert annotations[0].get_verticalalignment() == "bottom"


def test_patch_labelling_vertical_negative_bar_places_label_below() -> None:
    """A vertical bar with a negative value gets its label below the bar (va="top")."""
    _fig, ax = plt.subplots()
    (bar,) = ax.bar(["a"], [-5])

    patch_labelling(ax, bar, "-5")

    annotations = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
    assert annotations[0].get_verticalalignment() == "top"


def test_patch_labelling_does_not_mutate_default_textparams_across_calls() -> None:
    """The textparams=None default must not become a shared mutable dict across calls."""
    _fig, ax = plt.subplots()
    (bar1,) = ax.bar(["a"], [1])
    (bar2,) = ax.bar(["b"], [2])

    patch_labelling(ax, bar1, "1")
    patch_labelling(ax, bar2, "2")  # would fail oddly if the first call's state leaked

    annotations = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
    assert {a.get_text() for a in annotations} == {"1", "2"}
