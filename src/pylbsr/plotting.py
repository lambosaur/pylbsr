"""Matplotlib/seaborn styling and color helpers."""

import colorsys
import math
import zlib
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: F401


def setup_mpl_everything(
    font: str = "Arial",
    flag_font_fallback_to_default: bool = False,
    seaborn_default_palette: str = "Set2",
) -> None:
    """Set matplotlib/seaborn rcParams (font, sizes, grid style, palette) for consistent plots."""
    # This should yield '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
    matplotlib.font_manager.findfont(
        font,
        fallback_to_default=flag_font_fallback_to_default,
    )

    # Assert font is correctly identified.
    # ------------------------------------
    plt.figure()
    plt.text(0.5, 0.5, "test", fontfamily="Arial")
    plt.draw()
    plt.gca().texts[0].get_fontfamily()
    _font_used_detected = plt.gca().texts[0].get_fontproperties().get_name()
    plt.close()
    if _font_used_detected != font:
        if flag_font_fallback_to_default is True:
            print(f"Font '{font}' not found. Fallback to default font '{_font_used_detected}'.")
        else:
            raise ValueError(
                f"Font '{font}' was requested but detected font is '{_font_used_detected}'."
            )

    mpl.rcParams["font.sans-serif"] = [_font_used_detected]
    rc("text", usetex=False)
    # rc("font", **{"family": "serif", "serif": ["Arial"]})

    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["figure.titlesize"] = 16

    sns.set_style("whitegrid")

    # Grid dots
    mpl.rcParams["grid.linestyle"] = ":"
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["grid.color"] = "grey"
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette(seaborn_default_palette)


def hex_luminance(hex_color: str) -> float:
    """Get a [0-1] luminance value from a hex color string (>0.5 is light)."""
    # Assert format of the input
    assert hex_color.startswith("#") and len(hex_color) == 7, (
        "Input must be a hex color string like '#RRGGBB'"
    )
    # Convert to RGB (0-1)
    rgb = mpl.colors.to_rgb(hex_color)
    # Relative luminance (Rec. 709)
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return lum


def adjust_lightness(hex_color: str, lightness: float) -> str:
    """Return ``hex_color`` with its HLS lightness replaced by ``lightness`` (0-1).

    Hue and saturation unchanged -- e.g. a lighter/darker variant of the same color for
    encoding a secondary variable (confidence, support level, ...) via shade while hue
    still encodes a categorical variable (gene, group, ...).

    Uses ``colorsys`` (stdlib), not ``matplotlib.colors.rgb_to_hsv``/``hsv_to_rgb``:
    HSV's "Value" and HLS's "Lightness" are different channels that respond
    differently to the same RGB triple, so they are not interchangeable here.

    Args:
        hex_color: e.g. ``"#66c2a5"``.
        lightness: Target HLS lightness, 0 (black) to 1 (white).

    Returns:
        Hex color string, same format as the input.

    Example:
        >>> adjust_lightness("#66c2a5", 0.85)  # a washed-out variant of the same teal
        '#c8e9df'
    """
    r, g, b = mpl.colors.to_rgb(hex_color)
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return mpl.colors.to_hex((r2, g2, b2))


def stable_categorical_color(key: str, palette: list[str]) -> str:
    """Deterministically map ``key`` to one color in ``palette``, stable across processes and runs.

    Unlike Python's built-in ``hash()``, which is randomized per-process for strings
    (``PYTHONHASHSEED``) and would give a different bucket every run. Collisions across
    many keys against a small palette are expected and fine for "visually distinguish
    most categories at a glance" use cases; use a real categorical encoding (e.g. a
    legend) instead when every key must be unique.

    Args:
        key: The category to color, e.g. a gene name.
        palette: Candidate colors to choose from, e.g. ``SET2_HEX``.

    Returns:
        One entry of ``palette``.

    Example:
        >>> stable_categorical_color("AGO2", ["#66c2a5", "#fc8d62", "#8da0cb"])
        '#66c2a5'
    """
    return palette[zlib.crc32(key.encode()) % len(palette)]


def make_categorical_palette(category_ids: Iterable[Any], palette: str = "tab20") -> dict[Any, str]:
    """Deterministically map a known, full set of category IDs to hex colors.

    Assigns colors by sorted position, so the same ``category_ids`` set always gets the
    same mapping, with no collisions within that set. Unlike `stable_categorical_color`
    (hash-based, one key at a time, no need to know the full set upfront, but can collide
    across a small palette), use this when the full set of categories is known in advance
    and zero collisions within it matters -- e.g. coloring clusters consistently across
    several plots of the same clustering run.

    Args:
        category_ids: The categories to assign colors to.
        palette: A seaborn categorical palette name.

    Returns:
        Mapping of each category ID to a hex color string.
    """
    sorted_ids = sorted(category_ids)
    colors = sns.color_palette(palette=palette, n_colors=len(sorted_ids))
    return dict(zip(sorted_ids, [mpl.colors.rgb2hex(c) for c in colors]))


def plot_resizelabel(label: str, separator: str = " ", max_len: int = 20) -> str:
    """Insert a newline roughly halfway through a long, multi-token label.

    Only wraps if `label` is longer than `max_len` and has at least 4 tokens (3
    occurrences of `separator`) -- short or single-word labels are left untouched.

    Args:
        label: The label to (maybe) wrap.
        separator: Token separator, e.g. `" "`.
        max_len: Labels no longer than this are left untouched.

    Returns:
        `label`, with a newline inserted at the token roughly halfway through if it
        was wrapped, otherwise unchanged.
    """
    if len(label) > max_len and label.count(separator) >= 3:
        tokens = label.split(separator)
        mid = math.ceil(len(tokens) / 2)
        return separator.join(tokens[:mid]) + "\n" + separator.join(tokens[mid:])
    return label


def create_regular_grid_axes(
    n_tot: int, n_cols: int, height_row: float, width: float
) -> tuple[Figure, list[Axes]]:
    """Create a figure with `n_tot` axes laid out in a grid of up to `n_cols` columns.

    Args:
        n_tot: Number of axes to create.
        n_cols: Maximum number of axes per row.
        height_row: Figure height per row, in inches.
        width: Figure width, in inches.

    Returns:
        `(fig, axs)`, `axs` in row-major reading order (left-to-right, top-to-bottom).
    """
    n_rows = math.ceil(n_tot / n_cols)
    fig = plt.figure(figsize=(width, height_row * n_rows))
    axs = [fig.add_subplot(n_rows, n_cols, pos) for pos in range(1, n_tot + 1)]
    return (fig, axs)


def bin_to_labels(
    series: pd.Series, thresholds: Sequence[float], labels: Sequence[str], default: str = ""
) -> pd.Series:
    """Map each value to the label of the tightest threshold it satisfies (value < threshold).

    Args:
        series: Numeric values to label.
        thresholds: Threshold values; order doesn't matter -- internally sorted so the
            smallest (tightest) satisfied threshold's label wins over looser ones.
        labels: One label per threshold, same length as `thresholds`, in the same order
            (i.e. `labels[i]` goes with `thresholds[i]`).
        default: Label used where no threshold is satisfied.

    Returns:
        One label per value in `series`.

    Example:
        >>> bin_to_labels(
        ...     pd.Series([0.2, 0.03, 0.005, 0.0005]), [0.05, 0.01, 0.001], ["*", "**", "***"]
        ... ).tolist()
        ['', '*', '**', '***']
    """
    assert len(thresholds) == len(labels), "thresholds and labels must be the same length"
    # Loosest threshold first, so each subsequent (tighter) match overrides it.
    ordered = sorted(zip(thresholds, labels), reverse=True)

    result = pd.Series(default, index=series.index)
    for threshold, label in ordered:
        result = result.where(series >= threshold, label)
    return result


def pval_stars(
    pvals: pd.Series,
    thresholds: Sequence[float] = (0.05, 0.01, 0.001),
    labels: Sequence[str] = ("*", "**", "***"),
) -> pd.Series:
    """Convert p-values to significance-star labels.

    Note: reducing a p-value to a star rating is generally discouraged in modern
    statistical practice (it discards effect size and the exact p-value) -- kept here
    for convenience/compatibility with older plots, not as an endorsement.

    Args:
        pvals: P-values to label.
        thresholds: Significance thresholds.
        labels: One label per threshold, same order (e.g. `"*"` for the loosest).

    Returns:
        One label per p-value (`""` if none of the thresholds are met).
    """
    return bin_to_labels(pvals, thresholds, labels)


def patch_labelling(
    ax: Axes,
    patch: mpl.patches.Rectangle,
    label: str,
    vertical: bool = True,
    space: float = 3,
    shift: float = 0,
    revert: bool = False,
    textparams: dict[str, Any] | None = None,
) -> None:
    """Annotate one bar of a bar plot with a text label just past its end.

    Args:
        ax: Axes the bar plot was drawn on.
        patch: The bar (one entry of `ax.patches`) to label.
        label: Text to draw.
        vertical: Whether the bars are vertical (value on the y-axis) or horizontal.
        space: Points between the bar's end and the label.
        shift: Extra shift along the bar's own axis (x for vertical, y for horizontal),
            e.g. to de-collide labels on grouped/dodged bars.
        revert: For horizontal bars, force the label to the left of the bar's end
            regardless of sign (vertical bars already do this automatically for
            negative values).
        textparams: Extra kwargs forwarded to `ax.annotate`.

    Example:
        >>> for tick_label, patch in zip(ax.get_xticklabels(), ax.patches):  # doctest: +SKIP
        ...     shift = patch.get_width() / 4
        ...     patch_labelling(ax, patch, label_for[tick_label.get_text()], shift=shift)
    """
    if textparams is None:
        textparams = {}

    if vertical:
        y_value = patch.get_height()
        x_value = patch.get_x() + patch.get_width() / 2 + shift

        ha = "center"
        rotation: float = 0

        va = "bottom"
        if y_value < 0:
            space *= -1
            va = "top"

        xytext: tuple[float, float] = (0, space)
    else:
        x_value = patch.get_width()
        y_value = patch.get_y() + patch.get_height() / 2 + shift

        ha = "center"
        va = "center"
        rotation = -90

        if x_value < 0 or revert:
            space *= -1
            rotation = 90

        xytext = (space, 0)

    ax.annotate(
        label,
        (x_value, y_value),
        xytext=xytext,
        xycoords="data",
        textcoords="offset points",
        ha=ha,
        va=va,
        rotation=rotation,
        **textparams,
    )
