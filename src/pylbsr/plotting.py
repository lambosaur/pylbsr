"""Matplotlib/seaborn styling and color helpers."""

import colorsys
import zlib
from collections.abc import Iterable
from typing import Any

import matplotlib as mpl
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
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
