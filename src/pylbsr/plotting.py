import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable


def setup_mpl_everything(
    font: str = "Arial",
    flag_font_fallback_to_default: bool = False,
    seaborn_default_palette: str = "Set2",
) -> None:
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
    # Convert to RGB (0–1)
    rgb = mpl.colors.to_rgb(hex_color)
    # Relative luminance (Rec. 709)
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return lum
