"""Nucleotide colourings for secondary-structure drawings.

Two conventions, as known from the ViennaRNA tools:

- RNAfold web server: a paired nucleotide gets the ensemble probability of *its drawn pair*,
  an unpaired one its probability of being unpaired; rainbow scale purple (0) -> red (1).
- RNAalifold `alirna.ps`: both partners of a pair get the pair's colour; hue = number of
  distinct canonical pair types in the alignment (1..6), paler = more sequences that cannot
  form the pair; unpaired nucleotides white.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .layout import RNA, pair_list, pair_table

RGBA = tuple[float, float, float, float]

RNAFOLD_CMAP = mcolors.ListedColormap(
    [mcolors.hsv_to_rgb((0.8 * (1 - x), 0.85, 1.0)) for x in np.linspace(0, 1, 256)], name="rnafold"
)
# RNAalifold pair-type palette, by number of distinct pair types (1..6)
PAIR_TYPE_COLORS = ["#d32f2f", "#c79100", "#2e7d32", "#00838f", "#1565c0", "#7b1fa2"]
CANONICAL_PAIRS = frozenset({"AU", "UA", "GC", "CG", "GU", "UG"})
MISSING_COLOR = "0.82"


def pairing_probability(
    seq: str,
    structure: str,
    soft_pairs: Iterable[tuple[int, int]] = (),
    bonus: float = 1.0,
    model: RNA.md | None = None,
) -> np.ndarray:
    """Return, per nucleotide, the ensemble probability of its state in `structure`.

    Paired nucleotide: probability of the pair drawn in `structure`. Unpaired nucleotide:
    probability of being unpaired. The ensemble is RNAfold's partition function of `seq`.

    Args:
        seq: RNA or DNA sequence (T is read as U by ViennaRNA).
        structure: Dot-bracket string of the same length (the structure that is drawn).
        soft_pairs: 0-based pairs that get a `-bonus` kcal/mol soft constraint, e.g. the
            consensus pairs of a comparative fold.
        bonus: Soft-constraint bonus, in kcal/mol.
        model: ViennaRNA model details (default: ViennaRNA defaults, 37 °C).

    Returns:
        Float array of shape (len(seq),) in [0, 1].
    """
    if len(seq) != len(structure):
        raise ValueError(f"sequence ({len(seq)}) and structure ({len(structure)}) lengths differ")
    fc = RNA.fold_compound(seq, model) if model is not None else RNA.fold_compound(seq)
    for i, j in soft_pairs:
        fc.sc_add_bp(i + 1, j + 1, -bonus)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = np.array(fc.bpp())  # 1-based, upper triangle
    bpp = bpp + bpp.T
    table = pair_table(structure)
    out = np.empty(len(seq))
    for k, j in enumerate(table):
        out[k] = bpp[k + 1][j + 1] if j >= 0 else 1.0 - bpp[k + 1].sum()
    return np.clip(out, 0.0, 1.0)


def probability_colors(values: Sequence[float] | np.ndarray) -> list[RGBA]:
    """Map probabilities to the RNAfold rainbow scale; NaN gets light grey."""
    return [
        RNAFOLD_CMAP(float(v)) if np.isfinite(v) else mcolors.to_rgba(MISSING_COLOR) for v in values
    ]


def probability_colorbar(fig: Figure, ax: Axes, label: str | None = None) -> Colorbar:
    """Add a compact horizontal colour bar for `probability_colors` under `ax`."""
    cb = fig.colorbar(
        ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=RNAFOLD_CMAP),
        ax=ax,
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        aspect=40,
        shrink=0.35,
        anchor=(0.05, 1.0),
    )
    cb.set_label(
        label or "probability of the drawn pair (paired nt) / of being unpaired (unpaired nt)",
        fontsize=9,
    )
    cb.ax.tick_params(labelsize=8)
    return cb


def pair_conservation(alignment: Iterable[str], ci: int, cj: int) -> tuple[int, float]:
    """Return (distinct canonical pair types, fraction of rows that cannot pair) at two columns.

    Rows gapped at both columns are ignored. Rows are upper-cased; T is read as U.

    Args:
        alignment: Aligned sequences of equal length, gap character `-`.
        ci: 0-based alignment column of the 5' partner.
        cj: 0-based alignment column of the 3' partner.
    """
    types, bad, n = set(), 0, 0
    for row in alignment:
        x, y = row[ci].upper().replace("T", "U"), row[cj].upper().replace("T", "U")
        if x == "-" and y == "-":
            continue
        n += 1
        if x + y in CANONICAL_PAIRS:
            types.add(x + y)
        else:
            bad += 1
    return len(types), (bad / n if n else 1.0)


def pair_color(n_types: int, frac_bad: float) -> RGBA:
    """Return the RNAalifold-style colour of a pair (grey if no row can form it)."""
    if n_types == 0:
        return mcolors.to_rgba("0.7")
    base = np.array(mcolors.to_rgb(PAIR_TYPE_COLORS[min(n_types, 6) - 1]))
    fade = min(frac_bad * 2.0, 0.85)  # paler with more non-pairing rows
    r, g, b = base + (1 - base) * fade
    return (float(r), float(g), float(b), 1.0)


def covariation_colors(
    structure: str, alignment: Mapping[str, str] | Sequence[str], reference: str
) -> list[RGBA]:
    """Colour the nucleotides of `structure` by the covariation of their pairs.

    Args:
        structure: Dot-bracket string of the *ungapped* reference sequence (what is drawn).
        alignment: Aligned rows (dict name -> row, or a list of rows), gap character `-`.
        reference: The aligned row of the drawn sequence; its ungapped length must equal
            `len(structure)`. It may or may not also be one of the `alignment` rows.

    Returns:
        One RGBA colour per nucleotide of `structure`.
    """
    columns = [c for c, x in enumerate(reference) if x != "-"]
    if len(columns) != len(structure):
        raise ValueError(
            f"reference has {len(columns)} ungapped positions, structure has {len(structure)}"
        )
    rows = list(alignment.values()) if isinstance(alignment, Mapping) else list(alignment)
    colors = [mcolors.to_rgba("white")] * len(structure)
    for i, j in pair_list(structure):
        colors[i] = colors[j] = pair_color(*pair_conservation(rows, columns[i], columns[j]))
    return colors


def covariation_handles(n_sequences: int) -> list[Patch]:
    """Return legend patches for `covariation_colors`."""
    return [
        Patch(color=c, label=f"{k + 1} pair type{'s' if k else ''}")
        for k, c in enumerate(PAIR_TYPE_COLORS)
    ] + [
        Patch(
            facecolor="white",
            edgecolor="0.45",
            label=f"unpaired (pair colour: types across {n_sequences} sequences; "
            "paler = more cannot pair)",
        )
    ]
