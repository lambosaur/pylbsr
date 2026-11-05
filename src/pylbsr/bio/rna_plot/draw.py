"""Annotated drawings of a secondary structure on a flat exterior loop.

On top of `layout.flat_coordinates`, `draw_structure` can show:

- base letters on nodes, with an optional colour per nucleotide (see `colors`);
- a focus span: nucleotides outside it are drawn as flank (grey, dashed outline);
- loop motifs (`Loop`): loop nucleotides circled; the closing pair, when it is formed,
  drawn as filled squares;
- tracks (`Track`), e.g. CLIP sites or reads: lines running beside the backbone, always on
  the left of the 5'->3' direction. In this layout that side is above the exterior backbone,
  on the outer side of both strands of every helix and outside hairpin loops; it never holds
  a base-pair line, so a track never crosses the backbone. One layer per track kind keeps
  overlapping tracks readable;
- a constraint structure: its pairs that were formed drawn thick, the others dashed;
- highlighted nucleotides (e.g. mutations), with a thick red outline;
- position numbers, placed on the side away from the pairing partner.

All positions are 0-based; spans are half-open [start, end), as elsewhere in pylbsr.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .layout import Engine, flat_coordinates, pair_list, pair_table

ColorLike = str | tuple[float, float, float] | tuple[float, float, float, float]

CONSTRAINT_COLOR = "#2e7d32"
HIGHLIGHT_COLOR = "#d50000"
FLANK_COLOR = "0.82"
TRACK_OFFSET = 0.62  # distance of layer 0 from the backbone, in backbone steps
TRACK_LAYER_STEP = 0.30


@dataclass(frozen=True)
class Loop:
    """A loop motif to circle, e.g. a hairpin triloop.

    Attributes:
        start: 0-based first loop nucleotide.
        end: 0-based end (exclusive). The closing pair is (start - 1, end) when it is formed.
        label: Legend label; loops with the same label share one legend entry.
        color: Outline colour.
    """

    start: int
    end: int
    label: str
    color: ColorLike = "#00838f"


@dataclass(frozen=True)
class Track:
    """A span drawn as a line beside the backbone, e.g. a CLIP binding site.

    Attributes:
        start: 0-based first nucleotide.
        end: 0-based end (exclusive).
        label: Legend label; tracks with the same label share one legend entry.
        color: Line colour.
        layer: 0 = nearest to the backbone; use one layer per track kind.
    """

    start: int
    end: int
    label: str
    color: ColorLike = "#1565c0"
    layer: int = 0


@dataclass
class StructurePlot:
    """Result of `draw_structure`.

    Attributes:
        xy: Node coordinates, shape (n, 2), in backbone steps.
        handles: Legend handles for the annotations that were drawn.
    """

    xy: np.ndarray
    handles: list[Artist] = field(default_factory=list)


def hairpin_loops(structure: str, sizes: Sequence[int] | None = None) -> list[tuple[int, int]]:
    """Return the hairpin loops of `structure` as 0-based half-open spans.

    Example:
        >>> hairpin_loops("((...))..((((....))))")
        [(2, 5), (13, 17)]
        >>> hairpin_loops("((...))..((((....))))", sizes=[3])
        [(2, 5)]

    Args:
        structure: Dot-bracket string.
        sizes: Keep only loops of these lengths (e.g. `[3]` for triloops).
    """
    loops = []
    for i, j in pair_list(structure):
        inner = structure[i + 1 : j]
        if inner and set(inner) == {"."} and (sizes is None or len(inner) in sizes):
            loops.append((i + 1, j))
    return loops


def suggest_figsize(xy: np.ndarray, inches_per_step: float = 0.28) -> tuple[float, float]:
    """Return a (width, height) in inches that keeps letters readable at default sizes."""
    span = xy.max(axis=0) - xy.min(axis=0) + 4.0
    return float(span[0] * inches_per_step), float(span[1] * inches_per_step)


def _away_normals(xy: np.ndarray, table: list[int]) -> np.ndarray:
    """Return unit normals to the backbone, pointing away from what is next to each nt.

    Away from the pairing partner (paired nt), down (unpaired nt on the exterior backbone), or
    away from the local centre (unpaired nt in a loop).
    """
    n = len(xy)
    out = np.zeros((n, 2))
    for k in range(n):
        t = xy[min(k + 1, n - 1)] - xy[max(k - 1, 0)]
        nrm = np.array([-t[1], t[0]]) / (np.linalg.norm(t) or 1.0)
        if table[k] >= 0:
            away = xy[k] - xy[table[k]]
        elif abs(xy[k, 1]) < 1e-9:
            away = np.array([0.0, -1.0])
        else:
            away = xy[k] - xy[max(k - 4, 0) : k + 5].mean(axis=0)
        out[k] = nrm if nrm @ away >= 0 else -nrm
    return out


def _left_normals(xy: np.ndarray, start: int, end: int) -> np.ndarray:
    """Normals along [start, end) on the left of the 5'->3' direction, smoothed over 3 nt."""
    n = len(xy)
    out = []
    for k in range(start, end):
        t = xy[min(k + 1, n - 1)] - xy[max(k - 1, 0)]
        out.append(np.array([-t[1], t[0]]) / (np.linalg.norm(t) or 1.0))
    arr = np.array(out)
    if len(arr) >= 3:
        sm = arr.copy()
        sm[1:-1] = (arr[:-2] + arr[1:-1] + arr[2:]) / 3
        arr = sm / np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-9)
    return arr


def _check_span(start: int, end: int, n: int, what: str) -> None:
    if not 0 <= start < end <= n:
        raise ValueError(f"{what} [{start}, {end}) is outside [0, {n})")


def draw_structure(  # noqa: C901 -- one drawing pass per annotation kind, kept together
    ax: Axes,
    seq: str,
    structure: str,
    *,
    xy: np.ndarray | None = None,
    engine: Engine = "auto",
    node_colors: Sequence[ColorLike] | None = None,
    focus: tuple[int, int] | None = None,
    flank_label: str = "flank",
    loops: Sequence[Loop] = (),
    tracks: Sequence[Track] = (),
    constraint: str | None = None,
    highlight: Sequence[int] = (),
    highlight_label: str = "highlighted nucleotide",
    numbering: int | Sequence[int | None] | None = 1,
    number_every: int = 10,
    letter_size: float = 8.5,
    node_size: float = 120.0,
    show_ends: bool = True,
) -> StructurePlot:
    """Draw `structure` of `seq` on `ax`, exterior loop flat, helices above it.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> res = draw_structure(ax, "GGGAAACCC", "(((...)))", loops=[Loop(3, 6, "triloop")])
        >>> res.xy.shape
        (9, 2)

    Args:
        ax: Target axes (its aspect is set to equal and its axis turned off).
        seq: Sequence; one letter is drawn per node.
        structure: Nested dot-bracket string of the same length.
        xy: Precomputed coordinates (default: `flat_coordinates(structure, engine)`).
        engine: Layout engine passed to `flat_coordinates`.
        node_colors: One fill colour per nucleotide (default: white, flank grey).
        focus: 0-based half-open span; nucleotides outside it are drawn as flank.
        flank_label: Legend label of the flank.
        loops: Loop motifs to circle.
        tracks: Spans to draw beside the backbone.
        constraint: Dot-bracket of constrained pairs (e.g. consensus pairs of a guided fold).
        highlight: 0-based positions to outline in red (e.g. mutated nucleotides).
        highlight_label: Legend label of the highlighted nucleotides.
        numbering: Number of the first nucleotide (int), or one label per nucleotide (None =
            no label, e.g. an inserted base), or None for no numbers. Numbers divisible by
            `number_every` are drawn.
        number_every: Numbering step.
        letter_size: Font size of the base letters.
        node_size: Marker area of the nodes, in points^2.
        show_ends: Write 5' and 3' at the ends.

    Returns:
        Coordinates and legend handles (pass `handles` to `ax.legend`).
    """
    n = len(structure)
    if len(seq) != n:
        raise ValueError(f"sequence ({len(seq)}) and structure ({n}) lengths differ")
    if xy is None:
        xy = flat_coordinates(structure, engine=engine)
    table = pair_table(structure)
    pairs = set(pair_list(structure))
    cons = set(pair_list(constraint)) if constraint else set()
    handles: list[Artist] = []

    # backbone and pairs
    ax.plot(xy[:, 0], xy[:, 1], "-", color="0.6", lw=0.9, zorder=1)
    for i, j in pairs:
        if (i, j) in cons:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], "-", color=CONSTRAINT_COLOR, lw=2.6, zorder=1)
        else:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], "-", color="0.35", lw=1.0, zorder=1)
    for i, j in cons - pairs:
        ax.plot(
            xy[[i, j], 0], xy[[i, j], 1], "--", color=CONSTRAINT_COLOR, lw=1.0, alpha=0.7, zorder=1
        )

    # tracks beside the backbone, drawn above nodes and letters
    seen: dict[str, Artist] = {}
    for t in tracks:
        _check_span(t.start, t.end, n, "track")
        dist = TRACK_OFFSET + TRACK_LAYER_STEP * t.layer
        pts = xy[t.start : t.end] + _left_normals(xy, t.start, t.end) * dist
        if len(pts) == 1:  # a 1-nt track: short tick so it stays visible
            pts = np.vstack([pts, pts + 1e-3])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=t.color,
            lw=3.5,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=10,
        )
        seen.setdefault(t.label, Line2D([], [], color=t.color, lw=3.5, label=t.label))
    handles += list(seen.values())

    # nodes: focus span solid outline, flank dashed (visible under any colouring)
    in_focus = np.ones(n, bool)
    if focus is not None:
        _check_span(*focus, n, "focus")
        in_focus[:] = False
        in_focus[focus[0] : focus[1]] = True
    fill = (
        list(node_colors)
        if node_colors is not None
        else ["white" if f else FLANK_COLOR for f in in_focus]
    )
    if len(fill) != n:
        raise ValueError(f"node_colors has {len(fill)} entries, expected {n}")
    fill_rgba = [mcolors.to_rgba(c) for c in fill]
    for mask, edge, lw, ls in ((in_focus, "0.45", 0.6, "-"), (~in_focus, "0.3", 0.9, "--")):
        idx = np.flatnonzero(mask)
        if len(idx):
            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=node_size,
                c=[fill_rgba[k] for k in idx],
                edgecolors=edge,
                linewidths=lw,
                linestyles=ls,
                zorder=2,
            )
    for k, (x, y) in enumerate(xy):
        ax.text(x, y, seq[k], ha="center", va="center", fontsize=letter_size, zorder=3)
    if focus is not None:
        handles.append(
            Patch(facecolor=FLANK_COLOR, edgecolor="0.3", linestyle="--", label=flank_label)
        )

    # loop motifs
    loop_labels: dict[str, Artist] = {}
    closing_drawn = False
    for lp in loops:
        _check_span(lp.start, lp.end, n, "loop")
        ax.scatter(
            xy[lp.start : lp.end, 0],
            xy[lp.start : lp.end, 1],
            s=node_size * 2.4,
            facecolor="none",
            edgecolor=lp.color,
            linewidth=2.2,
            zorder=4,
        )
        closing = (lp.start - 1, lp.end)
        if closing in pairs:
            closing_drawn = True
            ax.scatter(
                xy[list(closing), 0],
                xy[list(closing), 1],
                s=node_size * 1.6,
                marker="s",
                facecolor=lp.color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.55,
                zorder=4,
            )
        loop_labels.setdefault(
            lp.label,
            Line2D(
                [], [], marker="o", ls="none", mfc="none", mec=lp.color, mew=2, ms=11, label=lp.label
            ),
        )
    handles += list(loop_labels.values())
    if closing_drawn:
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                ls="none",
                mfc="0.5",
                mec="black",
                ms=9,
                label="closing pair of that loop",
            )
        )

    for k in highlight:
        ax.scatter(
            xy[k, 0],
            xy[k, 1],
            s=node_size * 1.9,
            facecolor="none",
            edgecolor=HIGHLIGHT_COLOR,
            linewidth=2.6,
            zorder=5,
        )
    if len(highlight):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                mfc="none",
                mec=HIGHLIGHT_COLOR,
                mew=2.6,
                ms=10,
                label=highlight_label,
            )
        )

    # numbering, on the side away from the partner
    if numbering is not None:
        labels = [numbering + k for k in range(n)] if isinstance(numbering, int) else list(numbering)
        if len(labels) != n:
            raise ValueError(f"numbering has {len(labels)} labels, expected {n}")
        normals = _away_normals(xy, table)
        for k, num in enumerate(labels):
            if num is not None and num % number_every == 0:
                x, y = xy[k] + normals[k] * 1.55
                ax.text(
                    x, y, f"{num:,}", ha="center", va="center", fontsize=letter_size - 2, color="0.35"
                )

    if show_ends and n:
        ax.text(xy[0, 0] - 0.9, xy[0, 1], "5'", ha="right", va="center", fontsize=letter_size + 1)
        ax.text(xy[-1, 0] + 0.9, xy[-1, 1], "3'", ha="left", va="center", fontsize=letter_size + 1)
    if constraint:
        handles += [
            Line2D([], [], color=CONSTRAINT_COLOR, lw=2.6, label="pair in the constraint"),
            Line2D([], [], color="0.35", lw=1.0, label="pair not in the constraint"),
            Line2D(
                [], [], color=CONSTRAINT_COLOR, lw=1.0, ls="--", label="constraint pair not formed"
            ),
        ]
    ax.set_aspect("equal")  # required, or helices shear
    ax.axis("off")
    ax.margins(0.03, 0.14)
    return StructurePlot(xy=xy, handles=handles)
