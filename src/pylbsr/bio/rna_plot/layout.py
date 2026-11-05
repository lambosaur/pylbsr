"""Flat-exterior-loop layout of an RNA secondary structure.

The exterior loop is a straight horizontal backbone read 5'->3' from left to right, and every
helix rooted on it is drawn above it (the VARNA "Radiate + flat exterior loop" style).
ViennaRNA has no such layout (its layouts all curve the exterior loop), so each helix subtree
is laid out by a ViennaRNA layout engine and then rigidly placed on the backbone.

Spacing rules, all in units of one backbone step:

- an unpaired exterior nucleotide advances the cursor by `spacing`;
- a helix subtree advances the cursor past its *full* width (a layout can lean a helix
  outside its closing pair), plus `gap`.

Consecutive exterior nucleotides are therefore unevenly spaced around helices, which is what
keeps neighbouring helices from overlapping. Each subtree is scaled uniformly (scaling only x
would shear the helix).

Only nested structures in `(`/`)`/`.` dot-bracket notation are supported (no pseudoknots).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import numpy as np

try:
    import RNA
except ImportError as e:
    raise ImportError(
        "ViennaRNA is required for pylbsr.bio.rna_plot. "
        "Install with `pip install pylbsr[rna]` or `conda install -c bioconda viennarna`."
    ) from e

Engine = Literal["auto", "naview", "simple", "puzzler"]

# ViennaRNA plot types, as in RNA.PLOT_TYPE_* (missing as named constants before 2.7).
_PLOT_TYPE = {"simple": 0, "naview": 1, "puzzler": 4}


def pair_table(structure: str) -> list[int]:
    """Return the 0-based partner of each position, or -1 if unpaired.

    Args:
        structure: Nested dot-bracket string; only `(`, `)` and `.` are allowed.

    Returns:
        List of length `len(structure)`.

    Raises:
        ValueError: if the brackets are unbalanced or another character is present.
    """
    table = [-1] * len(structure)
    stack: list[int] = []
    for k, c in enumerate(structure):
        if c == "(":
            stack.append(k)
        elif c == ")":
            if not stack:
                raise ValueError(f"unbalanced ')' at position {k}")
            i = stack.pop()
            table[i], table[k] = k, i
        elif c != ".":
            raise ValueError(f"unsupported character {c!r} at position {k}")
    if stack:
        raise ValueError(f"unbalanced '(' at position {stack[-1]}")
    return table


def pair_list(structure: str) -> list[tuple[int, int]]:
    """Return the base pairs as 0-based (i, j) tuples with i < j, in 5'->3' order of i."""
    return [(i, j) for i, j in enumerate(pair_table(structure)) if j > i]


@contextmanager
def _plot_type(engine: str) -> Iterator[None]:
    """Temporarily set ViennaRNA's global layout type (read by RNA.get_xy_coordinates)."""
    old = RNA.cvar.rna_plot_type
    RNA.cvar.rna_plot_type = _PLOT_TYPE[engine]
    try:
        yield
    finally:
        RNA.cvar.rna_plot_type = old


def engine_coordinates(structure: str, engine: Literal["naview", "simple", "puzzler"]) -> np.ndarray:
    """Return the raw ViennaRNA layout of `structure`, scaled so one backbone step = 1.

    Args:
        structure: Dot-bracket string.
        engine: ViennaRNA layout: `naview`, `simple` (radial) or `puzzler` (RNApuzzler,
            free of overlaps by construction).

    Returns:
        Float array of shape (len(structure), 2).
    """
    with _plot_type(engine):
        c = RNA.get_xy_coordinates(structure)
        p = np.array([[c.get(k).X, c.get(k).Y] for k in range(len(structure))], dtype=float)
    if len(p) < 2:
        return p
    return p / max(float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1))), 1e-9)


def min_distance(xy: np.ndarray) -> float:
    """Return the smallest distance between two distinct nucleotides (inf if fewer than 2)."""
    if len(xy) < 2:
        return float("inf")
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _oriented(p: np.ndarray) -> np.ndarray:
    """Return a subtree layout with its closing pair on the x axis, 5' base at the origin.

    The helix grows upward (positive y).
    """
    p = p - (p[0] + p[-1]) / 2
    v = p[-1] - p[0]
    ang = -np.arctan2(v[1], v[0])  # closing pair along +x
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    p = p @ rot.T
    if p[:, 1].mean() < 0:  # grow upward
        p[:, 1] *= -1
    return np.asarray(p - p[0])


def _defects(
    p: np.ndarray, sub: str, min_dist: float, sides: tuple[bool, bool]
) -> tuple[int, int, float]:
    """Return (crossings, nucleotides under the backbone, -min distance) of a subtree.

    All three are minimal when the layout is clean. `sides` tells whether an exterior
    backbone segment enters the subtree on the left (5') / leaves it on the right (3'); a
    nucleotide below the backbone on such a side, outside the closing pair, would cross it.
    """
    inner = p[1:-1]
    low = inner[:, 1] < -0.05
    below = int((low & sides[0] & (inner[:, 0] < p[0, 0])).sum())
    below += int((low & sides[1] & (inner[:, 0] > p[-1, 0])).sum())
    return segment_crossings(p, sub), below, -min(min_distance(p), min_dist)


def _subtree_coordinates(
    sub: str, engine: Engine, min_dist: float, sides: tuple[bool, bool] = (True, True)
) -> np.ndarray:
    """Lay out and orient one helix subtree.

    `auto` tries naview, then the radial layout, then RNApuzzler, and keeps the first clean
    layout (no crossing, nothing under the backbone next to it, no nucleotides closer than
    `min_dist`);
    if none is clean, the one with the fewest defects. naview packs multiloops with few
    unpaired bases so tightly that nodes collide; the radial layout is roomier for small
    domains, RNApuzzler is safe for large ones.
    """
    if engine != "auto":
        return _oriented(engine_coordinates(sub, engine))
    best, best_defects = None, None
    for candidate in ("naview", "simple", "puzzler"):
        p = _oriented(engine_coordinates(sub, candidate))
        defects = _defects(p, sub, min_dist, sides)
        if defects == (0, 0, -min_dist):
            return p
        if best_defects is None or defects < best_defects:
            best, best_defects = p, defects
    assert best is not None
    return best


def flat_coordinates(
    structure: str,
    engine: Engine = "auto",
    spacing: float = 1.0,
    gap: float = 2.6,
    min_dist: float = 0.6,
) -> np.ndarray:
    """Return 2D coordinates with the exterior loop on y = 0 and helices above it.

    Example:
        >>> xy = flat_coordinates("..((((...))))..")
        >>> xy.shape
        (15, 2)
        >>> bool((xy[[0, 1, 13, 14], 1] == 0).all())
        True

    Args:
        structure: Nested dot-bracket string.
        engine: Layout of each helix subtree. `auto` (default) uses naview and falls back,
            per subtree, to the radial layout and then to RNApuzzler while the subtree has
            two nucleotides closer than `min_dist`, crossing segments, or nucleotides under
            the backbone. A fixed engine is used as is.
        spacing: Length of one backbone step, in plot units.
        gap: Extra space after each helix subtree, in backbone steps.
        min_dist: Collision threshold of the `auto` fallback, in backbone steps.

    Returns:
        Float array of shape (len(structure), 2).
    """
    table = pair_table(structure)
    n = len(structure)
    xy = np.zeros((n, 2))
    x, i = 0.0, 0
    while i < n:
        j = table[i]
        if j < 0:  # unpaired exterior nucleotide: on the backbone
            xy[i] = (x, 0.0)
            x += spacing
            i += 1
            continue

        sides = (i > 0, j < n - 1)  # exterior backbone segments next to the subtree
        p = _subtree_coordinates(structure[i : j + 1], engine, min_dist, sides) * spacing
        p[:, 0] += x - min(p[:, 0].min(), 0.0)  # 5' base at the cursor, subtree right of it
        xy[i : j + 1] = p
        x = p[:, 0].max() + gap * spacing  # advance past its full width
        i = j + 1
    return xy


def segment_crossings(xy: np.ndarray, structure: str) -> int:
    """Count crossings between drawn segments (backbone steps and base-pair lines).

    Segments that share a nucleotide are not counted. Useful to check a layout.

    Args:
        xy: Coordinates from `flat_coordinates`.
        structure: The dot-bracket string the coordinates were computed for.

    Returns:
        Number of crossing segment pairs.
    """
    segs = np.array([(k, k + 1) for k in range(len(xy) - 1)] + pair_list(structure))
    if len(segs) < 2:
        return 0
    a, b = xy[segs[:, 0]], xy[segs[:, 1]]

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        cross_z = (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )
        return np.asarray(np.sign(cross_z))

    p, q = a[:, None], b[:, None]
    r, s = a[None], b[None]
    cross = (orient(p, q, r) * orient(p, q, s) < 0) & (orient(r, s, p) * orient(r, s, q) < 0)
    shared = (segs[:, None, :, None] == segs[None, :, None, :]).any(axis=(2, 3))
    return int(np.triu(cross & ~shared, 1).sum())
