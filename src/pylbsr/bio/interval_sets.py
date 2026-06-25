"""Set comparison utilities for named collections of genomic intervals (BedTool objects)."""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools as pbt
import seaborn as sns
import upsetplot

# ── helpers ───────────────────────────────────────────────────────────────────


def _validate_stranded(named_bedtools: dict[str, pbt.BedTool]) -> None:
    """Raise ValueError if any BedTool lacks a strand column (< 6 fields).

    Checks only the first non-empty interval of each BedTool.

    Args:
        named_bedtools: Mapping of set name to BedTool to validate.

    Raises:
        ValueError: If a BedTool has fewer than 6 fields.
    """
    for key, bt in named_bedtools.items():
        for interval in bt:
            if len(interval.fields) < 6:
                raise ValueError(
                    f"stranded=True requires >=6-column BED input (BED6); "
                    f"set '{key}' has {len(interval.fields)} field(s). "
                    f"Pass stranded=False for unstranded BED3/BED4 input."
                )
            break  # only inspect the first interval


# ── standalone functions ──────────────────────────────────────────────────────


def build_membership(
    named_bedtools: dict[str, pbt.BedTool],
    frac: float,
    stranded: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build per-region set-membership tables with reciprocal-overlap filtering.

    For each named BedTool, returns a DataFrame where each row is one region and
    boolean columns indicate whether that region overlaps each other set at ``>=frac``
    reciprocal coverage (bedtools ``-f frac -F frac -u``).

    Args:
        named_bedtools: Mapping of set name to BedTool. Key order is preserved.
        frac: Minimum reciprocal overlap fraction (0–1). Both the query region and
            the matching subject region must be covered by at least this fraction.
        stranded: If ``True``, only same-strand overlaps are counted (bedtools
            ``-s`` flag). All BedTools must have at least 6 columns (BED6) with a
            valid strand field (``+``, ``-``, or ``.``).

    Returns:
        A dict keyed by set name. Each value is a DataFrame with columns
        ``chrom``, ``start``, ``end``, and one boolean column per set name.
        When ``stranded=True``, a ``strand`` column is also included between
        ``end`` and the indicator columns. The source-set column is always
        ``True``.

    Raises:
        ValueError: If ``stranded=True`` and any BedTool has fewer than 6 fields.
    """
    if stranded:
        _validate_stranded(named_bedtools)

    intersect_kw: dict[str, Any] = {"u": True, "f": frac, "F": frac}
    if stranded:
        intersect_kw["s"] = True

    result: dict[str, pd.DataFrame] = {}
    for query_key, bt_query in named_bedtools.items():
        df_raw = bt_query.to_dataframe()

        if stranded:
            # Extract chrom, start, end, strand (columns 0,1,2,5) and add integer name.
            df = df_raw.iloc[:, [0, 1, 2, 5]].copy()
            df.columns = pd.Index(["chrom", "start", "end", "strand"])
            df.insert(3, "name", range(len(df)))
            # BED6 layout required for -s: chrom start end name score strand
            df_bed = pd.DataFrame({
                "chrom": df["chrom"],
                "start": df["start"],
                "end": df["end"],
                "name": df["name"],
                "score": 0,
                "strand": df["strand"],
            })
            out_names = ["chrom", "start", "end", "name", "score", "strand"]
        else:
            # BED3 → BED4: chrom start end name
            df = df_raw.iloc[:, :3].copy()
            df.columns = pd.Index(["chrom", "start", "end"])
            df.insert(3, "name", range(len(df)))
            df_bed = df
            out_names = ["chrom", "start", "end", "name"]

        bt_named = pbt.BedTool.from_dataframe(df_bed)

        df_result = df.copy()
        df_result[query_key] = True

        for subject_key, bt_subject in named_bedtools.items():
            if subject_key == query_key:
                continue
            overlapping: set[Any] = set(
                bt_named.intersect(bt_subject, **intersect_kw)
                .to_dataframe(names=out_names)["name"]
            )
            df_result[subject_key] = df_result["name"].isin(overlapping)

        result[query_key] = df_result.drop(columns=["name"])

    return result


def pairwise_intersect_counts(
    named_bedtools: dict[str, pbt.BedTool],
    frac: float,
    stranded: bool = False,
) -> tuple[dict[str, int], np.ndarray]:
    """Compute pairwise reciprocal-overlap counts between named BedTool sets.

    Args:
        named_bedtools: Mapping of set name to BedTool. Key order determines
            the row/column order of the returned matrix.
        frac: Minimum reciprocal overlap fraction (0–1).
        stranded: If ``True``, only same-strand overlaps are counted (bedtools
            ``-s`` flag). All BedTools must have at least 6 columns (BED6).

    Returns:
        A tuple ``(counts, inter)`` where ``counts[name]`` is the total number of
        regions in that set and ``inter[i, j]`` is the count of set-i regions that
        overlap at least one set-j region at ``>=frac`` reciprocal coverage.
        The diagonal satisfies ``inter[i, i] == counts[name_i]``.

    Raises:
        ValueError: If ``stranded=True`` and any BedTool has fewer than 6 fields.
    """
    if stranded:
        _validate_stranded(named_bedtools)

    set_names = list(named_bedtools.keys())
    n = len(set_names)
    counts = {name: bt.count() for name, bt in named_bedtools.items()}
    inter = np.zeros((n, n), dtype=int)

    intersect_kw: dict[str, Any] = {"u": True, "f": frac, "F": frac}
    if stranded:
        intersect_kw["s"] = True

    for i, name_i in enumerate(set_names):
        for j, name_j in enumerate(set_names):
            if i == j:
                inter[i, j] = counts[name_i]
            else:
                inter[i, j] = (
                    named_bedtools[name_i]
                    .intersect(named_bedtools[name_j], **intersect_kw)
                    .count()
                )
    return counts, inter


# ── class ─────────────────────────────────────────────────────────────────────


class BedSetComparison:
    """Compare named collections of genomic intervals with pairwise reciprocal-overlap analysis.

    Background — why interval set comparison is non-trivial
    --------------------------------------------------------
    With **exact sets** (gene IDs, transcript names, categories) every element has a
    unique identity and ``upsetplot.from_indicators`` works trivially.

    With **genomic intervals**, the "same biological event" may be represented by
    slightly different coordinates across runs (different threshold estimation, window
    merging, etc.).  There is no canonical identity — it must be constructed by
    choosing a definition of "same region":

    - **Sub-interval** (``bedtools multiinter``): the genome is split at every set
      boundary; each resulting fragment is labelled with the sets that cover it.
      This is the position-as-identity view.  ``bedtools multiinter`` does not support
      ``-f``/``-F`` fraction thresholds.  An additional complication is that a single
      original region from set A overlapping a region from set B may be fragmented
      into several sub-intervals, none of which individually satisfies a fraction
      threshold even if the total aggregate overlap would; correctly post-processing
      this requires re-merging fragments back to original region pairs — which is
      equivalent to running ``bedtools intersect`` directly.
    - **Per-region membership** (``build_membership``): for each region in a query
      set, ``bedtools intersect -f frac -F frac -u`` is used to determine which
      subject regions overlap it at the specified reciprocal fraction.  Fractions are
      computed at the original-region level (not sub-interval level), so the
      multi-fragment case above is handled correctly with no extra post-processing.
    - **Anchor view** (``plot_upset(anchor=...)``: the anchor set's regions are the
      rows; columns show membership in each other set.  Answers "which anchor-set
      regions are recovered elsewhere?" but other-set-only regions are invisible.
    - **Non-redundant catalog** (``plot_upset(anchor=None)``): each region is
      attributed to the highest-priority set (first key in ``named_bedtools``) that
      claims it; lower-priority sets contribute only their novel regions.  All
      2ᴺ−1 combination bins are visible and each region appears exactly once.

    All expensive bedtools intersections are deferred until first property access and
    cached.

    Example:
        >>> # Unstranded (BED3/BED4), complete upset view
        >>> cmp = BedSetComparison(
        ...     named_bedtools={"full": bt_full, "rep1": bt_rep1, "rep2": bt_rep2},
        ...     frac=0.66,
        ... )
        >>> fig, axes = cmp.plot_upset(label="differential")          # complete view
        >>> fig, axes = cmp.plot_upset(anchor="full", label="diff")   # anchored view
        >>> fig, ax   = cmp.plot_heatmap(label="differential")
        >>> df        = cmp.cross_check(anchor="full")
        >>> # Stranded (BED6)
        >>> cmp_stranded = BedSetComparison(
        ...     named_bedtools={"full": bt_full, "rep1": bt_rep1},
        ...     frac=0.66,
        ...     stranded=True,
        ... )
    """

    def __init__(
        self,
        named_bedtools: dict[str, pbt.BedTool],
        frac: float,
        stranded: bool = False,
    ) -> None:
        """Initialize the comparison.

        Args:
            named_bedtools: Mapping of set name to BedTool. Key insertion order is
                preserved and used as the row/column order in matrices and plots.
            frac: Minimum reciprocal overlap fraction (0–1). An overlap counts only
                when the intersecting segment covers at least this fraction of both
                the query region (bedtools ``-f``) and the subject region (``-F``).
            stranded: If ``True``, only same-strand overlaps are counted (bedtools
                ``-s`` flag). All BedTools must be BED6 (>=6 columns with a valid
                strand field). Default is ``False`` (unstranded; BED3/BED4/BED6 all
                accepted, strand column ignored).
        """
        self.named_bedtools = named_bedtools
        self.set_names: list[str] = list(named_bedtools.keys())
        self.frac = frac
        self.stranded = stranded
        self._membership: dict[str, pd.DataFrame] | None = None
        self._counts: dict[str, int] | None = None
        self._inter: np.ndarray | None = None

    # ── lazy computed properties ──────────────────────────────────────────────

    @property
    def membership(self) -> dict[str, pd.DataFrame]:
        """Per-region membership tables, computed lazily on first access.

        Returns:
            Dict keyed by set name. Each DataFrame has columns ``chrom``,
            ``start``, ``end``, optionally ``strand`` (when ``stranded=True``),
            and one boolean column per set name. The source-set column is
            always ``True``.
        """
        if self._membership is None:
            self._membership = build_membership(
                self.named_bedtools, self.frac, self.stranded
            )
        return self._membership

    @property
    def _pairwise(self) -> tuple[dict[str, int], np.ndarray]:
        if self._counts is None:
            self._counts, self._inter = pairwise_intersect_counts(
                self.named_bedtools, self.frac, self.stranded
            )
        return self._counts, self._inter  # type: ignore[return-value]

    @property
    def counts(self) -> dict[str, int]:
        """Total region count per set.

        Returns:
            Dict mapping set name to the number of regions in that set.
        """
        return self._pairwise[0]

    @property
    def inter(self) -> np.ndarray:
        """Pairwise reciprocal-overlap count matrix.

        Returns:
            Integer array of shape ``(n_sets, n_sets)`` where ``inter[i, j]`` is
            the count of set-i regions that overlap at least one set-j region at the
            configured fraction threshold. The diagonal equals the set sizes.
        """
        return self._pairwise[1]

    @property
    def frac_matrix(self) -> np.ndarray:
        """Row-normalised fraction matrix.

        Returns:
            Float array of shape ``(n_sets, n_sets)`` where ``frac_matrix[i, j]``
            is the fraction of set-i regions found in set-j. Diagonal is ``1.0``.
        """
        counts, inter = self._pairwise
        row_totals = np.array([counts[k] for k in self.set_names], dtype=float)
        mat = inter.astype(float) / row_totals[:, np.newaxis]
        np.fill_diagonal(mat, 1.0)
        return mat

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pooled_complete(self) -> pd.DataFrame:
        """Build a non-redundant catalog of all regions across all sets.

        Each region is attributed to the **highest-priority set** (the set that
        appears first in ``named_bedtools``) that contains it.  Lower-priority
        sets contribute only their **novel** regions — those with no reciprocal
        overlap with any higher-priority set.

        Concretely, for sets ordered s0, s1, ..., sN-1:

        1. All regions from s0 are included (with membership indicators for s1…sN-1).
        2. Regions from s1 where ``s0=False`` are appended.
        3. Regions from s2 where ``s0=False`` AND ``s1=False`` are appended.
        4. And so on.

        This guarantees that each genomic region appears in exactly one row and
        that all 2^N - 1 non-empty combination bins are representable in the
        resulting upset plot.  No new bedtools calls are made — this is pure
        filtering of the already-computed ``membership`` tables.

        Returns:
            A DataFrame with the same column schema as each ``membership`` value,
            with rows drawn non-redundantly across all sets.
        """
        mem = self.membership
        frames = [mem[self.set_names[0]]]
        for k in range(1, len(self.set_names)):
            higher = self.set_names[:k]
            novel_mask = ~mem[self.set_names[k]][higher].any(axis=1)
            frames.append(mem[self.set_names[k]].loc[novel_mask])
        return pd.concat(frames, ignore_index=True)

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot_upset(
        self,
        anchor: str | None = None,
        label: str = "",
        figsize: tuple[float, float] = (14, 6),
        **upset_kwargs: Any,
    ) -> tuple[plt.Figure, dict]:  # type: ignore[type-arg]
        """Plot an UpSet diagram of set membership across all named sets.

        Two modes are available via ``anchor``:

        - ``anchor=None`` **(default, complete view)**: uses the non-redundant
          catalog (``_pooled_complete``).  All 2^N - 1 combination bins are visible,
          including set-exclusive regions.  Totals bars show the raw size of each
          set.  Regions are attributed to the highest-priority set (key order in
          ``named_bedtools``).
        - ``anchor="<set_name>"`` **(anchored view)**: only rows from that set are
          used.  Answers "which anchor-set regions are recovered in the other sets?"
          Regions exclusive to other sets are not shown.  Totals bars show how many
          anchor-set regions overlap each other set.

        Args:
            anchor: Set name for the anchored view, or ``None`` (default) for the
                complete view.
            label: Optional descriptor inserted into the figure title (e.g.
                ``"differential"`` or ``"unchanged"``).
            figsize: Figure size passed to ``plt.figure``.
            **upset_kwargs: Additional keyword arguments forwarded to
                ``upsetplot.UpSet``. Default values for ``show_counts``,
                ``sort_by``, and ``subset_size`` can be overridden here.

        Returns:
            A tuple ``(fig, axes)`` where ``axes`` is the dict returned by
            ``upsetplot.UpSet.plot()``.
        """
        if anchor is None:
            pooled = self._pooled_complete()
            true_counts = {k: self.counts[k] for k in self.set_names}
        else:
            pooled = self.membership[anchor]
            true_counts = {k: int(pooled[k].sum()) for k in self.set_names}

        r_bool = pooled[self.set_names].astype(bool)
        upset_data = upsetplot.from_indicators(r_bool)

        defaults: dict[str, Any] = {
            "show_counts": True,
            "sort_by": "cardinality",
            "subset_size": "count",
        }
        defaults.update(upset_kwargs)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            fig = plt.figure(figsize=figsize)
            upset_obj = upsetplot.UpSet(upset_data, **defaults)
            axes = upset_obj.plot(fig=fig)

        inter_ax = axes["intersections"]
        for text in inter_ax.texts:
            text.set_fontsize(8)

        totals_ax = axes["totals"]
        bar_labels = [t.get_text() for t in totals_ax.get_yticklabels()]

        for bar, lbl in zip(totals_ax.patches, bar_labels):
            bar.set_width(true_counts.get(lbl, bar.get_width()))

        bar_info: dict[float, tuple[int, float]] = {
            bar.get_y() + bar.get_height() / 2: (
                true_counts.get(lbl, int(bar.get_width())),
                bar.get_width(),
            )
            for bar, lbl in zip(totals_ax.patches, bar_labels)
        }
        for text in totals_ax.texts:
            ty = text.get_position()[1]
            closest_y = min(bar_info, key=lambda cy: abs(cy - ty))
            true_val, bar_width = bar_info[closest_y]
            text.set_text(str(true_val))
            text.set_fontsize(8)
            text.set_x(bar_width)

        if bar_labels:
            max_val = max(true_counts[lbl] for lbl in bar_labels if lbl in true_counts)
            totals_ax.set_xlim(max_val * 1.15, 0)
        totals_ax.set_xlabel("Set size\n(# regions)")

        strand_note = ", stranded" if self.stranded else ""
        title_type = f" – {label}" if label else ""
        if anchor is None:
            anchor_note = "complete"
            subtitle = "all regions attributed to highest-priority set (key order)"
        else:
            anchor_note = f"{anchor}-anchored"
            subtitle = f"bars: of the {anchor}-run regions, how many overlap the other runs"
        fig.suptitle(
            f"Upset{title_type} ({anchor_note}, ≥{self.frac:.0%} reciprocal{strand_note})\n"
            f"{subtitle}",
            y=1.02,
        )
        plt.tight_layout()
        return fig, axes

    def plot_heatmap(
        self,
        label: str = "",
        highlight_top_right: bool = False,
        figsize: tuple[float, float] = (6, 6),
        **heatmap_kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a pairwise intersection fraction heatmap.

        Each cell ``(i, j)`` shows the fraction of the row set found in the column
        set. Diagonal cells show the set size instead of a fraction.

        Args:
            label: Optional descriptor inserted into the figure title.
            highlight_top_right: Whether to highlight the top-right triangle of the heatmap.
            figsize: Figure size passed to ``plt.subplots``.
            **heatmap_kwargs: Additional keyword arguments forwarded to
                ``seaborn.heatmap``. Default values can be overridden here.

        Returns:
            A tuple ``(fig, ax)``.
        """
        counts = self.counts
        frac = self.frac_matrix
        n = len(self.set_names)

        frac_df = pd.DataFrame(frac, index=self.set_names, columns=self.set_names)

        hm_defaults: dict[str, Any] = {
            "annot": True,
            "fmt": ".3f",
            "cmap": "Blues",
            "vmin": 0,
            "vmax": 1,
            "linewidths": 0.5,
            "square": True,
            "cbar_kws": {"shrink": 0.5},
        }
        hm_defaults.update(heatmap_kwargs)

        fig, ax = plt.subplots(figsize=figsize)
        g = sns.heatmap(frac_df, ax=ax, **hm_defaults)

        g.figure.axes[-1].set_title("Fraction", fontsize=12)
        g.figure.axes[-1].add_patch(
            mpl.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", lw=3.5)
        )

        for i in range(n):
            ax.texts[i * n + i].set_text("")
            ax.text(
                i + 0.5,
                i + 0.5,
                f"N={counts[self.set_names[i]]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

        if highlight_top_right:
            for i in range(n):
                for j in range(i + 1, n):
                    ax.add_patch(
                        mpl.patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=3.5)
                    )

        strand_note = ", stranded" if self.stranded else ""
        title_type = f" – {label}" if label else ""
        ax.set_title(
            f"Pairwise intersection fractions{title_type}"
            f" (≥{self.frac:.0%} reciprocal{strand_note})\n"
            "each cell: fraction of the ROW set found in the column set",
            fontsize=12,
        )
        plt.tight_layout()
        return fig, ax

    def cross_check(self, anchor: str) -> pd.DataFrame:
        """Compare upset-derived fractions against heatmap fractions for a given anchor.

        Both code paths compute the same underlying bedtools intersect operation.
        A mismatch indicates that ``membership`` and the pairwise counts were computed
        with different parameters (e.g. due to a stale notebook execution).

        Args:
            anchor: Name of the anchor set. Must be a key in ``named_bedtools``.

        Returns:
            A DataFrame with columns ``other``, ``upset_n``, ``upset_frac``,
            ``heatmap_frac``, and ``match`` (bool). One row per non-anchor set.
        """
        anchor_idx = self.set_names.index(anchor)
        n_anchor = len(self.membership[anchor])
        fm = self.frac_matrix

        rows = []
        for other_key in [k for k in self.set_names if k != anchor]:
            other_idx = self.set_names.index(other_key)
            upset_n = int(self.membership[anchor][other_key].sum())
            upset_frac = upset_n / n_anchor if n_anchor > 0 else 0.0
            hm_frac = float(fm[anchor_idx, other_idx])
            rows.append(
                {
                    "other": other_key,
                    "upset_n": upset_n,
                    "upset_frac": round(upset_frac, 9),
                    "heatmap_frac": round(hm_frac, 9),
                    "match": abs(upset_frac - hm_frac) < 1e-9,
                }
            )

        return pd.DataFrame(rows)
