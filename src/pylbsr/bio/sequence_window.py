"""Windowed sequence extraction from genomic intervals.

Provides utilities to extract a fill-character-padded sequence of exactly
length W centred on a query BED interval, with optional masking against
a set of restraint intervals.
"""

from typing import Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pybedtools as pbt
import pyfaidx
from Bio.Seq import Seq
from pandera.typing.pandas import DataFrame

from .span_masks import coordinates_from_binary_mask, intervals_to_span_masks


class WindowSegmentsModel(pa.DataFrameModel):
    """Pandera schema for the output of decompose_query_window.

    Each row describes a contiguous sub-region of the W-length window,
    ordered left-to-right by window_start.
    """

    window_start: pa.typing.Int64
    window_end: pa.typing.Int64
    genomic_chrom: pa.typing.String | None
    genomic_start: pa.typing.Int64 | None
    genomic_end: pa.typing.Int64 | None
    is_padding: pa.typing.Bool
    is_masked: pa.typing.Bool
    reverse_complement: pa.typing.Bool

    class Config:
        """Pandera model configuration."""

        coerce = True

    @pa.dataframe_check
    @classmethod
    def check_window_end_gt_start(cls, df: pd.DataFrame) -> pd.Series:
        """window_end must be strictly greater than window_start."""
        return df["window_end"] > df["window_start"]

    @pa.check("window_start")
    @classmethod
    def check_window_start_non_negative(
        cls, series: pa.typing.Series[pa.typing.Int64]
    ) -> pa.typing.Series[pa.typing.Int64]:
        """window_start must be non-negative."""
        return series[series >= 0]

    @pa.check("genomic_start")
    @classmethod
    def check_genomic_start_non_negative(
        cls, series: pa.typing.Series[pa.typing.Int64]
    ) -> pa.typing.Series[pa.typing.Int64]:
        """genomic_start must be non-negative when not null."""
        return series[series >= 0]


def decompose_query_window(
    chrom: str,
    start: int,
    end: int,
    strand: Literal["+", "-"],
    chrom_sizes: dict[str, int],
    window_size: int,
    restraint_intervals: pbt.BedTool | None = None,
    restraint_mode: Literal["keep", "exclude"] = "keep",
) -> DataFrame[WindowSegmentsModel]:
    """Decompose a query BED interval into an ordered table of window segments.

    The window of length `window_size` is centred on the midpoint of [start, end).
    Each row of the returned DataFrame describes a contiguous sub-region, ordered
    left-to-right within the window.

    Args:
        chrom: Chromosome name.
        start: 0-based interval start (inclusive).
        end: 0-based interval end (exclusive); must be > start.
        strand: Interval strand, "+" or "-".
        chrom_sizes: Mapping of chromosome name to its total length.
        window_size: Exact length W of the output window.
        restraint_intervals: Optional BedTool of constraint intervals.
        restraint_mode: "keep" → only restraint-covered positions are valid;
            "exclude" → restraint-covered positions are masked.

    Returns:
        DataFrame with columns window_start, window_end, genomic_chrom,
        genomic_start, genomic_end, is_padding, is_masked, reverse_complement.
        Rows are ordered by window_start and together span [0, window_size).

    Raises:
        ValueError: On invalid inputs (unknown chrom, bad window_size, etc.).
    """
    if chrom not in chrom_sizes:
        raise ValueError(f"Chromosome '{chrom}' not found in chrom_sizes.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if start < 0 or start >= end:
        raise ValueError(f"Require 0 <= start < end, got start={start}, end={end}.")

    center = (start + end) // 2
    win_start = center - window_size // 2
    win_end = win_start + window_size

    chrom_len = chrom_sizes[chrom]
    left_pad = max(0, -win_start)
    right_pad = max(0, win_end - chrom_len)
    gen_start = max(0, win_start)
    gen_end = min(chrom_len, win_end)

    rc = strand == "-"
    rows: list[dict] = []

    def _pad_row(ws: int, we: int) -> dict:
        return {
            "window_start": ws,
            "window_end": we,
            "genomic_chrom": None,
            "genomic_start": None,
            "genomic_end": None,
            "is_padding": True,
            "is_masked": False,
            "reverse_complement": rc,
        }

    def _genomic_row(ws: int, we: int, gs: int, ge: int, masked: bool) -> dict:
        return {
            "window_start": ws,
            "window_end": we,
            "genomic_chrom": chrom,
            "genomic_start": gs,
            "genomic_end": ge,
            "is_padding": False,
            "is_masked": masked,
            "reverse_complement": rc,
        }

    # Entire window outside chromosome → all padding
    if gen_start >= gen_end:
        rows.append(_pad_row(0, window_size))
        return pd.DataFrame(rows)

    # Left chromosome padding
    if left_pad > 0:
        rows.append(_pad_row(0, left_pad))

    # Genomic region — determine valid sub-intervals via pybedtools
    gen_len = gen_end - gen_start

    if restraint_intervals is None:
        valid_segs = np.array([[0, gen_len]], dtype=np.int64)
    else:
        window_bt = pbt.BedTool(f"{chrom}\t{gen_start}\t{gen_end}", from_string=True)
        if restraint_mode == "keep":
            result_bt = window_bt.intersect(restraint_intervals)
        else:  # "exclude"
            result_bt = window_bt.subtract(restraint_intervals)

        valid_list = [[iv.start - gen_start, iv.end - gen_start] for iv in result_bt]
        if valid_list:
            starts_v = np.array([s for s, _ in valid_list], dtype=np.int64)
            ends_v = np.array([e for _, e in valid_list], dtype=np.int64)
            # Merge overlapping/adjacent intervals via binary mask
            valid_mask = intervals_to_span_masks(starts_v, ends_v, gen_len).any(axis=0).astype(np.int64)
            valid_segs = coordinates_from_binary_mask(valid_mask)
        else:
            valid_segs = np.empty((0, 2), dtype=np.int64)

    # Build masked/valid segment rows
    prev = 0
    for seg_start, seg_end in valid_segs:
        seg_start = int(seg_start)
        seg_end = int(seg_end)
        if seg_start > prev:
            # Gap before this valid segment → masked
            gs, ge = gen_start + prev, gen_start + seg_start
            ws, we = gs - win_start, ge - win_start
            rows.append(_genomic_row(ws, we, gs, ge, masked=True))
        gs, ge = gen_start + seg_start, gen_start + seg_end
        ws, we = gs - win_start, ge - win_start
        rows.append(_genomic_row(ws, we, gs, ge, masked=False))
        prev = seg_end

    # Trailing masked segment after last valid segment
    if prev < gen_len:
        gs, ge = gen_start + prev, gen_end
        ws, we = gs - win_start, ge - win_start
        rows.append(_genomic_row(ws, we, gs, ge, masked=True))

    # Right chromosome padding
    if right_pad > 0:
        rows.append(_pad_row(window_size - right_pad, window_size))

    return pd.DataFrame(rows)


def fetch_windowed_sequence(
    chrom: str,
    start: int,
    end: int,
    strand: Literal["+", "-"],
    chrom_sizes: dict[str, int],
    window_size: int,
    fasta: pyfaidx.Fasta,
    fill_char: str,
    restraint_intervals: pbt.BedTool | None = None,
    restraint_mode: Literal["keep", "exclude"] = "keep",
) -> str:
    """Fetch a fill-character-padded sequence of exact length W for a BED interval.

    The window is centred on the midpoint of [start, end). Positions outside the
    chromosome are padded with `fill_char`. Positions masked by `restraint_intervals`
    are also replaced by `fill_char`. For strand="-", the full result is
    reverse-complemented using `Bio.Seq.reverse_complement()`.

    Note:
        When strand="-", `fill_char` must be recognised by BioPython's IUPAC complement
        table (e.g. "N" is safe; arbitrary characters will raise a TranslationError).

    Args:
        chrom: Chromosome name.
        start: 0-based interval start (inclusive).
        end: 0-based interval end (exclusive).
        strand: Interval strand, "+" or "-".
        chrom_sizes: Mapping of chromosome name to its total length.
        window_size: Exact output sequence length W.
        fasta: Open pyfaidx.Fasta object backed by an indexed FASTA file.
        fill_char: Single character used for padding and masking.
        restraint_intervals: Optional BedTool of constraint intervals.
        restraint_mode: "keep" or "exclude" (see decompose_query_window).

    Returns:
        Nucleotide string of length exactly window_size.

    Raises:
        ValueError: If the assembled sequence length does not equal window_size.
    """
    df = decompose_query_window(
        chrom=chrom,
        start=start,
        end=end,
        strand=strand,
        chrom_sizes=chrom_sizes,
        window_size=window_size,
        restraint_intervals=restraint_intervals,
        restraint_mode=restraint_mode,
    )

    parts: list[str] = []
    for _, row in df.iterrows():
        seg_len = int(row["window_end"]) - int(row["window_start"])
        if row["is_padding"] or row["is_masked"]:
            parts.append(fill_char * seg_len)
        else:
            parts.append(str(fasta[row["genomic_chrom"]][int(row["genomic_start"]):int(row["genomic_end"])]))

    result = "".join(parts)

    if df["reverse_complement"].iloc[0]:
        result = str(Seq(result).reverse_complement())

    if len(result) != window_size:
        raise ValueError(
            f"Assembled sequence length {len(result)} does not match window_size {window_size}."
        )

    return result
