"""Arbitrary-interval genome sequence fetching, backed by FASTA or .2bit files.

Unlike `pylbsr.bio.sequence_window`, which fetches a fixed-width window centred on
a query interval, `fetch_interval_sequence` fetches an interval exactly as given —
no forced width, no centering. Coordinates are 0-based, half-open (BED-style),
matching the rest of `pylbsr`.
"""

from collections.abc import Mapping
from typing import Literal, Protocol

import py2bit
import pyfaidx
from Bio.Seq import Seq


class IndexedGenomeSequence(Protocol):
    """Common interface for point-querying a genome sequence by 0-based half-open interval."""

    @property
    def chromsizes(self) -> Mapping[str, int]:
        """Mapping of chromosome name to its total length."""
        ...

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Return raw forward-strand sequence for [start, end); no padding/strand handling."""
        ...


class PyfaidxIndexedSequence:
    """FASTA-backed IndexedGenomeSequence, wrapping an already-open pyfaidx.Fasta."""

    def __init__(self, fasta: pyfaidx.Fasta) -> None:
        """Wrap an already-open pyfaidx.Fasta, precomputing chromsizes."""
        self._fasta = fasta
        self._chromsizes = {name: len(record) for name, record in fasta.items()}

    @classmethod
    def from_path(cls, fasta_path: str) -> "PyfaidxIndexedSequence":
        """Open `fasta_path` (with its .fai index) and wrap it."""
        return cls(pyfaidx.Fasta(str(fasta_path)))

    @property
    def chromsizes(self) -> Mapping[str, int]:
        """Mapping of chromosome name to its total length."""
        return self._chromsizes

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Return raw forward-strand sequence for [start, end)."""
        return str(self._fasta[chrom][start:end])

    def close(self) -> None:
        """Close the underlying pyfaidx.Fasta handle."""
        self._fasta.close()


class TwoBitIndexedSequence:
    """.2bit-backed IndexedGenomeSequence, wrapping py2bit (already 0-based half-open)."""

    def __init__(self, twobit_path: str) -> None:
        """Open the .2bit file at `twobit_path`."""
        self._tb = py2bit.open(str(twobit_path))

    @property
    def chromsizes(self) -> Mapping[str, int]:
        """Mapping of chromosome name to its total length."""
        chromsizes: Mapping[str, int] = self._tb.chroms()
        return chromsizes

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Return raw forward-strand sequence for [start, end)."""
        sequence: str = self._tb.sequence(chrom, start, end)
        return sequence

    def close(self) -> None:
        """Close the underlying py2bit handle."""
        self._tb.close()


def fetch_interval_sequence(
    chrom: str,
    start: int,
    end: int,
    strand: Literal["+", "-"],
    source: IndexedGenomeSequence,
    fill_char: str = "N",
) -> str:
    """Fetch an arbitrary [start, end) interval (0-based, half-open), any backend.

    Reverse-complemented if strand=="-". Portions outside chromosome bounds are
    padded with `fill_char`. Unlike `sequence_window.fetch_windowed_sequence`, the
    interval is used exactly as given — no forced fixed width, no centering.

    Args:
        chrom: Chromosome name.
        start: 0-based interval start (inclusive); may be negative.
        end: 0-based interval end (exclusive); must be > start.
        strand: Interval strand, "+" or "-".
        source: An IndexedGenomeSequence backend (PyfaidxIndexedSequence or
            TwoBitIndexedSequence).
        fill_char: Single character used to pad any portion of the interval that
            falls outside the chromosome.

    Returns:
        Nucleotide string of length exactly (end - start).

    Raises:
        ValueError: If end <= start, or chrom is not in source.chromsizes.
    """
    if end <= start:
        raise ValueError(f"Require start < end, got start={start}, end={end}.")
    if chrom not in source.chromsizes:
        raise ValueError(f"Chromosome '{chrom}' not found in source.chromsizes.")

    chrom_len = source.chromsizes[chrom]
    left_pad = max(0, -start)
    right_pad = max(0, end - chrom_len)
    gen_start, gen_end = max(0, start), min(chrom_len, end)

    if gen_start >= gen_end:
        seq = fill_char * (end - start)
    else:
        seq = fill_char * left_pad + source.fetch(chrom, gen_start, gen_end) + fill_char * right_pad

    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
    return seq
