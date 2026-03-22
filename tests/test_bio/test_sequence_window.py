"""Tests for pylbsr.bio.sequence_window."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from pylbsr.bio.sequence_window import decompose_query_window, fetch_windowed_sequence


CHROM_SIZES = {"chr1": 1000}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window_cols() -> list[str]:
    return [
        "window_start",
        "window_end",
        "genomic_chrom",
        "genomic_start",
        "genomic_end",
        "is_padding",
        "is_masked",
        "reverse_complement",
    ]


def _assert_covers_window(df: pd.DataFrame, window_size: int) -> None:
    """Assert that segment rows are contiguous and cover exactly [0, window_size)."""
    assert list(df.columns) == _window_cols()
    assert df["window_start"].iloc[0] == 0
    assert df["window_end"].iloc[-1] == window_size
    for i in range(len(df) - 1):
        assert df["window_end"].iloc[i] == df["window_start"].iloc[i + 1]


# ---------------------------------------------------------------------------
# decompose_query_window
# ---------------------------------------------------------------------------


class TestDecomposeQueryWindow:
    def test_basic_centering_no_padding(self) -> None:
        """Query exactly fills the window → one valid segment, no padding."""
        df = decompose_query_window("chr1", 100, 200, "+", CHROM_SIZES, window_size=100)
        _assert_covers_window(df, 100)
        assert len(df) == 1
        row = df.iloc[0]
        assert not row["is_padding"]
        assert not row["is_masked"]
        assert row["genomic_start"] == 100
        assert row["genomic_end"] == 200
        assert not row["reverse_complement"]

    def test_left_chromosome_clip(self) -> None:
        """Window extends before position 0 → left padding rows."""
        # center = 15, win_start = 15 - 50 = -35, win_end = 65
        df = decompose_query_window("chr1", 10, 20, "+", CHROM_SIZES, window_size=100)
        _assert_covers_window(df, 100)
        pad = df[df["is_padding"]]
        assert len(pad) == 1
        assert pad.iloc[0]["window_start"] == 0
        assert pad.iloc[0]["window_end"] == 35  # left_pad = 35

    def test_right_chromosome_clip(self) -> None:
        """Window extends past chromosome end → right padding."""
        # center = 995, win_start = 945, win_end = 1045
        df = decompose_query_window("chr1", 990, 1000, "+", CHROM_SIZES, window_size=100)
        _assert_covers_window(df, 100)
        pad = df[df["is_padding"]]
        assert len(pad) == 1
        assert pad.iloc[0]["window_end"] == 100

    def test_entirely_outside_chromosome(self) -> None:
        """Query centred way past chromosome end → all padding."""
        df = decompose_query_window("chr1", 900, 910, "+", {"chr1": 100}, window_size=100)
        assert len(df) == 1
        assert df.iloc[0]["is_padding"]
        assert df.iloc[0]["window_start"] == 0
        assert df.iloc[0]["window_end"] == 100

    def test_reverse_complement_flag(self) -> None:
        df = decompose_query_window("chr1", 100, 200, "-", CHROM_SIZES, window_size=100)
        assert df["reverse_complement"].all()

    def test_forward_strand_no_rc(self) -> None:
        df = decompose_query_window("chr1", 100, 200, "+", CHROM_SIZES, window_size=100)
        assert not df["reverse_complement"].any()

    def test_unknown_chrom_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            decompose_query_window("chrX", 0, 10, "+", CHROM_SIZES, window_size=50)

    def test_bad_window_size_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            decompose_query_window("chr1", 0, 10, "+", CHROM_SIZES, window_size=0)

    def test_bad_coordinates_raises(self) -> None:
        with pytest.raises(ValueError, match="start < end"):
            decompose_query_window("chr1", 50, 50, "+", CHROM_SIZES, window_size=100)

    def test_keep_restraints(self) -> None:
        """Keep mode: only restraint-covered region is valid; gaps are masked."""
        import pybedtools as pbt

        restraints = pbt.BedTool("chr1\t120\t180", from_string=True)
        df = decompose_query_window(
            "chr1", 100, 200, "+", CHROM_SIZES, window_size=100,
            restraint_intervals=restraints, restraint_mode="keep",
        )
        _assert_covers_window(df, 100)
        valid = df[~df["is_padding"] & ~df["is_masked"]]
        assert len(valid) == 1
        assert valid.iloc[0]["genomic_start"] == 120
        assert valid.iloc[0]["genomic_end"] == 180
        masked = df[df["is_masked"]]
        assert len(masked) == 2  # [100,120) and [180,200)

    def test_exclude_restraints(self) -> None:
        """Exclude mode: restraint-covered region is masked; rest is valid."""
        import pybedtools as pbt

        restraints = pbt.BedTool("chr1\t120\t180", from_string=True)
        df = decompose_query_window(
            "chr1", 100, 200, "+", CHROM_SIZES, window_size=100,
            restraint_intervals=restraints, restraint_mode="exclude",
        )
        _assert_covers_window(df, 100)
        masked = df[df["is_masked"]]
        assert len(masked) == 1
        assert masked.iloc[0]["genomic_start"] == 120
        assert masked.iloc[0]["genomic_end"] == 180

    def test_no_restraints_single_valid_segment(self) -> None:
        df = decompose_query_window("chr1", 400, 600, "+", CHROM_SIZES, window_size=200)
        _assert_covers_window(df, 200)
        assert len(df) == 1
        assert not df.iloc[0]["is_masked"]

    def test_segment_window_coords_sum_to_window_size(self) -> None:
        df = decompose_query_window("chr1", 10, 20, "+", CHROM_SIZES, window_size=100)
        total = (df["window_end"] - df["window_start"]).sum()
        assert total == 100


# ---------------------------------------------------------------------------
# fetch_windowed_sequence
# ---------------------------------------------------------------------------


class TestFetchWindowedSequence:
    def _mock_fasta(self, seq: str) -> MagicMock:
        """Return a pyfaidx.Fasta mock where fasta[chrom][start:end] == seq[start:end]."""
        fasta = MagicMock()

        class _Seq:
            def __init__(self, full: str) -> None:
                self._seq = full

            def __getitem__(self, sl: slice) -> "_Seq":
                return _Seq(self._seq[sl])

            def __str__(self) -> str:
                return self._seq

        fasta.__getitem__ = lambda self_, chrom: _Seq(seq)
        return fasta

    def test_length_equals_window_size(self) -> None:
        fasta = self._mock_fasta("A" * 1000)
        result = fetch_windowed_sequence(
            "chr1", 100, 200, "+", CHROM_SIZES, window_size=100, fasta=fasta, fill_char="N"
        )
        assert len(result) == 100

    def test_fill_char_used_for_padding(self) -> None:
        fasta = self._mock_fasta("A" * 1000)
        # center=15, win_start=-35 → 35 Ns on the left
        result = fetch_windowed_sequence(
            "chr1", 10, 20, "+", CHROM_SIZES, window_size=100, fasta=fasta, fill_char="N"
        )
        assert result[:35] == "N" * 35
        assert len(result) == 100

    def test_fill_char_used_for_masked_regions(self) -> None:
        import pybedtools as pbt

        fasta = self._mock_fasta("A" * 1000)
        restraints = pbt.BedTool("chr1\t120\t180", from_string=True)
        result = fetch_windowed_sequence(
            "chr1", 100, 200, "+", CHROM_SIZES, window_size=100,
            fasta=fasta, fill_char="N",
            restraint_intervals=restraints, restraint_mode="keep",
        )
        # positions [0,20) and [80,100) in the window are masked → 'N'
        assert result[:20] == "N" * 20
        assert result[80:] == "N" * 20
        assert len(result) == 100

    def test_reverse_complement(self) -> None:
        # Build a fasta returning "ACGT" * 250 (1000 chars)
        fasta = self._mock_fasta("ACGT" * 250)
        result_fwd = fetch_windowed_sequence(
            "chr1", 100, 200, "+", CHROM_SIZES, window_size=100, fasta=fasta, fill_char="N"
        )
        result_rev = fetch_windowed_sequence(
            "chr1", 100, 200, "-", CHROM_SIZES, window_size=100, fasta=fasta, fill_char="N"
        )
        from Bio.Seq import Seq
        assert result_rev == str(Seq(result_fwd).reverse_complement())
        assert len(result_rev) == 100

    def test_fill_char_preserved_after_rc(self) -> None:
        """N fill chars survive reverse complement unchanged."""
        fasta = self._mock_fasta("A" * 1000)
        # center=15, left_pad=35 → first 35 chars are 'N', last 65 are 'A'
        # After RC: last 35 chars (were first) become 'N', first 65 become 'T'
        result = fetch_windowed_sequence(
            "chr1", 10, 20, "-", CHROM_SIZES, window_size=100, fasta=fasta, fill_char="N"
        )
        assert result[-35:] == "N" * 35
        assert len(result) == 100

    def test_entirely_padded_window(self) -> None:
        fasta = self._mock_fasta("A" * 100)
        result = fetch_windowed_sequence(
            "chr1", 900, 910, "+", {"chr1": 100}, window_size=100, fasta=fasta, fill_char="N"
        )
        assert result == "N" * 100
