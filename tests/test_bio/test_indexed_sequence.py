"""Tests for pylbsr.bio.indexed_sequence."""

from pathlib import Path
from unittest.mock import MagicMock

import pyfaidx
import pytest

from pylbsr.bio.indexed_sequence import (
    PyfaidxIndexedSequence,
    TwoBitIndexedSequence,
    fetch_interval_sequence,
)

# ---------------------------------------------------------------------------
# fetch_interval_sequence — pure logic, tested against a minimal fake backend
# ---------------------------------------------------------------------------


class _FakeSource:
    """Minimal IndexedGenomeSequence: chrom1 is "ACGTACGTAC..." repeated, len 1000."""

    def __init__(self) -> None:
        self._seq = "ACGTACGTAC" * 100

    @property
    def chromsizes(self) -> dict[str, int]:
        return {"chr1": len(self._seq)}

    def fetch(self, chrom: str, start: int, end: int) -> str:
        return self._seq[start:end]


class TestFetchIntervalSequence:
    def test_exact_interval_forward(self) -> None:
        source = _FakeSource()
        result = fetch_interval_sequence("chr1", 10, 20, "+", source)
        assert result == source._seq[10:20]
        assert len(result) == 10

    def test_reverse_complement(self) -> None:
        source = _FakeSource()
        fwd = fetch_interval_sequence("chr1", 10, 20, "+", source)
        rev = fetch_interval_sequence("chr1", 10, 20, "-", source)
        from Bio.Seq import Seq

        assert rev == str(Seq(fwd).reverse_complement())

    def test_left_edge_padding(self) -> None:
        source = _FakeSource()
        result = fetch_interval_sequence("chr1", -5, 5, "+", source, fill_char="N")
        assert result[:5] == "N" * 5
        assert result[5:] == source._seq[0:5]
        assert len(result) == 10

    def test_right_edge_padding(self) -> None:
        source = _FakeSource()
        chrom_len = len(source._seq)
        result = fetch_interval_sequence(
            "chr1", chrom_len - 5, chrom_len + 5, "+", source, fill_char="N"
        )
        assert result[:5] == source._seq[-5:]
        assert result[5:] == "N" * 5
        assert len(result) == 10

    def test_entirely_outside_chromosome(self) -> None:
        source = _FakeSource()
        chrom_len = len(source._seq)
        result = fetch_interval_sequence(
            "chr1", chrom_len + 100, chrom_len + 110, "+", source, fill_char="N"
        )
        assert result == "N" * 10

    def test_default_fill_char_is_n(self) -> None:
        source = _FakeSource()
        result = fetch_interval_sequence("chr1", -3, 0, "+", source)
        assert result == "NNN"

    def test_unknown_chrom_raises(self) -> None:
        source = _FakeSource()
        with pytest.raises(ValueError, match="not found"):
            fetch_interval_sequence("chrX", 0, 10, "+", source)

    def test_bad_interval_raises(self) -> None:
        source = _FakeSource()
        with pytest.raises(ValueError, match="start < end"):
            fetch_interval_sequence("chr1", 10, 10, "+", source)

    def test_works_with_magicmock_source(self) -> None:
        """Same logic against a MagicMock, matching the repo's existing test convention."""
        source = MagicMock()
        source.chromsizes = {"chr1": 1000}
        source.fetch.side_effect = lambda chrom, start, end: "A" * (end - start)
        result = fetch_interval_sequence("chr1", 100, 110, "+", source, fill_char="N")
        assert result == "A" * 10


# ---------------------------------------------------------------------------
# PyfaidxIndexedSequence — real file I/O (deliberate departure from mock-only
# convention: this class's entire job is wrapping real pyfaidx access, so a
# mock would only assert "the mock returns what I told it to").
# ---------------------------------------------------------------------------


class TestPyfaidxIndexedSequence:
    @pytest.fixture
    def fasta_path(self, tmp_path: Path) -> Path:
        fasta_file = tmp_path / "test.fa"
        fasta_file.write_text(">chr1\nACGTACGTAC\n>chr2\nGGGGCCCCGG\n")
        return fasta_file

    def test_chromsizes(self, fasta_path: Path) -> None:
        source = PyfaidxIndexedSequence.from_path(str(fasta_path))
        try:
            assert source.chromsizes == {"chr1": 10, "chr2": 10}
        finally:
            source.close()

    def test_fetch(self, fasta_path: Path) -> None:
        source = PyfaidxIndexedSequence.from_path(str(fasta_path))
        try:
            assert source.fetch("chr1", 0, 4) == "ACGT"
            assert source.fetch("chr2", 4, 10) == "CCCCGG"
        finally:
            source.close()

    def test_from_already_open_fasta(self, fasta_path: Path) -> None:
        fasta = pyfaidx.Fasta(str(fasta_path))
        source = PyfaidxIndexedSequence(fasta)
        assert source.chromsizes == {"chr1": 10, "chr2": 10}
        source.close()

    def test_integration_with_fetch_interval_sequence(self, fasta_path: Path) -> None:
        source = PyfaidxIndexedSequence.from_path(str(fasta_path))
        try:
            result = fetch_interval_sequence("chr1", -2, 4, "+", source, fill_char="N")
            assert result == "NNACGT"
        finally:
            source.close()


# ---------------------------------------------------------------------------
# TwoBitIndexedSequence — py2bit.open is mocked (no faToTwoBit tool available
# to generate a real .2bit file; only verifies the wrapper's API calls).
# ---------------------------------------------------------------------------


class TestTwoBitIndexedSequence:
    def test_chromsizes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_tb = MagicMock()
        mock_tb.chroms.return_value = {"chr1": 500}
        monkeypatch.setattr("pylbsr.bio.indexed_sequence.py2bit.open", lambda path: mock_tb)

        source = TwoBitIndexedSequence("genome.2bit")
        assert source.chromsizes == {"chr1": 500}
        mock_tb.chroms.assert_called_once()

    def test_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_tb = MagicMock()
        mock_tb.sequence.return_value = "ACGT"
        monkeypatch.setattr("pylbsr.bio.indexed_sequence.py2bit.open", lambda path: mock_tb)

        source = TwoBitIndexedSequence("genome.2bit")
        result = source.fetch("chr1", 10, 14)
        assert result == "ACGT"
        mock_tb.sequence.assert_called_once_with("chr1", 10, 14)

    def test_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_tb = MagicMock()
        monkeypatch.setattr("pylbsr.bio.indexed_sequence.py2bit.open", lambda path: mock_tb)

        source = TwoBitIndexedSequence("genome.2bit")
        source.close()
        mock_tb.close.assert_called_once()

    def test_integration_with_fetch_interval_sequence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_tb = MagicMock()
        mock_tb.chroms.return_value = {"chr1": 20}
        mock_tb.sequence.return_value = "ACGTACGTAC"
        monkeypatch.setattr("pylbsr.bio.indexed_sequence.py2bit.open", lambda path: mock_tb)

        source = TwoBitIndexedSequence("genome.2bit")
        result = fetch_interval_sequence("chr1", 5, 15, "+", source)
        assert result == "ACGTACGTAC"
        mock_tb.sequence.assert_called_once_with("chr1", 5, 15)
