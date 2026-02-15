"""Tests for gff2chain module - coordinate mapping between genomic and transcript coordinates."""

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pylbsr.bio.gff2chain import (
    ChainRecord,
    MappingTable,
    _assign_entity_query_coords,
    _per_entity_processing_to_nool_bed6,
    gff_to_chains,
    write_chains,
)

# ============================================================================
# FIXTURES - Forward Strand Transcript (ENST00000402432.4)
# ============================================================================


@pytest.fixture
def gff_forward_strand() -> pd.DataFrame:
    """GFF table for forward strand transcript ENST00000402432.4."""
    data = [
        {"score":0, "seqid": "chr5", "type": "transcript", "start": 160229506, "end": 160238716, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "exon", "start": 160229506, "end": 160229624, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "CDS", "start": 160229558, "end": 160229624, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "start_codon", "start": 160229558, "end": 160229560, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "exon", "start": 160232098, "end": 160232273, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "CDS", "start": 160232098, "end": 160232273, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "exon", "start": 160234820, "end": 160234909, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "CDS", "start": 160234820, "end": 160234909, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "exon", "start": 160238606, "end": 160238716, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "CDS", "start": 160238606, "end": 160238659, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "stop_codon", "start": 160238657, "end": 160238659, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "five_prime_UTR", "start": 160229506, "end": 160229557, "strand": "+", "transcript_id": "ENST00000402432.4"},
        {"score":0, "seqid": "chr5", "type": "three_prime_UTR", "start": 160238660, "end": 160238716, "strand": "+", "transcript_id": "ENST00000402432.4"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def expected_mapping_table_forward_transcript() -> pd.DataFrame:
    """Expected mapping table for forward strand transcript (after extracting 'transcript' type)."""
    data = [
        {
            "a.chrom": "chr5",
            "a.start": 160229505,  # 0-based
            "a.end": 160238716,
            "a.strand": "+",
            "b.chrom": "ENST00000402432.4",
            "b.start": 0,
            "b.end": 9211,
            "b.strand": "+",
            "block_len": 9211,
        }
    ]
    return pd.DataFrame(data)


@pytest.fixture
def expected_mapping_table_forward_exons() -> pd.DataFrame:
    """Expected mapping table for forward strand exons (after extracting 'exon' type and merging)."""
    # After merging overlapping exons, we expect 4 blocks
    data = [
        {
            "a.chrom": "chr5",
            "a.start": 160229505,  # 0-based
            "a.end": 160229624,
            "a.strand": "+",
            "b.chrom": "ENST00000402432.4",
            "b.start": 0,
            "b.end": 119,
            "b.strand": "+",
            "block_len": 119,
        },
        {
            "a.chrom": "chr5",
            "a.start": 160232097,  # 0-based
            "a.end": 160232273,
            "a.strand": "+",
            "b.chrom": "ENST00000402432.4",
            "b.start": 119,
            "b.end": 295,
            "b.strand": "+",
            "block_len": 176,
        },
        {
            "a.chrom": "chr5",
            "a.start": 160234819,  # 0-based
            "a.end": 160234909,
            "a.strand": "+",
            "b.chrom": "ENST00000402432.4",
            "b.start": 295,
            "b.end": 385,
            "b.strand": "+",
            "block_len": 90,
        },
        {
            "a.chrom": "chr5",
            "a.start": 160238605,  # 0-based
            "a.end": 160238716,
            "a.strand": "+",
            "b.chrom": "ENST00000402432.4",
            "b.start": 385,
            "b.end": 496,
            "b.strand": "+",
            "block_len": 111,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def expected_chain_forward_transcript_a_to_b() -> dict[str, Any]:
    """Expected chain header for forward strand transcript: genomic (a) → transcript (b)."""
    return {
        "score": 0,
        "t_chrom": "chr5",
        "t_size": 160238716,
        "t_strand": "+",
        "t_start": 160229505,
        "t_end": 160238716,
        "q_chrom": "ENST00000402432.4",
        "q_size": 9211,
        "q_strand": "+",
        "q_start": 0,
        "q_end": 9211,
        "chain_id": 1,
        "blocks": [(9211,)],
    }


@pytest.fixture
def expected_chain_forward_transcript_b_to_a() -> dict[str, Any]:
    """Expected chain header for forward strand transcript: transcript (b) → genomic (a)."""
    return {
        "score": 0,
        "t_chrom": "ENST00000402432.4",
        "t_size": 9211,
        "t_strand": "+",
        "t_start": 0,
        "t_end": 9211,
        "q_chrom": "chr5",
        "q_size": 160238716,
        "q_strand": "+",
        "q_start": 160229505,
        "q_end": 160238716,
        "chain_id": 1,
        "blocks": [(9211,)],
    }


# ============================================================================
# FIXTURES - Reverse Strand Transcript (ENST00000447513.7)
# ============================================================================


@pytest.fixture
def gff_reverse_strand() -> pd.DataFrame:
    """GFF table for reverse strand transcript ENST00000447513.7."""
    data = [
        {"score": 0, "seqid": "chr1", "type": "transcript", "start": 2403974, "end": 2412564, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "exon", "start": 2403974, "end": 2405834, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "three_prime_UTR", "start": 2403974, "end": 2405765, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "CDS", "start": 2405766, "end": 2405834, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "stop_codon", "start": 2405766, "end": 2405768, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "CDS", "start": 2406484, "end": 2406619, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "exon", "start": 2406484, "end": 2406619, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "CDS", "start": 2406720, "end": 2406895, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "exon", "start": 2406720, "end": 2406895, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "CDS", "start": 2408452, "end": 2408858, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "exon", "start": 2408452, "end": 2408858, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "CDS", "start": 2410371, "end": 2410451, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "exon", "start": 2410371, "end": 2410451, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "exon", "start": 2412391, "end": 2412564, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "CDS", "start": 2412391, "end": 2412502, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "start_codon", "start": 2412500, "end": 2412502, "strand": "-", "transcript_id": "ENST00000447513.7"},
        {"score": 0, "seqid": "chr1", "type": "five_prime_UTR", "start": 2412503, "end": 2412564, "strand": "-", "transcript_id": "ENST00000447513.7"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def expected_mapping_table_reverse_exons() -> pd.DataFrame:
    """Expected mapping table for reverse strand exons (after extracting 'exon' type)."""
    data = [
        {"a.chrom": "chr1", "a.start": 2403973, "a.end": 2405834, "a.strand": "-", "b.chrom": "ENST00000447513.7", "b.start": 974, "b.end": 2835, "b.strand": "+", "block_len": 1861},
        {"a.chrom": "chr1", "a.start": 2406483, "a.end": 2406619, "a.strand": "-", "b.chrom": "ENST00000447513.7", "b.start": 838, "b.end": 974, "b.strand": "+", "block_len": 136},
        {"a.chrom": "chr1", "a.start": 2406719, "a.end": 2406895, "a.strand": "-", "b.chrom": "ENST00000447513.7", "b.start": 662, "b.end": 838, "b.strand": "+", "block_len": 176},
        {"a.chrom": "chr1", "a.start": 2408451, "a.end": 2408858, "a.strand": "-", "b.chrom": "ENST00000447513.7", "b.start": 255, "b.end": 662, "b.strand": "+", "block_len": 407},
        {"a.chrom": "chr1", "a.start": 2410370, "a.end": 2410451, "a.strand": "-", "b.chrom": "ENST00000447513.7", "b.start": 174, "b.end": 255, "b.strand": "+", "block_len": 81},
        {"a.chrom": "chr1", "a.start": 2412390, "a.end": 2412564, "a.strand": "-", "b.chrom": "ENST00000447513.7", "b.start": 0, "b.end": 174, "b.strand": "+", "block_len": 174},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def expected_chain_reverse_exons_a_to_b() -> dict[str, Any]:
    """Expected chain for reverse strand exons: genomic (a) → transcript (b)."""
    return {
        "score": 0,
        "t_chrom": "chr1",
        "t_size": 2412564,
        "t_strand": "-",
        "t_start": 2403973,
        "t_end": 2412564,
        "q_chrom": "ENST00000447513.7",
        "q_size": 2835,
        "q_strand": "-",
        "q_start": 0,
        "q_end": 2835,
        "chain_id": 1,
        "blocks": [
            (1861, 649, 0),
            (136, 100, 0),
            (176, 1556, 0),
            (407, 1512, 0),
            (81, 1939, 0),
            (174,),
        ],
    }


@pytest.fixture
def expected_chain_reverse_exons_b_to_a() -> dict[str, Any]:
    """Expected chain for reverse strand exons: transcript (b) → genomic (a)."""
    return {
        "score": 0,
        "t_chrom": "ENST00000447513.7",
        "t_size": 2835,
        "t_strand": "+",
        "t_start": 0,
        "t_end": 2835,
        "q_chrom": "chr1",
        "q_size": 2412564,
        "q_strand": "-",
        "q_start": 0,
        "q_end": 8591,
        "chain_id": 1,
    }


# ============================================================================
# TESTS - ChainRecord Basic Functionality
# ============================================================================


def test_chain_record_creation() -> None:
    """Test basic ChainRecord creation and attributes."""
    chain = ChainRecord(
        t_chrom="chr1",
        t_size=1000,
        t_strand="+",
        t_start=100,
        t_end=900,
        q_chrom="transcript1",
        q_size=800,
        q_strand="+",
        q_start=0,
        q_end=800,
        chain_id=1,
        score=100,
    )
    assert chain.t_chrom == "chr1"
    assert chain.t_size == 1000
    assert chain.q_chrom == "transcript1"
    assert chain.chain_id == 1
    assert chain.score == 100


def test_chain_record_header() -> None:
    """Test ChainRecord header generation."""
    chain = ChainRecord(
        t_chrom="chr1",
        t_size=1000,
        t_strand="+",
        t_start=100,
        t_end=900,
        q_chrom="transcript1",
        q_size=800,
        q_strand="+",
        q_start=0,
        q_end=800,
        chain_id=1,
        score=100,
    )
    expected = "chain 100 chr1 1000 + 100 900 transcript1 800 + 0 800 1"
    assert chain.header == expected


def test_chain_record_add_block() -> None:
    """Test adding blocks to ChainRecord."""
    chain = ChainRecord(
        t_chrom="chr1",
        t_size=1000,
        t_strand="+",
        t_start=100,
        t_end=900,
        q_chrom="transcript1",
        q_size=800,
        q_strand="+",
        q_start=0,
        q_end=800,
        chain_id=1,
        score=100,
    )

    # Add non-terminal blocks (with dt, dq)
    chain.add_block(100, 10, 0)
    chain.add_block(200, 5, 0)

    # Add terminal block (no dt, dq)
    chain.add_block(500)

    assert len(chain.blocks) == 3
    assert chain.blocks[0] == (100, 10, 0)
    assert chain.blocks[1] == (200, 5, 0)
    assert chain.blocks[2] == (500,)


def test_chain_record_to_string() -> None:
    """Test ChainRecord serialization to UCSC chain format."""
    chain = ChainRecord(
        t_chrom="chr1",
        t_size=1000,
        t_strand="+",
        t_start=100,
        t_end=900,
        q_chrom="transcript1",
        q_size=800,
        q_strand="+",
        q_start=0,
        q_end=800,
        chain_id=1,
        score=100,
    )
    chain.add_block(100, 10, 0)
    chain.add_block(700)

    result = chain.to_string()
    lines = result.strip().split("\n")

    assert lines[0] == "chain 100 chr1 1000 + 100 900 transcript1 800 + 0 800 1"
    assert lines[1] == "100\t10\t0"
    assert lines[2] == "700"


# ============================================================================
# TESTS - MappingTable to Chain Conversion (Forward Strand)
# ============================================================================


def test_mapping_table_forward_transcript_a_to_b(
    expected_mapping_table_forward_transcript: pd.DataFrame,
    expected_chain_forward_transcript_a_to_b: dict[str, Any],
) -> None:
    """Test MappingTable.to_chain() for forward strand transcript: genomic → transcript."""
    mapping = MappingTable(expected_mapping_table_forward_transcript)
    chain = mapping.to_chain(target="a", query="b", chain_id=1, score=0)

    expected = expected_chain_forward_transcript_a_to_b

    assert chain.t_chrom == expected["t_chrom"]
    assert chain.t_size == expected["t_size"]
    assert chain.t_strand == expected["t_strand"]
    assert chain.t_start == expected["t_start"]
    assert chain.t_end == expected["t_end"]
    assert chain.q_chrom == expected["q_chrom"]
    assert chain.q_size == expected["q_size"]
    assert chain.q_strand == expected["q_strand"]
    assert chain.q_start == expected["q_start"]
    assert chain.q_end == expected["q_end"]
    assert chain.chain_id == expected["chain_id"]
    assert chain.blocks == expected["blocks"]


def test_mapping_table_forward_transcript_b_to_a(
    expected_mapping_table_forward_transcript: pd.DataFrame,
    expected_chain_forward_transcript_b_to_a: dict[str, Any]
) -> None:
    """Test MappingTable.to_chain() for forward strand transcript: transcript → genomic."""
    mapping = MappingTable(expected_mapping_table_forward_transcript)
    chain = mapping.to_chain(target="b", query="a", chain_id=1, score=0)

    expected = expected_chain_forward_transcript_b_to_a

    assert chain.t_chrom == expected["t_chrom"]
    assert chain.t_size == expected["t_size"]
    assert chain.t_strand == expected["t_strand"]
    assert chain.t_start == expected["t_start"]
    assert chain.t_end == expected["t_end"]
    assert chain.q_chrom == expected["q_chrom"]
    assert chain.q_size == expected["q_size"]
    assert chain.q_strand == expected["q_strand"]
    assert chain.q_start == expected["q_start"]
    assert chain.q_end == expected["q_end"]
    assert chain.chain_id == expected["chain_id"]
    assert chain.blocks == expected["blocks"]


# ============================================================================
# TESTS - MappingTable to Chain Conversion (Reverse Strand)
# ============================================================================


def test_mapping_table_reverse_exons_a_to_b(
    expected_mapping_table_reverse_exons: pd.DataFrame,
    expected_chain_reverse_exons_a_to_b: dict[str, Any]
) -> None:
    """Test MappingTable.to_chain() for reverse strand exons: genomic → transcript."""
    mapping = MappingTable(expected_mapping_table_reverse_exons)
    chain = mapping.to_chain(target="a", query="b", chain_id=1, score=0)

    expected = expected_chain_reverse_exons_a_to_b

    assert chain.t_chrom == expected["t_chrom"]
    assert chain.t_size == expected["t_size"]
    assert chain.t_strand == expected["t_strand"]
    assert chain.t_start == expected["t_start"]
    assert chain.t_end == expected["t_end"]
    assert chain.q_chrom == expected["q_chrom"]
    assert chain.q_size == expected["q_size"]
    assert chain.q_strand == expected["q_strand"]
    assert chain.q_start == expected["q_start"]
    assert chain.q_end == expected["q_end"]
    assert chain.chain_id == expected["chain_id"]
    assert chain.blocks == expected["blocks"]


def test_mapping_table_reverse_exons_blocks_dt_dq(
    expected_mapping_table_reverse_exons: pd.DataFrame
) -> None:
    """Test that reverse strand produces correct dt and dq values in blocks."""
    mapping = MappingTable(expected_mapping_table_reverse_exons)
    chain = mapping.to_chain(target="a", query="b", chain_id=1, score=0)

    # Verify the critical dt and dq values from the notebook
    # Block 0→1: dt=649, dq=0
    assert chain.blocks[0] == (1861, 649, 0)

    # Block 1→2: dt=100, dq=0
    assert chain.blocks[1] == (136, 100, 0)

    # Block 2→3: dt=1556, dq=0
    assert chain.blocks[2] == (176, 1556, 0)

    # Block 3→4: dt=1512, dq=0
    assert chain.blocks[3] == (407, 1512, 0)

    # Block 4→5: dt=1939, dq=0
    assert chain.blocks[4] == (81, 1939, 0)

    # Last block: only size
    assert chain.blocks[5] == (174,)


# ============================================================================
# TESTS - Helper Functions
# ============================================================================


def test_per_entity_processing_to_nool_bed6_forward_exons(
    gff_forward_strand: pd.DataFrame
) -> None:
    """Test _per_entity_processing_to_nool_bed6 for forward strand exons."""
    result = _per_entity_processing_to_nool_bed6(
        gff=gff_forward_strand,
        entity_id_column="transcript_id",
        subset_type="exon",
    )

    # Should merge overlapping exons into 4 blocks
    assert len(result) == 4
    assert result["chrom"].iloc[0] == "chr5"
    assert result["strand"].iloc[0] == "+"
    assert result["name"].iloc[0] == "ENST00000402432.4"

    # Check first exon (0-based coordinates)
    assert result["start"].iloc[0] == 160229505
    assert result["end"].iloc[0] == 160229624


def test_assign_entity_query_coords_forward(gff_forward_strand: pd.DataFrame) -> None:
    """Test _assign_entity_query_coords for forward strand."""
    bed6 = _per_entity_processing_to_nool_bed6(
        gff=gff_forward_strand,
        entity_id_column="transcript_id",
        subset_type="exon",
    )
    mapping = _assign_entity_query_coords(bed6)

    # Check structure
    assert "a.chrom" in mapping.columns
    assert "b.chrom" in mapping.columns
    assert "block_len" in mapping.columns

    # Check first block starts at 0 in transcript coords
    assert mapping["b.start"].iloc[0] == 0


def test_assign_entity_query_coords_reverse(gff_reverse_strand: pd.DataFrame) -> None:
    """Test _assign_entity_query_coords for reverse strand (transcription order)."""
    bed6 = _per_entity_processing_to_nool_bed6(
        gff=gff_reverse_strand,
        entity_id_column="transcript_id",
        subset_type="exon",
    )
    mapping = _assign_entity_query_coords(bed6)

    # For reverse strand, transcription order is reverse genomic order
    # First block in transcription order (5' end) is the last genomically
    assert len(mapping) == 6

    # Last exon genomically (chr1:2412390-2412564) should be first in transcript coords
    # Check it starts at b.start=0
    last_genomic = mapping[mapping["a.end"] == 2412564]
    assert last_genomic["b.start"].iloc[0] == 0
    assert last_genomic["b.end"].iloc[0] == 174


# ============================================================================
# TESTS - Full Workflow with gff_to_chains
# ============================================================================


def test_gff_to_chains_forward_transcript(gff_forward_strand: pd.DataFrame) -> None:
    """Test full gff_to_chains workflow for forward strand transcript."""
    chains = list(
        gff_to_chains(
            gff=gff_forward_strand,
            entity_id_column="transcript_id",
            subset_type="transcript",
            target="seqid",
            query="entity_id",
        )
    )

    assert len(chains) == 1
    chain = chains[0]

    assert chain.t_chrom == "chr5"
    assert chain.q_chrom == "ENST00000402432.4"
    assert chain.t_strand == "+"
    assert chain.q_strand == "+"
    assert len(chain.blocks) == 1
    assert chain.blocks[0] == (9211,)


def test_gff_to_chains_forward_exons(gff_forward_strand: pd.DataFrame) -> None:
    """Test full gff_to_chains workflow for forward strand exons."""
    chains = list(
        gff_to_chains(
            gff=gff_forward_strand,
            entity_id_column="transcript_id",
            subset_type="exon",
            target="seqid",
            query="entity_id",
        )
    )

    assert len(chains) == 1
    chain = chains[0]

    assert chain.t_chrom == "chr5"
    assert chain.q_chrom == "ENST00000402432.4"
    assert len(chain.blocks) == 4


def test_gff_to_chains_reverse_exons(gff_reverse_strand: pd.DataFrame) -> None:
    """Test full gff_to_chains workflow for reverse strand exons."""
    chains = list(
        gff_to_chains(
            gff=gff_reverse_strand,
            entity_id_column="transcript_id",
            subset_type="exon",
            target="seqid",
            query="entity_id",
        )
    )

    assert len(chains) == 1
    chain = chains[0]

    assert chain.t_chrom == "chr1"
    assert chain.q_chrom == "ENST00000447513.7"
    assert chain.t_strand == "-"
    assert chain.q_strand == "-"
    assert len(chain.blocks) == 6

    # Verify critical block structure
    assert chain.blocks[0] == (1861, 649, 0)
    assert chain.blocks[-1] == (174,)


# ============================================================================
# TESTS - write_chains Function
# ============================================================================


def test_write_chains_to_stringio(gff_forward_strand: pd.DataFrame) -> None:
    """Test write_chains with StringIO."""
    chains = gff_to_chains(
        gff=gff_forward_strand,
        entity_id_column="transcript_id",
        subset_type="transcript",
        target="seqid",
        query="entity_id",
    )

    buffer = StringIO()
    write_chains(chains, buffer)
    result = buffer.getvalue()

    assert "chain 0 chr5" in result
    assert "ENST00000402432.4" in result
    assert "9211" in result


def test_write_chains_to_file(gff_forward_strand: pd.DataFrame, tmp_path: Path) -> None:
    """Test write_chains with actual file."""
    chains = gff_to_chains(
        gff=gff_forward_strand,
        entity_id_column="transcript_id",
        subset_type="transcript",
        target="seqid",
        query="entity_id",
    )

    output_file = tmp_path / "test.chain"
    with open(output_file, "w") as f:
        write_chains(chains, f)

    # Verify file was created and contains expected content
    assert output_file.exists()
    content = output_file.read_text()
    assert "chain 0 chr5" in content
    assert "ENST00000402432.4" in content


# ============================================================================
# TESTS - Edge Cases and Validation
# ============================================================================


def test_chain_record_header_as_dict() -> None:
    """Test ChainRecord.header_as_dict() method."""
    chain = ChainRecord(
        t_chrom="chr1",
        t_size=1000,
        t_strand="+",
        t_start=100,
        t_end=900,
        q_chrom="transcript1",
        q_size=800,
        q_strand="+",
        q_start=0,
        q_end=800,
        chain_id=1,
        score=100,
    )

    header_dict = chain.header_as_dict()
    assert "chain" in header_dict
    assert header_dict["chain"]["t_chrom"] == "chr1"
    assert header_dict["chain"]["q_chrom"] == "transcript1"
    assert header_dict["chain"]["score"] == 100


def test_gff_to_chains_empty_subset() -> None:
    """Test gff_to_chains with subset_type that doesn't exist."""
    gff = pd.DataFrame([
        {"seqid": "chr1", "type": "exon", "start": 100, "end": 200, "strand": "+", "transcript_id": "T1"},
    ])

    with pytest.raises(AssertionError, match="must have annotations of type"):
        list(gff_to_chains(gff, "transcript_id", "CDS", "seqid", "entity_id"))


def test_mapping_table_invalid_target_query() -> None:
    """Test MappingTable.to_chain() with invalid target/query values."""
    data = pd.DataFrame([
        {"a.chrom": "chr1", "a.start": 0, "a.end": 100, "a.strand": "+",
            "b.chrom": "T1", "b.start": 0, "b.end": 100, "b.strand": "+", "block_len": 100}
    ])
    mapping = MappingTable(data)

    with pytest.raises(AssertionError, match="target and query must be"):
        mapping.to_chain(target="x", query="y", chain_id=1, score=0)
