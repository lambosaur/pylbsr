"""Tests for gff.bed2gtf -- BED-to-GTF table conversion."""

import pandas as pd
import pytest

from pylbsr.bio.gff import bed2gtf


def test_bed2gtf_requires_minimal_columns() -> None:
    """chrom/start/end are the minimal required columns."""
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0]})  # missing "end"
    with pytest.raises(ValueError, match="Missing minimal BED columns"):
        bed2gtf(df)


def test_bed2gtf_shifts_start_to_one_based() -> None:
    """0-based BED start becomes 1-based GTF start; end is unchanged."""
    df = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})

    result = bed2gtf(df)

    assert result.loc[0, "start"] == 101
    assert result.loc[0, "end"] == 200


def test_bed2gtf_chrom_maps_to_seqname() -> None:
    """chrom is automatically renamed to seqname."""
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]})

    result = bed2gtf(df)

    assert result.loc[0, "seqname"] == "chr1"
    assert "chrom" not in result.columns


def test_bed2gtf_fills_missing_columns_with_defaults() -> None:
    """GTF columns with no matching BED column get GTF_DEFAULT_VALUES."""
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]})

    result = bed2gtf(df)

    assert result.loc[0, "source"] == "."
    assert result.loc[0, "strand"] == "."
    assert result.loc[0, "score"] == 0


def test_bed2gtf_default_values_can_be_overridden() -> None:
    """default_values overrides individual entries of GTF_DEFAULT_VALUES."""
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]})

    result = bed2gtf(df, default_values={"score": 100})

    assert result.loc[0, "score"] == 100


def test_bed2gtf_no_extra_columns_does_not_crash() -> None:
    """Regression test: when every BED column maps directly to a GTF column (no leftover
    columns for "attribute"), the original code KeyError'd on gtf_to_bed_columns["attribute"]
    -- confirmed empirically against the unmodified pyutils source before this fix.
    """
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10], "score": [5], "strand": ["+"]})

    result = bed2gtf(df)

    assert result.loc[0, "score"] == 5
    assert result.loc[0, "strand"] == "+"
    assert result.loc[0, "attribute"] == "."  # default, since nothing was left over


def test_bed2gtf_extra_columns_become_attribute_string() -> None:
    """Columns not named after a GTF column are concatenated into 'key "value"; ...'."""
    df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [10],
            "gene_id": ["ENSG1"],
            "transcript_id": ["ENST1"],
        }
    )

    result = bed2gtf(df)

    assert result.loc[0, "attribute"] == 'gene_id "ENSG1"; transcript_id "ENST1"'


def test_bed2gtf_map_columns_bed_to_gtf_renames_onto_gtf_column() -> None:
    """map_columns_bed_to_gtf assigns a BED column's content to a named GTF column,
    instead of it being folded into "attribute".
    """
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10], "name": ["my_feature"]})

    result = bed2gtf(df, map_columns_bed_to_gtf={"name": "feature"})

    assert result.loc[0, "feature"] == "my_feature"
    assert result.loc[0, "attribute"] == "."  # nothing left over for attribute


def test_bed2gtf_rejects_renaming_collision() -> None:
    """Renaming a column onto a name already present under its own name is rejected."""
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10], "score": [5], "avg_score": [7]})

    with pytest.raises(ValueError, match="Conflict"):
        bed2gtf(df, map_columns_bed_to_gtf={"avg_score": "score"})


def test_bed2gtf_returns_columns_in_gtf_order() -> None:
    """Output columns always come back in the standard GTF column order."""
    df = pd.DataFrame({"end": [10], "chrom": ["chr1"], "start": [0]})

    result = bed2gtf(df)

    assert list(result.columns) == [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ]
