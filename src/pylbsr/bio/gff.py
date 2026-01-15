"""Module for reading and writing GFF3 files with validation using Pandera."""

import os

# Disable automatic backend detection for pandera that loads dask.
os.environ.setdefault("PANDERA_BACKEND", "pandas")

# sys.modules["dask"] = None
# sys.modules["dask.array"] = None
# sys.modules["dask.dataframe"] = None
# sys.modules["modin"] = None
# sys.modules["pyspark"] = None
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera as pa
import pandera.pandas as pa
import pybedtools as pbt
from pandera import DataFrameModel, Field, check
from pandera.pandas import DataFrameModel, Field, check
from pandera.typing import DataFrame, Series
from typing_extensions import Self


@dataclass
class GenomicInterval:
    chrom: str
    start: int
    end: int
    strand: str


class GFFSchema(DataFrameModel):
    """Pandera schema for GFF3 files."""

    seqid: Series[str] = Field(nullable=False)
    source: Series[str] = Field(nullable=True)
    type: Series[str] = Field(nullable=False)

    start: Series[pd.Int64Dtype] = Field(ge=1, nullable=False)
    end: Series[pd.Int64Dtype] = Field(ge=1, nullable=False)

    score: Series[pd.Float64Dtype] = Field(nullable=True)
    strand: Series[str] = Field(nullable=True)
    phase: Series[str] = Field(nullable=True)
    attributes: Series[str] = Field(nullable=True)

    # --- column-level checks ----------------------------------------------
    # TODO: not sure if there is a convention here.
    # @check("score", element_wise=True)
    # def score_range(cls, s):
    #    return s is None or (0 <= s <= 1)

    # TODO: what about "-1" and "+1" or "fwd" and "rev" encodings?
    @check("strand", element_wise=True)
    def _valid_strand(cls, s: str) -> bool:
        return pd.isna(s) or s in {"+", "-", "."}

    @check("phase", element_wise=True)
    def _valid_phase(cls, s: str) -> bool:
        return pd.isna(s) or s in {"0", "1", "2", "."}

    # --- dataframe-level checks -------------------------------------------
    @check
    def _end_ge_start(cls, df: pd.DataFrame) -> pd.Series:
        return df["end"] >= df["start"]

    class Config:
        """Pandera configuration for GFFSchema."""
        coerce = True  # automatically cast types
        strict = True  # no extra columns allowed


def read_gff(filepath: os.PathLike, validate: bool = False) -> DataFrame[GFFSchema]:
    """Read a GFF3 file into a Pandas DataFrame."""
    gff_columns = [
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]

    columns_to_nullable_values = {
        "score": [".", ""],
    }

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=gff_columns,
        comment="#",
        dtype=str,  # Delegate type conversion to pandera
        # DO NOT DETECT NA VALUES AUTOMATICALLY
        keep_default_na=True,
        na_filter=False,
    )

    # Post-process nullable values.
    for column, nullable_values in columns_to_nullable_values.items():
        df[column] = df[column].replace(nullable_values, pd.NA)

    if validate:
        df = GFFSchema.validate(df)
    else:
        # Manual type conversion.
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    return df


def split_attributes(col: pd.Series, kv_sep: str = "=", field_sep: str = ";") -> pd.DataFrame:
    """Split the GFF attributes column into a DataFrame of key-value pairs."""

    def parse(s: str) -> dict[str, str]:
        """Parse a single GFF attributes string into a dictionary."""
        if not isinstance(s, str):
            return {}
        items = []
        for kv in s.split(field_sep):
            kv = kv.strip()
            if not kv:
                continue

            if kv_sep in kv:
                k, v = kv.split(kv_sep, 1)
                items.append((k, v))

        return dict(items)

    return pd.json_normalize(col.apply(parse))

def write_gff(
    gff: pd.DataFrame,
    filepath: os.PathLike,
) -> None:
    """Write a GFF dataframe to a GFF file."""
    # Write GFF file without header and index.
    # Write expected GFF identifier header.
    with open(filepath, "w") as f:
        f.write("##gff-version 3\n")
        gff.to_csv(
            f,
            sep="\t",
            header=False,
            index=False,
            na_rep=".",
            mode="a",
        )

class ExtendedGFF:
    """Class representing a GFF file with split attributes."""

    def __init__(self, gff: pd.DataFrame, attributes: pd.DataFrame):
        """Initialize an ExtendedGFF instance."""
        self._gff = gff
        self._attributes = attributes
        self._extended = pd.concat(
            [gff.reset_index(drop=True), attributes.reset_index(drop=True)], axis=1
        )

    @property
    def gff(self) -> pd.DataFrame:
        """Return the original GFF dataframe."""
        return self._gff

    @property
    def attributes(self) -> pd.DataFrame:
        """Return the attributes dataframe."""
        return self._attributes

    @property
    def extended(self) -> pd.DataFrame:
        """Return the extended GFF dataframe with split attributes."""
        return self._extended

    @classmethod
    def from_filepath(cls, filepath: os.PathLike, validate: bool = False) -> Self:
        """Create an ExtendedGFF instance from a GFF file."""
        gff = read_gff(filepath, validate=validate)
        attributes = split_attributes(gff["attributes"])
        return cls(gff, attributes)

    @classmethod
    def from_gff(cls, gff: pd.DataFrame, validate: bool = False) -> Self:
        """Create an ExtendedGFF instance from a GFF dataframe."""
        if validate:
            gff = GFFSchema.validate(gff)
        attributes = split_attributes(gff["attributes"])
        return cls(gff, attributes)

    def write(self, filepath: os.PathLike) -> None:
        """Write the GFF dataframe to a GFF file."""
        write_gff(self._gff, filepath)


def gff_transcript_segments_to_bed(gff: pd.DataFrame) -> pd.DataFrame:
    """Produce a BED6 dataframe from a GFF transcript segments dataframe, including introns."""
    assert "transcript_id" in gff.columns, 'Column "transcript_id" not found in GFF!'

    # NOTE: we will use the "ID" column to identify unique features.
    # This has a pre-established format "{FEATURE_TYPE}:{TRANSCRIPT_ID}[:{EXON_INDEX}]"
    assert "ID" in gff.columns, "GFF must have an 'ID' column!"

    assert "type" in gff.columns, "GFF must have a 'type' column!"
    assert "transcript" in gff["type"].values, "GFF must have transcript annotations!"
    assert gff["type"].value_counts().get("transcript", 0) == 1, (
        "GFF must have exactly one transcript!"
    )
    # NOTE: wrong assertion `gff["type"].value_counts().get("exon", 0) > 1` for  single exon gffs.
    assert "exon" in gff["type"].values, "GFF must have exon annotations!"

    gff_transcript = gff.loc[lambda df: df["type"] == "transcript"]
    gff_segments = (
        gff.loc[lambda df: df["type"] != "transcript"]
        .copy()
        .sort_values(by=["start", "end"])
        .reset_index(drop=True)
    )

    selected_columns_for_bed6_format = ["seqid", "start", "end", "ID", "score", "strand"]

    # Extract the transcript
    bed_transcript = gff_transcript.loc[:, selected_columns_for_bed6_format].assign(
        start=lambda df: df["start"] - 1
    )
    # Extract the transcript ID
    transcript_id = bed_transcript["ID"].values[0]

    # Extract the segments (exons, UTRs, etc.)
    bed_segments = gff_segments.loc[:, selected_columns_for_bed6_format].assign(
        start=lambda df: df["start"] - 1
    )

    bt_transcript = pbt.BedTool.from_dataframe(bed_transcript)
    bt_segments = pbt.BedTool.from_dataframe(bed_segments)

    # Identify introns by subtracting exons from the transcript region.
    bed_introns = (
        bt_transcript.subtract(bt_segments, s=True)
        .to_dataframe()
        .assign(
            name=f"intron:{transcript_id}",
            score=np.nan,
        )
    )

    # Merge all segments back
    bed6_cols = ["chrom", "start", "end", "name", "score", "strand"]

    # Create empty BED6 dataframe explicitly if no introns are found.
    if bed_introns.shape[0] == 0:
        bed_introns = pd.DataFrame(columns=bed6_cols)

    bed = (
        pd.concat(
            [bed_segments.set_axis(bed6_cols, axis=1), bed_introns.set_axis(bed6_cols, axis=1)],
            axis=0,
        )
        .sort_values(by=["start", "end"])
        .reset_index(drop=True)
    )

    # Assert that the transcript coordinates match the min/max of the segments + introns.
    assert bed["start"].min() == bed_transcript["start"].values[0], (
        "Transcript start does not match min segment/intron start!"
    )
    assert bed["end"].max() == bed_transcript["end"].values[0], (
        "Transcript end does not match max segment/intron end!"
    )

    return bed


def get_transcript_boundaries_from_gff(gff: pd.DataFrame) -> GenomicInterval:
    """Get the (1-based) genomic boundaries of a transcript from its GFF annotations.

    From a provided GFF pandas dataframe, this function extracts the genomic boundaries,
    either using the explicit "transcript" annotation, or by inferring them from the provided
    segments (e.g., exons, UTRs, CDS, etc.).

    In this second case, it is assumed the segments cover the entire transcript.
    """
    if gff["transcript_id"].nunique() > 1:
        raise ValueError("GFF contains annotations for multiple transcripts!")

    if "transcript" in gff["type"].values:
        gff_transcript = gff.loc[lambda df: df["type"] == "transcript"]
        if not gff.shape[0] == 1:
            raise ValueError("GFF contains multiple transcript annotations!")

        gff_transcript = gff_transcript.iloc[0, :]

        transcript_boundaries = GenomicInterval(
            chrom=gff_transcript["seqid"],
            start=gff_transcript["start"],
            end=gff_transcript["end"],
            strand=gff_transcript["strand"],
        )
        return transcript_boundaries

    else:
        warnings.warn(
            "GFF does not contain a transcript annotation; inferring boundaries from segments."
        )

        transcript_boundaries = GenomicInterval(
            chrom=gff.iloc[0]["seqid"],
            start=gff["start"].min(),
            end=gff["end"].max(),
            strand=gff.iloc[0]["strand"],
        )
        return transcript_boundaries
