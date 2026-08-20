"""Module for reading and writing GFF3 files with validation using Pandera."""

import collections
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
from typing import cast

import numpy as np
import pandas as pd
import pybedtools as pbt
from pandera.pandas import DataFrameModel, Field, check, dataframe_check
from pandera.typing import DataFrame, Series
from typing_extensions import Self


@dataclass
class GenomicInterval:
    """A single genomic interval (chrom, start, end, strand)."""

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
    @dataframe_check()
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

    return cast(DataFrame[GFFSchema], df)


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

    return pd.json_normalize(list(col.apply(parse)))


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

    def __init__(self, gff: pd.DataFrame, attributes: pd.DataFrame) -> None:
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
    strand = gff_transcript["strand"].values[0]
    bed_introns = bt_transcript.subtract(bt_segments, s=True).to_dataframe().assign(score=np.nan)
    if bed_introns.shape[0] > 0:
        intron_order = (
            bed_introns["start"].rank(method="first", ascending=(strand != "-")).astype(int)
        )
        bed_introns["name"] = intron_order.map(lambda i: f"intron:{transcript_id}:{i}")
    else:
        bed_introns["name"] = pd.Series(dtype=str)

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
        if gff_transcript.shape[0] != 1:
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


GTF_COLUMNS = (
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
)

GTF_DEFAULT_VALUES: dict[str, object] = {
    "source": ".",
    "strand": ".",  # could also be '+'
    "frame": ".",  # could also be 0
    "attribute": ".",
    "feature": ".",
    "score": 0,
}


def bed2gtf(  # noqa: C901 -- cohesive column-mapping/validation pipeline, covered by tests
    table_bed: pd.DataFrame,
    map_columns_bed_to_gtf: dict[str, str] | None = None,
    default_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Convert a BED-formatted table to a GTF-formatted table.

    The minimal expected columns in `table_bed` are `chrom`, `start`, `end` (0-based
    coordinates). `chrom` is automatically mapped to `seqname`. Any other column not
    named after a GTF column has its content concatenated into the GTF `attribute`
    column as `key "value"; key "value"` pairs; provide `map_columns_bed_to_gtf` to
    map a BED column directly onto a GTF column instead (e.g. `name` -> `feature`).
    GTF columns absent from the result are filled from `GTF_DEFAULT_VALUES`
    (overridable via `default_values`).

    Args:
        table_bed: Table with at least `chrom`, `start`, `end` columns (0-based).
        map_columns_bed_to_gtf: Maps BED column names onto GTF column names.
        default_values: Overrides for `GTF_DEFAULT_VALUES`.

    Returns:
        A GTF-formatted table (1-based `start`), with columns `GTF_COLUMNS` in order.

    Raises:
        ValueError: `table_bed` is missing `chrom`/`start`/`end`, `map_columns_bed_to_gtf`
            renames a column onto one that's already present under its own name, or a
            required GTF column ends up with neither a source column nor a default value.
    """
    bed_to_gtf_columns = {"chrom": "seqname", "start": "start", "end": "end"}

    default_gtf_values = GTF_DEFAULT_VALUES.copy()
    if default_values is not None:
        default_gtf_values.update(default_values)

    missing_cols = [c for c in bed_to_gtf_columns if c not in table_bed.columns]
    if missing_cols:
        raise ValueError(f"Missing minimal BED columns: {missing_cols}")

    if map_columns_bed_to_gtf is None:
        map_columns_bed_to_gtf = {}

    # Sanity check: collisions between already-present GTF-named columns and renamings.
    # Apply the renaming first, in case a conflict is resolved by another renaming.
    bed_to_gtf_columns.update(map_columns_bed_to_gtf)
    renamed_columns = pd.Series([bed_to_gtf_columns.get(c, c) for c in table_bed.columns])
    collisions = renamed_columns.value_counts().loc[lambda s: s > 1]
    if len(collisions) > 0:
        conflicts = {k: v for k, v in bed_to_gtf_columns.items() if v in collisions.index}
        raise ValueError(f"Conflict between renamed columns and already-present columns: {conflicts}")

    # Now that the mapping is confirmed collision-free, add the remaining BED columns
    # that already happen to match a GTF column name.
    for col in GTF_COLUMNS:
        if col in table_bed.columns and col not in bed_to_gtf_columns:
            bed_to_gtf_columns[col] = col

    # Reverse mapping (GTF column -> BED column(s)); anything not explicitly mapped
    # above is accumulated under "attribute".
    gtf_to_bed_columns: dict[str, list[str]] = collections.defaultdict(list)
    for col in table_bed.columns:
        gtf_to_bed_columns[bed_to_gtf_columns.get(col, "attribute")].append(col)

    mapped_columns: dict[str, str | list[str]] = {
        k: (v[0] if len(v) == 1 and k != "attribute" else v) for k, v in gtf_to_bed_columns.items()
    }
    stray_lists = [v for k, v in mapped_columns.items() if k != "attribute" and isinstance(v, list)]
    if stray_lists:
        raise ValueError(f"Unexpected list of columns built from the input BED: {stray_lists}")

    gtf_table = table_bed.rename(columns=bed_to_gtf_columns).copy()

    # "attribute" is absent from gtf_to_bed_columns entirely (not just empty) whenever every
    # BED column mapped directly onto a GTF column -- .get(..., []) instead of [...] avoids a
    # KeyError in that case.
    attribute_source_columns = mapped_columns.get("attribute", [])
    if attribute_source_columns:

        def format_row_to_attribute(row: pd.Series) -> str:
            return "; ".join(f'{k} "{v}"' for k, v in row[attribute_source_columns].items())

        gtf_table["attribute"] = table_bed.apply(format_row_to_attribute, axis=1).values

    for col, default_v in default_gtf_values.items():
        if col not in gtf_table.columns and col in GTF_COLUMNS:
            gtf_table[col] = default_v

    missing_gtf_cols = [c for c in GTF_COLUMNS if c not in gtf_table.columns]
    if missing_gtf_cols:
        raise ValueError(f"Missing columns in the GTF-converted table: {missing_gtf_cols}")

    gtf_table = gtf_table.loc[:, list(GTF_COLUMNS)].copy()

    # BED start is 0-based; GTF start is 1-based. BED's half-open end already equals
    # GTF's inclusive end numerically, so only start needs shifting.
    gtf_table["start"] = gtf_table["start"] + 1

    return gtf_table
