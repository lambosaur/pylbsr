import os

# Disable automatic backend detection for pandera that loads dask.
os.environ.setdefault("PANDERA_BACKEND", "pandas")
import sys

# sys.modules["dask"] = None
# sys.modules["dask.array"] = None
# sys.modules["dask.dataframe"] = None
# sys.modules["modin"] = None
# sys.modules["pyspark"] = None
import pandas as pd
import pandera as pa
import pandera.pandas as pa
from pandera import DataFrameModel, Field, check
from pandera.pandas import DataFrameModel, Field, check
from pandera.typing import DataFrame, Series


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
    def valid_strand(cls, s):
        return pd.isna(s) or s in {"+", "-", "."}

    @check("phase", element_wise=True)
    def valid_phase(cls, s):
        return pd.isna(s) or s in {"0", "1", "2", "."}

    # --- dataframe-level checks -------------------------------------------
    @check
    def end_ge_start(cls, df: pd.DataFrame) -> pd.Series:
        return df["end"] >= df["start"]

    class Config:
        coerce = True  # automatically cast types
        strict = True  # no extra columns allowed


def read_gff(filepath: os.PathLike, validate: bool = False) -> DataFrame[GFFSchema]:
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
    def parse(s: str):
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
):
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
    def __init__(self, gff: pd.DataFrame, attributes: pd.DataFrame):
        self._gff = gff
        self._attributes = attributes
        self._extended = pd.concat(
            [gff.reset_index(drop=True), attributes.reset_index(drop=True)], axis=1
        )

        # Initialize the parent class with the extended dataframe.
        super().__init__(self._extended)

    @property
    def gff(self) -> pd.DataFrame:
        return self._gff

    @property
    def attributes(self) -> pd.DataFrame:
        return self._attributes

    @property
    def extended(self) -> pd.DataFrame:
        return self._extended

    @classmethod
    def from_filepath(cls, filepath: os.PathLike, validate: bool = False) -> "ExtendedGFF":
        gff = read_gff(filepath, validate=validate)
        attributes = split_attributes(gff["attributes"])
        return cls(gff, attributes)

    def write(self, filepath: os.PathLike):
        write_gff(self._gff, filepath)

