from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

bed6_cols = ["chrom", "start", "end", "name", "score", "strand"]

# https://macs3-project.github.io/MACS/docs/narrowPeak.html
narrowpeak_cols = [
    *bed6_cols,
    *[
    "SignalValue",
    "Pvalue",
    "Qvalue",
    "PeakSummitOffset",
]
]


@dataclass
class GenomicInterval:
    """BED6 interval."""

    chrom: str
    start: int
    end: int
    name: str
    score: int | float | str
    strand: Literal["+", "-"]

    def __post_init__(self):
        # Enforce [0, end) interval.
        if self.start < 0:
            raise ValueError(f"Start position {self.start} must be non-negative.")
        if self.start >= self.end:
            raise ValueError(
                f"Start position {self.start} must be less than end position {self.end}."
            )


class Bed6IntervalsModel(pa.DataFrameModel):
    """Dataframe Pandera Schema Model for BED6 intervals."""

    chrom: pa.typing.String
    start: pa.typing.Int64
    end: pa.typing.Int64
    name: pa.typing.String
    score: pa.typing.Int64
    strand: pa.typing.String

    @pa.check("start")
    def check_start_greater_than_zero(
        cls, series: pa.typing.Series[pa.typing.Int64]
    ) -> pa.typing.Series[pa.typing.Int64]:
        return series[series > 0]

    @pa.check("end")
    def check_end_greater_than_zero(
        cls, series: pa.typing.Series[pa.typing.Int64]
    ) -> pa.typing.Series[pa.typing.Int64]:
        return series[series > 0]

    @pa.check("end")
    def check_end_greater_than_start(
        cls, series: pa.typing.Series[pa.typing.Int64]
    ) -> pa.typing.Series[pa.typing.Int64]:
        return series[series > series["start"]]


@dataclass
class SequenceInterval:
    chrom: str
    start: int
    end: int
    strand: str


def parse_name_to_sequence_interval(name: str) -> SequenceInterval:
    """Parse the simple "chrom:start-end:strand" genomic identifiers to a SequenceInterval object."""
    chrom, start_end, strand = name.split(":")
    start, end = map(int, start_end.split("-"))
    return SequenceInterval(chrom=chrom, start=start, end=end, strand=strand)


def identifiers_to_bed6_dataframe(
    identifiers: Sequence[str],
) -> DataFrame[Bed6IntervalsModel]:
    """Parse a list of "chrom:start-end:strand" identifiers to a BED6 dataframe."""
    df = pd.DataFrame(
        map(
            lambda v: asdict(parse_name_to_sequence_interval(v)),
            identifiers,
        )
    )
    df["name"] = identifiers
    df = df.reset_index().rename(columns={"index": "score"})

    return df.loc[:, bed6_cols]

