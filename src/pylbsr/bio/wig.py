"""Wig and BigWig related utilities."""

import abc
import functools
import glob
import os
import string
from collections.abc import Iterator
from typing import IO

import numpy as np
import pandas as pd
import pyBigWig as pbw
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

# ============================================================================
# Wiggle Format
# ============================================================================


@functools.total_ordering
class WiggleBlock(BaseModel, abc.ABC):
    """Abstract base class for Wiggle blocks (fixedStep / variableStep)."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    chrom: str
    values: tuple[int | float, ...]
    span: PositiveInt = 1

    @property
    @abc.abstractmethod
    def start(self) -> int:
        """Start position of the block (1-based, inclusive)."""
        ...

    @property
    @abc.abstractmethod
    def stop(self) -> int:
        """Last position covered by the block (1-based, inclusive)."""
        ...

    @property
    @abc.abstractmethod
    def positions(self) -> tuple[int, ...]:
        """Positions with values (not all positions in the range)."""
        ...

    @property
    @abc.abstractmethod
    def indexed_values(self) -> tuple[tuple[int, int | float], ...]:
        """Return (position, value) pairs for each covered position."""
        ...

    @property
    @abc.abstractmethod
    def header(self) -> str:
        """The declaration line for this block."""
        ...

    @property
    @abc.abstractmethod
    def data_lines(self) -> tuple[str, ...]:
        """The data lines for this block."""
        ...

    def to_wig(self) -> str:
        """Format this block as a wiggle string."""
        return "\n".join((self.header, *self.data_lines))

    def as_series(self, full_range: bool = True, fill_value: float | int = np.nan) -> pd.Series:
        """Return the block as a pandas Series.

        If `full_range` is True, the series includes all positions from
        `start` to `stop` (inclusive), filling gaps with `fill_value`.
        """
        series = pd.Series(dict(self.indexed_values))
        if full_range:
            return series.reindex(
                pd.RangeIndex(start=self.start, stop=self.stop + 1),
                fill_value=fill_value,
            )
        return series

    def intersect(self, other: "WiggleBlock") -> tuple[int, int] | None:
        """Return the intersection range (start, stop) or None."""
        if not isinstance(other, WiggleBlock):
            raise TypeError("Other must be a WiggleBlock instance")
        if self.chrom != other.chrom:
            return None
        start = max(self.start, other.start)
        stop = min(self.stop, other.stop)
        if start > stop:
            return None
        return (start, stop)

    def overlaps(self, other: "WiggleBlock") -> bool:
        """Check if this block overlaps with another WiggleBlock."""
        return self.intersect(other) is not None

    def __len__(self) -> int:
        return self.stop - self.start + 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WiggleBlock):
            return NotImplemented
        return (self.chrom, self.start) == (other.chrom, other.start)

    def __lt__(self, other: "WiggleBlock") -> bool:
        if not isinstance(other, WiggleBlock):
            return NotImplemented
        if self.chrom != other.chrom:
            return self.chrom < other.chrom
        return self.start < other.start

    def __hash__(self) -> int:
        return hash((self.chrom, self.start))


class FixedStepBlock(WiggleBlock):
    """A fixed-step Wiggle block."""

    start_: PositiveInt = Field(alias="start")
    step: PositiveInt = 1

    @model_validator(mode="after")
    def validate_step_span(self):
        if self.span > self.step:
            raise ValueError(f"Span {self.span} cannot be larger than step {self.step}")
        return self

    @field_validator("values")
    @classmethod
    def check_non_empty(cls, v: tuple[int | float, ...]) -> tuple[int | float, ...]:
        if len(v) == 0:
            raise ValueError("values must contain at least one element.")
        return v

    @property
    def start(self) -> int:
        return self.start_

    @property
    def stop(self) -> int:
        return self.start + self.step * (len(self.values) - 1) + self.span - 1

    @property
    def indexed_values(self) -> tuple[tuple[int, int | float], ...]:
        i = np.arange(len(self.values))
        j = np.arange(self.span)
        idx = self.start + i[:, None] * self.step + j[None, :]
        idx = idx.ravel()
        vals = np.repeat(np.array(self.values), self.span)
        return tuple(zip(idx, vals))

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(v[0] for v in self.indexed_values)

    @property
    def header(self) -> str:
        base = f"fixedStep chrom={self.chrom} start={self.start} step={self.step}"
        if self.span > 1:
            return f"{base} span={self.span}"
        return base

    @property
    def data_lines(self) -> tuple[str, ...]:
        return tuple(str(v) for v in self.values)


class VariableStepBlock(WiggleBlock):
    """A variable-step Wiggle block."""

    positions_: tuple[PositiveInt, ...] = Field(alias="positions")

    @model_validator(mode="after")
    def validate_positions(self) -> "VariableStepBlock":
        if len(self.positions_) != len(self.values):
            raise ValueError("positions and values must have the same length.")
        if not all(
            self.positions_[i] < self.positions_[i + 1] for i in range(len(self.positions_) - 1)
        ):
            raise ValueError("Positions must be strictly increasing.")
        diffs = [b - a for a, b in zip(self.positions_[:-1], self.positions_[1:])]
        if any(diff < self.span for diff in diffs):
            raise ValueError("Span value leads to overlap of some of the provided positions.")
        return self

    @property
    def start(self) -> int:
        return self.positions_[0]

    @property
    def stop(self) -> int:
        return self.positions_[-1] + self.span - 1

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(
            np.repeat(self.positions_, self.span) + np.tile(range(self.span), len(self.positions_))
        )

    @property
    def indexed_values(self) -> tuple[tuple[int, int | float], ...]:
        pos = np.array(self.positions_)
        vals = np.array(self.values)
        idx = np.repeat(pos, self.span) + np.tile(range(self.span), len(vals))
        return tuple(zip(idx, np.repeat(vals, self.span)))

    @property
    def header(self) -> str:
        base = f"variableStep chrom={self.chrom}"
        if self.span > 1:
            return f"{base} span={self.span}"
        return base

    @property
    def data_lines(self) -> tuple[str, ...]:
        return tuple(f"{pos} {val}" for pos, val in zip(self.positions_, self.values))


class WigTrackDefinition(BaseModel):
    """A Wiggle track definition line."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    type_: str = Field(alias="type", default="wiggle_0")
    name: str = "wiggle_track"
    description: str = "Wiggle Track"
    priority: int | None = None
    color: str | None = None
    graph_type: str | None = Field(alias="graphType", default=None)

    @model_validator(mode="after")
    def validate_type(self) -> "WigTrackDefinition":
        if self.type_ != "wiggle_0":
            raise ValueError(f"Invalid type: {self.type_}. Expected 'wiggle_0'.")
        return self

    def to_wig(self) -> str:
        """Format this track definition as a wiggle track line."""
        parts = [
            f"track type={self.type_}",
            f'name="{self.name}"',
            f'description="{self.description}"',
        ]
        if self.priority is not None:
            parts.append(f"priority={self.priority}")
        if self.color is not None:
            parts.append(f"color={self.color}")
        if self.graph_type is not None:
            parts.append(f"graphType={self.graph_type}")
        return " ".join(parts)


class WigBlockCollection(BaseModel):
    """A collection of Wiggle blocks, optionally associated with a track definition."""

    blocks: list[WiggleBlock] = Field(default_factory=list)
    track_definition: WigTrackDefinition | None = None

    def add(self, block: WiggleBlock) -> None:
        """Add a block at the correct sorted position, checking for overlaps."""
        starts = [b.start for b in self.blocks]
        index_insert = next(
            (i for i, s in enumerate(starts) if s > block.start),
            len(self.blocks),
        )
        prev_block = self.blocks[index_insert - 1] if index_insert > 0 else None
        next_block = self.blocks[index_insert] if index_insert < len(self.blocks) else None
        if prev_block and prev_block.overlaps(block):
            raise ValueError("New block overlaps with the previous block.")
        if next_block and next_block.overlaps(block):
            raise ValueError("New block overlaps with the next block.")
        self.blocks.insert(index_insert, block)

    def to_wig(self, handle: IO[str]) -> None:
        """Write the collection in wiggle format to a file handle."""
        if self.track_definition is not None:
            handle.write(self.track_definition.to_wig())
            handle.write("\n")
        for i, block in enumerate(self.blocks):
            if i > 0 or self.track_definition is not None:
                handle.write("\n")
            handle.write(block.to_wig())
        handle.write("\n")

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index: int) -> WiggleBlock:
        return self.blocks[index]

    def as_bed3(self) -> tuple[tuple[str, int, int], ...]:
        """Return block ranges as BED3 tuples (0-based half-open)."""
        if not self.blocks:
            return tuple()
        return tuple((block.chrom, block.start - 1, block.stop) for block in self.blocks)


# ============================================================================
# Wiggle Reader
# ============================================================================


def _parse_value(s: str) -> int | float:
    """Parse a numeric string as int if possible, otherwise float."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def _build_block(
    block_type: str, params: dict[str, str], data_lines: list[str],
) -> WiggleBlock:
    """Construct a WiggleBlock from a parsed declaration and its data lines."""
    if block_type == "fixedStep":
        return FixedStepBlock(
            chrom=params["chrom"],
            start=int(params["start"]),
            step=int(params.get("step", "1")),
            span=int(params.get("span", "1")),
            values=tuple(_parse_value(line) for line in data_lines),
        )
    if block_type == "variableStep":
        positions = []
        values = []
        for line in data_lines:
            parts = line.split()
            positions.append(int(parts[0]))
            values.append(_parse_value(parts[1]))
        return VariableStepBlock(
            chrom=params["chrom"],
            span=int(params.get("span", "1")),
            positions=tuple(positions),
            values=tuple(values),
        )
    raise ValueError(f"Unknown block type: {block_type}")


def read_wig(handle: IO[str]) -> Iterator[WiggleBlock]:
    """Read wiggle blocks from a text file handle.

    Yields WiggleBlock instances (FixedStepBlock or VariableStepBlock).
    Track definition and browser lines are skipped.

    The handle must be opened in text mode (e.g. ``open(path)`` or
    ``gzip.open(path, 'rt')``).
    """
    block_type: str | None = None
    block_params: dict[str, str] = {}
    data_lines: list[str] = []

    for raw_line in handle:
        line = raw_line.strip()

        if not line or line.startswith("#") or line.startswith("browser "):
            continue

        if line.startswith("track "):
            # Flush any pending block before a new track section.
            if block_type is not None and data_lines:
                yield _build_block(block_type, block_params, data_lines)
                block_type = None
                data_lines = []
            continue

        if line.startswith("fixedStep ") or line.startswith("variableStep "):
            # Flush previous block.
            if block_type is not None and data_lines:
                yield _build_block(block_type, block_params, data_lines)

            # Parse the declaration line.
            parts = line.split()
            block_type = parts[0]
            block_params = {}
            for part in parts[1:]:
                key, val = part.split("=", 1)
                block_params[key] = val
            data_lines = []
        else:
            data_lines.append(line)

    # Flush the last block.
    if block_type is not None and data_lines:
        yield _build_block(block_type, block_params, data_lines)


# ============================================================================
# BigWig Utilities
# ============================================================================


class LazyLoaderBigWig:
    """Dict like object with depth >= 1 that loads bigwig files from an unformatted filepath.

    The filepath is an unformatted string that should contain a set of {KEYS}.
    NOTE: the keys' order will be used to define the structure of the dict. i.e.
    "path/{key1}/{key2}/key3.bw" will be stored under LazyLoaderBigWig["{key1}{sep}{key2}{sep}{key3}"]

    The keys should be associated with a list of possible values, defined in the
    expected_formatting_keyvalues argument.
    E.g. `{'key1': ['value1', 'value2'], 'key2': ['value3', 'value4']}`

    """

    def __init__(
        self,
        ufmt_filepath: str,
        expected_formatting_keyvalues: dict[str, list[str]],
        key_separator: str = "/",
    ):
        # Check that all expected keys are in the string to be formatted later.
        # Expected keys: {RBP_CT}, {STRAND_STR}

        # Ordered list of strings corresponding to keys to format in the filepath.
        if "{" not in ufmt_filepath or "}" not in ufmt_filepath:
            raise ValueError(f"Filepath must contain keys in the format {{key}}: {ufmt_filepath}")

        detected_keys: list[str] = []
        for i in string.Formatter().parse(ufmt_filepath):
            if i[1] is not None and i[1] not in detected_keys:
                detected_keys.append(i[1])

        # `expected_formatting_keyvalues` provides with the expected list of values for each key.
        if not all([key in expected_formatting_keyvalues for key in set(detected_keys)]):
            raise KeyError(
                f"Detected keys: {detected_keys} are not all associated to a "
                f"list of values in the expected_formatting_keyvalues: {expected_formatting_keyvalues}"
            )

        # The lazy-loader object can be queried as a dict with `lazyloader['key1{key_separator}key2...']`
        # but we need to make sure that the key separator is not part of the key strings.
        for key, values in expected_formatting_keyvalues.items():
            if any([key_separator in value for value in values]):
                raise ValueError(
                    f"The key separator '{key_separator}' is part of one of the values of key={key}"
                )

        self._key_separator = key_separator
        self._expected_formatting_keyvalues = expected_formatting_keyvalues
        self._detected_keys = detected_keys
        self._ufmt_filepath = ufmt_filepath
        self._bigwig = {}
        self._expected_key_format: str = self._key_separator.join(
            map(lambda v: "{" + v + "}", self._detected_keys)
        )

    # NOTE: Old version supporting nested dict. Keeping for reference.
    # def __getitem__(self, query) -> Dict:
    #    if query not in self._expected_formatting_keyvalues[0]:
    #        raise ValueError("Parent key should correspond to the first formatting field of the path.")

    #    if query not in self._bigwig:
    #        # Load all the bigwig files for this RBP_CT
    #        self._bigwig[query] = {}

    #        for children_key in self._detected_keys[1:]:
    #            for value in self._expected_formatting_keyvalues[children_key]:
    #                self._bigwig[query][children_key] = {}

    #        for signal in self.SIGNAL_LIST:
    #            self._bigwig[key][signal] = {}
    #            for strand_str in self.STRAND_STR_LIST:
    #                filepath = self.ufmt_filepath.format(
    #                    RBP_CT=key, SIGNAL=signal, STRAND_STR=strand_str
    #                )
    #                if not os.path.exists(filepath):
    #                    raise FileNotFoundError(f"File not found: {filepath}")
    #                self._bigwig[key][signal][strand_str] = pbw.open(filepath)
    #    return self._bigwig[key]

    @property
    def expected_key_format(self) -> str:
        """Return the expected key format for the lazy loader."""
        return self._expected_key_format

    def keys(self) -> list[str]:
        """Return the list of keys *currently loaded* in the lazy loader."""
        return list(self._bigwig.keys())

    @property
    def expected_keys(self) -> list[str]:
        """Return the list of expected keys for the lazy loader."""
        return self._detected_keys

    @property
    def expected_formatting_keyvalues(self) -> dict[str, list[str]]:
        """Return the expected formatting keyvalues for the lazy loader."""
        return self._expected_formatting_keyvalues

    def close(self):
        # Close all the bigwig files
        for key in self._bigwig:
            self._bigwig[key].close()

    def __getitem__(self, query: str) -> pbw.pyBigWig:
        key_values = query.split(self._key_separator)
        if not len(key_values) == len(self._detected_keys):
            raise ValueError(
                f"Query should contain {len(self._detected_keys)} keys, but got {len(key_values)}"
            )
        for key, key_value in zip(self._detected_keys, key_values):
            if key_value not in self._expected_formatting_keyvalues[key]:
                raise ValueError(
                    f"Query value '{key_value}' is not in the expected values for key {key}."
                )

        # Check if the bigwig file is already loaded
        if query not in self._bigwig:
            # Load the bigwig file

            filepath = self._ufmt_filepath.format(
                **{key: key_value for key, key_value in zip(self._detected_keys, key_values)},
            )

            if "*" in str(filepath):
                # NOTE: This is a bit of a hack to allow for wildcard path when the filename is not
                # consistent across all keys.
                detected_filepaths = list(glob.glob(filepath))
                if len(detected_filepaths) == 0:
                    raise FileNotFoundError(f"No files found for pattern: {filepath}")
                elif len(detected_filepaths) > 1:
                    raise ValueError(f"Multiple files found for pattern: {filepath}")
                else:
                    filepath = detected_filepaths[0]

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            self._bigwig[query] = pbw.open(filepath)

        return self._bigwig[query]

    def __del__(self):
        # Close all the bigwig files
        self.close()
        print("Closed all bigwig files.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
