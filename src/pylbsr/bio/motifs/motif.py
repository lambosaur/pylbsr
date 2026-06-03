# /usr/bin/env python3
# Documentation on motif formats:
# - TRANSFAC (MEME): https://meme-suite.org/meme/doc/transfac-format.html
# - MEME format: https://meme-suite.org/meme/doc/meme-format.html

from __future__ import annotations

import collections
import dataclasses
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from typing import Iterator, TextIO

import Bio.motifs
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Common alphabet constants — useful for comparison and as canonical references.
ALPHABET_DNA: tuple[str, ...] = ("A", "C", "G", "T")
ALPHABET_RNA: tuple[str, ...] = ("A", "C", "G", "U")

# Mapping between our matrix_type vocabulary and logomaker's type strings.
_MATRIX_TYPE_TO_LOGOMAKER: dict[str, str] = {
    "PCM": "counts",
    "PPM": "probability",
    "PWM": "weight",
    "ICM": "information",
}


class MotifError(Exception):
    """Raised for errors in motif parsing, validation, or conversion."""


@dataclass
class _ParsedMatrixLine:
    """Parsed components of a single TRANSFAC matrix line (P0 header or NN data row)."""

    key: str
    values: tuple[str, ...]
    consensus: str | None


@dataclass
class _ParsedMetadataLine:
    """Parsed key-value pair from a TRANSFAC metadata line."""

    key: str
    value: str


@dataclass
class Motif:
    """Representation of a transcription factor binding site motif.

    The position weight matrix is stored as a pandas DataFrame whose columns are
    the alphabet letters (parsed from the TRANSFAC P0 line) and whose index uses
    the TRANSFAC "01", "02", ... position labels.

    Args:
        id: Motif identifier.
        matrix: Position weight matrix; columns = alphabet letters, index = "01"…"NN".
        metadata: Header and footer metadata lines keyed by "header" / "footer".
        consensus: Optional consensus sequence parsed from matrix rows.
        matrix_type: Matrix type hint — one of "PCM", "PPM", "PWM", "ICM", "CWM", or None.
            Not set by the parser (TRANSFAC does not encode this); assign explicitly when known.
    """

    id: str
    matrix: pd.DataFrame
    metadata: dict[str, list[_ParsedMetadataLine]] = field(
        default_factory=lambda: defaultdict(list)
    )
    consensus: str | None = None
    matrix_type: str | None = None

    def __post_init__(self) -> None:
        if self.matrix.empty:
            raise MotifError("Dataframe matrix is empty.")

        if list(self.matrix.index)[0] != "01":
            raise MotifError(
                f"Motif PWM index should start at '01'; got {list(self.matrix.index)[0]!r}"
            )

        if "header" not in self.metadata:
            raise MotifError("Expected metadata list 'header' not found.")

        if "footer" not in self.metadata:
            raise MotifError("Expected metadata list 'footer' not found.")

    @property
    def alphabet(self) -> tuple[str, ...]:
        """Alphabet letters derived from matrix column names."""
        return tuple(self.matrix.columns)

    def get_metadata(
        self, key: str, subset: str | None = None
    ) -> defaultdict[str, list[_ParsedMetadataLine]]:
        """Return metadata lines matching *key*, optionally restricted to *subset*.

        Args:
            key: Metadata key to search for (e.g. "ID", "AC").
            subset: If given, restrict to "header" or "footer".

        Returns:
            defaultdict mapping subset name to matched lines.

        Raises:
            ValueError: if *subset* is not "header" or "footer".
        """
        subsets = ["header", "footer"]
        if subset is not None:
            if subset not in subsets:
                raise ValueError(f"subset must be 'header' or 'footer'; got {subset!r}")
            subsets = [subset]

        found: defaultdict[str, list[_ParsedMetadataLine]] = defaultdict(list)
        for s in subsets:
            for line in self.metadata[s]:
                if line.key == key:
                    found[s].append(line)
        return found

    def set_metadata(self, key: str, subset: str, value: str) -> None:
        """Update the value of a unique metadata entry identified by *key*.

        Args:
            key: Metadata key (must match exactly one line in *subset*).
            subset: "header" or "footer".
            value: New value to assign.

        Raises:
            ValueError: if zero or more than one line matches *key* in *subset*.
        """
        found = self.get_metadata(key, subset=subset)[subset]
        if len(found) == 0:
            raise ValueError(f"No metadata line with key {key!r} in {subset!r}.")
        if len(found) > 1:
            raise ValueError(
                f"Multiple metadata lines ({len(found)}) with key {key!r} in {subset!r}."
            )
        idx = self.metadata[subset].index(found[0])
        self.metadata[subset][idx].value = value

    def add_metadata(self, key: str, subset: str, value: str) -> None:
        """Append a new metadata line to *subset*.

        Args:
            key: Metadata key.
            subset: "header" or "footer".
            value: Metadata value.

        Raises:
            ValueError: if *subset* is not "header" or "footer".
        """
        if subset not in self.metadata:
            raise ValueError(f"subset must be 'header' or 'footer'; got {subset!r}")
        self.metadata[subset].append(_ParsedMetadataLine(key=key, value=value))

    def relabel_alphabet(self, mapping: dict[str, str]) -> Motif:
        """Return a new Motif with alphabet letters renamed and/or reordered.

        *mapping* must be a bijection on the current alphabet.
        Dict insertion order determines the new column order.

        Example — DNA to RNA::

            motif.relabel_alphabet({"A": "A", "C": "C", "G": "G", "T": "U"})

        Args:
            mapping: Dict mapping each current letter to its new name.

        Returns:
            New Motif with renamed/reordered columns.

        Raises:
            MotifError: if mapping keys don't match the current alphabet,
                or if the mapping is not injective (duplicate values).
        """
        if set(mapping.keys()) != set(self.alphabet):
            raise MotifError(
                f"mapping keys {set(mapping.keys())} != current alphabet {set(self.alphabet)}"
            )
        if len(set(mapping.values())) != len(mapping):
            raise MotifError("mapping is not injective (duplicate target letters).")
        new_matrix = self.matrix.rename(columns=mapping)[list(mapping.values())]
        return dataclasses.replace(self, matrix=new_matrix)

    def to_logomaker_df(self) -> pd.DataFrame:
        """Return the matrix as a logomaker-compatible DataFrame.

        logomaker expects an integer index named "pos".

        Returns:
            DataFrame with RangeIndex named "pos" and numeric values.
        """
        df = self.matrix.copy()
        df.index = pd.RangeIndex(len(df), name="pos")
        return df


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------


def _parse_transfac_metadata_line(line: str, key_value_separator: str) -> _ParsedMetadataLine:
    """Parse a TRANSFAC metadata line into a key-value pair.

    Args:
        line: A line in the format "{KEY}{key_value_separator}{VALUE}".
        key_value_separator: Separator between key and value.

    Returns:
        _ParsedMetadataLine with fields "key" and "value".
    """
    stripped = line.strip("\n")
    if key_value_separator in stripped:
        key, value = stripped.split(key_value_separator, maxsplit=1)
    else:
        # Line has no value (e.g. bare section marker like "XL"); store empty value.
        key, value = stripped, ""
    return _ParsedMetadataLine(key=key, value=value)


def _parse_transfac_matrix_line(
    line: str,
    matrix_key_value_separator: str,
    matrix_value_content_separator: str,
    alphabet_size: int | None = None,
) -> _ParsedMatrixLine:
    """Parse a single TRANSFAC matrix line (P0 header or NN data row).

    For the P0/PO line, *alphabet_size* must be None — the values define the alphabet.
    For data rows, *alphabet_size* must be provided to separate matrix values from the
    optional trailing consensus letter.

    Args:
        line: Matrix line in the format "{KEY}{matrix_key_value_separator}{VALUE...}".
        matrix_key_value_separator: Separator between the row key and its values.
        matrix_value_content_separator: Separator between individual values.
        alphabet_size: Number of alphabet letters; required for NN rows. Defaults to None.

    Returns:
        _ParsedMatrixLine with fields "key", "values", and "consensus".

    Raises:
        ValueError: if *alphabet_size* is missing for NN rows, or value count is unexpected.
    """
    key, value = line.strip("\n").split(matrix_key_value_separator, maxsplit=1)
    content = [x for x in value.split(matrix_value_content_separator) if x != ""]

    if key in ("P0", "PO"):
        return _ParsedMatrixLine(key=key, values=tuple(content), consensus=None)

    if not alphabet_size:
        raise ValueError("alphabet_size must be provided for NN (data) lines.")

    if len(content) not in (alphabet_size, alphabet_size + 1):
        raise ValueError(
            f"Expected {alphabet_size} (+1 optional consensus) values; "
            f"got {len(content)}: {content!r}  (line: {line!r})"
        )

    consensus = content[-1] if len(content) == alphabet_size + 1 else None
    return _ParsedMatrixLine(key=key, values=tuple(content[:alphabet_size]), consensus=consensus)


def _parse_transfac_matrix_lines(
    lines: list[str],
    matrix_key_value_separator: str,
    matrix_value_content_separator: str,
) -> list[_ParsedMatrixLine]:
    """Parse all lines of a TRANSFAC matrix section.

    The first line must be the P0/PO header; its value count defines the alphabet size.
    Subsequent lines are data rows, each optionally ending with a consensus letter.

    Args:
        lines: Lines corresponding to the motif matrix (P0 header + NN rows).
        matrix_key_value_separator: Separator between row key and values.
        matrix_value_content_separator: Separator between values within a row.

    Returns:
        list[_ParsedMatrixLine]: parsed header followed by data rows.

    Raises:
        ValueError: if fewer than two lines are given, the first line is not P0/PO,
            or alphabet size is zero.
    """
    if len(lines) == 0:
        raise ValueError("No lines provided for parsing.")
    if len(lines) == 1:
        raise ValueError("Only one line provided (need at least P0 + one data row).")
    if not (lines[0].startswith("P0") or lines[0].startswith("PO")):
        raise ValueError("First line must be the P0/PO header line.")

    parsed_header = _parse_transfac_matrix_line(
        line=lines[0],
        matrix_key_value_separator=matrix_key_value_separator,
        matrix_value_content_separator=matrix_value_content_separator,
        alphabet_size=None,
    )

    alphabet_size = len(parsed_header.values)
    if alphabet_size == 0:
        raise ValueError(f"Alphabet size must be ≥ 1; parsed {alphabet_size} from P0 line.")

    parsed_lines: list[_ParsedMatrixLine] = [parsed_header]
    for line in lines[1:]:
        parsed_lines.append(
            _parse_transfac_matrix_line(
                line=line,
                matrix_key_value_separator=matrix_key_value_separator,
                matrix_value_content_separator=matrix_value_content_separator,
                alphabet_size=alphabet_size,
            )
        )
    return parsed_lines


def _group_transfac_motif_lines(
    lines: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Partition TRANSFAC motif lines into header, matrix, and footer sections.

    Rules:
    - Everything before "P0" or "PO" is header.
    - From "P0"/"PO" onward is matrix, until the first "XX" line.
    - From that "XX" line onward is footer.

    Args:
        lines: All lines attributed to a single TRANSFAC motif.

    Returns:
        Tuple of (header_lines, matrix_lines, footer_lines).
    """
    lines_header: list[str] = []
    lines_matrix: list[str] = []
    lines_footer: list[str] = []

    matrix_found = False
    footer_found = False

    for line in lines:
        if footer_found:
            lines_footer.append(line)
        elif matrix_found:
            if line[:2] == "XX":
                footer_found = True
                lines_footer.append(line)
            else:
                lines_matrix.append(line)
        else:
            if line[:2] in ("P0", "PO"):
                matrix_found = True
                lines_matrix.append(line)
            else:
                lines_header.append(line)

    return lines_header, lines_matrix, lines_footer


def _group_handle_lines_per_motif(lines: Iterator[str]) -> list[list[str]]:
    """Split an iterator of lines into per-motif groups using "//" as separator.

    Args:
        lines: Iterator over file lines.

    Returns:
        list[list[str]]: one sub-list per motif (includes the "//" terminator line).

    Raises:
        MotifError: if the last group of lines is not terminated by "//".
    """
    result: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        current.append(line)
        if line.startswith("//"):
            result.append(current)
            current = []

    if current:
        if not (len(current) == 1 and current[0] == "\n"):
            raise MotifError(f"Last motif is not terminated by '//' line: {current}")

    return result


# ---------------------------------------------------------------------------
# Public parsing / writing API
# ---------------------------------------------------------------------------


def parse_transfac_motif_lines(
    lines: list[str],
    key_value_separator: str = "  ",
    matrix_key_value_separator: str = "  ",
    matrix_value_content_separator: str = " ",
    rename_columns: dict | None = None,
    force_numeric: bool = False,
) -> Motif:
    """Parse a list of TRANSFAC-formatted lines for a single motif.

    Steps:
    1. Split into header, matrix, and footer sections.
    2. Parse each section.
    3. Assemble and return a Motif object.

    Args:
        lines: Lines for a single motif (as read from a TRANSFAC file).
        key_value_separator: Separator between KEY and VALUE in metadata lines. Defaults to "  ".
        matrix_key_value_separator: Separator between KEY and VALUE in matrix lines. Defaults to "  ".
        matrix_value_content_separator: Separator between values within a matrix line. Defaults to " ".
        rename_columns: Optional dict to rename columns parsed from the P0 line. Defaults to None.
        force_numeric: If True, raise MotifError when matrix values cannot be cast to numeric.
            If False (default), warn and keep values as strings. Defaults to False.

    Returns:
        Motif parsed from the given lines.

    Raises:
        ValueError: if the ID field is missing or duplicated, or motif construction fails.
        MotifError: if force_numeric is True and matrix values are non-numeric.
    """
    lines_header, lines_matrix, lines_footer = _group_transfac_motif_lines(lines)

    parsed_matrix_content = _parse_transfac_matrix_lines(
        lines_matrix, matrix_key_value_separator, matrix_value_content_separator
    )

    matrix = pd.DataFrame(
        [line_content.values for line_content in parsed_matrix_content[1:]],
        columns=parsed_matrix_content[0].values,
    )
    matrix.index = pd.Index([f"{i + 1:02d}" for i in range(matrix.shape[0])])

    if rename_columns is not None:
        matrix = matrix.rename(columns=rename_columns)

    # Cast to numeric; int64 for count matrices, float64 for frequency/weight matrices.
    try:
        matrix = matrix.apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        if force_numeric:
            raise MotifError(f"Matrix values could not be cast to numeric: {exc}") from exc
        logger.warning("Matrix values could not be cast to numeric; keeping as strings.")

    consensus_chars = [
        line_content.consensus if line_content.consensus else ""
        for line_content in parsed_matrix_content[1:]
    ]
    consensus: str | None = "".join(consensus_chars) or None

    metadata_header = [
        _parse_transfac_metadata_line(line=line, key_value_separator=key_value_separator)
        for line in lines_header
        if not line.startswith("XX")
    ]
    metadata_footer = [
        _parse_transfac_metadata_line(line=line, key_value_separator=key_value_separator)
        for line in lines_footer
        if not (line.startswith("XX") or line.startswith("//"))
    ]

    motif_ids = [m.value for m in metadata_header if m.key == "ID"]
    if len(motif_ids) != 1:
        raise ValueError(f"Expected exactly one ID in header; found: {motif_ids}")

    motif_id = motif_ids[0].strip()
    metadata: dict[str, list[_ParsedMetadataLine]] = {
        "header": metadata_header,
        "footer": metadata_footer,
    }

    try:
        return Motif(id=motif_id, matrix=matrix, metadata=metadata, consensus=consensus)
    except ValueError as exc:
        raise ValueError(f"Could not construct Motif from parsed lines: {exc}") from exc


def read_motifs_transfac(
    handle: TextIO,
    key_value_separator: str = "  ",
    matrix_key_value_separator: str = "  ",
    matrix_value_content_separator: str = " ",
    rename_columns: dict | None = None,
    strict: bool = False,
    force_numeric: bool = False,
) -> list[Motif]:
    """Read and parse all TRANSFAC-formatted motifs from a file handle.

    Motifs must be separated by "//" lines.

    Args:
        handle: Readable text handle (file or StringIO).
        key_value_separator: Separator for metadata lines. Defaults to "  ".
        matrix_key_value_separator: Separator for matrix lines. Defaults to "  ".
        matrix_value_content_separator: Separator between values in matrix lines. Defaults to " ".
        rename_columns: Optional column renaming dict. Defaults to None.
        strict: If True, raise MotifError when any motif fails to parse. Defaults to False.
        force_numeric: If True, raise MotifError when matrix values are not numeric.
            Defaults to False.

    Returns:
        list[Motif]: successfully parsed motifs.

    Raises:
        MotifError: if strict=True and any motif cannot be parsed.
    """
    all_motif_lines = _group_handle_lines_per_motif(iter(handle))
    total = len(all_motif_lines)
    failed_indices: list[int] = []
    motifs: list[Motif] = []

    for idx, motif_lines in enumerate(all_motif_lines):
        try:
            motif = parse_transfac_motif_lines(
                lines=motif_lines,
                key_value_separator=key_value_separator,
                matrix_key_value_separator=matrix_key_value_separator,
                matrix_value_content_separator=matrix_value_content_separator,
                rename_columns=rename_columns,
                force_numeric=force_numeric,
            )
            motifs.append(motif)
        except (ValueError, MotifError) as exc:
            logger.warning("Failed to parse motif at index %d: %s", idx, exc)
            failed_indices.append(idx)

    n_failed = len(failed_indices)
    if n_failed > 0:
        msg = f"Failed to parse {n_failed:,} of {total:,} motifs (indices: {failed_indices})."
        if strict:
            raise MotifError(msg)
        warnings.warn(msg)

    logger.debug("Parsed %d/%d motifs (%d failed).", len(motifs), total, n_failed)
    return motifs


def _write_matrix_transfac(
    handle: TextIO,
    matrix: pd.DataFrame,
    consensus: str | None = None,
    format_matrix_values: str = "{value}",
    matrix_key_value_separator: str = "  ",
    matrix_value_content_separator: str = " ",
) -> None:
    """Write the matrix section of a TRANSFAC motif to *handle*.

    Args:
        handle: Writable text handle.
        matrix: Position weight matrix DataFrame.
        consensus: Optional consensus string (length must match matrix row count).
        format_matrix_values: Python format string for each matrix value. Defaults to "{value}".
        matrix_key_value_separator: Separator between row key and values. Defaults to "  ".
        matrix_value_content_separator: Separator between values. Defaults to " ".

    Raises:
        ValueError: if consensus length does not match matrix row count.
    """
    if consensus and len(consensus) != matrix.shape[0]:
        raise ValueError(
            f"Consensus length {len(consensus)} != matrix row count {matrix.shape[0]}."
        )

    header = f"P0{matrix_key_value_separator}" + matrix_value_content_separator.join(
        matrix.columns.values
    )
    handle.write(header + "\n")

    for int_idx, (row_key, row) in enumerate(matrix.iterrows()):
        values_formatted = matrix_value_content_separator.join(
            format_matrix_values.format(value=v) for v in row.values
        )
        row_line = f"{row_key}{matrix_key_value_separator}{values_formatted}"
        if consensus:
            row_line += f"{matrix_value_content_separator}{consensus[int_idx]}"
        handle.write(row_line + "\n")


def write_motif_transfac(
    handle: TextIO,
    motif: Motif,
    format_matrix_values: str = "{value}",
    minimal: bool = False,
    drop_keys: list[str] | None = None,
    key_value_separator: str = "  ",
    matrix_value_content_separator: str = " ",
) -> None:
    """Write a Motif in TRANSFAC format to *handle*.

    Args:
        handle: Writable text handle.
        motif: Motif to serialise.
        format_matrix_values: Python format string applied to each matrix value.
            Must contain "{value". Defaults to "{value}".
        minimal: If True, write only the ID line + matrix (no other metadata). Defaults to False.
        drop_keys: Metadata keys to omit when writing. Defaults to None (keep all).
        key_value_separator: Separator between KEY and VALUE for all lines. Defaults to "  ".
        matrix_value_content_separator: Separator between matrix values. Defaults to " ".

    Raises:
        ValueError: if format_matrix_values does not contain "{value", or matrix is empty.
    """
    if "{value" not in format_matrix_values:
        raise ValueError("format_matrix_values must contain '{value'.")
    if motif.matrix.empty:
        raise ValueError("Motif matrix is empty.")

    drop_keys = drop_keys or []
    header_lines: list[str] = []
    footer_lines: list[str] = []

    if minimal:
        header_lines = [f"ID{key_value_separator}{motif.id}\n", "XX\n"]
        footer_lines = ["XX\n", "//\n"]
    else:
        for md in motif.metadata["header"]:
            if md.key not in drop_keys:
                header_lines.append(f"{md.key}{key_value_separator}{md.value}\n")
        header_lines.append("XX\n")

        footer_lines.append("XX\n")
        for md in motif.metadata["footer"]:
            if md.key not in drop_keys:
                footer_lines.append(f"{md.key}{key_value_separator}{md.value}\n")
        footer_lines.append("XX\n")
        footer_lines.append("//\n")

    handle.writelines(header_lines)
    _write_matrix_transfac(
        handle=handle,
        matrix=motif.matrix,
        format_matrix_values=format_matrix_values,
        matrix_key_value_separator=key_value_separator,
        matrix_value_content_separator=matrix_value_content_separator,
    )
    handle.writelines(footer_lines)
    logger.debug("Wrote motif %r.", motif.id)


def motif_to_biopython_motif(motif: Motif) -> Bio.motifs.Motif:
    """Convert a Motif to a Biopython Bio.motifs.Motif via TRANSFAC round-trip.

    The motif alphabet must be ALPHABET_DNA = ('A', 'C', 'G', 'T').
    For RNA or custom alphabets, use relabel_alphabet() to remap first.

    Args:
        motif: Source motif with DNA alphabet.

    Returns:
        Bio.motifs.Motif parsed from TRANSFAC-formatted output.

    Raises:
        MotifError: if the motif alphabet is not ('A', 'C', 'G', 'T').
    """
    if motif.alphabet != ALPHABET_DNA:
        raise MotifError(
            f"Biopython requires alphabet {ALPHABET_DNA}; got {motif.alphabet}. "
            "Use motif.relabel_alphabet() to remap first."
        )
    handle = StringIO()
    write_motif_transfac(handle=handle, motif=motif, minimal=True)
    handle.seek(0)
    return Bio.motifs.read(handle, "transfac")


# ---------------------------------------------------------------------------
# Matrix type conversion (logomaker adaptor)
# ---------------------------------------------------------------------------


def convert_matrix_type(
    motif: Motif,
    to_type: str,
    background: dict[str, float] | None = None,
    pseudocount: float = 1.0,
) -> Motif:
    """Convert the motif matrix to a different type using logomaker.transform_matrix().

    Supported types: "PCM" (counts), "PPM" (probability), "PWM" (weight), "ICM" (information).

    Args:
        motif: Source motif. If motif.matrix_type is set, it is used as from_type;
            otherwise "PCM" is assumed.
        to_type: Target matrix type — one of "PCM", "PPM", "PWM", "ICM".
        background: Per-letter background frequencies as a dict keyed by alphabet letters.
            Defaults to None (uniform background).
        pseudocount: Pseudocount added during count-to-probability conversion. Defaults to 1.0.

    Returns:
        New Motif with converted matrix and matrix_type set to to_type.

    Raises:
        MotifError: if to_type is not a recognised matrix type.
        ImportError: if logomaker is not installed.
    """
    import logomaker  # core dependency

    valid_types = list(_MATRIX_TYPE_TO_LOGOMAKER.keys())
    if to_type not in valid_types:
        raise MotifError(f"to_type must be one of {valid_types}; got {to_type!r}.")

    from_type_key = motif.matrix_type or "PCM"
    if from_type_key not in _MATRIX_TYPE_TO_LOGOMAKER:
        raise MotifError(
            f"motif.matrix_type {from_type_key!r} is not a recognised type. "
            f"Set it to one of {valid_types} before converting."
        )

    from_logomaker = _MATRIX_TYPE_TO_LOGOMAKER[from_type_key]
    to_logomaker = _MATRIX_TYPE_TO_LOGOMAKER[to_type]

    df = motif.to_logomaker_df()

    # Background must be a list ordered by alphabet if provided.
    bg: list[float] | None = None
    if background is not None:
        bg = [background[letter] for letter in motif.alphabet]

    converted = logomaker.transform_matrix(
        df,
        from_type=from_logomaker,
        to_type=to_logomaker,
        background=bg,
        pseudocount=pseudocount,
    )

    # Restore TRANSFAC-style string index ("01", "02", ...)
    converted.index = pd.Index([f"{i + 1:02d}" for i in range(len(converted))])

    return dataclasses.replace(motif, matrix=converted, matrix_type=to_type)


# ---------------------------------------------------------------------------
# Collection utilities (from former utils.py)
# ---------------------------------------------------------------------------


def relabel_motif_collection(
    motif_collection: list[Motif],
) -> list[Motif]:
    """Ensure unique IDs within a motif collection by appending numeric suffixes to duplicates.

    For each duplicate ID, the first occurrence keeps its original ID; subsequent occurrences
    are renamed to "{id}_{N}" (N = 2, 3, …). A CC comment recording the original ID is added
    to the footer of each renamed motif.

    Args:
        motif_collection: List of motifs (may contain duplicate IDs).

    Returns:
        List of motifs with unique IDs (same order as input).
    """
    relabeled: list[Motif] = []
    id_counts: collections.defaultdict[str, int] = collections.defaultdict(int)

    for motif in motif_collection:
        id_counts[motif.id] += 1

        if id_counts[motif.id] > 1:
            original_id = motif.id
            new_id = f"{original_id}_{id_counts[original_id]}"
            motif.id = new_id
            motif.set_metadata(key="ID", subset="header", value=new_id)
            motif.add_metadata(subset="footer", key="CC", value=f"Original duplicated ID: {original_id}")

        relabeled.append(motif)

    return relabeled


def merge_rename_motif_collections(
    motif_collections: dict[str, list[Motif]],
) -> list[Motif]:
    """Merge multiple named motif collections into one, prefixing each motif ID.

    Each motif is renamed to "{collection_id}.{motif.id}". If an AC (accession) field
    exists in the motif header, it is also prefixed.

    Args:
        motif_collections: Dict mapping collection name to list of motifs.

    Returns:
        Single flat list of motifs with unique, prefixed IDs.

    Raises:
        ValueError: if any duplicate IDs remain after prefixing.
    """
    all_motifs: list[Motif] = []

    for collection_id, motif_list in motif_collections.items():
        for motif in motif_list:
            new_id = f"{collection_id}.{motif.id}"
            motif.id = new_id
            motif.set_metadata(key="ID", subset="header", value=new_id)

            # Rename AC field if present (bug fix: header is a list of _ParsedMetadataLine,
            # not a dict — must search by .key attribute).
            ac_hits = [m for m in motif.metadata["header"] if m.key == "AC"]
            if ac_hits:
                motif.set_metadata(
                    key="AC",
                    subset="header",
                    value=f"{collection_id}.{ac_hits[0].value}",
                )

            all_motifs.append(motif)

    id_counts = collections.Counter(m.id for m in all_motifs)
    duplicates = [(k, v) for k, v in id_counts.items() if v > 1]
    if duplicates:
        raise ValueError(f"Duplicate motif IDs after merging: {duplicates}")

    return all_motifs
