"""Library for mapping genomic coordinates from a GFF to relative coordinates through a chain file."""

from collections.abc import Callable, Iterable
from numbers import Number
from typing import Literal, TextIO

import pandas as pd
import pybedtools as pbt


class ChainRecord:
    """A class representing a single chain record in UCSC chain format."""

    def __init__(
        self,
        t_chrom: str,
        t_size: int,
        t_strand: str,
        t_start: int,
        t_end: int,
        q_chrom: str,
        q_size: int,
        q_strand: str,
        q_start: int,
        q_end: int,
        chain_id: int,
        score: Number,
    ) -> None:
        """Initialize a ChainRecord with the given parameters."""
        # Header fields as defined by UCSC chain format
        self.score = score
        self.t_chrom = t_chrom
        self.t_size = t_size
        self.t_strand = t_strand
        self.t_start = t_start
        self.t_end = t_end
        self.q_chrom = q_chrom
        self.q_size = q_size
        self.q_strand = q_strand
        self.q_start = q_start
        self.q_end = q_end
        self.chain_id = chain_id

        # Each block is (size,) or (size, dt, dq)
        self.blocks: list[tuple[int, ...]] = []

    def add_block(self, size: int, dt: int | None = None, dq: int | None = None) -> None:
        """Add a block to the chain.

        If dt and dq are None, it means that this is the last block and only size is provided.
        Otherwise, dt and dq represent the distance to the next block in the target and
        query sequences, respectively.

        Note: dt and dq are not provided for the last block, as they would be meaningless
        (no next block).

        The UCSC chain format allows for blocks to be represented with just size (for the last block)
        or with size, dt, dq (for all but the last block). This method handles both cases accordingly.

        Args:
            size (int): The size of the block (number of bases in the alignment).
            dt (int | None): The distance to the next block in the target sequence.
                                None if this is the last block.
            dq (int | None): The distance to the next block in the query sequence.
                                None if this is the last block.
        """
        # Last block has only size
        if dt is None or dq is None:
            self.blocks.append((size,))
        else:
            self.blocks.append((size, dt, dq))

    def header_as_dict(self) -> dict:
        """Return the header line of the chain record in UCSC format as a dictionary."""
        return {
            "chain": {
                "score": self.score,
                "t_chrom": self.t_chrom,
                "t_size": self.t_size,
                "t_strand": self.t_strand,
                "t_start": self.t_start,
                "t_end": self.t_end,
                "q_chrom": self.q_chrom,
                "q_size": self.q_size,
                "q_strand": self.q_strand,
                "q_start": self.q_start,
                "q_end": self.q_end,
                "chain_id": self.chain_id,
            }
        }

    @property
    def header(self) -> str:
        """Return the header line of the chain record in UCSC format."""
        return (
            f"chain {self.score} "
            f"{self.t_chrom} {self.t_size} {self.t_strand} {self.t_start} {self.t_end} "
            f"{self.q_chrom} {self.q_size} {self.q_strand} {self.q_start} {self.q_end} "
            f"{self.chain_id}"
        )

    def to_string(self) -> str:
        """Convert the ChainRecord to a string in UCSC chain format."""
        header = (
            f"chain {self.score} "
            f"{self.t_chrom} {self.t_size} {self.t_strand} {self.t_start} {self.t_end} "
            f"{self.q_chrom} {self.q_size} {self.q_strand} {self.q_start} {self.q_end} "
            f"{self.chain_id}"
        )

        block_lines = ["\t".join(map(str, block)) for block in self.blocks]

        return header + "\n" + "\n".join(block_lines) + "\n"



class MappingTable:
    """A class representing a mapping table between two sets of interval coordinates."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a MappingTable with a DataFrame containing the mapping information."""
        self.df = df.copy()  # one entity only

    def to_chain(self, target: str, query: str, chain_id: int, score: Number) -> ChainRecord:
        """Convert the mapping table to a UCSC chain format with the specified target and query.

        Per UCSC chain format specification:
        - Blocks must be in increasing order on BOTH query and query
        - When qStrand = "-": reverse-complement query coords AND reverse mapping within blocks
        - For genomic query: adjust q_size to make reverse-to-forward conversion a no-op

        Args:
            target: Which coordinate system becomes the query ('a' or 'b')
            query: Which coordinate system becomes the query ('a' or 'b')
            chain_id: Unique identifier for this chain
            score: Alignment score
        """
        assert target in {"a", "b"} and query in {"a", "b"} and target != query, (
            "target and query must be 'a' and 'b', and different from each other."
        )

        df = self.df.rename(
            columns={
                f"{target}.chrom": "tChrom",
                f"{target}.start": "tStart",
                f"{target}.end": "tEnd",
                f"{target}.strand": "tStrand",
                f"{query}.chrom": "qChrom",
                f"{query}.start": "qStart",
                f"{query}.end": "qEnd",
                f"{query}.strand": "qStrand",
            }
        ).copy()

        # Strands must be constant per entity
        t_strand = df["tStrand"].iloc[0]
        q_strand_orig = df["qStrand"].iloc[0]
        assert df["tStrand"].nunique() == 1, "query strand must be constant!"
        assert df["qStrand"].nunique() == 1, "Query strand must be constant!"

        # Sort by increasing query coordinates
        df = df.sort_values("tStart", ascending=True).reset_index(drop=True)

        # Determine if we need to reverse the mapping
        need_reverse = t_strand != q_strand_orig

        # Compute query size (needed for RC transformation)
        q_size_for_rc = int(df["qEnd"].max())

        # When we need to reverse, RC the query coordinates to make them increasing
        if need_reverse:
            df["qStart_rc"] = q_size_for_rc - df["qEnd"]
            df["qEnd_rc"] = q_size_for_rc - df["qStart"]
            df["qStart"] = df["qStart_rc"]
            df["qEnd"] = df["qEnd_rc"]

        # Compute final sizes
        t_size = int(df["tEnd"].max())

        # For genomic query with reversal: adjust q_size to make reverse-to-forward conversion a no-op
        if need_reverse and query == "a":  # Query is genomic 'a'
            # After RC, query coords are [0, q_size_for_rc)
            # When liftOver sees qStrand="-", it will do: qForward = q_size - qReverse
            # We want qForward to equal the ORIGINAL coords (before RC)
            # So: original_qStart = q_size - RC_qEnd
            #     original_qEnd = q_size - RC_qStart
            # This works when q_size = q_size_for_rc (which is the max of original qEnd)
            q_size = q_size_for_rc
        else:
            q_size = int(df["qEnd"].max())

        # Set query strand in the chain
        q_strand = "-" if need_reverse else "+"

        # Build header
        t_start = int(df["tStart"].min())
        t_end = int(df["tEnd"].max())
        q_start = int(df["qStart"].min())
        q_end = int(df["qEnd"].max())

        chain = ChainRecord(
            t_chrom=df["tChrom"].iloc[0],
            t_size=t_size,
            t_strand=t_strand,
            t_start=t_start,
            t_end=t_end,
            q_chrom=df["qChrom"].iloc[0],
            q_size=q_size,
            q_strand=q_strand,
            q_start=q_start,
            q_end=q_end,
            chain_id=chain_id,
            score=score,
        )

        # Build blocks - query coords are now in increasing order (after RC if needed)
        for i in range(len(df)):
            size = int(df.loc[i, "block_len"])
            if i == len(df) - 1:
                chain.add_block(size)
            else:
                dt = int(df.loc[i + 1, "tStart"] - df.loc[i, "tEnd"])
                dq = int(df.loc[i + 1, "qStart"] - df.loc[i, "qEnd"])
                chain.add_block(size, dt, dq)

        return chain



def _per_entity_processing_to_nool_bed6(
    gff: pd.DataFrame,
    entity_id_column: str,
    subset_type: list[str] | str,
) -> pd.DataFrame:
    """Convert GFF annotations to merged BED6 format for a single entity.

    Filters by feature type, converts to BED6 format, and merges overlapping intervals.

    Args:
        gff: GFF DataFrame for a single entity (e.g., one transcript)
        entity_id_column: Column name containing entity IDs
        subset_type: Feature type(s) to include (e.g., "exon", "CDS")

    Returns:
        Merged BED6 DataFrame with columns: chrom, start, end, name, score, strand
    """
    assert entity_id_column in gff.columns, f"Entity ID column '{entity_id_column}' not found in GFF!"
    assert "type" in gff.columns, "GFF must have a 'type' column!"

    # Subset on type
    subset_type_ = subset_type if isinstance(subset_type, list) else [subset_type]

    assert gff["type"].isin(subset_type_).any(), f"GFF must have annotations of type(s) {subset_type_}!"


    gff = gff.loc[lambda df: df["type"].isin(subset_type_)]
    if gff.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "strand",
                "name",
                "score",
                "strand",
            ]
        )

    selected_bed6_cols = ["seqid", "start", "end", entity_id_column, "score", "strand"]
    bed6 = gff[selected_bed6_cols].copy()
    bed6["start"] = bed6["start"] - 1  # Convert to 0-based coordinates
    bed6.columns = ["chrom", "start", "end", "name", "score", "strand"]

    assert bed6["name"].nunique() == 1, "GFF contains multiple entities with different IDs!"
    assert bed6["strand"].nunique() == 1, "GFF contains annotations on multiple strands!"
    assert bed6["chrom"].nunique() == 1, "GFF contains annotations on multiple chromosomes!"

    # Merge overlapping intervals.
    # First sort, then merge.
    bed6 = bed6.sort_values("start", ascending=True).reset_index(drop=True)
    bed6_merged = pbt.BedTool.from_dataframe(bed6).merge().to_dataframe().reset_index(drop=True)
    bed6_merged.columns = [
        "chrom",
        "start",
        "end",
    ]
    bed6_merged["name"] = bed6["name"].values[0]
    bed6_merged["score"] = "."
    bed6_merged["strand"] = bed6["strand"].values[0]

    # Make sure to sort
    bed6_merged = bed6_merged.sort_values("start", ascending=True).reset_index(drop=True)

    return bed6_merged


def _assign_entity_query_coords(bed: pd.DataFrame) -> pd.DataFrame:
    """Assign entity-relative coordinates to genomic intervals.

    Computes cumulative coordinates in transcription order (5' to 3'),
    accounting for strand direction. Returns a mapping table between
    genomic (a) and entity-relative (b) coordinate systems.

    Args:
        bed: BED6 DataFrame with genomic intervals for one entity

    Returns:
        Mapping table with columns: a.chrom, a.start, a.end, a.strand,
        b.chrom, b.start, b.end, b.strand, block_len
    """
    # bed is the merged, per-entity BED with columns:
    # chrom, start, end, strand

    # Assert that the bed is sorted.
    assert bed["start"].is_monotonic_increasing or bed["end"].is_monotonic_decreasing, (
        "BED must be sorted by start or end coordinates!"
    )
    # Assert that the bed has non-overlapping intervals.
    assert not ((bed["start"].shift(-1) < bed["end"]).fillna(False)).any(), (
        "BED must have non-overlapping intervals!"
    )

    if bed.empty:
        bed["q_start"] = []
        bed["q_end"] = []
        return bed

    strand = bed["strand"].iloc[0]

    # Order blocks by transcription order, not genomic order
    if strand == "+":
        bed = bed.sort_values("start", ascending=True)
    else:
        # For negative strand, transcription runs from high to low coordinates
        bed = bed.sort_values("end", ascending=False)

    # Block length in genomic coordinates (half-open BED)
    bed["block_len"] = bed["end"] - bed["start"]

    # Cumulative entity-relative coordinates
    # q_start is the sum of previous block lengths
    bed["q_start"] = bed["block_len"].cumsum().shift(fill_value=0)
    bed["q_end"] = bed["q_start"] + bed["block_len"]

    # Finally, create a mapping table with columns:
    # "a.chrom", "a.start", "a.end", "a.strand", "b.chrom", "b.start", "b.end", "b.strand", "block_len"
    # Here by default b.strand is "+".
    mapping_table = bed.copy()
    mapping_table = mapping_table.rename(
        columns={
            "chrom": "a.chrom",
            "start": "a.start",
            "end": "a.end",
            "strand": "a.strand",
            "block_len": "block_len",
            "name": "b.chrom",
            "q_start": "b.start",
            "q_end": "b.end",
        }
    )
    mapping_table["b.strand"] = "+"
    mapping_table = mapping_table[
        [
            "a.chrom",
            "a.start",
            "a.end",
            "a.strand",
            "b.chrom",
            "b.start",
            "b.end",
            "b.strand",
            "block_len",
        ]
    ]

    mapping_table = mapping_table.sort_values("a.start", ascending=True).reset_index(drop=True)

    return mapping_table



# Define a function signature that takes a mapping table and returns a float.
ScoreFunction = Callable[[pd.DataFrame], float]


def gff_to_chains(
    gff: pd.DataFrame,
    entity_id_column: str,
    subset_type: list[str] | str,
    target: Literal["seqid", "entity_id"],
    query: Literal["seqid", "entity_id"],
    scoring_fun: ScoreFunction | None = None,
) -> Iterable[ChainRecord]:
    """Convert a GFF to UCSC chain format records (lazy iterator).

    On Chain Format naming of "query" and "target": the target is the coordinate system that is then
    used as input to liftOver to retrieve coordinates in the "query" coordinate system.

    Args:
        gff: The input GFF as a DataFrame. This can be derived from the ExtendedGFF parsing of the GFF
                data structure so as to provide access to additional columns (e.g. "transcript_id").
        entity_id_column: The column in the GFF that contains the entity ID (e.g., transcript ID).
        subset_type: The type(s) of features to subset on (e.g., "exon", "CDS").
        target: Which coordinate system becomes the target ('seqid' or 'entity_id')
        query: Which coordinate system becomes the query ('seqid' or 'entity_id')
        scoring_fun: Optional function to compute chain score from mapping table. Defaults to 0.

    Yields:
        ChainRecord objects, one per entity (e.g., one per transcript).

    Example:
        >>> chains = gff_to_chains(gff_df, "transcript_id", "exon", "seqid", "entity_id")
        >>> with open("output.chain", "w") as f:
        ...     write_chains(chains, f)
    """
    assert entity_id_column in gff.columns, f"Entity ID column '{entity_id_column}' not found in GFF!"
    assert "type" in gff.columns, "GFF must have a 'type' column!"

    subset_type_ = subset_type if isinstance(subset_type, list) else [subset_type]

    assert gff["type"].isin(subset_type_).any(), f"GFF must have annotations of type(s) {subset_type_}!"
    assert target in {"seqid", "entity_id"} and query in {"seqid", "entity_id"} and target != query, (
        "target and query must be 'seqid' and 'entity_id', and different from each other."
    )

    # Map the "seqid" and "entity_id" literals to the actual column names in the mapping table
    _target = "a" if target == "seqid" else "b"
    _query = "b" if query == "entity_id" else "a"

    grouped_gff = gff.groupby(entity_id_column)
    for idx_entity, (entity_id, group) in enumerate(grouped_gff):
        bed6 = _per_entity_processing_to_nool_bed6(group, entity_id_column, subset_type)
        mapping_table = _assign_entity_query_coords(bed6)
        score = scoring_fun(mapping_table) if scoring_fun is not None else 0
        chain = MappingTable(mapping_table).to_chain(
            target=_target,
            query=_query,
            chain_id=idx_entity,
            score=score
        )
        yield chain


def write_chains(chains: Iterable[ChainRecord], handle: TextIO) -> None:
    """Write chain records to a file handle in UCSC chain format.

    Args:
        chains: Iterable of ChainRecord objects to write
        handle: File handle or file-like object (must be opened in text mode)

    Example:
        >>> chains = gff_to_chains(gff_df, "transcript_id", "exon", "seqid", "entity_id")
        >>> with open("output.chain", "w") as f:
        ...     write_chains(chains, f)

        >>> # Or with StringIO:
        >>> from io import StringIO
        >>> buffer = StringIO()
        >>> write_chains(chains, buffer)
        >>> result = buffer.getvalue()
    """
    for chain in chains:
        handle.write(chain.to_string())
        # IMPORTANT: separate chains by a blank line
        handle.write("\n")
