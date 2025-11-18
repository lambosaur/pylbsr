import os
from collections.abc import Callable, Iterator

from Bio import SeqIO

from ..misc import get_open_func


def fasta_chunk_iterator(
    fasta_filepath: os.PathLike,
    chunk_size: int
) -> Iterator[list[SeqIO.SeqRecord]]:
    """Generator to yield chunks of seq-records from a FASTA file."""
    open_func: Callable = get_open_func(fasta_filepath)
    # TODO: better way to decide of the access mode?
    access_mode = "rt" if str(fasta_filepath).endswith(".gz") else "r"

    with open_func(fasta_filepath, access_mode) as f:
        chunk: list[SeqIO.SeqRecord] = []
        for record in SeqIO.parse(f, "fasta"):
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk: list[SeqIO.SeqRecord] = []
        if chunk:
            yield chunk
