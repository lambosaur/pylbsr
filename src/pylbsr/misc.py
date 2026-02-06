
import glob
import gzip
import logging
import os
import random
import string
import tempfile
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import bgzf
from dotmap import DotMap


def sanitize_dotmap(dm: DotMap) -> DotMap:
    """Recursively removes all keys starting with '_' from a DotMap.

    Owing to tab completion features in REPL environments, DotMaps may get populated with
    keys starting with '_'.  This function removes them all recursively.

    Args:
        dm (dotmap.DotMap): The DotMap to sanitize.

    Return:
        dotmap.DotMap: The sanitized DotMap.
    """
    keys_to_delete = [k for k in dm if k.startswith("_")]
    for k in keys_to_delete:
        del dm[k]
    for k, v in dm.items():
        if isinstance(v, DotMap):
            sanitize_dotmap(v)
    return dm


def set_seed(seed: int = 42) -> None:
    """Set seed to all possible random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Seed set to {seed}")


def create_randomized_tmp_dir(parent_dir: os.PathLike | None = None) -> str:
    """Create a randomized temporary directory."""
    # Get the parent tmp dir where to create a randomized tmp dir.

    if parent_dir is not None:
        parent_tmp_dir = Path(parent_dir)
        # Make the directory
        if not parent_tmp_dir.exists():
            parent_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Default system value
    if parent_tmp_dir is None:
        parent_tmp_dir = Path(tempfile.gettempdir())

    if "params_tmp_dir" in globals():
        params_tmp_dir = globals()["params_tmp_dir"]
        if params_tmp_dir is not None:
            parent_tmp_dir = Path(params_tmp_dir)

    if os.getenv("TMP_DIR") is not None:
        parent_tmp_dir = Path(os.getenv("TMP_DIR"))

    if os.getenv("TMP") is not None:
        parent_tmp_dir = Path(os.getenv("TMP"))

    if parent_tmp_dir is None:
        raise ValueError("No temporary directory specified or found.")

    # Test write access to the parent tmp dir
    if not parent_tmp_dir.is_dir():
        raise NotADirectoryError(f"Parent tmp dir is not a directory: {parent_tmp_dir}")

    if not os.access(parent_tmp_dir, os.W_OK):
        raise PermissionError(f"Parent tmp dir is not writable: {parent_tmp_dir}")

    # Make a randomized tmp dir
    tmp_dir = tempfile.mkdtemp(dir=parent_tmp_dir)
    return tmp_dir


def init_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Initialize or retrieve a logger that works well in Jupyter notebooks.

    Avoids duplicate handlers across cells.

    Args:
        name (str): Name of the logger.
        level (int | str): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # prevent adding multiple handlers on re-run
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def resolve_path(filepath: str | Path, base: Path | None = None) -> Path:
    """Resolve a file path to an absolute Path, handling relative and absolute paths.

    Args:
        filepath: Path string or Path object, absolute or relative.
        base: Base path to resolve relative paths against (default: cwd).

    Returns:
        Path: Absolute resolved path.
    """
    filepath = Path(filepath)
    base = Path(base) if base is not None else Path.cwd()

    if filepath.is_absolute():
        return filepath.resolve()
    else:
        return (base / filepath).resolve()


def extract_fields_from_formatted_string(
    unformatted_string: str, formatted_string: str
) -> dict[str, str]:
    """Extract values from candidate string based on format pattern.

    Given a format pattern with fields (e.g. "/path/to/data/{sample_id}/{run_id}/results.txt")
    and a candidate string (e.g. "/path/to/data/SAMPLE123/RUN456/results.txt"),
    this function extracts the values corresponding to each field.

    The candidate string must match the pattern structure and serves to fill in the fields.

    Essentially the function can be used to reverse the formatting operation, i.e. given
    a pattern and a formatted string, it retrieves the values used to format the string.

    Args:
        unformatted_string (str): The format pattern with fields.
        formatted_string (str): The candidate string to extract values from.

    Returns:
        dict[str, str]: A dictionary mapping field names to their extracted values.

    """
    parts = list(string.Formatter().parse(unformatted_string))
    values = {}
    pos = 0

    for i, (lit, field, _, _) in enumerate(parts):
        # match literal
        if lit:
            assert formatted_string.startswith(lit, pos), f"literal {lit!r} not found"
            pos += len(lit)

        if field:
            # next literal (if any)
            next_lit = None
            for j in range(i + 1, len(parts)):
                if parts[j][0]:  # has literal
                    next_lit = parts[j][0]
                    break

            if next_lit:
                end = formatted_string.index(next_lit, pos)
                values[field] = formatted_string[pos:end]
                pos = end
            else:
                # last field consumes the rest
                values[field] = formatted_string[pos:]
                pos = len(formatted_string)

    return values


def glob_wildcards(unformatted_filepath: str) -> dict[str, list[str]]:
    """Return all matching values for each field in the unformatted filepath pattern.

    Given a string representing an unformatted filepath with fields (e.g.
    "/path/to/data/{sample_id}/{run_id}/results.txt"), this function finds all
    matching file paths and extracts the values for each field.

    The result is a dictionary where each key is a field name and the value
    is a list of all unique values found for that field across the matching file paths.

    Args:
        unformatted_filepath (str): The unformatted filepath pattern with fields.

    Returns:
        dict[str, list[str]]: A dictionary mapping field names to lists of matching values.

    """
    parts = list(string.Formatter().parse(unformatted_filepath))
    fields = [f for _, f, _, _ in parts if f]
    glob_pattern = unformatted_filepath.format(**{f: "*" for f in fields})

    matches = {f: [] for f in fields}
    for filepath in glob.glob(glob_pattern):
        vals = extract_fields_from_formatted_string(unformatted_filepath, Path(filepath).as_posix())
        for f in fields:
            matches[f].append(vals[f])

    return matches


def get_open_func(filepath: os.PathLike) -> Callable:
    """Return the appropriate open function based on file extension."""
    # TODO: Check better way ; e.g. https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed

    if str(filepath).endswith(".gz"):
        return gzip.open
    elif str(filepath).endswith(".bgz"):
        return bgzf.open
    else:
        return open




def chunked(lst: Sequence, n: int) -> Iterable:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


@contextmanager
def no_scientific_notation(precision: int = 6) -> Generator[None, None, None]:
    """Suppress scientific notation temporarily for NumPy arrays and Pandas objects.

    Params:
        precision (int): Number of decimal places to display.
    """
    # Save old NumPy print options
    old_np_opts = np.get_printoptions()
    np.set_printoptions(suppress=True, precision=precision)

    # Save old Pandas float format
    old_pd_fmt = pd.get_option("display.float_format")
    pd.set_option("display.float_format", f"{{:.{precision}f}}".format)

    try:
        yield
    finally:
        # Restore old settings
        np.set_printoptions(**old_np_opts)
        pd.set_option("display.float_format", old_pd_fmt)
