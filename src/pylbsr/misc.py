"""General-purpose helpers not specific to any bioinformatics domain."""

import collections
import datetime
import functools
import glob
import gzip
import logging
import os
import random
import shutil
import string
import tempfile
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import bgzf
from dotmap import DotMap

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def recursive_defaultdict_to_dict(d: object) -> object:
    """Recursively convert nested defaultdicts to plain dicts.

    Plain dicts raise KeyError on missing keys instead of silently creating new
    nested defaultdicts.
    """
    if isinstance(d, collections.defaultdict):
        d = {k: recursive_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def tryint(s: str) -> int | str:
    """Convert `s` to int if possible, otherwise return it unchanged.

    Useful as a per-token key function for natural sorting.
    """
    try:
        return int(s)
    except ValueError:
        return s


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


def create_randomized_tmp_dir(parent_dir: os.PathLike | None = None) -> str:  # noqa: C901 -- barely over threshold, straightforward fallback chain
    """Create a randomized temporary directory."""
    # Get the parent tmp dir where to create a randomized tmp dir.
    parent_tmp_dir = None

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

    tmp_dir_env = os.getenv("TMP_DIR")
    if tmp_dir_env is not None:
        parent_tmp_dir = Path(tmp_dir_env)

    tmp_env = os.getenv("TMP")
    if tmp_env is not None:
        parent_tmp_dir = Path(tmp_env)

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

    matches: dict[str, list[str]] = {f: [] for f in fields}
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


def set_seed(seed: int = 42) -> None:
    """Seed Python's `random` and numpy's global RNG for reproducibility.

    Covers stdlib `random` and numpy's legacy global RNG -- e.g. seaborn's jitter/strip-plot
    offsets, `DataFrame.sample`, `np.random.choice`, or anything else drawing from either of
    these without its own explicit `Generator` instance. For torch-based reproducibility
    (including CUDA/cuDNN determinism), use `pylbsr.torch_utils.set_seed` instead, which
    calls this plus seeds torch.

    Args:
        seed: Seed value applied to both RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seed set to %d", seed)


def slice_range_overlapping(
    start: int, end: int, step: int, windowsize: int
) -> Iterator[tuple[int, int]]:
    """Yield overlapping (window_start, window_end) index windows tiling [start, end).

    Args:
        start: First window's start.
        end: Windows stop once one would extend past this.
        step: Distance between consecutive windows' starts.
        windowsize: Width of each window.

    Yields:
        (window_start, window_end) pairs, each windowsize wide.
    """
    for i in range(start, end - windowsize + step, step):
        yield (i, i + windowsize)


def make_experiment_outputdir(
    parent_path: Path | str,
    name: str | None = None,
    symlink_as_latest: bool = True,
    reuse: bool = False,
    replace: bool = False,
    increment: bool = False,
) -> Path:
    """Use or create a timestamped output directory, optionally with a "latest" symlink.

    Args:
        parent_path: Parent directory of the output directory to create.
        name: Explicit name for the output directory (e.g. an existing formatted-date
            name, to reuse it); defaults to the current timestamp (YYYY-mm-dd_HH:MM:ss).
        symlink_as_latest: Create/update a "latest" symlink to the output directory, in
            the same parent directory.
        reuse: Reuse an existing directory instead of raising, keeping its contents.
        replace: Remove and recreate an existing directory.
        increment: If the directory already exists and neither `reuse` nor `replace`
            apply, append a "_{i}" suffix (i = 1..99) instead of raising.

    Returns:
        The output directory (created or reused).

    Raises:
        FileExistsError: The directory already exists and none of `reuse`/`replace`/
            `increment` apply.
        ValueError: `increment` was set but all suffixes "_1".."_99" are already taken.
    """
    if name:
        outputdir = Path(parent_path) / name
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # noqa: DTZ005 -- local wall-clock time is the intent for a directory name, not an unambiguous instant
        outputdir = Path(parent_path, date)

    try:
        outputdir.mkdir(parents=True, exist_ok=reuse)
    except FileExistsError:
        if replace:
            shutil.rmtree(outputdir)
            outputdir.mkdir(parents=False, exist_ok=False)
        elif increment:
            i = 1
            dirname = outputdir.name
            while outputdir.exists() and i < 100:
                outputdir = Path(parent_path) / f"{dirname}_{i}"
                i += 1
            if outputdir.exists():
                raise ValueError(
                    f"Incremented dirname until '_99'; check your output parent "
                    f"directory (e.g. '{outputdir}' exists)."
                ) from None
            outputdir.mkdir(parents=False, exist_ok=False)
        else:
            raise FileExistsError(
                f"{outputdir} already exists; no clear instruction from reuse, replace, or increment."
            ) from None

    logger.info("Output directory: %s", outputdir)

    if symlink_as_latest:
        symlink = Path(parent_path) / "latest"
        if symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(outputdir.relative_to(symlink.parent))
        logger.info("Symlink: %s", symlink)

    return outputdir


def silent_try_convert(
    var: object, requested_type: Callable[..., object], return_val: object = None
) -> object:
    """Cast `var` to `requested_type`, returning `return_val` (or `var` unchanged) if it fails.

    Args:
        var: Value to cast.
        requested_type: Type/callable to cast to, e.g. `int` or `float`.
        return_val: Value to return if the cast raises `ValueError`/`TypeError`. If
            None, returns `var` unchanged instead.

    Returns:
        The cast value, or the fallback.

    Example:
        >>> silent_try_convert(float("nan"), int, 0)
        0
    """
    try:
        return requested_type(var)
    except (ValueError, TypeError):
        return var if return_val is None else return_val


try_int = functools.partial(silent_try_convert, requested_type=int)
try_float = functools.partial(silent_try_convert, requested_type=float)


def drop_multiple_columns(df: pd.DataFrame, regex_col_list: list[str]) -> pd.DataFrame:
    """Drop every column whose name matches any of several regex patterns.

    Args:
        df: Table to drop columns from.
        regex_col_list: Regex patterns; any column name matching any of them is dropped.

    Returns:
        `df` with the matching columns removed.
    """
    for regex_col in regex_col_list:
        df = df.drop(df.filter(regex=regex_col).columns, axis=1)
    return df


def explode_df_from_multivalue_columns(
    df: pd.DataFrame, lst_cols: str | list[str], fill_value: object = ""
) -> pd.DataFrame:
    """One row per value across list-valued columns, aligned across all of `lst_cols`.

    Args:
        df: Table with one or more list-valued columns.
        lst_cols: Column name(s) holding list values, exploded together in lockstep
            (row `["a","b"], [1,2]` explodes to `("a",1)` and `("b",2)`, not a cross
            product -- all listed columns must have matching list lengths per row). A
            row whose lists are empty produces one output row filled with `fill_value`.
        fill_value: Value used for the columns of a row whose lists were empty.

    Returns:
        The exploded table, in the same column order as `df`.
    """
    if not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    exploded = df.explode(lst_cols)
    with pd.option_context("future.no_silent_downcasting", True):
        exploded = exploded.fillna(fill_value)
    return exploded.loc[:, df.columns]
