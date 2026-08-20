"""Tests for the pyutils-ported misc.py additions.

Only covers what was newly ported here (slice_range_overlapping, make_experiment_outputdir,
silent_try_convert/try_int/try_float, drop_multiple_columns, explode_df_from_multivalue_columns)
plus set_seed (moved here from torch_utils.py -- see torch_utils.set_seed for the torch-aware
version that also seeds CUDA/cuDNN); the rest of misc.py had no test coverage before this and
is out of scope for this change.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylbsr.misc import (
    drop_multiple_columns,
    explode_df_from_multivalue_columns,
    make_experiment_outputdir,
    set_seed,
    silent_try_convert,
    slice_range_overlapping,
    try_float,
    try_int,
)


def test_set_seed_makes_random_and_numpy_draws_reproducible() -> None:
    """Calling set_seed with the same value reproduces the same random/numpy draws."""
    set_seed(42)
    py_draw_1, np_draw_1 = random.random(), np.random.rand()

    set_seed(42)
    py_draw_2, np_draw_2 = random.random(), np.random.rand()

    assert py_draw_1 == py_draw_2
    assert np_draw_1 == np_draw_2


def test_slice_range_overlapping_tiles_the_full_range() -> None:
    """Consecutive windows overlap by (windowsize - step), covering [start, end)."""
    windows = list(slice_range_overlapping(start=0, end=10, step=2, windowsize=4))

    assert windows == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_slice_range_overlapping_non_overlapping_when_step_equals_windowsize() -> None:
    """step == windowsize produces back-to-back, non-overlapping windows."""
    windows = list(slice_range_overlapping(start=0, end=8, step=4, windowsize=4))

    assert windows == [(0, 4), (4, 8)]


def test_make_experiment_outputdir_creates_named_dir(tmp_path: Path) -> None:
    """A named, non-existing directory is created as-is."""
    outputdir = make_experiment_outputdir(tmp_path, name="run1", symlink_as_latest=False)

    assert outputdir == tmp_path / "run1"
    assert outputdir.is_dir()


def test_make_experiment_outputdir_raises_by_default_if_exists(tmp_path: Path) -> None:
    """No reuse/replace/increment given, and the dir already exists -> FileExistsError."""
    (tmp_path / "run1").mkdir()

    with pytest.raises(FileExistsError):
        make_experiment_outputdir(tmp_path, name="run1", symlink_as_latest=False)


def test_make_experiment_outputdir_reuse_does_not_raise(tmp_path: Path) -> None:
    """reuse=True accepts an already-existing directory."""
    existing = tmp_path / "run1"
    existing.mkdir()
    (existing / "marker.txt").write_text("keep me")

    outputdir = make_experiment_outputdir(tmp_path, name="run1", reuse=True, symlink_as_latest=False)

    assert outputdir == existing
    assert (outputdir / "marker.txt").exists()  # contents preserved, not wiped


def test_make_experiment_outputdir_replace_wipes_existing(tmp_path: Path) -> None:
    """replace=True removes and recreates an existing directory."""
    existing = tmp_path / "run1"
    existing.mkdir()
    (existing / "marker.txt").write_text("should be gone")

    outputdir = make_experiment_outputdir(
        tmp_path, name="run1", replace=True, symlink_as_latest=False
    )

    assert outputdir == existing
    assert not (outputdir / "marker.txt").exists()


def test_make_experiment_outputdir_increment_appends_suffix(tmp_path: Path) -> None:
    """increment=True finds the first free '_{i}' suffix instead of raising.

    Regression test: the original had `exists_ok=False` (a typo for `exist_ok`) on the
    increment path's mkdir call, which would TypeError on every call -- this exercises
    that exact path, so it's a real regression guard, not just a happy-path smoke test.
    """
    (tmp_path / "run1").mkdir()

    outputdir = make_experiment_outputdir(
        tmp_path, name="run1", increment=True, symlink_as_latest=False
    )

    assert outputdir == tmp_path / "run1_1"
    assert outputdir.is_dir()


def test_make_experiment_outputdir_symlinks_as_latest(tmp_path: Path) -> None:
    """symlink_as_latest=True (default) creates a "latest" symlink pointing at the new dir."""
    outputdir = make_experiment_outputdir(tmp_path, name="run1")

    latest = tmp_path / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == outputdir.resolve()


def test_silent_try_convert_success() -> None:
    """A convertible value is cast normally."""
    assert silent_try_convert("42", int, 0) == 42


def test_silent_try_convert_failure_returns_fallback() -> None:
    """An unconvertible value returns return_val when one is given."""
    assert silent_try_convert(float("nan"), int, 0) == 0


def test_silent_try_convert_failure_returns_var_when_no_fallback() -> None:
    """With return_val=None (the default), failure returns var unchanged, not None."""
    assert silent_try_convert("not-a-number", int) == "not-a-number"


def test_try_int_and_try_float_partials() -> None:
    """try_int/try_float are silent_try_convert pre-bound to int/float."""
    assert try_int("5") == 5
    assert try_float("5.5") == 5.5
    assert try_int("nope", return_val=-1) == -1


def test_drop_multiple_columns_matches_regex() -> None:
    """Every column matching any of the given regex patterns is dropped."""
    df = pd.DataFrame({"conservation_a": [1], "tf_b": [2], "keep_me": [3]})

    result = drop_multiple_columns(df, ["^conservation_", "^tf_"])

    assert list(result.columns) == ["keep_me"]


def test_explode_df_from_multivalue_columns_basic() -> None:
    """Aligned list columns explode in lockstep, not as a cross product."""
    df = pd.DataFrame({"id": [1, 2], "a": [[1, 2], [3]], "b": [["x", "y"], ["z"]]})

    result = explode_df_from_multivalue_columns(df, ["a", "b"])

    assert list(result["id"]) == [1, 1, 2]
    assert list(result["a"]) == [1, 2, 3]
    assert list(result["b"]) == ["x", "y", "z"]


def test_explode_df_from_multivalue_columns_empty_list_uses_fill_value() -> None:
    """A row with an empty list produces one row filled with fill_value, not a dropped row.

    Regression test: the original used DataFrame.append (removed in pandas 2.0) specifically
    on this empty-list code path -- this exercises exactly that path.
    """
    df = pd.DataFrame({"id": [1, 2], "a": [[1, 2], []], "b": [["x", "y"], []]})

    result = explode_df_from_multivalue_columns(df, ["a", "b"], fill_value="MISSING")

    assert len(result) == 3  # 2 exploded rows from id=1, 1 filled row from id=2
    empty_row = result[result["id"] == 2]
    assert list(empty_row["a"]) == ["MISSING"]
    assert list(empty_row["b"]) == ["MISSING"]


def test_explode_df_from_multivalue_columns_accepts_single_column_name() -> None:
    """lst_cols may be a bare string, not just a list."""
    df = pd.DataFrame({"id": [1], "a": [[10, 20]]})

    result = explode_df_from_multivalue_columns(df, "a")

    assert list(result["a"]) == [10, 20]
