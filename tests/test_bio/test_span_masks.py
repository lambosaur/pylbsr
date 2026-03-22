"""Tests for pylbsr.bio.span_masks."""

import numpy as np
import pandas as pd
import pytest

from pylbsr.bio.span_masks import (
    combine_span_masks_on_identifiers,
    coordinates_from_binary_mask,
    coordinates_from_binary_masks,
    intervals_to_span_masks,
    relative_coordinates_to_scattered_span_masks,
    scatter_span_masks,
)


class TestCoordinatesFromBinaryMask:
    def test_single_region(self) -> None:
        mask = np.array([0, 1, 1, 1, 0, 0])
        result = coordinates_from_binary_mask(mask)
        np.testing.assert_array_equal(result, [[1, 4]])

    def test_two_regions(self) -> None:
        mask = np.array([1, 1, 0, 0, 1, 0])
        result = coordinates_from_binary_mask(mask)
        np.testing.assert_array_equal(result, [[0, 2], [4, 5]])

    def test_empty_mask(self) -> None:
        mask = np.zeros(5, dtype=int)
        result = coordinates_from_binary_mask(mask)
        assert result.shape == (0, 2)

    def test_full_mask(self) -> None:
        mask = np.ones(4, dtype=int)
        result = coordinates_from_binary_mask(mask)
        np.testing.assert_array_equal(result, [[0, 4]])

    def test_round_trip_with_intervals_to_span_masks(self) -> None:
        starts = np.array([0, 5, 9])
        ends = np.array([3, 8, 11])
        combined = intervals_to_span_masks(starts, ends, length=12).any(axis=0).astype(int)
        result = coordinates_from_binary_mask(combined)
        np.testing.assert_array_equal(result, [[0, 3], [5, 8], [9, 11]])


class TestCoordinatesFromBinaryMasks:
    def test_2d_input(self) -> None:
        masks = np.array([[1, 1, 0], [0, 1, 1]])
        result = coordinates_from_binary_masks(masks)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result[0], [[0, 2]])
        np.testing.assert_array_equal(result[1], [[1, 3]])


class TestIntervalsToSpanMasks:
    def test_basic(self) -> None:
        starts = np.array([0, 4, 7])
        ends = np.array([4, 8, 10])
        result = intervals_to_span_masks(starts, ends, length=11)
        expected = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        ], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_shape(self) -> None:
        starts = np.array([1, 3])
        ends = np.array([2, 5])
        result = intervals_to_span_masks(starts, ends, length=8)
        assert result.shape == (2, 8)
        assert result.dtype == np.int64

    def test_empty_intervals(self) -> None:
        result = intervals_to_span_masks(np.array([]), np.array([]), length=5)
        assert result.shape == (0, 5)


class TestCombineSpanMasksOnIdentifiers:
    def test_merge_two_into_one(self) -> None:
        masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int64)
        result = combine_span_masks_on_identifiers(masks, ["a", "a"])
        np.testing.assert_array_equal(result, [[1, 1, 1, 1]])

    def test_keep_distinct(self) -> None:
        masks = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
        result = combine_span_masks_on_identifiers(masks, ["a", "b"])
        assert result.shape == (2, 3)

    def test_length_mismatch_raises(self) -> None:
        masks = np.zeros((3, 5), dtype=np.int64)
        with pytest.raises(ValueError, match="identifiers length"):
            combine_span_masks_on_identifiers(masks, ["a", "b"])


class TestScatterSpanMasks:
    def test_basic_scatter(self) -> None:
        masks = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
        result = scatter_span_masks(masks, ["a", "b"], ["c", "b", "a"])
        np.testing.assert_array_equal(result[0], [0, 0, 0])  # "c" → zero
        np.testing.assert_array_equal(result[1], [0, 1, 0])  # "b"
        np.testing.assert_array_equal(result[2], [1, 0, 0])  # "a"

    def test_zero_fill_missing(self) -> None:
        masks = np.array([[1, 1]], dtype=np.int64)
        result = scatter_span_masks(masks, ["a"], ["a", "b", "c"])
        np.testing.assert_array_equal(result[1], [0, 0])
        np.testing.assert_array_equal(result[2], [0, 0])

    def test_non_subset_raises(self) -> None:
        masks = np.array([[1, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="must be present"):
            scatter_span_masks(masks, ["z"], ["a", "b"])

    def test_duplicate_all_identifiers_raises(self) -> None:
        masks = np.array([[1, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="unique"):
            scatter_span_masks(masks, ["a"], ["a", "a"])


class TestRelativeCoordinatesToScatteredSpanMasks:
    def test_basic(self) -> None:
        df = pd.DataFrame({"name": ["q1", "q1", "q2"], "start": [0, 5, 2], "end": [3, 8, 4]})
        result = relative_coordinates_to_scattered_span_masks(df, ["q1", "q2"], length=10, allow_empty=True)
        assert result.shape == (2, 10)
        assert result[0, 0] == 1  # q1 covers [0,3)
        assert result[0, 6] == 1  # q1 covers [5,8)
        assert result[1, 2] == 1  # q2 covers [2,4)
        assert result[1, 0] == 0  # q2 doesn't cover [0,2)

    def test_empty_disallowed_raises(self) -> None:
        df = pd.DataFrame({"name": pd.Series([], dtype=str), "start": pd.Series([], dtype="int64"), "end": pd.Series([], dtype="int64")})
        with pytest.raises(ValueError, match="allow_empty=False"):
            relative_coordinates_to_scattered_span_masks(df, ["q1"], length=5, allow_empty=False)


