"""Functions to create and manipulate binary span masks from intervals.

A span mask is a binary array of length L with 1s at positions covered by
the intervals and 0s elsewhere.
"""

from collections.abc import Hashable, Sequence

import numpy as np
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame


def coordinates_from_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Return [start, end) coordinates of contiguous `1` regions in a binary mask.

    Args:
        binary_mask: 1-D integer array of shape (length,).

    Returns:
        Int64 array of shape (n_regions, 2); each row is [start, end).
    """
    return np.where(np.diff(binary_mask, prepend=0, append=0) != 0)[0].reshape(-1, 2).astype(np.int64)


def coordinates_from_binary_masks(binary_masks: np.ndarray) -> np.ndarray:
    """Return [start, end) coordinate arrays for each mask along the last axis.

    Args:
        binary_masks: Integer array with shape (..., length).

    Returns:
        Object array with the same leading shape; each element is a (k, 2) int64 array.
    """
    return np.vectorize(
        coordinates_from_binary_mask,
        signature="(n)->()",
        otypes=[object],
    )(binary_masks)


def intervals_to_span_masks(
    starts: np.ndarray,
    ends: np.ndarray,
    length: int,
) -> np.ndarray:
    """Produce binary arrays with 1s at positions covered by each (start, end) interval.

    Example:
        >>> starts = np.array([0, 4, 7])
        >>> ends = np.array([4, 8, 10])
        >>> intervals_to_span_masks(starts, ends, length=11)
        array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=int64)

    Args:
        starts: Integer array of shape (n_intervals,) with 0-based starts (inclusive).
        ends: Integer array of shape (n_intervals,) with exclusive ends.
        length: Length of each output mask.

    Returns:
        Int64 array of shape (n_intervals, length).
    """
    positions = np.arange(length)[:, np.newaxis]  # (length, 1)
    return ((positions >= starts) & (positions < ends)).T.astype(np.int64)


def combine_span_masks_on_identifiers(
    span_masks: np.ndarray,
    identifiers: Sequence[Hashable],
) -> np.ndarray:
    """OR-combine span masks that share the same identifier.

    Args:
        span_masks: Integer array of shape (n_intervals, length).
        identifiers: Sequence of length n_intervals; masks sharing an identifier are merged.

    Returns:
        Int64 array of shape (n_unique_identifiers, length), ordered by sorted unique identifier.

    Raises:
        ValueError: If len(identifiers) != span_masks.shape[0].
    """
    if len(identifiers) != span_masks.shape[0]:
        raise ValueError(
            f"identifiers length ({len(identifiers)}) must match "
            f"span_masks rows ({span_masks.shape[0]})."
        )
    unique_ids, inverse = np.unique(identifiers, return_inverse=True)
    combined = np.zeros((len(unique_ids), span_masks.shape[1]), dtype=int)
    np.add.at(combined, inverse, span_masks)
    return (combined > 0).astype(np.int64)


def scatter_span_masks(
    span_masks: np.ndarray,
    span_masks_identifiers: Sequence[Hashable],
    all_identifiers: Sequence[Hashable],
) -> np.ndarray:
    """Scatter combined masks into the full identifier space, zero-filling missing entries.

    Args:
        span_masks: Integer array of shape (n_unique, length).
        span_masks_identifiers: Unique identifiers corresponding to rows of span_masks.
        all_identifiers: Full ordered set of identifiers; determines the output row order.

    Returns:
        Int64 array of shape (len(all_identifiers), length).

    Raises:
        ValueError: If identifiers are not unique or span_masks_identifiers is not a
            subset of all_identifiers.
    """
    if span_masks.shape[0] != len(span_masks_identifiers):
        raise ValueError(
            f"span_masks rows ({span_masks.shape[0]}) must match "
            f"span_masks_identifiers length ({len(span_masks_identifiers)})."
        )
    if len(span_masks_identifiers) != len(set(span_masks_identifiers)):
        raise ValueError("span_masks_identifiers must be unique.")
    if len(all_identifiers) != len(set(all_identifiers)):
        raise ValueError("all_identifiers must be unique.")
    if not set(span_masks_identifiers).issubset(set(all_identifiers)):
        raise ValueError("All span_masks_identifiers must be present in all_identifiers.")

    id_to_row = {id_: i for i, id_ in enumerate(span_masks_identifiers)}
    result = np.zeros((len(all_identifiers), span_masks.shape[1]), dtype=np.int64)
    for i, id_ in enumerate(all_identifiers):
        if id_ in id_to_row:
            result[i] = span_masks[id_to_row[id_]]
    return result


class RelativeCoordinatesIntervals(pa.DataFrameModel):
    """Pandera schema for named intervals with positions relative to a fixed length."""

    name: pa.typing.String
    start: pa.typing.Int64
    end: pa.typing.Int64


def relative_coordinates_to_scattered_span_masks(
    relative_coordinates: DataFrame[RelativeCoordinatesIntervals],
    all_identifiers: Sequence[str],
    length: int,
    allow_empty: bool,
) -> np.ndarray:
    """Convert a (name, start, end) DataFrame into a (N, L) binary mask matrix.

    Pipeline:
        1. Build per-interval span masks.
        2. OR-combine masks with the same name.
        3. Scatter into the full identifier space (zero-fill missing names).

    Args:
        relative_coordinates: DataFrame with columns name, start, end.
            Coordinates must be relative to `length`.
        all_identifiers: All possible identifier values; determines the output row order.
        length: Length of each output mask (== L).
        allow_empty: If False, raises when relative_coordinates is empty.

    Returns:
        Int64 array of shape (len(all_identifiers), length).

    Raises:
        ValueError: If empty and allow_empty is False, or unknown identifiers are found.
    """
    if relative_coordinates.empty:
        if not allow_empty:
            raise ValueError("relative_coordinates is empty, but allow_empty=False.")
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
        identifiers: np.ndarray = np.array([], dtype=str)
    else:
        starts = relative_coordinates["start"].to_numpy(dtype=np.int64)
        ends = relative_coordinates["end"].to_numpy(dtype=np.int64)
        identifiers = relative_coordinates["name"].to_numpy()

    unexpected = set(identifiers) - set(all_identifiers)
    if unexpected:
        raise ValueError(f"Unexpected identifiers: {unexpected}")

    span_masks = intervals_to_span_masks(starts, ends, length)
    combined = combine_span_masks_on_identifiers(span_masks, identifiers)
    return scatter_span_masks(combined, list(np.unique(identifiers)), all_identifiers)


