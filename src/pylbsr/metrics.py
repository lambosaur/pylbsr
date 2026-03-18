from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class PairwiseValidResult:
    """Data structure for storing the result of pairwise valid filtering."""
    x: np.ndarray
    y: np.ndarray
    valid_fraction: float

def pairwise_valid(x: np.ndarray, y: np.ndarray) -> PairwiseValidResult:
    """Returns the entries where both arrays are valid (not NaN) as a dataclass."""
    mask = ~(np.isnan(x) | np.isnan(y))
    return PairwiseValidResult(x=x[mask], y=y[mask], valid_fraction=mask.mean())

@dataclass
class MetricResult:
    """Data structure for storing the result of a metric computation."""
    value: object | None  # can be float, array, tuple, etc.
    valid_fraction: float

def nanmetric(
    x: np.ndarray,
    y: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], object],
    missing_value: object | None = None,
) -> MetricResult:
    """Compute a metric on x and y ignoring NaNs.

    Args:
        x (np.ndarray): Input array.
        y (np.ndarray): Input array.
        func (callable): Function that takes two arrays and returns a scalar metric.
        missing_value (object, optional): Value used in `MetricResult` if there are no valid entries.
            Defaults to None.

    Returns:
        MetricResult: Dataclass with `value` and `valid_fraction`.
    """
    result = pairwise_valid(x, y)
    if len(result.x) == 0:
        return MetricResult(value=missing_value, valid_fraction=0.0)
    return MetricResult(value=func(result.x, result.y), valid_fraction=result.valid_fraction)

