import warnings
from enum import Enum

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class SliceCoordinateSystem(str, Enum):
    """Coordinate system for defining slices.

    This class is used by the SliceConfig class to specify how to produce a slice of
    [start:end] coordinates within a defined length L, considering additional parameters
    specific to each mode. See SliceConfig.to_slice() for details.
    """

    FROM_CENTER = "from_center"
    ABSOLUTE = "absolute"
    FROM_ANCHOR_POSITION = "from_anchor_position"


class SliceConfig(BaseModel):
    """Configuration for defining slices of sequences using `to_slice()` method.

    Modes and required parameters/attributes:
    - FROM_CENTER:
        - define slice from the center of the sequence of length L, extending left and right
            by defined amounts. The center is defined as L//2 for odd L, and (L//2)+1 for even L.
        - parameters: extend_left, extend_right

    - ABSOLUTE:
        - define slice using absolute start and end coordinates within the sequence of length L.
        - parameters: start, end (provided directly as attributes of the instance)

    - FROM_ANCHOR_POSITION:
        - define slice from a given anchor position within the sequence of length L,
            extending left and right by defined amounts extend_left and extend_right.
        - parameters: anchor_position, extend_left, extend_right

    """

    mode: SliceCoordinateSystem

    # NOTE: if set to True, will raise an error if the resulting slice length
    # does not match the expected length (as defined by the provided parameters.)
    # Otherwise: resolve according to boundaries (0, L), and only warn.
    require_expected_slice_length: bool

    # common / resolved
    start: int | None = None
    end: int | None = None

    # from_center / from_anchor_position params
    extend_left: int = 0
    extend_right: int = 0

    @model_validator(mode="after")
    def resolve_coordinates(self) -> Self:
        """Resolve coordinates depending on the selected mode."""
        if self.mode == SliceCoordinateSystem.ABSOLUTE and (
            self.start is not None and self.end is not None
        ):
            raise ValueError("absolute mode requires start and end")

        elif self.mode in [
            SliceCoordinateSystem.FROM_CENTER,
            SliceCoordinateSystem.FROM_ANCHOR_POSITION,
        ] and (self.start is not None or self.end is not None):
            # start/end computed later when L is known
            raise ValueError("Selected mode must not set start/end")

        return self

    def to_slice(self, L: int, anchor_position: int | None = None) -> slice:
        """Get the slice object for the given L (length of the sequence)."""
        if self.mode == SliceCoordinateSystem.ABSOLUTE:
            start = self.start
            end = self.end

            expected_length = end - start

        elif self.mode == SliceCoordinateSystem.FROM_CENTER:
            center = L // 2 if L % 2 == 1 else (L // 2) + 1
            start = center - self.extend_left
            end = center + self.extend_right + 1  # +1 because end is exclusive

        elif self.mode == SliceCoordinateSystem.FROM_ANCHOR_POSITION:
            if anchor_position is None:
                raise ValueError("anchor_position must be provided for FROM_ANCHOR_POSITION mode")

            # Validate
            if anchor_position < 0 or anchor_position >= L:
                raise ValueError(f"anchor_position out of bounds: {anchor_position} not in [0, {L})")

            #
            start = anchor_position - self.extend_left
            end = anchor_position + self.extend_right + 1  # +1 because end is exclusive

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        expected_length = end - start

        # Post-process by setting to min/max boundaries considering L.
        start = max(start, 0)
        end = min(end, L)

        actual_length = end - start
        if actual_length != expected_length:
            if self.require_expected_slice_length:
                raise ValueError(
                    f"Slice length mismatch: expected {expected_length}, got {actual_length}"
                )
            else:
                warnings.warn(
                    f"Slice length mismatch: expected {expected_length}, got {actual_length}"
                )

        return slice(start, end)

