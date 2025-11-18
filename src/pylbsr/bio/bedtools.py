from enum import Enum
from typing import Literal

import pybedtools as pbt


def set_name(interval: pbt.cbedtools.Interval, name: str) -> pbt.cbedtools.Interval:
    """Set the `name` field of a pyBedTools Interval object. To be used with `pbt.BedTool.each()`."""
    interval.name = name
    return interval


def set_name_from_coordinates(interval: pbt.cbedtools.Interval) -> pbt.cbedtools.Interval:
    """Set the `name` field of a pyBedTools Interval object to a str representation of its coordinates.

    To be used with `pbt.BedTool.each()`.
    """
    interval.name = f"{interval.chrom}:{interval.start}-{interval.end}:{interval.strand}"
    return interval

def bt_center_interval_on(
    interval: pbt.cbedtools.Interval,
    center_on: Literal["5p", "center", "3p"],
    stranded: bool,
) -> pbt.cbedtools.Interval:
    """Bedtools function to generate 1nt intervals from input interval centered on desired location."""
    positive_strand_encoding = ["+", "+1", 1, "1", "forward", "fwd", "plus", "pos"]
    negative_strand_encoding = ["-", "-1", -1, "reverse", "rev", "minus", "neg"]

    if center_on == "center":
        half_length = (interval.end - interval.start) // 2
        interval.start = interval.start + half_length
        interval.end = interval.start + 1

    elif center_on == "5p":
        if stranded:
            if interval.strand in [*positive_strand_encoding, "."]:
                interval.end = interval.start + 1

            elif interval.strand in negative_strand_encoding:
                interval.start = interval.end - 1
            else:
                raise ValueError(f"Unexpected strand value: {interval.strand}")
        else:
            interval.end = interval.start + 1

    elif center_on == "3p":
        if stranded:
            if interval.strand in [*positive_strand_encoding, "."]:
                interval.start = interval.end - 1
            elif interval.strand in negative_strand_encoding:
                interval.end = interval.start + 1
            else:
                raise ValueError(f"Unexpected strand value: {interval.strand}")
        else:
            interval.start = interval.end - 1
    else:
        raise ValueError(
            f"Unexpected value for `center_on`: {center_on}. Expected one of ['5p', 'center', '3p']."
        )

    return interval
