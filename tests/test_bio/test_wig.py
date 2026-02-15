"""Tests for pylbsr.bio.wig wiggle format classes."""

from io import StringIO

import pytest

from pylbsr.bio.wig import (
    FixedStepBlock,
    VariableStepBlock,
    WigBlockCollection,
    WigTrackDefinition,
    read_wig,
)


# ============================================================================
# FixedStepBlock
# ============================================================================


class TestFixedStepBlock:
    """Tests for FixedStepBlock."""

    def test_basic_construction(self):
        block = FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0))
        assert block.chrom == "chr1"
        assert block.start == 1
        assert block.step == 1
        assert block.span == 1
        assert block.stop == 3
        assert len(block) == 3

    def test_with_step_and_span(self):
        block = FixedStepBlock(
            chrom="chr1", start=101, values=(0.1, 0.2, 0.3), step=3, span=2
        )
        assert block.start == 101
        # stop = 101 + 3*(3-1) + 2-1 = 108
        assert block.stop == 108
        assert len(block) == 8

    def test_indexed_values_simple(self):
        block = FixedStepBlock(chrom="chr1", start=1, values=(10.0, 20.0))
        assert block.indexed_values == ((1, 10.0), (2, 20.0))

    def test_indexed_values_with_span(self):
        block = FixedStepBlock(
            chrom="chr1", start=1, values=(10.0, 20.0), step=2, span=2
        )
        # Value 10.0 at positions 1,2; value 20.0 at positions 3,4
        assert block.indexed_values == ((1, 10.0), (2, 10.0), (3, 20.0), (4, 20.0))

    def test_positions(self):
        block = FixedStepBlock(chrom="chr1", start=5, values=(1.0, 2.0, 3.0), step=2)
        # With span=1: positions 5, 7, 9
        assert block.positions == (5, 7, 9)

    def test_header_without_span(self):
        block = FixedStepBlock(chrom="chr1", start=1, values=(1.0,), step=1)
        assert block.header == "fixedStep chrom=chr1 start=1 step=1"

    def test_header_with_span(self):
        block = FixedStepBlock(
            chrom="chr1", start=100, values=(1.0,), step=5, span=3
        )
        assert block.header == "fixedStep chrom=chr1 start=100 step=5 span=3"

    def test_to_wig_simple(self):
        block = FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0))
        expected = "fixedStep chrom=chr1 start=1 step=1\n1.0\n2.0\n3.0"
        assert block.to_wig() == expected

    def test_to_wig_with_span(self):
        block = FixedStepBlock(
            chrom="chr1", start=100, values=(10.0,), step=5, span=3
        )
        expected = "fixedStep chrom=chr1 start=100 step=5 span=3\n10.0"
        assert block.to_wig() == expected

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="at least one element"):
            FixedStepBlock(chrom="chr1", start=1, values=())

    def test_span_greater_than_step_raises(self):
        with pytest.raises(ValueError, match="cannot be larger than step"):
            FixedStepBlock(chrom="chr1", start=1, values=(1.0,), step=2, span=3)

    def test_as_series_full_range(self):
        block = FixedStepBlock(
            chrom="chr1", start=1, values=(10.0, 20.0), step=3, span=1
        )
        series = block.as_series(full_range=True)
        assert series[1] == 10.0
        assert series[4] == 20.0
        assert len(series) == 4  # positions 1..4

    def test_frozen(self):
        block = FixedStepBlock(chrom="chr1", start=1, values=(1.0,))
        with pytest.raises(Exception):
            block.chrom = "chr2"


# ============================================================================
# VariableStepBlock
# ============================================================================


class TestVariableStepBlock:
    """Tests for VariableStepBlock."""

    def test_basic_construction(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(100, 200, 300), values=(1.0, 2.0, 3.0)
        )
        assert block.start == 100
        assert block.stop == 300
        assert block.span == 1

    def test_with_span(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(100, 200), values=(1.0, 2.0), span=3
        )
        assert block.start == 100
        assert block.stop == 202  # 200 + 3 - 1

    def test_indexed_values(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(10, 20), values=(1.5, 2.5)
        )
        assert block.indexed_values == ((10, 1.5), (20, 2.5))

    def test_indexed_values_with_span(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(10, 20), values=(1.5, 2.5), span=2
        )
        assert block.indexed_values == (
            (10, 1.5), (11, 1.5), (20, 2.5), (21, 2.5)
        )

    def test_header_without_span(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(100,), values=(1.0,)
        )
        assert block.header == "variableStep chrom=chr2"

    def test_header_with_span(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(100,), values=(1.0,), span=5
        )
        assert block.header == "variableStep chrom=chr2 span=5"

    def test_to_wig(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(100, 200), values=(1.5, 2.5)
        )
        expected = "variableStep chrom=chr2\n100 1.5\n200 2.5"
        assert block.to_wig() == expected

    def test_to_wig_with_span(self):
        block = VariableStepBlock(
            chrom="chr2", positions=(100, 200), values=(1.5, 2.5), span=3
        )
        expected = "variableStep chrom=chr2 span=3\n100 1.5\n200 2.5"
        assert block.to_wig() == expected

    def test_non_increasing_positions_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            VariableStepBlock(
                chrom="chr1", positions=(200, 100), values=(1.0, 2.0)
            )

    def test_span_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            VariableStepBlock(
                chrom="chr1", positions=(100, 101), values=(1.0, 2.0), span=3
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            VariableStepBlock(
                chrom="chr1", positions=(100, 200, 300), values=(1.0, 2.0)
            )


# ============================================================================
# Block comparison & overlap
# ============================================================================


class TestBlockComparison:
    """Tests for ordering, equality, and overlap logic."""

    def test_ordering_same_chrom(self):
        b1 = FixedStepBlock(chrom="chr1", start=100, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr1", start=200, values=(2.0,))
        assert b1 < b2
        assert sorted([b2, b1]) == [b1, b2]

    def test_ordering_different_chrom(self):
        b1 = FixedStepBlock(chrom="chr1", start=100, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr2", start=1, values=(2.0,))
        assert b1 < b2

    def test_equality(self):
        b1 = FixedStepBlock(chrom="chr1", start=100, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr1", start=100, values=(999.0,), step=5)
        assert b1 == b2  # equality is (chrom, start) only

    def test_overlaps_true(self):
        b1 = FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0))
        b2 = FixedStepBlock(chrom="chr1", start=2, values=(4.0,))
        assert b1.overlaps(b2)

    def test_overlaps_false(self):
        b1 = FixedStepBlock(chrom="chr1", start=1, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr1", start=2, values=(2.0,))
        assert not b1.overlaps(b2)

    def test_overlaps_different_chrom(self):
        b1 = FixedStepBlock(chrom="chr1", start=1, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr2", start=1, values=(2.0,))
        assert not b1.overlaps(b2)

    def test_intersect(self):
        b1 = FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0))
        b2 = FixedStepBlock(chrom="chr1", start=2, values=(4.0, 5.0))
        assert b1.intersect(b2) == (2, 3)

    def test_intersect_none(self):
        b1 = FixedStepBlock(chrom="chr1", start=1, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr1", start=10, values=(2.0,))
        assert b1.intersect(b2) is None

    def test_cross_type_overlap(self):
        fb = FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0))
        vb = VariableStepBlock(chrom="chr1", positions=(2,), values=(4.0,))
        assert fb.overlaps(vb)


# ============================================================================
# WigTrackDefinition
# ============================================================================


class TestWigTrackDefinition:
    """Tests for WigTrackDefinition."""

    def test_default(self):
        td = WigTrackDefinition()
        wig = td.to_wig()
        assert wig.startswith("track type=wiggle_0")
        assert 'name="wiggle_track"' in wig
        assert 'description="Wiggle Track"' in wig

    def test_custom(self):
        td = WigTrackDefinition(
            name="my_track",
            description="My Track",
            priority=10,
            color="255,0,0",
            graphType="points",
        )
        wig = td.to_wig()
        assert 'name="my_track"' in wig
        assert "priority=10" in wig
        assert "color=255,0,0" in wig
        assert "graphType=points" in wig

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid type"):
            WigTrackDefinition(type="invalid")

    def test_optional_fields_omitted(self):
        td = WigTrackDefinition()
        wig = td.to_wig()
        assert "priority" not in wig
        assert "color" not in wig
        assert "graphType" not in wig


# ============================================================================
# WigBlockCollection
# ============================================================================


class TestWigBlockCollection:
    """Tests for WigBlockCollection."""

    def test_add_maintains_order(self):
        coll = WigBlockCollection()
        b1 = FixedStepBlock(chrom="chr1", start=100, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr1", start=1, values=(2.0,))
        coll.add(b1)
        coll.add(b2)
        assert coll[0].start == 1
        assert coll[1].start == 100
        assert len(coll) == 2

    def test_overlap_raises(self):
        coll = WigBlockCollection()
        b1 = FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0))
        b2 = FixedStepBlock(chrom="chr1", start=2, values=(4.0,))
        coll.add(b1)
        with pytest.raises(ValueError, match="overlaps"):
            coll.add(b2)

    def test_mixed_block_types(self):
        coll = WigBlockCollection()
        fb = FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0))
        vb = VariableStepBlock(
            chrom="chr1", positions=(100, 200), values=(3.0, 4.0)
        )
        coll.add(fb)
        coll.add(vb)
        assert len(coll) == 2

    def test_to_wig_without_track(self):
        coll = WigBlockCollection()
        coll.add(FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0)))
        buf = StringIO()
        coll.to_wig(buf)
        assert buf.getvalue() == "fixedStep chrom=chr1 start=1 step=1\n1.0\n2.0\n"

    def test_to_wig_with_track(self):
        td = WigTrackDefinition(name="test")
        coll = WigBlockCollection(track_definition=td)
        coll.add(FixedStepBlock(chrom="chr1", start=1, values=(1.0,)))
        buf = StringIO()
        coll.to_wig(buf)
        lines = buf.getvalue().split("\n")
        assert lines[0].startswith("track type=wiggle_0")
        assert lines[1] == ""  # blank line separating track def from block
        assert lines[2] == "fixedStep chrom=chr1 start=1 step=1"
        assert lines[3] == "1.0"

    def test_to_wig_multiple_blocks(self):
        coll = WigBlockCollection()
        coll.add(FixedStepBlock(chrom="chr1", start=1, values=(1.0,)))
        coll.add(
            VariableStepBlock(chrom="chr1", positions=(100,), values=(2.0,))
        )
        buf = StringIO()
        coll.to_wig(buf)
        content = buf.getvalue()
        assert "fixedStep" in content
        assert "variableStep" in content

    def test_to_wig_to_file(self, tmp_path):
        coll = WigBlockCollection()
        coll.add(FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0)))
        filepath = tmp_path / "test.wig"
        with open(filepath, "w") as f:
            coll.to_wig(f)
        content = filepath.read_text()
        assert content.startswith("fixedStep chrom=chr1")
        assert content.endswith("\n")

    def test_as_bed3(self):
        coll = WigBlockCollection()
        coll.add(FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0)))
        bed = coll.as_bed3()
        # 1-based [1,3] → 0-based half-open [0,3)
        assert bed == (("chr1", 0, 3),)

    def test_iter(self):
        coll = WigBlockCollection()
        b1 = FixedStepBlock(chrom="chr1", start=1, values=(1.0,))
        b2 = FixedStepBlock(chrom="chr1", start=10, values=(2.0,))
        coll.add(b1)
        coll.add(b2)
        blocks = list(coll)
        assert len(blocks) == 2

    def test_empty_collection(self):
        coll = WigBlockCollection()
        assert len(coll) == 0
        buf = StringIO()
        coll.to_wig(buf)
        assert buf.getvalue() == "\n"
        assert coll.as_bed3() == ()


# ============================================================================
# read_wig
# ============================================================================


class TestReadWig:
    """Tests for read_wig iterator."""

    def test_read_fixed_step(self):
        content = "fixedStep chrom=chr1 start=100 step=5\n10.0\n20.0\n30.0\n"
        blocks = list(read_wig(StringIO(content)))
        assert len(blocks) == 1
        block = blocks[0]
        assert isinstance(block, FixedStepBlock)
        assert block.chrom == "chr1"
        assert block.start == 100
        assert block.step == 5
        assert block.values == (10.0, 20.0, 30.0)

    def test_read_fixed_step_with_span(self):
        content = "fixedStep chrom=chr1 start=1 step=3 span=2\n1.0\n2.0\n"
        block = next(read_wig(StringIO(content)))
        assert block.span == 2
        assert block.step == 3

    def test_read_variable_step(self):
        content = "variableStep chrom=chr2 span=3\n100 1.5\n200 2.5\n"
        blocks = list(read_wig(StringIO(content)))
        assert len(blocks) == 1
        block = blocks[0]
        assert isinstance(block, VariableStepBlock)
        assert block.chrom == "chr2"
        assert block.span == 3
        assert block.positions_ == (100, 200)
        assert block.values == (1.5, 2.5)

    def test_read_multiple_blocks(self):
        content = (
            "fixedStep chrom=chr1 start=1 step=1\n1.0\n2.0\n"
            "fixedStep chrom=chr1 start=100 step=1\n3.0\n"
        )
        blocks = list(read_wig(StringIO(content)))
        assert len(blocks) == 2
        assert blocks[0].start == 1
        assert blocks[0].values == (1.0, 2.0)
        assert blocks[1].start == 100
        assert blocks[1].values == (3.0,)

    def test_read_mixed_types(self):
        content = (
            "fixedStep chrom=chr1 start=1 step=1\n1.0\n"
            "variableStep chrom=chr2\n100 2.0\n"
        )
        blocks = list(read_wig(StringIO(content)))
        assert len(blocks) == 2
        assert isinstance(blocks[0], FixedStepBlock)
        assert isinstance(blocks[1], VariableStepBlock)

    def test_skips_track_lines(self):
        content = (
            'track type=wiggle_0 name="test"\n'
            "fixedStep chrom=chr1 start=1 step=1\n1.0\n"
        )
        blocks = list(read_wig(StringIO(content)))
        assert len(blocks) == 1

    def test_track_line_flushes_previous_block(self):
        content = (
            "fixedStep chrom=chr1 start=1 step=1\n1.0\n"
            'track type=wiggle_0 name="track2"\n'
            "fixedStep chrom=chr2 start=1 step=1\n2.0\n"
        )
        blocks = list(read_wig(StringIO(content)))
        assert len(blocks) == 2
        assert blocks[0].chrom == "chr1"
        assert blocks[1].chrom == "chr2"

    def test_skips_comments_and_empty_lines(self):
        content = (
            "# comment\n"
            "\n"
            "fixedStep chrom=chr1 start=1 step=1\n"
            "1.0\n"
            "\n"
            "# another comment\n"
            "2.0\n"
        )
        blocks = list(read_wig(StringIO(content)))
        assert len(blocks) == 1
        assert blocks[0].values == (1.0, 2.0)

    def test_integer_values_preserved(self):
        content = "fixedStep chrom=chr1 start=1 step=1\n10\n20\n"
        block = next(read_wig(StringIO(content)))
        assert block.values == (10, 20)
        assert isinstance(block.values[0], int)

    def test_empty_input(self):
        blocks = list(read_wig(StringIO("")))
        assert blocks == []

    def test_round_trip(self):
        coll = WigBlockCollection()
        coll.add(FixedStepBlock(chrom="chr1", start=1, values=(1.0, 2.0, 3.0)))
        coll.add(
            VariableStepBlock(
                chrom="chr1", positions=(100, 200), values=(4.0, 5.0)
            )
        )
        buf = StringIO()
        coll.to_wig(buf)
        buf.seek(0)

        blocks = list(read_wig(buf))
        assert len(blocks) == 2
        assert blocks[0].chrom == "chr1"
        assert blocks[0].start == 1
        assert blocks[0].values == (1.0, 2.0, 3.0)
        assert blocks[1].positions_ == (100, 200)
        assert blocks[1].values == (4.0, 5.0)
