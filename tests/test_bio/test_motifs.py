"""Tests for pylbsr.bio.motifs."""

from io import StringIO

import Bio.motifs
import pandas as pd
import pytest

from pylbsr.bio.motifs import (
    ALPHABET_DNA,
    Motif,
    MotifError,
    merge_rename_motif_collections,
    motif_to_biopython_motif,
    parse_transfac_motif_lines,
    read_motifs_transfac,
    relabel_motif_collection,
    write_motif_transfac,
)
from pylbsr.bio.motifs.motif import (
    _parse_transfac_matrix_line,
    _parse_transfac_matrix_lines,
    _group_transfac_motif_lines,
    _write_matrix_transfac,
)
from pylbsr.bio.motifs._fixtures import (
    EmptyTransfacMotif,
    GenericMinimalTransfacMotif,
    JasparTransfacMotif,
    MCrossTransfacMotif,
    MalformedTransfacMotif,
    RsatTransfacMotif,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        MCrossTransfacMotif,
        RsatTransfacMotif,
        JasparTransfacMotif,
        GenericMinimalTransfacMotif,
    ]
)
def correct_motif_example(request):
    yield request.param


# RSAT has unusual whitespace — round-trip output won't match input exactly.
@pytest.fixture(
    params=[
        MCrossTransfacMotif,
        GenericMinimalTransfacMotif,
    ]
)
def correct_motif_writable_example(request):
    yield request.param


@pytest.fixture(
    params=[
        MalformedTransfacMotif,
        EmptyTransfacMotif,
    ]
)
def incorrect_motif_example(request):
    yield request.param


def _make_motif(fixture_cls) -> Motif:
    """Parse a Motif from a fixture class using its declared separators."""
    return parse_transfac_motif_lines(
        lines=fixture_cls.lines_motif,
        key_value_separator=fixture_cls.key_value_separator,
        matrix_key_value_separator=fixture_cls.matrix_key_value_separator,
        matrix_value_content_separator=fixture_cls.matrix_value_content_separator,
    )


# ---------------------------------------------------------------------------
# Parsing — low-level
# ---------------------------------------------------------------------------


def test_parse_matrix_line(correct_motif_example):
    for matrix_line, expected in correct_motif_example.get_lines_matrix_for_test(N=3):
        parsed = _parse_transfac_matrix_line(
            line=matrix_line,
            matrix_key_value_separator=correct_motif_example.matrix_key_value_separator,
            matrix_value_content_separator=correct_motif_example.matrix_value_content_separator,
            alphabet_size=len(correct_motif_example.alphabet),
        )
        assert parsed.values == expected.values


def test_parse_matrix_lines(correct_motif_example):
    result = _parse_transfac_matrix_lines(
        lines=correct_motif_example.lines_matrix,
        matrix_key_value_separator=correct_motif_example.matrix_key_value_separator,
        matrix_value_content_separator=correct_motif_example.matrix_value_content_separator,
    )
    assert result  # non-empty list returned


def test_group_transfac_motif_lines(correct_motif_example):
    assert _group_transfac_motif_lines(lines=correct_motif_example.lines_motif) == (
        correct_motif_example.lines_header,
        correct_motif_example.lines_matrix,
        correct_motif_example.lines_footer,
    )


def test_fail_parse_matrix_lines(incorrect_motif_example):
    with pytest.raises(ValueError):
        _parse_transfac_matrix_lines(
            lines=incorrect_motif_example.lines_matrix,
            matrix_key_value_separator=incorrect_motif_example.matrix_key_value_separator,
            matrix_value_content_separator=incorrect_motif_example.matrix_value_content_separator,
        )


# ---------------------------------------------------------------------------
# Parsing — end-to-end
# ---------------------------------------------------------------------------


def test_parse_transfac_motif_lines(correct_motif_example):
    motif = _make_motif(correct_motif_example)
    assert isinstance(motif, Motif)
    assert motif.id
    assert not motif.matrix.empty


def test_matrix_values_are_numeric(correct_motif_example):
    motif = _make_motif(correct_motif_example)
    for dtype in motif.matrix.dtypes:
        assert dtype != object, f"Column dtype should be numeric, got {dtype}"


def test_force_numeric_raises():
    """If matrix values are non-numeric, force_numeric=True must raise MotifError."""
    # "abc" is a single token so it passes the alphabet-size check but fails pd.to_numeric.
    lines = [
        "ID  test_motif\n",
        "XX\n",
        "P0  A C G T\n",
        "01  abc 0.1 0.1 0.2\n",
        "XX\n",
        "//\n",
    ]
    with pytest.raises(MotifError):
        parse_transfac_motif_lines(lines, force_numeric=True)


def test_read_motifs_transfac_two_motifs():
    """read_motifs_transfac returns one Motif per '//' block."""
    block = (
        "ID  motif_A\n"
        "XX\n"
        "P0  A C G T\n"
        "01  10 2 3 5\n"
        "XX\n"
        "//\n"
        "ID  motif_B\n"
        "XX\n"
        "P0  A C G T\n"
        "01  1 6 2 1\n"
        "XX\n"
        "//\n"
    )
    motifs = read_motifs_transfac(StringIO(block))
    assert len(motifs) == 2
    assert motifs[0].id == "motif_A"
    assert motifs[1].id == "motif_B"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_matrix(correct_motif_example):
    handle = StringIO()
    _write_matrix_transfac(
        handle=handle,
        matrix=correct_motif_example.get_matrix_as_dataframe(),
        consensus=correct_motif_example.consensus,
        format_matrix_values="{value}",
        matrix_key_value_separator=correct_motif_example.matrix_key_value_separator,
        matrix_value_content_separator=correct_motif_example.matrix_value_content_separator,
    )
    handle.seek(0)
    assert handle.read()


def test_write_matrix_content(correct_motif_writable_example):
    handle = StringIO()
    _write_matrix_transfac(
        handle=handle,
        matrix=correct_motif_writable_example.get_matrix_as_dataframe(),
        consensus=correct_motif_writable_example.consensus,
        format_matrix_values="{value}",
        matrix_key_value_separator=correct_motif_writable_example.matrix_key_value_separator,
        matrix_value_content_separator=correct_motif_writable_example.matrix_value_content_separator,
    )
    handle.seek(0)
    assert handle.readlines() == correct_motif_writable_example.lines_matrix


def test_write_motif_transfac_roundtrip():
    """Write then re-read a motif; matrix values must be preserved.

    write_motif_transfac normalises to its default separators ("  " / " "),
    so we read back with those same defaults.
    """
    original = _make_motif(MCrossTransfacMotif)
    buf = StringIO()
    write_motif_transfac(buf, original)  # defaults: kv_sep="  ", val_sep=" "
    buf.seek(0)
    # Read back using the write defaults (two-space kv, single-space value sep).
    recovered = read_motifs_transfac(buf)
    assert len(recovered) == 1
    pd.testing.assert_frame_equal(original.matrix, recovered[0].matrix)


# ---------------------------------------------------------------------------
# Motif.relabel_alphabet
# ---------------------------------------------------------------------------


def test_relabel_alphabet_rename():
    motif = _make_motif(GenericMinimalTransfacMotif)
    assert motif.alphabet == ("A", "C", "G", "T")
    rna = motif.relabel_alphabet({"A": "A", "C": "C", "G": "G", "T": "U"})
    assert rna.alphabet == ("A", "C", "G", "U")
    assert list(rna.matrix.columns) == ["A", "C", "G", "U"]


def test_relabel_alphabet_reorder():
    motif = _make_motif(GenericMinimalTransfacMotif)
    reordered = motif.relabel_alphabet({"T": "T", "G": "G", "C": "C", "A": "A"})
    assert list(reordered.matrix.columns) == ["T", "G", "C", "A"]


def test_relabel_alphabet_wrong_keys():
    motif = _make_motif(GenericMinimalTransfacMotif)
    with pytest.raises(MotifError):
        motif.relabel_alphabet({"X": "A", "C": "C", "G": "G", "T": "T"})


def test_relabel_alphabet_not_injective():
    motif = _make_motif(GenericMinimalTransfacMotif)
    with pytest.raises(MotifError):
        motif.relabel_alphabet({"A": "Z", "C": "Z", "G": "G", "T": "T"})


# ---------------------------------------------------------------------------
# relabel_motif_collection
# ---------------------------------------------------------------------------


def test_relabel_motif_collection_unique_ids():
    motif_a = _make_motif(GenericMinimalTransfacMotif)
    motif_b = _make_motif(GenericMinimalTransfacMotif)
    assert motif_a.id == motif_b.id  # same fixture → same ID

    result = relabel_motif_collection([motif_a, motif_b])
    assert result[0].id == "generic_minimal_motif"
    assert result[1].id == "generic_minimal_motif_2"


def test_relabel_motif_collection_cc_records_original():
    motif_a = _make_motif(GenericMinimalTransfacMotif)
    motif_b = _make_motif(GenericMinimalTransfacMotif)
    original_id = motif_a.id

    result = relabel_motif_collection([motif_a, motif_b])
    cc_lines = [m.value for m in result[1].metadata["footer"] if m.key == "CC"]
    assert any(original_id in v for v in cc_lines), (
        "CC footer should record the original (pre-rename) ID"
    )


# ---------------------------------------------------------------------------
# merge_rename_motif_collections
# ---------------------------------------------------------------------------


def _make_motif_with_ac() -> Motif:
    """Return a parsed JasparTransfacMotif (has an AC field)."""
    return parse_transfac_motif_lines(
        lines=JasparTransfacMotif.lines_motif,
        key_value_separator=JasparTransfacMotif.key_value_separator,
        matrix_key_value_separator=JasparTransfacMotif.matrix_key_value_separator,
        matrix_value_content_separator=JasparTransfacMotif.matrix_value_content_separator,
    )


def test_merge_rename_motif_collections_prefix():
    m1 = _make_motif(GenericMinimalTransfacMotif)
    result = merge_rename_motif_collections({"DB": [m1]})
    assert result[0].id == "DB.generic_minimal_motif"


def test_merge_rename_motif_collections_ac_renamed():
    """AC field must be prefixed (this tests the bug-fix path)."""
    motif = _make_motif_with_ac()
    original_ac = [m.value for m in motif.metadata["header"] if m.key == "AC"][0]

    result = merge_rename_motif_collections({"JASPAR": [motif]})
    new_ac = [m.value for m in result[0].metadata["header"] if m.key == "AC"][0]
    assert new_ac == f"JASPAR.{original_ac}"


def test_merge_rename_motif_collections_no_ac():
    """merge_rename_motif_collections works for motifs without an AC field."""
    motif = _make_motif(GenericMinimalTransfacMotif)
    result = merge_rename_motif_collections({"DB": [motif]})
    assert len(result) == 1


def test_merge_rename_motif_collections_two_collections():
    m1 = _make_motif(GenericMinimalTransfacMotif)
    m2 = _make_motif_with_ac()
    result = merge_rename_motif_collections({"DB1": [m1], "DB2": [m2]})
    ids = [m.id for m in result]
    assert len(ids) == len(set(ids)), "All merged motif IDs must be unique"


# ---------------------------------------------------------------------------
# Biopython integration
# ---------------------------------------------------------------------------


def test_motif_to_biopython_motif():
    motif = _make_motif_with_ac()
    bio_motif = motif_to_biopython_motif(motif)
    assert isinstance(bio_motif, Bio.motifs.Motif)
    assert len(bio_motif) == len(motif.matrix)


def test_motif_to_biopython_motif_wrong_alphabet():
    motif = _make_motif(GenericMinimalTransfacMotif)
    rna_motif = motif.relabel_alphabet({"A": "A", "C": "C", "G": "G", "T": "U"})
    with pytest.raises(MotifError):
        motif_to_biopython_motif(rna_motif)


# ---------------------------------------------------------------------------
# to_logomaker_df
# ---------------------------------------------------------------------------


def test_to_logomaker_df():
    motif = _make_motif(GenericMinimalTransfacMotif)
    df = motif.to_logomaker_df()
    assert df.index.name == "pos"
    assert list(df.index) == list(range(len(motif.matrix)))
    assert list(df.columns) == list(motif.matrix.columns)
