"""Tests for plotting.make_categorical_palette.

Only covers what was newly added here; the rest of plotting.py (setup_mpl_everything,
hex_luminance, adjust_lightness, stable_categorical_color) was untested before this and is
out of scope for this change.
"""

from pylbsr.plotting import make_categorical_palette


def test_make_categorical_palette_covers_every_id_with_a_hex_color() -> None:
    """Every input ID gets a mapped, valid-looking hex color string."""
    palette = make_categorical_palette(["b", "a", "c"])

    assert set(palette) == {"a", "b", "c"}
    for color in palette.values():
        assert color.startswith("#")
        assert len(color) == 7


def test_make_categorical_palette_is_deterministic() -> None:
    """Same input set (any order) always produces the same mapping."""
    p1 = make_categorical_palette(["cluster_2", "cluster_0", "cluster_1"])
    p2 = make_categorical_palette(["cluster_1", "cluster_2", "cluster_0"])

    assert p1 == p2


def test_make_categorical_palette_no_collisions_within_set() -> None:
    """Assigning from a large-enough palette, distinct IDs get distinct colors."""
    palette = make_categorical_palette(list(range(10)), palette="tab20")

    assert len(set(palette.values())) == 10
