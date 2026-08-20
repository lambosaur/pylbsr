"""Tests for pylbsr.ml's deprecated top-level re-export shim (ml/__init__.py's __getattr__).

Requires the `ml` extra (scikit-learn), since the shim resolves into ml.roc_prc.
"""

import pytest

pytest.importorskip("sklearn")

import pylbsr.ml


@pytest.mark.parametrize("name", ["ROCresults", "PRCresults", "ListROCresults", "ListPRCresults"])
def test_deprecated_names_still_resolve_and_warn(name: str) -> None:
    """pylbsr.ml.X still works for the four legacy names, but warns once per name."""
    # Reset the module's per-name warn-once cache so this test is independent of import order/
    # whatever earlier tests in the same process may have already triggered.
    pylbsr.ml._already_warned.discard(name)

    with pytest.warns(DeprecationWarning, match=f"pylbsr.ml.{name} is deprecated"):
        resolved = getattr(pylbsr.ml, name)

    from pylbsr.ml import roc_prc

    assert resolved is getattr(roc_prc, name)


def test_deprecated_name_warns_only_once_per_name() -> None:
    """A second access of the same already-warned name doesn't warn again."""
    pylbsr.ml._already_warned.discard("ROCresults")

    with pytest.warns(DeprecationWarning):
        _ = pylbsr.ml.ROCresults

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = pylbsr.ml.ROCresults
        assert len(w) == 0


def test_unknown_attribute_still_raises_attributeerror() -> None:
    """Only the four legacy names are special-cased; anything else is a real AttributeError."""
    with pytest.raises(AttributeError, match="no attribute 'NotARealThing'"):
        _ = pylbsr.ml.NotARealThing


def test_pca_and_clustering_are_not_reachable_at_package_level() -> None:
    """pca/clustering were never part of the deprecated shim -- they need explicit submodule
    imports (from pylbsr.ml.pca import ..., from pylbsr.ml.clustering import ...), matching
    bio/__init__.py's existing convention of not eagerly re-exporting every submodule.
    """
    with pytest.raises(AttributeError):
        _ = pylbsr.ml.plot_cumulative_variance_pca
    with pytest.raises(AttributeError):
        _ = pylbsr.ml.hierarchical_clustering_cut_tree
