"""Tests for pylbsr.ml -- the deprecated top-level re-export shim (ml.py's __getattr__).

No optional extra required: scikit-learn is a core dependency (see pyproject.toml), since
none of ml_stats' content needs torch.
"""

import pytest

import pylbsr.ml


@pytest.mark.parametrize("name", ["ROCresults", "PRCresults", "ListROCresults", "ListPRCresults"])
def test_deprecated_names_still_resolve_and_warn(name: str) -> None:
    """pylbsr.ml.X still works for the four legacy names, but warns once per name."""
    # Reset the module's per-name warn-once cache so this test is independent of import order/
    # whatever earlier tests in the same process may have already triggered.
    pylbsr.ml._already_warned.discard(name)

    with pytest.warns(DeprecationWarning, match=f"pylbsr.ml.{name} is deprecated"):
        resolved = getattr(pylbsr.ml, name)

    from pylbsr.ml_stats import roc_prc

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


def test_pca_and_clustering_are_not_reachable_via_the_shim() -> None:
    """pca/clustering were never part of the deprecated shim -- they need explicit submodule
    imports (from pylbsr.ml_stats.pca import ..., from pylbsr.ml_stats.clustering import ...).
    pylbsr.ml only ever special-cased the four original ROC/PRC names.
    """
    with pytest.raises(AttributeError):
        _ = pylbsr.ml.plot_cumulative_variance_pca
    with pytest.raises(AttributeError):
        _ = pylbsr.ml.hierarchical_clustering_cut_tree
