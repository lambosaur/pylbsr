"""Deprecated compatibility shim for the four legacy `pylbsr.ml.X` names.

The real content moved to `pylbsr.ml_stats` (ROC/PRC/PCA/clustering evaluation and
diagnostic plotting live alongside the GO-enrichment/group-comparison statistics
there -- none of it needs torch, so it never belonged next to `torch_utils.py`).
Import from the specific submodule instead:

    from pylbsr.ml_stats.roc_prc import ROCresults
    from pylbsr.ml_stats.pca import plot_cumulative_variance_pca
    from pylbsr.ml_stats.clustering import hierarchical_clustering_cut_tree

The four names below remain reachable as `pylbsr.ml.X` for backward compatibility with
existing external consumers, via a deprecated, lazily-resolved __getattr__ -- new code
should use the explicit submodule import above instead.
"""

import warnings
from typing import Any

_DEPRECATED_ROC_PRC_NAMES = frozenset(
    {"ROCresults", "PRCresults", "ListROCresults", "ListPRCresults"}
)

# `from pylbsr.ml import X` triggers __getattr__ twice per statement (once via the import
# system's internal hasattr() check, once via the actual IMPORT_FROM binding) -- track which
# names have already warned so a single import statement doesn't double-print.
_already_warned: set[str] = set()


def __getattr__(name: str) -> Any:  # noqa: ANN401 -- PEP 562 module __getattr__ is inherently dynamic
    """Resolve legacy `pylbsr.ml.X` access to `pylbsr.ml_stats.roc_prc.X`, warning once per name."""
    if name in _DEPRECATED_ROC_PRC_NAMES:
        if name not in _already_warned:
            _already_warned.add(name)
            warnings.warn(
                f"pylbsr.ml.{name} is deprecated; import it from pylbsr.ml_stats.roc_prc instead "
                f"(`from pylbsr.ml_stats.roc_prc import {name}`).",
                DeprecationWarning,
                stacklevel=2,
            )
        from pylbsr.ml_stats import roc_prc

        return getattr(roc_prc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
