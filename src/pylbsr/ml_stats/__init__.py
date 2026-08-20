"""Statistical/ML result reformatting and diagnostic plotting -- no torch required.

Submodules: go_enrichment, group_comparison, roc_prc, pca, clustering. Import directly
from the specific submodule (e.g. `from pylbsr.ml_stats.roc_prc import ROCresults`) --
nothing here is re-exported at the package level, so `import pylbsr.ml_stats` doesn't
eagerly pull in matplotlib/seaborn/scikit-learn/scipy.cluster.hierarchy just because some
submodule happens to need them.
"""

from . import go_enrichment as go_enrichment
