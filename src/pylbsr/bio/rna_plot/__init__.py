"""RNA secondary-structure drawings with a flat exterior loop (helices above the backbone).

Requires ViennaRNA (`pip install pylbsr[rna]`). Not imported by `pylbsr.bio`, so the rest of
the package works without it.

- `layout`: coordinates (`flat_coordinates`) and layout checks.
- `draw`: annotated drawing (`draw_structure`, `Loop`, `Track`).
- `colors`: RNAfold pair-probability and RNAalifold covariation colourings.
"""

from .colors import (
    covariation_colors as covariation_colors,
    covariation_handles as covariation_handles,
    pairing_probability as pairing_probability,
    probability_colorbar as probability_colorbar,
    probability_colors as probability_colors,
)
from .draw import (
    Loop as Loop,
    StructurePlot as StructurePlot,
    Track as Track,
    draw_structure as draw_structure,
    hairpin_loops as hairpin_loops,
    suggest_figsize as suggest_figsize,
)
from .layout import (
    flat_coordinates as flat_coordinates,
    min_distance as min_distance,
    pair_list as pair_list,
    pair_table as pair_table,
    segment_crossings as segment_crossings,
)
