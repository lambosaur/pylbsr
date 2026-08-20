"""GO-enrichment result reformatting, built on top of the `goatools` package."""

import logging
import warnings
from collections.abc import Iterable, Mapping
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GOEnrichmentResult(Protocol):
    """Interface for a single goatools GO-enrichment result record, as used here."""

    def get_field_values(self, fields: list[str]) -> list[object]:
        """Return this record's values for the given field names, in order."""
        ...

    def get_pvalue(self) -> float:
        """Return this record's (uncorrected) p-value, used for sorting."""
        ...


_GOTERM_FIELDS = ["GO", "name", "enrichment", "ratio_in_study", "ratio_in_pop", "p_fdr_bh"]


def goatools_go_list_to_table(
    goterms_list: Iterable[GOEnrichmentResult],
    remove_empty_sets: bool = True,
) -> pd.DataFrame | None:
    """Reformat a list of goatools GO-enrichment results into a DataFrame.

    Adds `N_genes` (parsed from `ratio_in_study`), `ratio_study`, `ratio_pop`
    (both as fractions), and `enrich` (`ratio_study / ratio_pop`).

    Args:
        goterms_list: goatools GO-enrichment result records.
        remove_empty_sets: Drop records with zero genes in the study set.

    Returns:
        One row per surviving GO term, or None if none survive (empty input,
        or all dropped as empty sets / malformed records).
    """
    selected_goterms: list[pd.Series] = []

    for go_term in goterms_list:
        try:
            info_goterm = pd.Series(go_term.get_field_values(_GOTERM_FIELDS), index=_GOTERM_FIELDS)
            info_goterm["N_genes"] = int(info_goterm["ratio_in_study"].split("/")[0])
        except (KeyError, ValueError, IndexError) as e:
            logger.warning("Skipping malformed GO-term record: %s", e)
            continue

        if remove_empty_sets and info_goterm["N_genes"] == 0:
            continue

        selected_goterms.append(info_goterm)

    if not selected_goterms:
        warnings.warn("No GO terms found; returning None.", UserWarning, stacklevel=2)
        return None

    table = pd.DataFrame(selected_goterms)

    table["ratio_study"] = table["ratio_in_study"].apply(
        lambda v: int(v.split("/")[0]) / int(v.split("/")[1])
    )
    table["ratio_pop"] = table["ratio_in_pop"].apply(
        lambda v: int(v.split("/")[0]) / int(v.split("/")[1])
    )
    table["enrich"] = table["ratio_study"] / table["ratio_pop"]

    return table


def goatools_go_clusters_to_table(
    go_clusters: Mapping[str, Mapping[str, Iterable[GOEnrichmentResult]]],
) -> pd.DataFrame | None:
    """Reformat per-cluster goatools GO-enrichment results into one long DataFrame.

    Args:
        go_clusters: Mapping of cluster name to a mapping containing at least a
            "significant" key with that cluster's significant GO-enrichment
            result records.

    Returns:
        One row per (cluster, GO term), tagged by a `cluster` column, sorted by
        p-value within each cluster; or None if no cluster yields any terms.
    """
    tables: list[pd.DataFrame] = []

    for cluster_name, cluster_results in go_clusters.items():
        sorted_go_terms = sorted(cluster_results["significant"], key=lambda gt: gt.get_pvalue())

        table = goatools_go_list_to_table(sorted_go_terms, remove_empty_sets=True)
        if table is None:
            continue
        tables.append(table.assign(cluster=cluster_name))

    if not tables:
        warnings.warn("No GO terms found in any cluster; returning None.", UserWarning, stacklevel=2)
        return None

    return pd.concat(tables, ignore_index=True)
