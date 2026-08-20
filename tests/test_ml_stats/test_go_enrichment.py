"""Tests for ml_stats.go_enrichment -- goatools GO-enrichment result reformatting."""

import pandas as pd
import pytest

from pylbsr.ml_stats.go_enrichment import goatools_go_clusters_to_table, goatools_go_list_to_table


class _FakeGOTerm:
    """Minimal stand-in for a goatools GO-enrichment result record."""

    def __init__(
        self,
        go: str,
        name: str,
        ratio_in_study: str,
        ratio_in_pop: str,
        p_fdr_bh: float,
        enrichment: str = "e",
    ) -> None:
        self._fields = {
            "GO": go,
            "name": name,
            "enrichment": enrichment,
            "ratio_in_study": ratio_in_study,
            "ratio_in_pop": ratio_in_pop,
            "p_fdr_bh": p_fdr_bh,
        }

    def get_field_values(self, fields: list[str]) -> list[object]:
        return [self._fields[f] for f in fields]

    def get_pvalue(self) -> float:
        return float(self._fields["p_fdr_bh"])


def test_go_list_to_table_basic() -> None:
    """N_genes/ratio_study/ratio_pop/enrich are computed correctly from ratio strings."""
    terms = [_FakeGOTerm("GO:0001", "term one", "5/100", "10/1000", 0.01)]

    table = goatools_go_list_to_table(terms)

    assert table is not None
    assert table.loc[0, "N_genes"] == 5
    assert table.loc[0, "ratio_study"] == pytest.approx(0.05)
    assert table.loc[0, "ratio_pop"] == pytest.approx(0.01)
    assert table.loc[0, "enrich"] == pytest.approx(5.0)


def test_go_list_to_table_removes_empty_sets_by_default() -> None:
    """A term with 0 genes in the study set is dropped when remove_empty_sets=True."""
    terms = [
        _FakeGOTerm("GO:0001", "empty", "0/100", "10/1000", 0.01),
        _FakeGOTerm("GO:0002", "nonempty", "5/100", "10/1000", 0.01),
    ]

    table = goatools_go_list_to_table(terms, remove_empty_sets=True)

    assert table is not None
    assert list(table["GO"]) == ["GO:0002"]


def test_go_list_to_table_keeps_empty_sets_when_disabled() -> None:
    """remove_empty_sets=False keeps zero-gene terms."""
    terms = [_FakeGOTerm("GO:0001", "empty", "0/100", "10/1000", 0.01)]

    table = goatools_go_list_to_table(terms, remove_empty_sets=False)

    assert table is not None
    assert list(table["GO"]) == ["GO:0001"]


def test_go_list_to_table_empty_input_warns_and_returns_none() -> None:
    """No terms at all -> a UserWarning and None, not an empty/malformed DataFrame."""
    with pytest.warns(UserWarning, match="No GO terms found"):
        table = goatools_go_list_to_table([])

    assert table is None


def test_go_list_to_table_skips_malformed_record() -> None:
    """A record with an unparseable ratio string is skipped, not a crash."""
    terms = [
        _FakeGOTerm("GO:0001", "bad", "not-a-ratio", "10/1000", 0.01),
        _FakeGOTerm("GO:0002", "good", "5/100", "10/1000", 0.01),
    ]

    table = goatools_go_list_to_table(terms)

    assert table is not None
    assert list(table["GO"]) == ["GO:0002"]


def test_go_clusters_to_table_merges_and_tags_cluster() -> None:
    """Multiple clusters are concatenated, each row tagged with its cluster, sorted by p-value."""
    go_clusters = {
        "clusterA": {
            "significant": [
                _FakeGOTerm("GO:0002", "second", "5/100", "10/1000", 0.05),
                _FakeGOTerm("GO:0001", "first", "5/100", "10/1000", 0.01),
            ]
        },
        "clusterB": {
            "significant": [_FakeGOTerm("GO:0003", "only", "5/100", "10/1000", 0.02)],
        },
    }

    table = goatools_go_clusters_to_table(go_clusters)

    assert table is not None
    assert set(table["cluster"]) == {"clusterA", "clusterB"}

    cluster_a = table.loc[table["cluster"] == "clusterA"].reset_index(drop=True)
    assert list(cluster_a["GO"]) == ["GO:0001", "GO:0002"]  # sorted ascending by p-value


def test_go_clusters_to_table_empty_dict_warns_and_returns_none() -> None:
    """No clusters at all -> a UserWarning and None, not a pd.concat crash on an empty list."""
    with pytest.warns(UserWarning, match="No GO terms found"):
        table = goatools_go_clusters_to_table({})

    assert table is None


def test_go_clusters_to_table_skips_clusters_with_no_surviving_terms() -> None:
    """A cluster whose only term is an empty set contributes nothing, others still return."""
    go_clusters = {
        "empty_cluster": {
            "significant": [_FakeGOTerm("GO:0001", "empty", "0/100", "10/1000", 0.01)],
        },
        "real_cluster": {
            "significant": [_FakeGOTerm("GO:0002", "real", "5/100", "10/1000", 0.01)],
        },
    }

    table = goatools_go_clusters_to_table(go_clusters)

    assert table is not None
    assert list(table["cluster"]) == ["real_cluster"]


def test_go_list_to_table_returns_dataframe_type() -> None:
    """Sanity: a non-empty result really is a DataFrame, not a Series or list."""
    terms = [_FakeGOTerm("GO:0001", "term", "5/100", "10/1000", 0.01)]

    table = goatools_go_list_to_table(terms)

    assert isinstance(table, pd.DataFrame)
