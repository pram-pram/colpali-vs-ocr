import pytest

from cvo.data import Query, SUBSETS, normalize_rows, validate
from fixtures import make_pages, row as r


def test_dedup_pages_and_skip_null_queries():
    rows = [r("p1", "what?"), r("p1", None), r("p2", "how?")]
    pages, queries = normalize_rows(rows)
    assert [p.id for p in pages] == ["p1", "p2"]
    assert [(q.id, q.text, q.relevant_page_ids) for q in queries] == [
        ("q0000", "what?", frozenset({"p1"})), ("q0001", "how?", frozenset({"p2"}))]


def test_blank_query_skipped():
    _, queries = normalize_rows([r("p1", "   "), r("p1", "ok")])
    assert [q.text for q in queries] == ["ok"]


def test_duplicate_query_text_merges_relevant():
    _, queries = normalize_rows([r("p1", "same"), r("p2", "same")])
    assert len(queries) == 1 and queries[0].relevant_page_ids == {"p1", "p2"}


def test_limit_drops_queries_whose_pages_were_cut():
    rows = [r("p1", "a"), r("p2", "b"), r("p3", "a")]
    pages, queries = normalize_rows(rows, limit=2)
    assert [p.id for p in pages] == ["p1", "p2"]
    assert {q.text: q.relevant_page_ids for q in queries} == {"a": {"p1"}, "b": {"p2"}}


def test_limit_drops_query_with_only_cut_pages():
    _, queries = normalize_rows([r("p1", "a"), r("p2", "b")], limit=1)
    assert [q.text for q in queries] == ["a"]


def test_validate_rejects_unknown_page():
    with pytest.raises(ValueError):
        validate([], [Query("q0", "x", frozenset({"p9"}))])


def test_validate_rejects_empty_relevant():
    with pytest.raises(ValueError):
        validate(make_pages(1), [Query("q0", "x", frozenset())])


def test_validate_accepts_good_data():
    validate(make_pages(2), [Query("q0", "x", frozenset({"p1"}))])


def test_subsets_mapping():
    assert SUBSETS == {
        "shift": "vidore/shiftproject_test",
        "tatdqa": "vidore/tatdqa_test",
        "infovqa": "vidore/infovqa_test_subsampled",
        "arxivqa": "vidore/arxivqa_test_subsampled",
    }
