import numpy as np
from fixtures import EMPTY_PAGE_ID, fake_ocr_fn, synthetic_corpus

from cvo.data import Query
from cvo.retrievers.base import max_per_page
from cvo.retrievers.bm25 import BM25Retriever


def test_max_per_page():
    assert max_per_page(["a", "a", "b"], np.array([1.0, 3.0, 2.0]), 5) == [("a", 3.0), ("b", 2.0)]


def test_bm25_finds_known_pages(tmp_path):
    pages, queries, texts = synthetic_corpus()
    r = BM25Retriever(fake_ocr_fn(texts), tmp_path)
    assert r.model_id == "bm25s"
    stats = r.index(pages)
    assert stats.num_pages == 5 and stats.bytes_stored > 0
    assert stats.pages_failed == 0 and stats.seconds >= 0.05
    top = r.search(queries, k=3)
    assert all(top[q.id][0][0] in q.relevant_page_ids for q in queries)


def test_empty_ocr_page_never_returned(tmp_path):  # Review Focus 1
    pages, queries, texts = synthetic_corpus()
    r = BM25Retriever(fake_ocr_fn(texts), tmp_path)
    r.index(pages)
    returned = {pid for hits in r.search(queries, k=5).values() for pid, _ in hits}
    assert EMPTY_PAGE_ID not in returned


def test_unknown_word_query_returns_empty(tmp_path):
    pages, queries, texts = synthetic_corpus()
    r = BM25Retriever(fake_ocr_fn(texts), tmp_path)
    r.index(pages)
    unknown = Query(id="qx", text="zzzxqv wibblefrotz", relevant_page_ids=frozenset())
    mixed = Query(id="qm", text="zzzxqv volcano", relevant_page_ids=frozenset({"s0"}))
    top = r.search([unknown, mixed], k=3)
    assert top["qx"] == []
    assert top["qm"][0][0] == "s0"


def test_all_pages_empty(tmp_path):
    pages, queries, _ = synthetic_corpus()
    r = BM25Retriever(fake_ocr_fn({p.id: "" for p in pages}), tmp_path)
    stats = r.index(pages)
    assert stats.num_pages == 5
    assert r.search(queries, k=3) == {q.id: [] for q in queries}
