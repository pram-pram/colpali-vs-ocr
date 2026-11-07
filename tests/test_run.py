import json

import pytest
from fixtures import fake_ocr_fn, synthetic_corpus

from cvo.retrievers.base import IndexStats
from cvo.retrievers.bm25 import BM25Retriever
from cvo.run import ARMS, check_same_pages, parse_args, run_arm


class FakeRetriever:
    name = "fake"
    model_id = "fake-model"

    def __init__(self, perfect: bool) -> None:
        self.perfect = perfect

    def index(self, pages):
        return IndexStats(seconds=2.0, bytes_stored=1000, num_pages=len(pages), pages_failed=0)

    def search(self, queries, k):
        out = {}
        for q in queries:
            rel = sorted(q.relevant_page_ids)
            others = ["x1", "x2"]
            ids = rel + others if self.perfect else others + rel
            out[q.id] = [(pid, 1.0 / (i + 1)) for i, pid in enumerate(ids)][:k]
        return out


def test_run_arm_writes_results(tmp_path):
    pages, queries, _ = synthetic_corpus()
    res = run_arm("tiny", FakeRetriever(perfect=True), pages, queries, tmp_path, limit=None)
    saved = json.loads((tmp_path / "tiny" / "fake.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["ndcg@5"] == 1.0 and set(saved["per_query"]) == {q.id for q in queries}
    assert saved["cost"]["bytes_per_page"] == res["index_stats"]["bytes_stored"] / len(pages)
    assert saved["cost"]["index_seconds_per_page"] == 2.0 / len(pages)
    assert saved["cost"]["query_ms"] >= 0
    assert saved["subset"] == "tiny" and saved["arm"] == "fake"
    assert saved["meta"]["model_id"] == "fake-model" and saved["meta"]["limit"] is None
    assert all(isinstance(s, float) for _, s in saved["per_query"]["q0"])


def test_check_same_pages_rejects_mismatch():
    with pytest.raises(ValueError):
        check_same_pages([{"index_stats": {"num_pages": 5}}, {"index_stats": {"num_pages": 4}}])
    check_same_pages([{"index_stats": {"num_pages": 5}}, {"index_stats": {"num_pages": 5}}])


def test_parse_args_defaults_and_validation():
    args = parse_args(["--subset", "all"])
    assert args.arms == list(ARMS) and args.limit is None
    assert str(args.artifacts) == "artifacts" and str(args.results) == "results"
    assert parse_args(["--subset", "shift", "--arms", "bm25,dense"]).arms == ["bm25", "dense"]
    with pytest.raises(SystemExit):
        parse_args(["--subset", "shift", "--arms", "bogus"])


def test_bm25_reindex_does_not_double_count_bytes(tmp_path):
    pages, _, texts = synthetic_corpus()
    r = BM25Retriever(fake_ocr_fn(texts), tmp_path)
    first = r.index(pages).bytes_stored
    assert r.index(pages).bytes_stored == first


def test_run_arm_rejects_empty_queries_or_pages(tmp_path):
    pages, queries, _ = synthetic_corpus()
    with pytest.raises(ValueError, match="no queries"):
        run_arm("tiny", FakeRetriever(perfect=True), pages, [], tmp_path, limit=None)
    with pytest.raises(ValueError, match="no pages"):
        run_arm("tiny", FakeRetriever(perfect=True), [], queries, tmp_path, limit=None)
