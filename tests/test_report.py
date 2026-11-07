import json

import pytest

from cvo.report import load_results, plot_all, select_gallery, summary_table


def _result(subset, arm, ndcg, per_query=None):
    return {
        "subset": subset,
        "arm": arm,
        "metrics": {"ndcg@5": ndcg, "recall@1": 0.5, "recall@5": 0.7, "recall@10": 0.9},
        "cost": {"index_seconds_per_page": 0.25, "query_ms": 12.0, "bytes_per_page": 2048.0},
        "index_stats": {"seconds": 5.0, "bytes_stored": 40960, "num_pages": 20, "pages_failed": 0},
        "per_query": per_query or {},
        "meta": {},
    }


def _results():
    return {
        s: {a: _result(s, a, 0.4 + 0.1 * i) for i, a in enumerate(["bm25", "dense", "colqwen"])}
        for s in ["shift", "tatdqa"]
    }


def test_load_results(tmp_path):
    for subset, arms in _results().items():
        (tmp_path / subset).mkdir()
        for arm, res in arms.items():
            (tmp_path / subset / f"{arm}.json").write_text(json.dumps(res))
    loaded = load_results(tmp_path)
    assert set(loaded) == {"shift", "tatdqa"}
    assert set(loaded["shift"]) == {"bm25", "dense", "colqwen"}
    assert loaded["tatdqa"]["colqwen"]["metrics"]["ndcg@5"] == pytest.approx(0.6)


def test_load_results_missing_dir(tmp_path):
    assert load_results(tmp_path / "nope") == {}


def test_select_gallery():
    a = _result("s", "colqwen", 0, {
        "q1": [["p1", 1.0], ["p2", 0.5]],           # a hit at rank 1
        "q2": [["p9", 1.0], ["p8", 0.5]],           # a miss
        "q3": [["p3", 1.0]],                        # both hit
        "q4": [["p4", 1.0]],                        # a hit at rank 1
    })
    b = _result("s", "bm25", 0, {
        "q1": [["x", 1.0], ["y", 0.5]],             # none in top 5
        "q2": [["p2", 1.0]],                        # b hit at rank 1
        "q3": [["p3", 1.0]],
        "q4": [["z", 1.0], ["w", 1.0], ["v", 1.0], ["u", 1.0], ["t", 1.0], ["p4", 0.1]],  # rank 6
    })
    relevant = {"q1": {"p1"}, "q2": {"p2"}, "q3": {"p3"}, "q4": {"p4"}}
    assert select_gallery(a, b, relevant) == {"a_wins": ["q1", "q4"], "b_wins": ["q2"]}
    assert select_gallery(a, b, relevant, n=1) == {"a_wins": ["q1"], "b_wins": ["q2"]}


def test_summary_table():
    table = summary_table(_results())
    lines = table.splitlines()
    assert lines[0].startswith("| Subset | Arm | nDCG@5 | R@1 | R@5 | R@10 | s/page | ms/query | KB/page |")
    assert set(lines[1].replace("|", "").strip()) <= {"-", " ", ":"}
    assert len(lines) == 2 + 6
    assert "| shift | colqwen | 0.600 |" in table
    assert "| 2.0 |" in lines[2]  # 2048 bytes -> 2.0 KB


def test_plot_all(tmp_path):
    paths = plot_all(_results(), tmp_path / "plots")
    assert [p.name for p in paths] == ["ndcg_by_subset.png", "quality_vs_cost.png", "storage_per_page.png"]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert p.stat().st_size > 1000
