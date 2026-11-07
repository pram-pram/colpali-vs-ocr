from pytest import approx
from cvo.eval import ndcg_at_k, hit_at_k, evaluate

def test_ndcg_single_relevant_at_rank2():
    assert ndcg_at_k(["a", "b", "c"], {"b"}, 5) == approx(0.6309, abs=1e-4)

def test_ndcg_two_relevant():
    assert ndcg_at_k(["a", "b", "c"], {"a", "c"}, 5) == approx(1.5 / 1.6309, abs=1e-4)

def test_ndcg_relevant_beyond_k_is_zero():
    assert ndcg_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

def test_hit_at_k():
    assert hit_at_k(["a", "b"], {"b"}, 1) == 0.0
    assert hit_at_k(["a", "b"], {"b"}, 2) == 1.0

def test_ranking_shorter_than_k():  # Review Focus 5
    assert ndcg_at_k(["a"], {"a"}, 5) == 1.0
    assert hit_at_k(["a"], {"a"}, 10) == 1.0

def test_evaluate_means_and_missing_query():
    m = evaluate({"q1": ["a", "b"]}, {"q1": {"a"}, "q2": {"x"}})
    assert m == {"ndcg@5": 0.5, "recall@1": 0.5, "recall@5": 0.5, "recall@10": 0.5}
