"""Retrieval metrics: binary-relevance nDCG@k, hit@k, and aggregate evaluation."""
import math


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(doc_id in relevant for doc_id in ranked_ids[:k]) else 0.0


def evaluate(
    rankings: dict[str, list[str]], relevant: dict[str, set[str]]
) -> dict[str, float]:
    n = len(relevant)
    if n == 0:
        return {"ndcg@5": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}

    def mean(fn, k):
        return sum(fn(rankings.get(q, []), rel, k) for q, rel in relevant.items()) / n

    return {
        "ndcg@5": mean(ndcg_at_k, 5),
        "recall@1": mean(hit_at_k, 1),
        "recall@5": mean(hit_at_k, 5),
        "recall@10": mean(hit_at_k, 10),
    }
