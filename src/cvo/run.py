"""Benchmark runner: one results JSON per (subset, arm)."""
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from cvo.artifacts import arm_dir, run_metadata, save_json
from cvo.data import SUBSETS, Page, Query, load_subset, validate
from cvo.eval import evaluate
from cvo.ocr import DoctrEngine, OcrResult, run_ocr
from cvo.retrievers.base import Retriever

ARMS = ("bm25", "dense", "colqwen")


def run_arm(
    subset: str,
    retriever: Retriever,
    pages: list[Page],
    queries: list[Query],
    out_dir: Path,
    limit: int | None,
    k: int = 10,
) -> dict:
    validate(pages, queries)
    stats = retriever.index(pages)

    start = time.perf_counter()
    hits = retriever.search(queries, k)
    query_ms = (time.perf_counter() - start) / len(queries) * 1000

    per_query = {
        qid: [[page_id, float(score)] for page_id, score in ranked]
        for qid, ranked in hits.items()
    }
    metrics = evaluate(
        {qid: [page_id for page_id, _ in ranked] for qid, ranked in per_query.items()},
        {q.id: set(q.relevant_page_ids) for q in queries},
    )
    result = {
        "subset": subset,
        "arm": retriever.name,
        "metrics": metrics,
        "cost": {
            "index_seconds_per_page": stats.seconds / stats.num_pages,
            "query_ms": query_ms,
            "bytes_per_page": stats.bytes_stored / stats.num_pages,
        },
        "index_stats": asdict(stats),
        "per_query": per_query,
        "meta": {**run_metadata(limit), "model_id": retriever.model_id},
    }
    out_dir = Path(out_dir) / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / f"{retriever.name}.json", result)
    return result


def check_same_pages(results: list[dict]) -> None:
    counts = {r["index_stats"]["num_pages"] for r in results}
    if len(counts) > 1:
        raise ValueError(f"arms indexed different page counts: {sorted(counts)}")


_ocr_engine: DoctrEngine | None = None


def _ocr(pages: list[Page], cache_dir: Path) -> OcrResult:
    """OCR with one DoctrEngine per process, created on first use."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = DoctrEngine()
    return run_ocr(pages, _ocr_engine, cache_dir)


def build_retriever(arm: str, subset: str, artifacts: Path) -> Retriever:
    # Heavy imports stay inside the branches so --help and tests need no extras.
    if arm == "bm25":
        from cvo.retrievers.bm25 import BM25Retriever

        ocr_fn = partial(_ocr, cache_dir=arm_dir(artifacts, subset, "ocr", "doctr"))
        return BM25Retriever(ocr_fn, arm_dir(artifacts, subset, "bm25", "bm25s"))
    if arm == "dense":
        from cvo.retrievers.dense import BgeM3Embedder, DenseRetriever

        ocr_fn = partial(_ocr, cache_dir=arm_dir(artifacts, subset, "ocr", "doctr"))
        return DenseRetriever(ocr_fn, BgeM3Embedder())
    if arm == "colqwen":
        from cvo.retrievers.colqwen import ColQwen2Encoder, ColQwenRetriever

        store = arm_dir(artifacts, subset, "colqwen", "vidore/colqwen2-v1.0")
        return ColQwenRetriever(ColQwen2Encoder(), store)
    raise ValueError(f"unknown arm: {arm}")


def _arm_list(value: str) -> list[str]:
    arms = [a.strip() for a in value.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad or not arms:
        raise argparse.ArgumentTypeError(f"arms must be a comma list from {list(ARMS)}, got {value!r}")
    return arms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m cvo.run", description=__doc__)
    p.add_argument("--subset", required=True, choices=[*SUBSETS, "all"])
    p.add_argument("--arms", type=_arm_list, default=list(ARMS), help="comma list of bm25,dense,colqwen")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--results", type=Path, default=Path("results"))
    return p.parse_args(argv)


def _free_gpu() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]
    for subset in subsets:
        pages, queries = load_subset(subset, args.limit)
        results = []
        for arm in args.arms:
            retriever = build_retriever(arm, subset, args.artifacts)
            res = run_arm(subset, retriever, pages, queries, args.results, args.limit)
            results.append(res)
            print(
                f"{subset} {arm} ndcg@5={res['metrics']['ndcg@5']:.4f} "
                f"s/page={res['cost']['index_seconds_per_page']:.3f}"
            )
            del retriever
            _free_gpu()
        check_same_pages(results)


if __name__ == "__main__":
    main()
