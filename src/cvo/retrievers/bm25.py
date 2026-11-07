"""OCR text -> chunks -> BM25 page retrieval."""
from __future__ import annotations

import time
from pathlib import Path

import bm25s
import numpy as np

from cvo.data import Page, Query
from cvo.retrievers.base import IndexStats, OcrFn, max_per_page, ocr_chunks


class BM25Retriever:
    name = "bm25"
    model_id = "bm25s"

    def __init__(self, ocr_fn: OcrFn, store_dir: Path) -> None:
        self._ocr_fn = ocr_fn
        self._store_dir = Path(store_dir)
        self._chunk_page_ids: list[str] = []
        self._vocab: dict[str, int] = {}
        self._bm25: bm25s.BM25 | None = None

    def index(self, pages: list[Page]) -> IndexStats:
        chunks, self._chunk_page_ids, ocr = ocr_chunks(pages, self._ocr_fn)

        start = time.perf_counter()
        self._bm25, self._vocab = None, {}
        if chunks:
            tokens = bm25s.tokenize(chunks, stopwords=None, show_progress=False)
            self._vocab = tokens.vocab
            self._bm25 = bm25s.BM25()
            self._bm25.index(tokens, show_progress=False)
            self._store_dir.mkdir(parents=True, exist_ok=True)
            self._bm25.save(self._store_dir)
        indexing_seconds = time.perf_counter() - start

        stored = (
            sum(f.stat().st_size for f in self._store_dir.rglob("*") if f.is_file())
            if self._store_dir.exists()
            else 0
        )
        return IndexStats(
            seconds=ocr.seconds + indexing_seconds,
            bytes_stored=stored,
            num_pages=len(pages),
            pages_failed=ocr.pages_failed,
        )

    def search(
        self, queries: list[Query], k: int
    ) -> dict[str, list[tuple[str, float]]]:
        results: dict[str, list[tuple[str, float]]] = {q.id: [] for q in queries}
        if self._bm25 is None:
            return results
        num_chunks = len(self._chunk_page_ids)
        for query in queries:
            # Same scheme as the corpus; tokens outside the index vocabulary are dropped.
            (words,) = bm25s.tokenize(
                [query.text], stopwords=None, return_ids=False, show_progress=False
            )
            known = [w for w in words if w in self._vocab]
            if not known:
                continue
            docs, scores = self._bm25.retrieve(
                [known], k=num_chunks, show_progress=False
            )
            page_ids = [self._chunk_page_ids[i] for i in docs[0]]
            results[query.id] = max_per_page(page_ids, np.asarray(scores[0]), k)
        return results
