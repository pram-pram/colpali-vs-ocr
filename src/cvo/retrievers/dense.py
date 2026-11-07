"""OCR text -> chunks -> bge-m3 embeddings -> brute-force cosine page retrieval."""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from cvo.data import Page, Query
from cvo.retrievers.base import IndexStats, OcrFn, max_per_page, ocr_chunks


class Embedder(Protocol):
    model_id: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized float32 row per text."""
        ...


class BgeM3Embedder:
    model_id = "BAAI/bge-m3"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).astype(np.float32)


class DenseRetriever:
    name = "dense"

    def __init__(self, ocr_fn: OcrFn, embedder: Embedder) -> None:
        self._ocr_fn = ocr_fn
        self._embedder = embedder
        self.model_id = embedder.model_id
        self._chunk_page_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    def index(self, pages: list[Page]) -> IndexStats:
        chunks, self._chunk_page_ids, ocr = ocr_chunks(pages, self._ocr_fn)

        start = time.perf_counter()
        self._embeddings = self._embedder.encode(chunks) if chunks else None
        indexing_seconds = time.perf_counter() - start

        return IndexStats(
            seconds=ocr.seconds + indexing_seconds,
            bytes_stored=self._embeddings.nbytes if self._embeddings is not None else 0,
            num_pages=len(pages),
            pages_failed=ocr.pages_failed,
        )

    def search(
        self, queries: list[Query], k: int
    ) -> dict[str, list[tuple[str, float]]]:
        results: dict[str, list[tuple[str, float]]] = {q.id: [] for q in queries}
        if self._embeddings is None or not queries:
            return results
        scores = self._embedder.encode([q.text for q in queries]) @ self._embeddings.T
        for query, row in zip(queries, scores):
            results[query.id] = max_per_page(self._chunk_page_ids, row, k)
        return results
