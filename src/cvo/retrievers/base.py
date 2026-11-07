"""Interface shared by every retrieval arm."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cvo.data import Page, Query
from cvo.ocr import OcrResult

OcrFn = Callable[[list[Page]], OcrResult]


@dataclass
class IndexStats:
    seconds: float
    bytes_stored: int
    num_pages: int
    pages_failed: int


class Retriever(Protocol):
    name: str
    model_id: str

    def index(self, pages: list[Page]) -> IndexStats: ...

    def search(
        self, queries: list[Query], k: int
    ) -> dict[str, list[tuple[str, float]]]: ...


def max_per_page(
    chunk_page_ids: list[str], chunk_scores: np.ndarray, k: int
) -> list[tuple[str, float]]:
    """Best chunk score per page, sorted descending, top k."""
    best: dict[str, float] = {}
    for page_id, score in zip(chunk_page_ids, chunk_scores):
        score = float(score)
        if page_id not in best or score > best[page_id]:
            best[page_id] = score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
