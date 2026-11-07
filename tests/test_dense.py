import zlib

import numpy as np
from fixtures import EMPTY_PAGE_ID, fake_ocr_fn, synthetic_corpus

from cvo.chunking import chunk_text
from cvo.retrievers.dense import DenseRetriever

DIM = 256


class HashingEmbedder:
    """Deterministic bag-of-words stand-in for bge-m3."""

    model_id = "hashing-256"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, zlib.crc32(word.encode()) % DIM] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def test_dense_finds_known_pages():
    pages, queries, texts = synthetic_corpus()
    r = DenseRetriever(fake_ocr_fn(texts), HashingEmbedder())
    assert r.name == "dense" and r.model_id == "hashing-256"
    stats = r.index(pages)
    n_chunks = sum(len(chunk_text(t)) for t in texts.values())
    assert stats.num_pages == 5 and stats.bytes_stored == n_chunks * DIM * 4
    assert stats.pages_failed == 0 and stats.seconds >= 0.05
    top = r.search(queries, k=3)
    assert all(top[q.id][0][0] in q.relevant_page_ids for q in queries)


def test_empty_ocr_page_never_returned():
    pages, queries, texts = synthetic_corpus()
    r = DenseRetriever(fake_ocr_fn(texts), HashingEmbedder())
    r.index(pages)
    returned = {pid for hits in r.search(queries, k=5).values() for pid, _ in hits}
    assert EMPTY_PAGE_ID not in returned


def test_all_pages_empty():
    pages, queries, _ = synthetic_corpus()
    r = DenseRetriever(fake_ocr_fn({p.id: "" for p in pages}), HashingEmbedder())
    stats = r.index(pages)
    assert stats.num_pages == 5 and stats.bytes_stored == 0
    assert r.search(queries, k=3) == {q.id: [] for q in queries}
