"""Page images -> ColQwen2 multi-vector embeddings (sharded on disk) -> MaxSim retrieval."""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image

from cvo.artifacts import atomic_write_bytes
from cvo.data import Page, Query
from cvo.retrievers.base import IndexStats


class MultiVectorEncoder(Protocol):
    model_id: str

    def encode_images(self, images: list[Image.Image]) -> list[torch.Tensor]:
        """One `[n_patches, d]` tensor per image."""
        ...

    def encode_queries(self, texts: list[str]) -> list[torch.Tensor]:
        """One `[n_tokens, d]` tensor per text."""
        ...


class ColQwen2Encoder:
    model_id = "vidore/colqwen2-v1.0"

    def __init__(self) -> None:
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self._model = ColQwen2.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="cuda:0"
        ).eval()
        self._processor = ColQwen2Processor.from_pretrained(self.model_id)

    def _embed(self, batch) -> list[torch.Tensor]:
        batch = batch.to(self._model.device)
        with torch.no_grad():
            out = self._model(**batch)  # [batch, seq, 128], padding rows zeroed
        mask = batch["attention_mask"].bool()
        return [emb[m].to("cpu", torch.float16) for emb, m in zip(out, mask)]

    def encode_images(self, images: list[Image.Image]) -> list[torch.Tensor]:
        return self._embed(self._processor.process_images(images))

    def encode_queries(self, texts: list[str]) -> list[torch.Tensor]:
        return self._embed(self._processor.process_queries(texts))


def maxsim(query: torch.Tensor, docs: list[torch.Tensor]) -> torch.Tensor:
    """score_i = sum over query rows of the max dot product with any row of doc i (float32)."""
    q = query.float()
    return torch.stack([(q @ d.float().T).max(dim=1).values.sum() for d in docs])


class ColQwenRetriever:
    name = "colqwen"

    def __init__(
        self,
        encoder: MultiVectorEncoder,
        store_dir: Path,
        batch_size: int = 4,
        shard_size: int = 100,
    ) -> None:
        self._encoder = encoder
        self.model_id = encoder.model_id
        self._store_dir = Path(store_dir)
        self.batch_size = batch_size
        self._shard_size = shard_size
        self._page_ids: list[str] = []
        self._embeddings: list[torch.Tensor] = []

    def _encode_pages(self, pages: list[Page]) -> list[torch.Tensor]:
        """Encode in batches; on CUDA OOM halve the batch size and retry from the same page."""
        embeddings: list[torch.Tensor] = []
        while len(embeddings) < len(pages):
            start = len(embeddings)
            batch = pages[start : start + self.batch_size]
            try:
                embeddings.extend(self._encoder.encode_images([p.image for p in batch]))
            except torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.batch_size //= 2
        return embeddings

    def _shard(self, i: int, pages: list[Page]) -> dict:
        path = self._store_dir / f"shard_{i:05d}.pt"
        page_ids = [p.id for p in pages]
        if path.exists():
            shard = torch.load(path, weights_only=True)
            if shard["page_ids"] == page_ids:
                return shard
        start = time.perf_counter()
        embeddings = self._encode_pages(pages)
        shard = {
            "page_ids": page_ids,
            "embeddings": embeddings,
            "seconds": time.perf_counter() - start,
        }
        buf = io.BytesIO()
        torch.save(shard, buf)
        atomic_write_bytes(path, buf.getvalue())
        return shard

    def index(self, pages: list[Page]) -> IndexStats:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._page_ids, self._embeddings = [], []
        seconds = 0.0
        for i, start in enumerate(range(0, len(pages), self._shard_size)):
            shard = self._shard(i, pages[start : start + self._shard_size])
            self._page_ids.extend(shard["page_ids"])
            self._embeddings.extend(shard["embeddings"])
            seconds += shard["seconds"]
        return IndexStats(
            seconds=seconds,
            bytes_stored=sum(e.nbytes for e in self._embeddings),
            num_pages=len(pages),
            pages_failed=0,
        )

    def search(
        self, queries: list[Query], k: int
    ) -> dict[str, list[tuple[str, float]]]:
        results: dict[str, list[tuple[str, float]]] = {q.id: [] for q in queries}
        if not self._embeddings or not queries:
            return results
        query_embeddings: list[torch.Tensor] = []
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            query_embeddings.extend(self._encoder.encode_queries([q.text for q in batch]))
        for query, q_emb in zip(queries, query_embeddings):
            scores = maxsim(q_emb, self._embeddings)
            top = torch.topk(scores, min(k, len(scores)))
            results[query.id] = [
                (self._page_ids[i], float(s)) for s, i in zip(top.values, top.indices)
            ]
        return results
