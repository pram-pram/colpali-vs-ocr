"""Load ViDoRe subsets and normalize them into pages and queries."""
from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

SUBSETS: dict[str, str] = {
    "shift": "vidore/shiftproject_test",
    "tatdqa": "vidore/tatdqa_test",
    "infovqa": "vidore/infovqa_test_subsampled",
    "arxivqa": "vidore/arxivqa_test_subsampled",
}


@dataclass(frozen=True)
class Page:
    id: str
    # Encoded image bytes (as loaded from the dataset) or an in-memory PIL image.
    image: Image.Image | bytes = field(compare=False, repr=False)

    def pil(self) -> Image.Image:
        """A freshly decoded RGB image; encoded pages are never kept decoded in RAM."""
        if isinstance(self.image, bytes):
            return Image.open(io.BytesIO(self.image)).convert("RGB")
        return self.image.convert("RGB")


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant_page_ids: frozenset[str]


def normalize_rows(
    rows: Iterable[dict], limit: int | None = None
) -> tuple[list[Page], list[Query]]:
    """Dedupe pages by image_filename; merge relevant pages per distinct query text.

    With `limit`, keep the first N pages (first-seen order) and drop relevance
    links to cut pages; queries left with no relevant page are dropped.
    """
    pages: dict[str, Page] = {}
    relevant: dict[str, set[str]] = {}
    for row in rows:
        page_id = row["image_filename"]
        if page_id not in pages:
            image = row["image"]
            if isinstance(image, dict):  # undecoded datasets.Image: {"bytes", "path"}
                image = image["bytes"] or Path(image["path"]).read_bytes()
            pages[page_id] = Page(id=page_id, image=image)
        text = row.get("query")
        if text is None or not text.strip():
            continue
        relevant.setdefault(text, set()).add(page_id)

    page_list = list(pages.values())
    if limit is not None:
        page_list = page_list[:limit]
    kept = {p.id for p in page_list}

    queries: list[Query] = []
    for text, ids in relevant.items():
        ids &= kept
        if ids:
            queries.append(Query(f"q{len(queries):04d}", text, frozenset(ids)))
    return page_list, queries


def load_subset(key: str, limit: int | None = None) -> tuple[list[Page], list[Query]]:
    from datasets import Image as ImageFeature
    from datasets import load_dataset

    ds = load_dataset(SUBSETS[key], split="test")
    # Keep images encoded; decoding every page up front can exceed Colab RAM.
    ds = ds.cast_column("image", ImageFeature(decode=False))
    return normalize_rows(ds, limit)


def validate(pages: list[Page], queries: list[Query]) -> None:
    page_ids = {p.id for p in pages}
    for q in queries:
        if not q.relevant_page_ids:
            raise ValueError(f"query {q.id} has no relevant pages")
        missing = q.relevant_page_ids - page_ids
        if missing:
            raise ValueError(f"query {q.id} references unknown pages: {sorted(missing)}")
