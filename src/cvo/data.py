"""Load ViDoRe subsets and normalize them into pages and queries."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

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
    image: Image.Image = field(compare=False, repr=False)


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
            pages[page_id] = Page(id=page_id, image=row["image"])
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
    from datasets import load_dataset

    return normalize_rows(load_dataset(SUBSETS[key], split="test"), limit)


def validate(pages: list[Page], queries: list[Query]) -> None:
    page_ids = {p.id for p in pages}
    for q in queries:
        if not q.relevant_page_ids:
            raise ValueError(f"query {q.id} has no relevant pages")
        missing = q.relevant_page_ids - page_ids
        if missing:
            raise ValueError(f"query {q.id} references unknown pages: {sorted(missing)}")
