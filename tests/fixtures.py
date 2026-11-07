from PIL import Image, ImageDraw

from cvo.data import Page, Query
from cvo.ocr import OcrResult
from cvo.retrievers.base import OcrFn


def row(image_filename: str, query: str | None) -> dict:
    return {
        "image_filename": image_filename,
        "query": query,
        "image": Image.new("RGB", (8, 8)),
    }


def make_pages(n: int) -> list[Page]:
    """Pages p0..p{n-1}, each a distinct solid colour."""
    return [
        Page(id=f"p{i}", image=Image.new("RGB", (8, 8), (i % 256, (i // 256) % 256, 7)))
        for i in range(n)
    ]


EMPTY_PAGE_ID = "s4"

_SYNTHETIC_TEXTS = {
    "s0": "The volcano erupted and lava flowed down the mountainside near the village.",
    "s1": "Quarterly revenue increased because the bakery sold more croissants and baguettes.",
    "s2": "Astronomers photographed a distant nebula using the orbiting telescope.",
    "s3": "The football tournament final ended with a penalty shootout at the stadium.",
    EMPTY_PAGE_ID: "",
}


def _draw_page(page_id: str, text: str) -> Page:
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 50:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for i, content in enumerate(lines):
        draw.text((10, 10 + 20 * i), content, fill="black")
    return Page(id=page_id, image=image)


def synthetic_corpus() -> tuple[list[Page], list[Query], dict[str, str]]:
    """Five pages with text drawn on them (s4 is blank), three keyword-unambiguous queries."""
    texts = dict(_SYNTHETIC_TEXTS)
    pages = [_draw_page(pid, text) for pid, text in texts.items()]
    queries = [
        Query(id="q0", text="volcano lava", relevant_page_ids=frozenset({"s0"})),
        Query(id="q1", text="croissants bakery revenue", relevant_page_ids=frozenset({"s1"})),
        Query(id="q2", text="telescope nebula", relevant_page_ids=frozenset({"s2"})),
    ]
    return pages, queries, texts


def fake_ocr_fn(texts: dict[str, str]) -> OcrFn:
    def ocr(pages: list[Page]) -> OcrResult:
        return OcrResult(
            texts={p.id: texts[p.id] for p in pages},
            seconds=0.01 * len(pages),
            pages_failed=0,
        )

    return ocr
