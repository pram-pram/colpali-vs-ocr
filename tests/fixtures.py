from PIL import Image

from cvo.data import Page


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
