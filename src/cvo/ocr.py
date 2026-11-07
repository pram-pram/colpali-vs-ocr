"""Cached OCR shared by the text retrieval arms."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from cvo.artifacts import load_json, page_key, save_json
from cvo.data import Page

log = logging.getLogger(__name__)


class OcrEngine(Protocol):
    def read(self, image: Image.Image) -> str: ...


class DoctrEngine:
    def __init__(self) -> None:
        import torch
        from doctr.models import ocr_predictor

        self._predictor = ocr_predictor(pretrained=True)
        if torch.cuda.is_available():
            self._predictor = self._predictor.cuda()

    def read(self, image: Image.Image) -> str:
        import numpy as np

        result = self._predictor([np.asarray(image.convert("RGB"))])
        return result.render()


@dataclass
class OcrResult:
    texts: dict[str, str]
    seconds: float
    pages_failed: int


def run_ocr(
    pages: list[Page], engine_factory: Callable[[], OcrEngine], cache_dir: Path
) -> OcrResult:
    """OCR each page once, caching per page. The engine is built only on the first cache miss."""
    cache_dir = Path(cache_dir)
    engine: OcrEngine | None = None
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    seconds = 0.0
    failed = 0
    for page in pages:
        path = cache_dir / f"{page_key(page.id)}.json"
        if path.exists():
            rec = load_json(path)
        else:
            if engine is None:
                engine = engine_factory()
            start = time.perf_counter()
            try:
                rec = {"text": engine.read(page.pil()), "failed": False}
            except Exception as exc:
                log.warning("OCR failed for page %s: %s", page.id, exc)
                rec = {"text": "", "failed": True}
            rec["seconds"] = time.perf_counter() - start
            save_json(path, rec)
        texts[page.id] = rec["text"]
        seconds += rec["seconds"]
        failed += bool(rec["failed"])
    return OcrResult(texts=texts, seconds=seconds, pages_failed=failed)
