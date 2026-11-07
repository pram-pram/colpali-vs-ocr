"""Artifact paths, crash-safe writes and run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# result key -> distribution name (differs from import name for some packages)
_DISTRIBUTIONS = {
    "torch": "torch",
    "transformers": "transformers",
    "colpali_engine": "colpali-engine",
    "doctr": "python-doctr",
    "sentence_transformers": "sentence-transformers",
    "bm25s": "bm25s",
    "datasets": "datasets",
}


def page_key(page_id: str) -> str:
    return hashlib.sha1(page_id.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def save_json(path: Path, obj) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    atomic_write_bytes(path, text.encode("utf-8"))


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm_dir(root: Path, subset: str, arm: str, model_id: str) -> Path:
    d = Path(root) / subset / arm / model_id.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def run_metadata(limit: int | None) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": {k: _version(d) for k, d in _DISTRIBUTIONS.items()},
        "gpu": _gpu_name(),
        "limit": limit,
    }
