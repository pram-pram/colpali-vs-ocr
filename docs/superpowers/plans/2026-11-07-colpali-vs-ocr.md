# ColPali vs OCR Retrieval Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible benchmark comparing BM25-on-OCR, dense-on-OCR and ColQwen2 page retrieval on four ViDoRe subsets, reporting quality (nDCG@5, Recall@k) and cost (index time, query latency, storage).

**Architecture:** A small `cvo` package where every arm implements one `Retriever` protocol; text arms share a cached docTR OCR step; ColQwen2 embeddings are sharded to disk for resume. Heavy models sit behind injectable protocols (`OcrEngine`, `Embedder`, `MultiVectorEncoder`) so all logic is tested on CPU with stubs; real model wrappers are thin and exercised on Colab.

**Tech Stack:** Python ≥3.11, torch, numpy, Pillow, `datasets`, `bm25s`, `sentence-transformers` (bge-m3), `python-doctr`, `colpali-engine` (ColQwen2), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-colpali-vs-ocr-design.md`

## Global Constraints

- `requires-python = ">=3.11"` (python-doctr 1.x floor; Colab runs 3.12).
- Core deps (CPU, installed for tests): `torch`, `numpy`, `pillow`, `datasets>=3`, `bm25s>=0.2`, `matplotlib`, `pytest`. Extras: `ocr = ["python-doctr[torch]>=1.0"]`, `dense = ["sentence-transformers>=3"]`, `vision = ["colpali-engine>=0.3.4"]`.
- Arm names are exactly `bm25`, `dense`, `colqwen`. Subset keys are exactly `shift`, `tatdqa`, `infovqa`, `arxivqa` mapping to `vidore/shiftproject_test`, `vidore/tatdqa_test`, `vidore/infovqa_test_subsampled`, `vidore/arxivqa_test_subsampled` (split `test`).
- Models: `BAAI/bge-m3`, `vidore/colqwen2-v1.0` (fp16, image batch size 4). OCR: docTR `ocr_predictor(pretrained=True)`.
- Chunking: 256 whitespace tokens, 32-token overlap. ColQwen shards: 100 pages.
- Metrics: nDCG@5 (binary relevance), Recall@1/5/10 as hit rate.
- Reported index time = sum of recorded fresh-compute seconds (OCR included for text arms), never cache-load time.
- All tests pass with `pytest -m "not gpu"` on CPU without the extras installed (import heavy libs lazily inside the real wrappers only).

## Review Focus

1. **Relevant page with empty OCR text** (figure-only page, OCR failure) → text arms simply never return it; query scores 0, no crash. Test in Task 6.
2. **`--limit` smaller than the corpus** → queries whose relevant pages were all cut are excluded, not counted as misses; partially-cut relevant sets keep only surviving ids. Test in Task 2.
3. **Same query text on multiple rows** → one `Query` with the union of relevant pages. Test in Task 2.
4. **Process killed mid-shard / reusing a shard from a `--limit` run** → no partial shard file is ever read; a shard whose stored page ids differ from the current pages is recomputed. Tests in Tasks 4 and 8.
5. **Fewer pages than k** (smoke runs) → rankings shorter than k; metrics still well-defined. Test in Task 1.

---

### Task 1: Project scaffold + metrics

**Files:**
- Create: `pyproject.toml`, `.gitignore` (`artifacts/`, `__pycache__/`, `*.egg-info/`, `.venv/`), `src/cvo/__init__.py`, `src/cvo/eval.py`, `tests/conftest.py`, `tests/test_eval.py`

**Interfaces:**
- Produces:
  - `ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float`
  - `hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float` (1.0 or 0.0)
  - `evaluate(rankings: dict[str, list[str]], relevant: dict[str, set[str]]) -> dict[str, float]` with keys `"ndcg@5", "recall@1", "recall@5", "recall@10"`, each the mean over queries in `relevant` (missing ranking = empty list).
  - pytest marker `gpu` registered in `pyproject.toml`; `tests/conftest.py` skips `gpu` tests when `torch.cuda.is_available()` is False.

- [ ] **Step 1: Write `pyproject.toml`** (setuptools, `src` layout, deps/extras from Global Constraints, `[tool.pytest.ini_options] markers = ["gpu: needs CUDA"]`, `pythonpath = ["src"]`), `.gitignore`, empty `__init__.py`, and `conftest.py`. Create a venv and `pip install -e .`.

- [ ] **Step 2: Write failing tests in `tests/test_eval.py`**

```python
from pytest import approx
from cvo.eval import ndcg_at_k, hit_at_k, evaluate

def test_ndcg_single_relevant_at_rank2():
    assert ndcg_at_k(["a", "b", "c"], {"b"}, 5) == approx(0.6309, abs=1e-4)

def test_ndcg_two_relevant():
    assert ndcg_at_k(["a", "b", "c"], {"a", "c"}, 5) == approx(1.5 / 1.6309, abs=1e-4)

def test_ndcg_relevant_beyond_k_is_zero():
    assert ndcg_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

def test_hit_at_k():
    assert hit_at_k(["a", "b"], {"b"}, 1) == 0.0
    assert hit_at_k(["a", "b"], {"b"}, 2) == 1.0

def test_ranking_shorter_than_k():  # Review Focus 5
    assert ndcg_at_k(["a"], {"a"}, 5) == 1.0
    assert hit_at_k(["a"], {"a"}, 10) == 1.0

def test_evaluate_means_and_missing_query():
    m = evaluate({"q1": ["a", "b"]}, {"q1": {"a"}, "q2": {"x"}})
    assert m == {"ndcg@5": 0.5, "recall@1": 0.5, "recall@5": 0.5, "recall@10": 0.5}
```

- [ ] **Step 3: Run** `pytest tests/test_eval.py -v` — Expected: FAIL (ImportError).
- [ ] **Step 4: Implement `src/cvo/eval.py`.** DCG = Σ 1/log2(i+1) over relevant hits at 1-based rank i ≤ k; IDCG uses min(|relevant|, k) ideal hits.
- [ ] **Step 5: Run** `pytest -v` — Expected: all PASS.
- [ ] **Step 6: Commit** `git add -A && git commit -m "feat: project scaffold and retrieval metrics"`

---

### Task 2: Data loading and normalization

**Files:**
- Create: `src/cvo/data.py`, `tests/test_data.py`, `tests/fixtures.py`

**Interfaces:**
- Produces (test helpers in `tests/fixtures.py`, reused by later tasks): `row(image_filename: str, query: str | None) -> dict` (image = `Image.new("RGB", (8, 8))`) and `make_pages(n: int) -> list[Page]` (ids `"p0"…`, each image a distinct solid colour so stubs can derive deterministic outputs).
- Produces:
  - `@dataclass(frozen=True) class Page: id: str; image: PIL.Image.Image` (`eq=False` on image comparison is fine — compare ids in tests)
  - `@dataclass(frozen=True) class Query: id: str; text: str; relevant_page_ids: frozenset[str]`
  - `SUBSETS: dict[str, str]` (key → HF dataset name, per Global Constraints)
  - `normalize_rows(rows: Iterable[dict], limit: int | None = None) -> tuple[list[Page], list[Query]]` — rows have keys `query` (str or None), `image`, `image_filename`. Pages deduped by `image_filename` in first-seen order; `limit` keeps the first N pages; query ids are `"q0000"`, `"q0001"`… in first-seen order of distinct query text.
  - `load_subset(key: str, limit: int | None = None) -> tuple[list[Page], list[Query]]` — `datasets.load_dataset(SUBSETS[key], split="test")` then `normalize_rows`.
  - `validate(pages: list[Page], queries: list[Query]) -> None` — raises `ValueError` if any query has empty `relevant_page_ids` or references an id not in `pages`.

- [ ] **Step 1: Write failing tests** (`r = row` from `tests/fixtures.py`):

```python
def test_dedup_pages_and_skip_null_queries():
    rows = [r("p1", "what?"), r("p1", None), r("p2", "how?")]
    pages, queries = normalize_rows(rows)
    assert [p.id for p in pages] == ["p1", "p2"]
    assert [(q.id, q.text, q.relevant_page_ids) for q in queries] == [
        ("q0000", "what?", frozenset({"p1"})), ("q0001", "how?", frozenset({"p2"}))]

def test_duplicate_query_text_merges_relevant():  # Review Focus 3
    _, queries = normalize_rows([r("p1", "same"), r("p2", "same")])
    assert len(queries) == 1 and queries[0].relevant_page_ids == {"p1", "p2"}

def test_limit_drops_queries_whose_pages_were_cut():  # Review Focus 2
    rows = [r("p1", "a"), r("p2", "b"), r("p3", "a")]
    pages, queries = normalize_rows(rows, limit=2)
    assert [p.id for p in pages] == ["p1", "p2"]
    assert {q.text: q.relevant_page_ids for q in queries} == {"a": {"p1"}, "b": {"p2"}}

def test_validate_rejects_unknown_page():
    with pytest.raises(ValueError):
        validate([], [Query("q0", "x", frozenset({"p9"}))])
```

- [ ] **Step 2: Run** `pytest tests/test_data.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `src/cvo/data.py`.** Skip rows whose `query` is None or blank after `.strip()`.
- [ ] **Step 4: Run** `pytest -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: ViDoRe loading and normalization`

---

### Task 3: Chunking

**Files:**
- Create: `src/cvo/chunking.py`, `tests/test_chunking.py`

**Interfaces:**
- Produces: `chunk_text(text: str, size: int = 256, overlap: int = 32) -> list[str]` — whitespace split; windows start every `size - overlap` tokens; stop once a window reaches the last token (no window fully contained in the previous one); empty/whitespace text → `[]`; chunks re-joined with single spaces.

- [ ] **Step 1: Write failing tests**

```python
def test_empty_text(): assert chunk_text("   ") == []
def test_short_text_single_chunk(): assert chunk_text("a b c", size=4, overlap=1) == ["a b c"]
def test_overlap_windows():
    toks = " ".join(str(i) for i in range(10))
    assert chunk_text(toks, size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
def test_no_redundant_tail():
    assert chunk_text("0 1 2 3 4", size=4, overlap=1) == ["0 1 2 3", "3 4"]
```

- [ ] **Step 2: Run** — Expected: FAIL. **Step 3: Implement.** **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat: overlapping text chunking`

---

### Task 4: Artifact store

**Files:**
- Create: `src/cvo/artifacts.py`, `tests/test_artifacts.py`

**Interfaces:**
- Produces:
  - `page_key(page_id: str) -> str` — sha1 hex of the id (filenames safe on Windows/Drive).
  - `atomic_write_bytes(path: Path, data: bytes) -> None` — write to `path.with_suffix(path.suffix + ".tmp")`, then `os.replace`.
  - `save_json(path: Path, obj) -> None` / `load_json(path: Path)` — via `atomic_write_bytes`, UTF-8, `ensure_ascii=False`, indent 2.
  - `arm_dir(root: Path, subset: str, arm: str, model_id: str) -> Path` — `root/subset/arm/<model_id with "/" → "__">`, created.
  - `run_metadata(limit: int | None) -> dict` — keys `timestamp` (ISO UTC), `python`, `versions` (dict for torch, transformers, colpali_engine, doctr, sentence_transformers, bm25s, datasets; `None` when not installed), `gpu` (`torch.cuda.get_device_name(0)` or `None`), `limit`.

- [ ] **Step 1: Write failing tests**

```python
def test_page_key_is_filename_safe():
    assert re.fullmatch(r"[0-9a-f]{40}", page_key("a/b:c.png"))

def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "x.json"; save_json(p, {"é": 1})
    assert load_json(p) == {"é": 1} and not list(tmp_path.glob("*.tmp"))

def test_crash_before_replace_leaves_no_target(tmp_path, monkeypatch):  # Review Focus 4
    def boom(*a): raise KeyboardInterrupt
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        save_json(tmp_path / "x.json", {"a": 1})
    assert not (tmp_path / "x.json").exists()

def test_arm_dir_sanitizes_model_id(tmp_path):
    d = arm_dir(tmp_path, "tatdqa", "colqwen", "vidore/colqwen2-v1.0")
    assert d == tmp_path / "tatdqa" / "colqwen" / "vidore__colqwen2-v1.0" and d.is_dir()

def test_run_metadata_keys():
    assert {"timestamp", "python", "versions", "gpu", "limit"} <= run_metadata(50).keys()
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (versions via `importlib.metadata.version`, catching `PackageNotFoundError`). **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: artifact paths, atomic writes, run metadata`

---

### Task 5: Cached OCR

**Files:**
- Create: `src/cvo/ocr.py`, `tests/test_ocr.py`

**Interfaces:**
- Consumes: `Page` (Task 2), `page_key`, `save_json`, `load_json` (Task 4).
- Produces:
  - `class OcrEngine(Protocol): def read(self, image: Image.Image) -> str`
  - `class DoctrEngine` — `__init__` lazily imports doctr, builds `ocr_predictor(pretrained=True)` on CUDA if available; `read` runs it on `np.asarray(image.convert("RGB"))` and returns `result.render()`.
  - `@dataclass class OcrResult: texts: dict[str, str]; seconds: float; pages_failed: int`
  - `run_ocr(pages: list[Page], engine: OcrEngine, cache_dir: Path) -> OcrResult` — per page, cache file `cache_dir/<page_key>.json` = `{"text", "seconds", "failed"}`. Cached pages are loaded, not re-run. An engine exception → text `""`, `failed: true`, logged via `logging.warning`. `seconds` = sum of cached per-page `seconds` (fresh + previously recorded); `pages_failed` = count of `failed`.

- [ ] **Step 1: Write failing tests** with a `CountingEngine` stub (returns `f"text {n}"`, counts calls, raises for a chosen page size):

```python
def test_ocr_caches_and_reuses(tmp_path):
    eng = CountingEngine(); pages = make_pages(3)
    first = run_ocr(pages, eng, tmp_path); second = run_ocr(pages, eng, tmp_path)
    assert eng.calls == 3 and first.texts == second.texts
    assert second.seconds == approx(first.seconds)  # recorded time, not cache-load time

def test_ocr_failure_recorded_not_dropped(tmp_path):
    res = run_ocr(make_pages(2), FailingEngine(fail_on=1), tmp_path)
    assert set(res.texts) == {"p0", "p1"} and res.texts["p1"] == "" and res.pages_failed == 1
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: cached docTR OCR with failure accounting`

---

### Task 6: Retriever protocol + BM25 arm

**Files:**
- Create: `src/cvo/retrievers/__init__.py`, `src/cvo/retrievers/base.py`, `src/cvo/retrievers/bm25.py`, `tests/test_bm25.py`
- Modify: `tests/fixtures.py` (add `synthetic_corpus`, `fake_ocr_fn`, `EMPTY_PAGE_ID`)

**Interfaces:**
- Consumes: `Page`, `Query`, `chunk_text`, `OcrResult`.
- Produces:
  - `base.py`: `IndexStats` (fields per spec §4.1: `seconds: float, bytes_stored: int, num_pages: int, pages_failed: int`), `Retriever` Protocol (`name: str`, `index(pages) -> IndexStats`, `search(queries, k) -> dict[str, list[tuple[str, float]]]`), and `max_per_page(chunk_page_ids: list[str], chunk_scores: np.ndarray, k: int) -> list[tuple[str, float]]` (best chunk score per page, sorted desc, top k).
  - `OcrFn = Callable[[list[Page]], OcrResult]` in `base.py`.
  - `bm25.py`: `class BM25Retriever(name="bm25")`, `__init__(self, ocr_fn: OcrFn, store_dir: Path)`. `index`: OCR → chunk → `bm25s.tokenize(chunks, stopwords=None)` → `bm25s.BM25().index`; saves to `store_dir` and sets `bytes_stored` = total file size there; `seconds` = OCR seconds + indexing seconds. `search`: retrieve all chunks per query (`k=len(chunks)`), then `max_per_page`.
  - `tests/fixtures.py`: `synthetic_corpus() -> tuple[list[Page], list[Query], dict[str, str]]` — 5 pages whose images have the topical text drawn on them with `ImageDraw` (so the real GPU test in Task 8 has something to read), one of them blank (`EMPTY_PAGE_ID`); 3 queries with known answers; plus the page-id→text map. `fake_ocr_fn(texts) -> OcrFn` returns those texts with `seconds=0.01 * len(pages)`, `pages_failed=0`.

- [ ] **Step 1: Write failing tests**

```python
def test_max_per_page():
    assert max_per_page(["a", "a", "b"], np.array([1.0, 3.0, 2.0]), 5) == [("a", 3.0), ("b", 2.0)]

def test_bm25_finds_known_pages(tmp_path):
    pages, queries, texts = synthetic_corpus()
    r = BM25Retriever(fake_ocr_fn(texts), tmp_path)
    stats = r.index(pages)
    assert stats.num_pages == 5 and stats.bytes_stored > 0
    top = r.search(queries, k=3)
    assert all(top[q.id][0][0] in q.relevant_page_ids for q in queries)

def test_empty_ocr_page_never_returned(tmp_path):  # Review Focus 1
    pages, queries, texts = synthetic_corpus()
    r = BM25Retriever(fake_ocr_fn(texts), tmp_path); r.index(pages)
    returned = {pid for hits in r.search(queries, k=5).values() for pid, _ in hits}
    assert EMPTY_PAGE_ID not in returned
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: retriever protocol and BM25 arm`

---

### Task 7: Dense arm

**Files:**
- Create: `src/cvo/retrievers/dense.py`, `tests/test_dense.py`

**Interfaces:**
- Consumes: `OcrFn`, `IndexStats`, `max_per_page`, `chunk_text`, `synthetic_corpus`.
- Produces:
  - `class Embedder(Protocol): def encode(self, texts: list[str]) -> np.ndarray` (rows L2-normalized, float32).
  - `class BgeM3Embedder` — lazily loads `SentenceTransformer("BAAI/bge-m3")`, `encode(..., normalize_embeddings=True, batch_size=32)`.
  - `class DenseRetriever(name="dense")`, `__init__(self, ocr_fn: OcrFn, embedder: Embedder)`. Cosine = dot product, brute force; `bytes_stored` = `embeddings.nbytes`; `seconds` = OCR + embedding time.

- [ ] **Step 1: Write failing tests** using a `HashingEmbedder` stub (bag-of-words hashed into 256 dims, normalized) — same assertions as BM25's `test_bm25_finds_known_pages` and `test_empty_ocr_page_never_returned`, plus `stats.bytes_stored == n_chunks * 256 * 4`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: dense bge-m3 arm`

---

### Task 8: ColQwen2 arm

**Files:**
- Create: `src/cvo/retrievers/colqwen.py`, `tests/test_colqwen.py`

**Interfaces:**
- Consumes: `Page`, `IndexStats`, `page_key`, `atomic_write_bytes`, `load_json`/`save_json`.
- Produces:
  - `class MultiVectorEncoder(Protocol)`: `encode_images(images: list[Image.Image]) -> list[torch.Tensor]` (each `[n_patches, d]`), `encode_queries(texts: list[str]) -> list[torch.Tensor]`.
  - `class ColQwen2Encoder` — lazily loads `ColQwen2.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.float16, device_map="cuda:0").eval()` and `ColQwen2Processor`; returns per-item tensors on CPU in fp16 with padding rows stripped (use the attention mask).
  - `maxsim(query: torch.Tensor, docs: list[torch.Tensor]) -> torch.Tensor` — score_i = Σ_q max_p ⟨query_q, doc_i,p⟩ computed in float32.
  - `class ColQwenRetriever(name="colqwen")`, `__init__(self, encoder: MultiVectorEncoder, store_dir: Path, batch_size: int = 4, shard_size: int = 100)`.
    - Shard `i` = `pages[i*100:(i+1)*100]` → `shard_{i:05d}.pt` (via `torch.save` to bytes + `atomic_write_bytes`) containing `{"page_ids": [...], "embeddings": [...], "seconds": float}`. An existing shard is reused only if its `page_ids` equal the current slice; otherwise recomputed.
    - OOM: on `torch.cuda.OutOfMemoryError` halve `batch_size` and retry the batch; at batch size 1 re-raise.
    - `IndexStats.seconds` = sum of shard `seconds`; `bytes_stored` = Σ tensor `nbytes`.

- [ ] **Step 1: Write failing tests** with a `StubEncoder` (deterministic tensors from page pixel values; counts images encoded; optional `oom_above` batch size that raises `torch.cuda.OutOfMemoryError`):

```python
def test_maxsim_hand_computed():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = [torch.tensor([[1.0, 0.0], [0.5, 0.5]]), torch.tensor([[0.0, 2.0]])]
    assert maxsim(q, d).tolist() == [1.5, 2.0]

def test_resume_skips_completed_shards(tmp_path):  # Review Focus 4
    pages = make_pages(250); enc = StubEncoder()
    ColQwenRetriever(enc, tmp_path, shard_size=100).index(pages[:100])
    enc.count = 0
    stats = ColQwenRetriever(enc, tmp_path, shard_size=100).index(pages)
    assert enc.count == 150 and stats.num_pages == 250

def test_shard_with_different_pages_is_recomputed(tmp_path):  # Review Focus 4
    enc = StubEncoder()
    ColQwenRetriever(enc, tmp_path, shard_size=100).index(make_pages(50))   # --limit 50 run
    enc.count = 0
    ColQwenRetriever(enc, tmp_path, shard_size=100).index(make_pages(100))
    assert enc.count == 100

def test_oom_halves_batch(tmp_path):
    enc = StubEncoder(oom_above=1)
    r = ColQwenRetriever(enc, tmp_path, batch_size=4); r.index(make_pages(5))
    assert r.batch_size == 1

@pytest.mark.gpu
def test_real_colqwen_smoke(tmp_path):
    pages, queries, _ = synthetic_corpus()
    r = ColQwenRetriever(ColQwen2Encoder(), tmp_path); r.index(pages)
    assert len(r.search(queries, k=3)) == len(queries)
```

- [ ] **Step 2: Run** `pytest tests/test_colqwen.py -v -m "not gpu"` — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: ColQwen2 arm with sharded resume and OOM backoff`

---

### Task 9: Benchmark runner + CLI

**Files:**
- Create: `src/cvo/run.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run_arm(subset: str, retriever: Retriever, pages: list[Page], queries: list[Query], out_dir: Path, limit: int | None, k: int = 10) -> dict` — calls `validate`, `index`, then times one `search` call over all queries; writes and returns `out_dir/<subset>/<retriever.name>.json`:
    `{"subset", "arm", "metrics": evaluate(...), "cost": {"index_seconds_per_page", "query_ms", "bytes_per_page"}, "index_stats": asdict(stats), "per_query": {qid: [[page_id, score], ...]}, "meta": run_metadata(limit)}`.
  - `check_same_pages(results: list[dict]) -> None` — raises `ValueError` if `index_stats["num_pages"]` differs between arms of one subset.
  - `build_retriever(arm: str, subset: str, artifacts: Path) -> Retriever` — wires real engines: OCR cache at `arm_dir(artifacts, subset, "ocr", "doctr")`, shared by `bm25` and `dense`.
  - CLI `python -m cvo.run --subset {shift,tatdqa,infovqa,arxivqa,all} --arms bm25,dense,colqwen [--limit N] [--artifacts PATH] [--results results]`; defaults `--artifacts artifacts`, `--results results`. For each subset: load once, run each arm, free GPU memory (`del` + `torch.cuda.empty_cache()`) between arms, then call `check_same_pages` on that subset's results.

- [ ] **Step 1: Write failing tests** with a `FakeRetriever` (fixed rankings, fixed `IndexStats`):

```python
def test_run_arm_writes_results(tmp_path):
    pages, queries, _ = synthetic_corpus()
    res = run_arm("tiny", FakeRetriever(perfect=True), pages, queries, tmp_path, limit=None)
    saved = json.loads((tmp_path / "tiny" / "fake.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["ndcg@5"] == 1.0 and set(saved["per_query"]) == {q.id for q in queries}
    assert saved["cost"]["bytes_per_page"] == res["index_stats"]["bytes_stored"] / len(pages)

def test_check_same_pages_rejects_mismatch():
    with pytest.raises(ValueError):
        check_same_pages([{"index_stats": {"num_pages": 5}}, {"index_stats": {"num_pages": 4}}])
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `pytest -v -m "not gpu"` — all PASS.
- [ ] **Step 5: Commit** `feat: benchmark runner and CLI`

---

### Task 10: Report, notebook and README

**Files:**
- Create: `src/cvo/report.py`, `tests/test_report.py`, `notebooks/benchmark.ipynb`, `README.md`

**Interfaces:**
- Consumes: results JSON schema from Task 9; `load_subset` for gallery images.
- Produces:
  - `load_results(results_dir: Path) -> dict[str, dict[str, dict]]` (subset → arm → result).
  - `summary_table(results) -> str` — Markdown table, one row per (subset, arm), columns nDCG@5, R@1, R@5, R@10, s/page, ms/query, KB/page.
  - `plot_all(results, out_dir: Path) -> list[Path]` — writes `ndcg_by_subset.png` (grouped bars), `quality_vs_cost.png` (x = s/page log scale, y = nDCG@5), `storage_per_page.png`.
  - `select_gallery(result_a: dict, result_b: dict, relevant: dict[str, set[str]], n: int = 2) -> dict[str, list[str]]` — `{"a_wins": [...], "b_wins": [...]}`: query ids where one arm has the relevant page at rank 1 and the other has none in its top 5.

- [ ] **Step 1: Write failing tests** on two hand-made result dicts: `select_gallery` picks the expected ids; `summary_table` contains a header row and one row per arm; `plot_all` writes 3 non-empty PNGs to `tmp_path` (use `matplotlib.use("Agg")`).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Write `notebooks/benchmark.ipynb`** — cells: GPU check; `git clone` + `pip install -e ".[ocr,dense,vision]"`; optional Drive mount setting `ARTIFACTS`; smoke run `!python -m cvo.run --subset tatdqa --arms bm25,dense,colqwen --limit 50 --artifacts $ARTIFACTS`; full run per subset (one cell each, so a disconnect resumes cleanly); `summary_table` + `plot_all` + gallery rendering with page images.
- [ ] **Step 6: Write `README.md`** with section headings from spec §9 (TL;DR table placeholder filled after the Colab run, How to run, Method, Results, Failure analysis, Limitations — limitations text written now from spec §9).
- [ ] **Step 7: Run** `pytest -v -m "not gpu"` — all PASS.
- [ ] **Step 8: Commit** `feat: report, Colab notebook, README`

---

### Task 11: Colab runs and write-up (human-in-the-loop)

Needs a Colab GPU session; the user runs the notebook.

- [ ] **Step 1:** Push the repo to GitHub; open the notebook in Colab (T4).
- [ ] **Step 2:** Run `pytest -v` there (includes the `gpu` test) — Expected: all PASS.
- [ ] **Step 3:** Smoke run (`--limit 50`) — Expected: 3 result JSONs, `pages_failed` small, nDCG@5 > 0 for every arm.
- [ ] **Step 4:** Full runs for all four subsets; commit `results/` JSON + plots.
- [ ] **Step 5:** Fill README TL;DR table and Results/Failure analysis from `summary_table` and the gallery; commit `docs: benchmark results and write-up`.
