# ColPali vs OCR Retrieval Benchmark — Design

**Date:** 2026-11-07
**Status:** Approved in conversation; pending written-spec review

## 1. Purpose

A learning / portfolio project that answers: *on visually rich documents, how does
vision-native retrieval (ColQwen2) compare with a classic OCR → text retrieval pipeline,
in quality and in cost?*

Deliverable: a public GitHub repo containing a small Python package, a Colab notebook
that reproduces every number, result JSON + charts, and a README write-up.

**Success criteria**

- A reader can open the notebook on a free Colab T4 and reproduce the reported numbers
  within one or two sessions.
- The write-up shows *where* each method wins (per document type), not just an average.
- Every reported number is traceable to a results file that records how it was produced.

**Out of scope (YAGNI):** vector databases, fine-tuning, hybrid fusion, additional vision
models, a web UI, private/internal documents.

## 2. Constraints

- Compute: free Colab or Kaggle GPU (T4, 16 GB VRAM, no bf16). Sessions can disconnect.
- Local development and tests run CPU-only on Windows.
- Data must be public and labeled.

## 3. Data

Four ViDoRe test subsets from Hugging Face, chosen to contrast document types:

| Subset key | HF dataset | Rows | Character |
|---|---|---|---|
| `shift` | `vidore/shiftproject_test` | 1000 | Text-heavy reports (French) |
| `tatdqa` | `vidore/tatdqa_test` | 1663 | Financial tables |
| `infovqa` | `vidore/infovqa_test_subsampled` | 500 | Infographics |
| `arxivqa` | `vidore/arxivqa_test_subsampled` | 500 | Scientific figures |

Each row is a `(query, image)` pair. Normalization in `data.py`:

- **Corpus:** unique pages keyed by `image_filename` → `Page(id, image)`.
- **Queries:** rows with a non-empty `query` → `Query(id, text, relevant_page_ids)`,
  where `relevant_page_ids` = the row's `image_filename` (merged if the same query text
  appears on multiple rows).
- `--limit N` restricts to the first N pages (and only queries whose relevant pages
  survive) for smoke tests.

## 4. Architecture

```
colpali-vs-ocr/
  src/cvo/
    data.py          # load + normalize a ViDoRe subset
    ocr.py           # docTR OCR, per-page disk cache
    chunking.py      # page text -> overlapping chunks
    retrievers/
      base.py        # Retriever protocol + IndexStats
      bm25.py        # bm25s over OCR chunks
      dense.py       # bge-m3 over OCR chunks
      colqwen.py     # ColQwen2 multi-vector + MaxSim
    eval.py          # metrics
    artifacts.py     # artifact paths, sharded save/load, run metadata
    run.py           # CLI entry point
  notebooks/benchmark.ipynb
  results/           # <subset>/<arm>.json, plots/
  tests/
  pyproject.toml
  README.md
```

### 4.1 Retriever interface

```python
class Retriever(Protocol):
    name: str
    def index(self, pages: list[Page]) -> IndexStats: ...
    def search(self, queries: list[Query], k: int) -> dict[str, list[tuple[str, float]]]: ...

@dataclass
class IndexStats:
    seconds: float        # fresh compute time, incl. OCR for text arms
    bytes_stored: int
    num_pages: int
    pages_failed: int
```

`search` returns, per query id, a page-level ranking of `(page_id, score)`.

### 4.2 Text arms (shared OCR)

- **OCR:** docTR (`ocr_predictor`, pretrained, GPU) → page text. Cached per page as JSON
  under the artifact dir; re-runs skip cached pages.
- **Chunking:** ~256-token chunks with 32-token overlap (whitespace tokenization is
  sufficient). Pages with empty text produce zero chunks.
- **Page scoring:** page score = max score over its chunks; ranking is page-level.
- **BM25 arm:** `bm25s` index over the chunks.
- **Dense arm:** `BAAI/bge-m3` dense embeddings (multilingual, needed for `shift`), cosine
  similarity, brute force.
- **Timing fairness:** both text arms' `IndexStats.seconds` include total OCR time for
  the subset (recorded once at OCR time, reused even when loaded from cache).

### 4.3 Vision arm

- **Model:** `vidore/colqwen2-v1.0` via `colpali-engine`, fp16, image batch size 4.
- **Index:** per-page multi-vector embeddings (fp16), saved in shards of 100 pages.
- **Search:** brute-force MaxSim using the processor's `score_multi_vector`.
- **Storage:** reported as actual bytes per page.

## 5. Metrics (`eval.py`)

Computed identically for every arm:

- **Quality:** nDCG@5 (headline, matches ViDoRe), Recall@1, @5, @10 (hit rate, since most
  queries have one relevant page).
- **Cost:** indexing seconds/page, query latency ms/query (mean over all queries, model
  already loaded), storage bytes/page.
- **Per-query output:** top-10 ranking per query per arm saved for failure analysis.

## 6. Artifacts, caching and resume

- One artifact root (default `./artifacts`, or a mounted Google Drive path via
  `--artifacts`), keyed `<subset>/<arm>/<model-id>/`.
- OCR cache: one file per page. ColQwen embeddings: one file per 100-page shard, plus a
  sidecar recording each shard's compute seconds. Resume skips completed pages/shards.
- Reported indexing time = sum of recorded fresh-compute times, never cache-load time.
- Results file `results/<subset>/<arm>.json` contains: metrics, IndexStats, per-query
  top-10, and metadata (library versions, model ids, GPU name, `--limit`, timestamp).

## 7. Error handling

- **OCR failure on a page:** store empty text, log it, increment `pages_failed`; never
  drop the page silently.
- **CUDA OOM during embedding:** halve batch size and retry, down to 1; then raise.
- **Fail fast at startup:** a query with no relevant pages, or arms whose indexed page
  sets differ, aborts the run with a clear message.

## 8. Testing

CPU-only `pytest`:

- `eval.py` against hand-computed rankings (known nDCG@5 / recall values).
- `chunking.py` boundaries and overlap.
- `data.py` normalization on a tiny in-memory fixture shaped like ViDoRe rows.
- BM25 and dense retrievers end-to-end on ~5 synthetic PIL-rendered pages with 3 known
  queries; the dense arm takes an injectable embedder (tiny model in tests, `bge-m3` in
  real runs). OCR is stubbed in these tests.
- Resume: interrupt after shard 1 (stub embedder), re-run, assert no recompute.
- ColQwen tests marked `@pytest.mark.gpu`, skipped without CUDA.

Before any full run: `python -m cvo.run --subset tatdqa --arms bm25,dense,colqwen --limit 50` on Colab.

## 9. Write-up

- Charts: nDCG@5 per subset grouped by arm; quality vs indexing cost scatter; storage
  per page bar chart.
- Failure gallery: 3–4 queries where ColQwen clearly won or lost, with the page each arm
  returned.
- README: TL;DR table → how to run → method → results → failure analysis → limitations
  (ViDoRe ≠ real-world data, mostly single relevant page, OCR-engine dependence, no
  fine-tuning).
