# ColPali vs OCR: retrieval on visually rich documents

On visually rich documents, how does vision-native retrieval (ColQwen2) compare with a
classic OCR then text retrieval pipeline, in quality and in cost? This repo benchmarks
three arms on four [ViDoRe](https://huggingface.co/vidore) subsets and reports where each
one wins.

| Arm | Pipeline |
|---|---|
| `bm25` | docTR OCR, chunking, BM25 (`bm25s`), page score = best chunk |
| `dense` | docTR OCR, chunking, `BAAI/bge-m3` dense embeddings, page score = best chunk |
| `colqwen` | `vidore/colqwen2-v1.0` multi-vector page embeddings, MaxSim |

## TL;DR

_Filled in after the Colab run — see Task 11_

## How to run

### Colab notebook (recommended)

Open `notebooks/benchmark.ipynb` in Colab with a free T4 GPU runtime, set `REPO_URL` in the
first code cell to your copy of this repo, and run the cells in order. The notebook installs
the package, runs the tests and a 50-page smoke run, then runs each subset in its own cell
so a disconnect resumes from the cached artifacts (mount Google Drive to keep them).

### Command line

```bash
pip install -e ".[ocr,dense,vision]" "colpali-engine>=0.3.4,<0.4"
python -m pytest -m "not gpu"      # CPU-only tests
python -m pytest -v                # includes the GPU ColQwen test
```

```bash
# smoke run, then the full benchmark
python -m cvo.run --subset tatdqa --arms bm25,dense,colqwen --limit 50 --results results_smoke
python -m cvo.run --subset all --arms bm25,dense,colqwen --artifacts artifacts --results results
```

Results land in `results/<subset>/<arm>.json`. Build the table and charts from them:

```bash
python -c "
from pathlib import Path
from cvo.report import load_results, plot_all, summary_table
r = load_results(Path('results'))
print(summary_table(r))
plot_all(r, Path('results/plots'))
"
```

## Method

**Data.** Four ViDoRe test subsets chosen to contrast document types: `shift` (text-heavy
French reports), `tatdqa` (financial tables), `infovqa` (infographics) and `arxivqa`
(scientific figures). Each row is a (query, page image) pair; pages are deduplicated by
image filename and queries are linked to their relevant page(s).

**Text arms.** One docTR pass per page (cached), whitespace-tokenized chunks of about 256
tokens with 32 overlap, page score = max over its chunks. OCR failures store empty text and
are counted in `pages_failed`. Both text arms include the total OCR time in their indexing
cost.

**Vision arm.** ColQwen2 in fp16, batch size 4 (halved on CUDA OOM), per-page multi-vector
embeddings stored in shards of 100 pages, brute-force MaxSim search.

**Metrics.** Quality: nDCG@5 (headline, as in ViDoRe) and Recall@1/5/10. Cost: indexing
seconds per page (fresh compute only, never cache-load time), query latency in ms
(mean, model already loaded), and stored bytes per page. Every results JSON also records
library versions, model ids, GPU name, `--limit` and timestamp.

## Results

_Filled in after the Colab run — see Task 11_

Charts (written to `results/plots/`): `ndcg_by_subset.png`, `quality_vs_cost.png`,
`storage_per_page.png`.

## Failure analysis

_Filled in after the Colab run — see Task 11_

The notebook renders a gallery of queries where ColQwen2 is right at rank 1 and BM25 has
nothing in its top 5 (and the reverse), with the page each arm returned.

## Limitations

- **ViDoRe is not real-world data.** The queries were largely generated against the
  pages, so results may not transfer to real user questions or private documents.
- **Mostly a single relevant page per query.** Recall is effectively a hit rate and the
  benchmark says little about multi-page or multi-document questions.
- **Results depend on the OCR engine.** The text arms use docTR with pretrained models;
  a different engine (or a commercial one) could shift their numbers considerably.
- **No fine-tuning and no hybrid fusion.** All models are used off the shelf, and the arms
  are compared individually rather than combined.
- **Costs are measured on a Colab T4.** Absolute timings vary with hardware and session
  load; the relative picture is more reliable than the absolute numbers.
