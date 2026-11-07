"""Tables, charts and failure-gallery selection from results JSON."""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ARM_ORDER = ("bm25", "dense", "colqwen")
SUBSET_ORDER = ("shift", "tatdqa", "infovqa", "arxivqa")
ARM_COLORS = {"bm25": "#4C78A8", "dense": "#F58518", "colqwen": "#54A24B"}
ARM_MARKERS = {"bm25": "o", "dense": "s", "colqwen": "^"}
SUBSET_COLORS = {"shift": "#4C78A8", "tatdqa": "#E45756", "infovqa": "#B279A2", "arxivqa": "#72B7B2"}


def load_results(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Read results/<subset>/<arm>.json into {subset: {arm: result}}."""
    results_dir = Path(results_dir)
    out: dict[str, dict[str, dict]] = {}
    if not results_dir.is_dir():
        return out
    for path in sorted(results_dir.glob("*/*.json")):
        out.setdefault(path.parent.name, {})[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _ordered(results: dict[str, dict[str, dict]]) -> list[tuple[str, str, dict]]:
    """(subset, arm, result) rows in canonical order; unknown names sort last."""
    def key(order: tuple[str, ...], name: str) -> tuple[int, str]:
        return (order.index(name) if name in order else len(order), name)

    return [
        (s, a, results[s][a])
        for s in sorted(results, key=lambda n: key(SUBSET_ORDER, n))
        for a in sorted(results[s], key=lambda n: key(ARM_ORDER, n))
    ]


def summary_table(results: dict[str, dict[str, dict]]) -> str:
    header = "| Subset | Arm | nDCG@5 | R@1 | R@5 | R@10 | s/page | ms/query | KB/page |"
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for subset, arm, r in _ordered(results):
        m, c = r["metrics"], r["cost"]
        lines.append(
            f"| {subset} | {arm} | {m['ndcg@5']:.3f} | {m['recall@1']:.3f} | {m['recall@5']:.3f} "
            f"| {m['recall@10']:.3f} | {c['index_seconds_per_page']:.3f} | {c['query_ms']:.1f} "
            f"| {c['bytes_per_page'] / 1024:.1f} |"
        )
    return "\n".join(lines)


def _save(fig: Figure, path: Path) -> None:
    FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _arms_present(results: dict[str, dict[str, dict]]) -> list[str]:
    return [a for a in ARM_ORDER if any(a in arms for arms in results.values())]


def _grouped_bars(results, value, ylabel, title, path: Path) -> None:
    subsets = [s for s in SUBSET_ORDER if s in results] + sorted(s for s in results if s not in SUBSET_ORDER)
    arms = _arms_present(results)
    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot()
    width = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        xs = [j + (i - (len(arms) - 1) / 2) * width for j in range(len(subsets))]
        ys = [value(results[s][arm]) if arm in results[s] else 0.0 for s in subsets]
        ax.bar(xs, ys, width, label=arm, color=ARM_COLORS[arm])
    ax.set_xticks(range(len(subsets)), subsets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="arm")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def _quality_vs_cost(results, path: Path) -> None:
    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot()
    for subset, arm, r in _ordered(results):
        ax.scatter(
            r["cost"]["index_seconds_per_page"],
            r["metrics"]["ndcg@5"],
            marker=ARM_MARKERS.get(arm, "D"),
            color=SUBSET_COLORS.get(subset, "#888888"),
            s=80,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Indexing cost (seconds per page, log scale)")
    ax.set_ylabel("nDCG@5")
    ax.set_title("Quality vs indexing cost")
    arm_handles = [
        Line2D([], [], marker=ARM_MARKERS[a], linestyle="", color="#555555", label=a)
        for a in _arms_present(results)
    ]
    subset_handles = [
        Line2D([], [], marker="o", linestyle="", color=SUBSET_COLORS.get(s, "#888888"), label=s)
        for s in [s for s in SUBSET_ORDER if s in results]
    ]
    ax.legend(handles=arm_handles + subset_handles, fontsize=8, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_all(results: dict[str, dict[str, dict]], out_dir: Path) -> list[Path]:
    """Write the three report charts (no pyplot global state) and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "ndcg_by_subset.png", out_dir / "quality_vs_cost.png", out_dir / "storage_per_page.png"]
    _grouped_bars(results, lambda r: r["metrics"]["ndcg@5"], "nDCG@5", "nDCG@5 by subset", paths[0])
    _quality_vs_cost(results, paths[1])
    _grouped_bars(
        results, lambda r: r["cost"]["bytes_per_page"] / 1024, "KB per page (stored)",
        "Index storage per page", paths[2],
    )
    return paths


def _rank1_hit(ranked: list, rel: set[str]) -> bool:
    return bool(ranked) and ranked[0][0] in rel


def _top5_miss(ranked: list, rel: set[str]) -> bool:
    return not any(page_id in rel for page_id, _ in ranked[:5])


def select_gallery(
    result_a: dict, result_b: dict, relevant: dict[str, set[str]], n: int = 2
) -> dict[str, list[str]]:
    """Query ids where one arm is right at rank 1 and the other has nothing in its top 5."""
    pa, pb = result_a["per_query"], result_b["per_query"]
    a_wins: list[str] = []
    b_wins: list[str] = []
    for qid in pa:
        if qid not in pb or qid not in relevant:
            continue
        rel = relevant[qid]
        if _rank1_hit(pa[qid], rel) and _top5_miss(pb[qid], rel):
            a_wins.append(qid)
        elif _rank1_hit(pb[qid], rel) and _top5_miss(pa[qid], rel):
            b_wins.append(qid)
    return {"a_wins": a_wins[:n], "b_wins": b_wins[:n]}
