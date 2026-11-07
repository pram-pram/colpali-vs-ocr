import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "benchmark.ipynb"


def test_notebook_code_cells_parse():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        src = "\n".join(ln for ln in lines if not ln.lstrip().startswith(("!", "%")))
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"cell {i} ({cell.get('id')}): {e}") from e


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_src_on_sys_path_before_cvo_import():
    cells = _code_cells()
    first_import = next(
        i for i, src in enumerate(cells)
        if any(ln.lstrip().startswith(("from cvo", "import cvo")) for ln in src.splitlines())
    )
    assert any('sys.path.insert(0, "src")' in src for src in cells[:first_import])


def test_full_runs_write_results_explicitly():
    runs = [src for src in _code_cells() if "cvo.run" in src and "--limit" not in src]
    assert runs and all("--results" in src for src in runs)
