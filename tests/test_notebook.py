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
