"""Assemble and run the notebooks from their plain-text sources.

The real source of every notebook is a ``.py`` file in ``tools/nb_src``. Cells
are separated by ``# --- code`` and ``# --- markdown`` banner lines. Keeping
the source as text means a reviewer reads Python rather than notebook JSON,
while the committed ``.ipynb`` still carries the outputs for the marker.

    python tools/make_notebooks.py                   # all notebooks
    python tools/make_notebooks.py project1_population
    python tools/make_notebooks.py --dry-run         # assemble without running
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCES = HERE / "nb_src"
NOTEBOOKS = PROJECT / "notebooks"

BANNER = re.compile(r"^# ---\s*(markdown|code)\s*$")
EXECUTION_TIMEOUT = 900

#: How a cell of each kind is turned into a notebook node.
CELL_FACTORIES = {
    "markdown": nbformat.v4.new_markdown_cell,
    "code": nbformat.v4.new_code_cell,
}


def uncomment(lines: list[str]) -> list[str]:
    """Strip the leading ``# `` from the lines of a markdown cell."""
    return [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines]


def trim(lines: list[str]) -> list[str]:
    """Drop blank lines from both ends of a cell."""
    body = list(lines)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return body


def parse(text: str) -> list[tuple[str, str]]:
    """Split a banner-delimited script into ``(kind, source)`` pairs."""
    kinds: list[str] = []
    blocks: list[list[str]] = []
    for line in text.splitlines():
        banner = BANNER.match(line)
        if banner:
            kinds.append(banner.group(1))
            blocks.append([])
        elif blocks:
            blocks[-1].append(line)
    if not blocks:
        raise ValueError("no '# --- code' or '# --- markdown' banners found")

    cells = []
    for kind, block in zip(kinds, blocks):
        body = trim(uncomment(block) if kind == "markdown" else block)
        if body:
            cells.append((kind, "\n".join(body)))
    return cells


def assemble(text: str) -> nbformat.NotebookNode:
    """Build a v4 notebook node from a banner-delimited script."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3",
                       "language": "python"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    })
    notebook.cells = [CELL_FACTORIES[kind](source) for kind, source in parse(text)]
    return notebook


def render(source: Path, run: bool = True) -> Path:
    """Assemble one notebook and, unless told otherwise, execute it."""
    notebook = assemble(source.read_text(encoding="utf-8"))
    if run:
        executor = ExecutePreprocessor(timeout=EXECUTION_TIMEOUT, kernel_name="python3")
        try:
            executor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOKS)}})
        except CellExecutionError as failure:
            raise SystemExit(
                f"{source.name}: execution failed, nothing written\n{failure}")
    NOTEBOOKS.mkdir(exist_ok=True)
    target = NOTEBOOKS / f"{source.stem}.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the project notebooks.")
    parser.add_argument("stems", nargs="*", help="script names to build; default all")
    parser.add_argument("--dry-run", action="store_true",
                        help="assemble the notebooks without executing them")
    args = parser.parse_args()

    wanted = ([SOURCES / f"{s}.py" for s in args.stems] if args.stems
              else sorted(SOURCES.glob("*.py")))
    for script in wanted:
        if not script.exists():
            raise SystemExit(f"no such script: {script}")
    for script in wanted:
        print(f"built {render(script, run=not args.dry_run).relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
