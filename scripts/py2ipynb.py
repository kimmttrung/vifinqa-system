"""Chuyển file `# %%` (percent format) → .ipynb để upload thẳng lên Kaggle.

    python scripts/py2ipynb.py notebooks/stage_d_dense.py notebooks/stage_e_planner.py

Không cần jupytext — chỉ ghi JSON notebook v4 trực tiếp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(path: Path) -> Path:
    lines = path.read_text(encoding="utf-8").splitlines()
    cells, buf, kind = [], [], "code"

    def flush():
        if not buf:
            return
        src = "\n".join(buf).strip("\n")
        if not src.strip():
            return
        if kind == "markdown":
            body = [l[2:] if l.startswith("# ") else l.lstrip("#").lstrip()
                    for l in src.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": "\n".join(body)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                          "execution_count": None, "source": src})

    for line in lines:
        s = line.strip()
        if s.startswith("# %%"):
            flush()
            buf, kind = [], "markdown" if "[markdown]" in s else "code"
            continue
        buf.append(line)
    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = path.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for a in sys.argv[1:]:
        p = convert(Path(a))
        n = len(json.loads(p.read_text(encoding="utf-8"))["cells"])
        print(f"{a}  →  {p}   ({n} cell)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
