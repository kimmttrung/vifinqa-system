"""Xây index truy xuất bảng: passages + BM25 (CPU, ~3-5 phút cho 146k bảng).

    python scripts/build_index.py

Sinh ra:
    artifacts/table_meta.parquet   metadata nhẹ, thứ tự hàng == thứ tự index
    artifacts/passages.txt.zst?    (không) — passages ghi thẳng vào table_meta
    artifacts/bm25.tf.npz / .meta.npz / .vocab.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bm25
from common import ARTIFACTS
from passages import table_passage

NEEDED = ["doc_id", "ticker", "year", "stmt_type", "table_idx", "char_start", "line_no",
          "page_no", "caption", "section", "n_rows", "n_cols", "n_header_rows",
          "has_ma_so", "unit_scale", "unit_source", "n_numeric_cells", "col_periods",
          "grid"]


def main() -> int:
    src = ARTIFACTS / "tables.parquet"
    if not src.exists():
        print(f"Không thấy {src} — chạy Stage A trước.")
        return 1

    t0 = time.time()
    pf = pq.ParquetFile(src)
    meta_rows, passages = [], []
    for batch in pf.iter_batches(batch_size=8000, columns=NEEDED):
        df = batch.to_pandas()
        for row in df.itertuples(index=False):
            grid = json.loads(row.grid) if row.grid else []
            passages.append(table_passage(row, grid))
            meta_rows.append({
                "doc_id": row.doc_id, "ticker": row.ticker, "year": row.year,
                "stmt_type": row.stmt_type, "table_idx": row.table_idx,
                "char_start": row.char_start, "line_no": row.line_no,
                "page_no": row.page_no, "section": row.section,
                "caption": row.caption, "n_rows": row.n_rows, "n_cols": row.n_cols,
                "n_header_rows": row.n_header_rows, "has_ma_so": row.has_ma_so,
                "unit_scale": row.unit_scale, "n_numeric_cells": row.n_numeric_cells,
                "col_periods": row.col_periods,
            })
        print(f"  {len(meta_rows):7d} bảng · {time.time()-t0:5.1f}s", flush=True)

    meta = pd.DataFrame(meta_rows)
    meta["row"] = range(len(meta))
    meta["passage"] = passages
    meta.to_parquet(ARTIFACTS / "table_meta.parquet", compression="zstd", index=False)
    print(f"table_meta.parquet: {len(meta)} dòng · {time.time()-t0:.1f}s")

    idx = bm25.build(passages)
    idx.save(ARTIFACTS / "bm25")
    print(f"BM25: {idx.n_docs} bảng × {len(idx.vocab)} từ vựng · {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
