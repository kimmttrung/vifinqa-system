"""Đọc lưới của một tập bảng cụ thể từ tables.parquet.

tables.parquet nặng 62 MB nén (grid + raw_html ≈ kích thước corpus gốc), nạp cả
vào RAM là lãng phí: một lần chạy chỉ chạm tới vài nghìn bảng. Ở đây quét theo
batch và chỉ giữ đúng những khoá được yêu cầu.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pyarrow.parquet as pq

from common import ARTIFACTS

_COLS = ["doc_id", "table_idx", "grid", "n_header_rows", "unit_scale",
         "section", "col_periods", "caption"]


@dataclass
class TableData:
    doc_id: str
    table_idx: int
    grid: list[list[str]]
    n_header_rows: int
    unit_scale: float
    section: str
    col_periods: str
    caption: str


def load_grids(keys) -> dict[tuple[str, int], TableData]:
    want = {(str(d), int(i)) for d, i in keys}
    if not want:
        return {}
    out: dict[tuple[str, int], TableData] = {}
    pf = pq.ParquetFile(ARTIFACTS / "tables.parquet")
    for batch in pf.iter_batches(batch_size=8000, columns=_COLS):
        cols = {n: batch.column(n).to_pylist() for n in _COLS}
        for i in range(batch.num_rows):
            k = (cols["doc_id"][i], int(cols["table_idx"][i]))
            if k not in want or k in out:
                continue
            out[k] = TableData(
                doc_id=k[0], table_idx=k[1],
                grid=json.loads(cols["grid"][i]) if cols["grid"][i] else [],
                n_header_rows=int(cols["n_header_rows"][i] or 1),
                unit_scale=float(cols["unit_scale"][i] or 1.0),
                section=cols["section"][i] or "OTHER",
                col_periods=cols["col_periods"][i] or "[]",
                caption=cols["caption"][i] or "",
            )
        if len(out) == len(want):
            break
    return out
