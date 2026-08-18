"""Biểu diễn một bảng thành đoạn text để index (BM25 và dense dùng chung).

KHÔNG embed toàn bộ ô: 146k bảng × mọi ô là vô ích và gây nhiễu — con số
không mang thông tin truy xuất. Tín hiệu thật nằm ở:
  metadata (công ty/năm/loại BC/section) + caption + header + NHÃN DÒNG.

Đây chính là ý "Trích xuất bảng cung cấp cấu trúc, metadata cung cấp ngữ cảnh".
"""
from __future__ import annotations

import json

_SECTION_VI = {
    "CDKT": "BANG CAN DOI KE TOAN TAI SAN NGUON VON",
    "KQKD": "BAO CAO KET QUA HOAT DONG KINH DOANH DOANH THU CHI PHI LOI NHUAN",
    "LCTT": "BAO CAO LUU CHUYEN TIEN TE DONG TIEN",
    "VCSH": "THAY DOI VON CHU SO HUU",
    "TM": "THUYET MINH BAO CAO TAI CHINH",
    "OTHER": "",
}

_UNIT_VI = {1.0: "VND", 1e3: "NGHIN VND", 1e6: "TRIEU VND", 1e9: "TY VND", 1e12: "NGHIN TY VND"}

MAX_LABEL_ROWS = 60


def row_labels(grid: list[list[str]], n_header: int = 1, limit: int = MAX_LABEL_ROWS):
    """Nhãn của mỗi dòng dữ liệu = ô text đầu tiên không phải mã số/số."""
    out = []
    for r in grid[n_header:]:
        for cell in r[:3]:
            s = cell.strip()
            if len(s) >= 3 and not s.replace(".", "").replace(",", "").isdigit():
                out.append(s)
                break
        if len(out) >= limit:
            break
    return out


def table_passage(row, grid=None) -> str:
    """row = một dòng của tables.parquet (namedtuple/Series)."""
    if grid is None:
        grid = json.loads(row.grid) if isinstance(row.grid, str) else (row.grid or [])
    n_head = max(int(getattr(row, "n_header_rows", 1) or 1), 1)
    header = " ".join(c for r in grid[:n_head] for c in r if c.strip())
    labels = " ; ".join(row_labels(grid, n_head))
    unit = _UNIT_VI.get(float(row.unit_scale), "")
    return (
        f"{row.ticker} {row.year} {row.stmt_type} {row.section} "
        f"{_SECTION_VI.get(row.section, '')} {unit}\n"
        f"{row.caption}\n{header}\n{labels}"
    )
