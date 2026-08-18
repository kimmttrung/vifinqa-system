"""Stage A — Corpus normalization: 1.973 file .txt → docs/tables.parquet.

Trích từ notebook đã nghiệm thu (11/08/2026): 146.246 bảng, 100% lưới hợp lệ,
0 file crash. Các primitive dùng chung (`fold`, `parse_number`, `html_to_grid`)
nằm ở `common.py`; ở đây chỉ còn phần riêng của Stage A.

Bốn ứng viên mã hoá `<vị_trí_bảng>` đều được lưu (`table_idx` / `char_start` /
`line_no` / `page_no`) dù probe đã chốt là `line_no` — giữ lại để đổi convention
không phải parse lại 363 MiB.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common import fold, html_to_grid, parse_number

# ───────────────────────── suy đơn vị ─────────────────────────
# Dùng (?<![A-Z]) thay cho \b ở đầu: OCR hay dính liền số với đơn vị
# ("31.12.2022Triệu VND"), lúc đó giữa '2' và 'T' KHÔNG có word boundary nên
# \bTRIEU sẽ trượt và rơi xuống match 'VND' = 1.0.
# (?<!CONG ) chặn "Công ty VND" bị đọc thành đơn vị tỷ — sau khi bỏ dấu thì
# "tỷ" và "ty" trùng nhau.
_UNIT_RULES = [
    (re.compile(r"(?<![A-Z])NGHIN\s*TY\b"), 1e12),
    (re.compile(r"(?<![A-Z])(?<!CONG )TY\s*(VND|DONG)\b"), 1e9),
    (re.compile(r"(?<![A-Z])TRIEU\s*(VND|DONG)\b"), 1e6),
    (re.compile(r"(?<![A-Z])(NGHIN|NGAN)\s*(VND|DONG)\b"), 1e3),
    (re.compile(r"(?<![A-Z])(VND|DONG)\b"), 1.0),
]


def _detect_unit(folded_text: str):
    for rx, scale in _UNIT_RULES:
        if rx.search(folded_text):
            return scale
    return None


def detect_unit(caption: str, header_text: str) -> tuple[float, str]:
    """→ (unit_scale, unit_source ∈ {header_cell, caption, default}).

    Đơn vị nằm ở 2 nơi: dòng phía trên bảng (Đơn vị: Triệu VND) hoặc trong ô
    header (31.12.2022 Triệu VND — kiểu ngân hàng). OCR làm hỏng cả chữ
    "Đơn vị" (đã gặp thật: Dom vi: VND) nên KHÔNG dò nhãn, chỉ dò token đơn vị.
    """
    h = _detect_unit(fold(header_text))
    c = _detect_unit(fold(caption))
    if h is not None and h > 1:
        return h, "header_cell"
    if c is not None:
        return c, "caption"
    if h is not None:
        return h, "header_cell"
    return 1.0, "default"


# ───────────────────── ngữ nghĩa cột (stock / flow) ─────────────────────
# Không dùng \b quanh năm: OCR hay dính năm với đơn vị ("2018VND") — giữa '8'
# và 'V' không có word boundary nên \b(20\d{2})\b trượt và cả cột mất mốc.
_DATE_IN = re.compile(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(20\d{2})")
_NAM_YYYY = re.compile(r"(?<![A-Z])NAM\s*(20\d{2})(?![0-9])")
_BARE_YYYY = re.compile(r"(?<![0-9])(20\d{2})(?![0-9])")


def classify_period(header_text: str, doc_year, section: str) -> dict | None:
    """→ {period_type, year, is_begin} hoặc None nếu cột không mang mốc thời gian."""
    f = fold(header_text)
    if not f:
        return None
    flow_default = section in ("KQKD", "LCTT")

    m = _DATE_IN.search(f)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return {"period_type": "stock", "year": y, "is_begin": (d == 1 and mo == 1)}
    m = _NAM_YYYY.search(f)
    if m:
        return {"period_type": "flow", "year": int(m.group(1)), "is_begin": False}
    if re.search(r"\b(SO\s*)?(CUOI\s*(NAM|KY|QUY))\b", f):
        return {"period_type": "stock", "year": doc_year, "is_begin": False}
    if re.search(r"\b(SO\s*)?(DAU\s*(NAM|KY|QUY))\b", f):
        return {"period_type": "stock", "year": doc_year, "is_begin": True}
    if re.search(r"\b(NAM|KY)\s*NAY\b", f):
        return {"period_type": "flow", "year": doc_year, "is_begin": False}
    if re.search(r"\b(NAM|KY)\s*TRUOC\b", f):
        return {"period_type": "flow", "year": (doc_year - 1) if doc_year else None,
                "is_begin": False}
    m = _BARE_YYYY.search(f)
    if m:
        return {"period_type": "flow" if flow_default else "stock",
                "year": int(m.group(1)), "is_begin": False}
    return None


# ───────────────────── phân loại section ─────────────────────
# Mã biểu mẫu là tín hiệu mạnh nhất: 1.613/1.973 doc (81,8%) có.
# NGÂN HÀNG LỆCH MỘT BẬC so với doanh nghiệp — B02/TCTD là CĐKT, không phải KQKD.
# [-–—/] chứ không chỉ [-/]: corpus dùng cả gạch ngang dài ("Mẫu số B 03 – DN/HN").
# Bug này im lặng vì fallback keyword vẫn bắt được khi bảng có heading —
# chỉ hỏng ở bảng KHÔNG có heading, đúng chỗ khó phát hiện nhất.
FORM_RE = re.compile(r"\bB\s*0?(\d{1,2})\s*[-–—/]\s*(DN|TCTD|CTCK|DNBH)", re.I)

_SECTION_BY_FORM = {
    ("DN", 1): "CDKT", ("DN", 2): "KQKD", ("DN", 3): "LCTT", ("DN", 4): "VCSH",
    ("DN", 9): "TM",
    ("DNBH", 1): "CDKT", ("DNBH", 2): "KQKD", ("DNBH", 3): "LCTT", ("DNBH", 9): "TM",
    ("CTCK", 1): "CDKT", ("CTCK", 2): "KQKD", ("CTCK", 3): "LCTT", ("CTCK", 4): "VCSH",
    ("CTCK", 5): "TM", ("CTCK", 9): "TM",
    ("TCTD", 2): "CDKT", ("TCTD", 3): "KQKD", ("TCTD", 4): "LCTT",
    ("TCTD", 5): "TM", ("TCTD", 6): "TM",
}

_HEADING_RULES = [
    ("BANG CAN DOI KE TOAN", "CDKT"),
    ("BAO CAO TINH HINH TAI CHINH", "CDKT"),
    ("KET QUA HOAT DONG KINH DOANH", "KQKD"),
    ("KET QUA KINH DOANH", "KQKD"),
    ("LUU CHUYEN TIEN TE", "LCTT"),
    ("THAY DOI VON CHU SO HUU", "VCSH"),
    ("BIEN DONG VON CHU SO HUU", "VCSH"),
    ("THUYET MINH", "TM"),
]


def section_from_form(line: str) -> str | None:
    m = FORM_RE.search(line)
    return _SECTION_BY_FORM.get((m.group(2).upper(), int(m.group(1)))) if m else None


def section_from_heading(folded_line: str) -> str | None:
    for kw, sec in _HEADING_RULES:
        if kw in folded_line:
            return sec
    return None


# ───────────────────── quét document ─────────────────────
PAGE_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$")
TABLE_RE = re.compile(r"<table\b.*?</table>", re.I | re.S)
STMT_TYPES = ("consolidated", "separate", "aggregated")


def parse_doc_meta(txt_path: Path, data_root: Path) -> dict:
    doc_dir = txt_path.parent
    doc_id = doc_dir.name
    year_dir = doc_dir.parent.name
    ticker = doc_dir.parent.parent.name
    low = doc_id.lower()
    # phân loại bằng SUBSTRING, không dùng hậu tố: hậu tố bỏ sót 24 doc bị tách _1/_2
    stmt_type = next((s for s in STMT_TYPES if s in low), "other")
    m = re.search(r"(20\d{2})", doc_id)
    year = int(m.group(1)) if m else (int(year_dir) if year_dir.isdigit() else None)
    return {
        "doc_id": doc_id,
        "file_stem": txt_path.name[:-4],
        "ticker": ticker,
        "year": year,
        "year_dir": int(year_dir) if year_dir.isdigit() else None,
        "stmt_type": stmt_type,
        "rel_path": str(txt_path.relative_to(data_root)).replace("\\", "/"),
    }


def _line_index(text: str):
    starts, lines, pos = [], [], 0
    for line in text.split("\n"):
        starts.append(pos)
        lines.append(line)
        pos += len(line) + 1
    return starts, lines


def _bisect(sorted_vals, x: int) -> int:
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def _header_row_count(grid, max_header: int = 3) -> int:
    """Số hàng header = số hàng đầu chưa phải hàng dữ liệu.

    Dừng ở hàng banner (một ô phủ HẾT chiều rộng). Banner phải phủ hết n_cols —
    nếu không, header 2 tầng kiểu ngân hàng ("Tại ngày" colspan=3 trên 5 cột)
    sẽ bị cắt mất tầng chứa đơn vị "Triệu VND".
    """
    if not grid:
        return 0
    n_cols = len(grid[0])
    n = 0
    for row in grid[:max_header]:
        cells = [c for c in row if c.strip()]
        if not cells:
            n += 1
            continue
        if n_cols > 1 and len(cells) == n_cols and len(set(cells)) == 1:
            break
        if any(parse_number(c) is not None for c in row) and len(cells) >= 2:
            break
        if n >= 1 and n_cols >= 4 and len(cells) <= 2:
            break
        n += 1
    return max(n, 1)


def process_document(txt_path: Path, data_root: Path) -> tuple[dict, list[dict]]:
    """→ (doc_record, [table_record…])"""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    meta = parse_doc_meta(txt_path, data_root)
    doc_year = meta["year"]
    starts, lines = _line_index(text)

    page_offsets, page_nums = [], []
    sec_offsets, sec_values = [], []
    for i, line in enumerate(lines):
        if len(line) > 300:            # dòng dài = chính là bảng, bỏ qua cho nhanh
            continue
        s = line.strip()
        if not s:
            continue
        m = PAGE_RE.match(s)
        if m:
            page_offsets.append(starts[i])
            page_nums.append(int(m.group(1)))
            continue
        sec = section_from_form(s) or section_from_heading(fold(s))
        if sec is not None:
            sec_offsets.append(starts[i])
            sec_values.append(sec)

    tables = []
    for idx, m in enumerate(TABLE_RE.finditer(text), start=1):
        cs, ce = m.start(), m.end()
        li = _bisect(starts, cs)
        pi = _bisect(page_offsets, cs)
        si = _bisect(sec_offsets, cs)

        caption_lines = []
        for j in range(li - 1, max(li - 12, -1), -1):
            s = lines[j].strip()
            if not s or PAGE_RE.match(s) or len(s) > 300:
                continue
            caption_lines.append(s)
            if len(caption_lines) == 3:
                break
        caption = " | ".join(reversed(caption_lines))

        raw_html = m.group(0)
        grid = html_to_grid(raw_html)
        n_rows, n_cols = len(grid), (len(grid[0]) if grid else 0)
        n_head = _header_row_count(grid) if grid else 0
        header_text = " ".join(c for r in grid[:n_head] for c in r)
        section = sec_values[si] if si >= 0 else "OTHER"
        unit_scale, unit_source = detect_unit(caption, header_text)

        col_periods = [classify_period(" ".join(grid[r][c] for r in range(n_head)),
                                       doc_year, section) for c in range(n_cols)]

        tables.append({
            "doc_id": meta["doc_id"], "ticker": meta["ticker"], "year": doc_year,
            "stmt_type": meta["stmt_type"],
            "table_idx": idx,                                 # ứng viên vị trí #1
            "char_start": cs, "char_end": ce,                 # #2
            "line_no": li + 1,                                # #3 ← convention THẬT
            "page_no": page_nums[pi] if pi >= 0 else None,    # #4
            "caption": caption[:500], "section": section,
            "n_rows": n_rows, "n_cols": n_cols, "n_header_rows": n_head,
            "has_ma_so": "MA SO" in fold(header_text),
            "unit_scale": unit_scale, "unit_source": unit_source,
            "n_numeric_cells": sum(1 for r in grid for c in r
                                   if parse_number(c) is not None),
            "col_periods": json.dumps(col_periods, ensure_ascii=False),
            "grid": json.dumps(grid, ensure_ascii=False),
            "raw_html": raw_html,
        })

    meta.update({"n_chars": len(text), "n_pages": len(page_offsets),
                 "n_tables": len(tables)})
    return meta, tables
