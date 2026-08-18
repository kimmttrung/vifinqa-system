"""Stage B' — Doc filter thuần metadata (KHÔNG embedding).

Tên file đã chứa TICKER_financial_statements_YEAR_type nên lọc metadata là đủ:
1.973 doc → thường 1–8 doc. 82,5% câu hỏi chỉ cần ≤4 document.

Bốn cái bẫy đã đo trên corpus, xử lý hết ở đây:
  · 4 ticker EVF/FTS/HND/MBS không có hậu tố loại ở MỌI năm → 1 báo cáo/năm,
    phải coi là thoả CẢ consolidated lẫn separate, nếu không recall = 0.
  · 24 doc bị tách đôi `_1`/`_2` ở 8 ticker → phải lấy cả hai mảnh.
  · doc "thuyết minh rời" của MCH/PRT/SSH và VPB_2023 nằm ở stmt_type='other'.
  · cột `01/01/Y` của báo cáo năm Y == cột `31/12` của báo cáo năm Y−1
    → khi thiếu doc năm Y vẫn cứu được bằng doc năm Y±1.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from common import load_docs
from qparse import QueryIR


@lru_cache(maxsize=1)
def _docs_index():
    docs = load_docs()
    by_tk_year: dict[tuple[str, int], pd.DataFrame] = {}
    for (tk, yr), g in docs.groupby(["ticker", "year"], dropna=False):
        by_tk_year[(tk, int(yr) if pd.notna(yr) else -1)] = g
    years_by_tk = {tk: sorted(int(y) for y in g.year.dropna().unique())
                   for tk, g in docs.groupby("ticker")}
    # ticker không bao giờ có hậu tố loại → 1 báo cáo/năm, thoả cả hai loại
    typeless = {tk for tk, g in docs.groupby("ticker")
                if (g.stmt_type == "other").all()}
    return docs, by_tk_year, years_by_tk, typeless


TYPE_PRIORITY = {
    "separate": ["separate", "other", "aggregated", "consolidated"],
    "consolidated": ["consolidated", "other", "aggregated", "separate"],
    None: ["consolidated", "other", "aggregated", "separate"],
}


def docs_for(ticker: str, year: int, stmt_type: str | None,
             year_slack: bool = True) -> list[str]:
    """→ danh sách doc_id cho (ticker, năm, loại báo cáo)."""
    _, by_tk_year, years_by_tk, typeless = _docs_index()
    g = by_tk_year.get((ticker, year))
    if g is None or g.empty:
        if not year_slack:
            return []
        # thiếu hẳn báo cáo năm Y: dùng năm liền kề — cột đầu kỳ/cuối kỳ bắc cầu được
        avail = years_by_tk.get(ticker, [])
        near = sorted(avail, key=lambda y: (abs(y - year), -y))[:2]
        out: list[str] = []
        for y in near:
            out.extend(docs_for(ticker, y, stmt_type, year_slack=False))
        return out

    if ticker in typeless:
        return sorted(g.doc_id.tolist())

    chosen: list[str] = []
    for t in TYPE_PRIORITY.get(stmt_type, TYPE_PRIORITY[None]):
        hit = g[g.stmt_type == t]
        if not hit.empty:
            chosen = sorted(hit.doc_id.tolist())
            break
    if not chosen:
        chosen = sorted(g.doc_id.tolist())

    # doc 'other' cùng năm là thuyết minh rời / bản không hậu tố — luôn kèm theo,
    # toàn corpus chỉ có 55 cái nên gần như không thêm nhiễu.
    extra = g[(g.stmt_type == "other") & (~g.doc_id.isin(chosen))].doc_id.tolist()
    return sorted(set(chosen) | set(extra))


def _years_needed(ir: QueryIR) -> list[int]:
    ys = set(ir.years)
    if not ys:
        return []
    # tăng trưởng / bình quân số dư cần thêm năm liền trước
    if {"GROWTH", "AVERAGE"} & set(ir.ops):
        ys |= {y - 1 for y in ir.years}
    # số đầu năm Y có thể nằm ở báo cáo năm Y (cột 01/01) — không cần mở rộng thêm
    return sorted(ys)


def filter_docs(ir: QueryIR, max_docs: int = 40) -> list[str]:
    """QueryIR → danh sách doc_id ứng viên (đã khử trùng, cắt trần)."""
    out: list[str] = []
    years = _years_needed(ir)
    _, _, years_by_tk, _ = _docs_index()
    for tk in ir.tickers:
        yy = years or years_by_tk.get(tk, [])
        for y in yy:
            for d in docs_for(tk, y, ir.stmt_type):
                if d not in out:
                    out.append(d)
    return out[:max_docs]
