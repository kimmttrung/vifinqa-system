"""Fast-path tra cứu trên facts.parquet — thay hẳn BM25 cho câu tra cứu giá trị.

Vì sao nó mạnh hơn retrieval+locate:

  · BM25 xếp hạng BẢNG rồi mới dò ô trong top-k. Nếu bảng đúng rơi khỏi top-k
    thì mất trắng — đây chính là 34,7% lỗi "insufficient evidence" của mentor.
  · Fact table đã trải phẳng MỌI (nhãn × kỳ) của MỌI bảng trong doc. Tra thẳng
    trên nhãn nên không phụ thuộc thứ hạng bảng: bảng đúng được tìm ra *vì* nó
    chứa nhãn đúng, chứ không phải vì caption của nó khớp câu hỏi.
  · Trả về đúng một table_idx ⇒ nộp k=1 ⇒ F₂ = 1.000.

BM25 vẫn giữ vai trò: câu hỏi mà fact table không khớp nhãn nào (chỉ tiêu diễn
đạt vòng, tên riêng trong thuyết minh) thì rơi xuống đường retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from common import ARTIFACTS, tokens
from locate import _col_unit_penalty, score_label
from qparse import QueryIR


@lru_cache(maxsize=1)
def _load():
    f = pd.read_parquet(ARTIFACTS / "facts.parquet").reset_index(drop=True)
    idx = {k: np.asarray(v) for k, v in
           f.groupby(["ticker", "period_year"]).indices.items()}
    return f, idx


def available() -> bool:
    return (ARTIFACTS / "facts.parquet").exists()


@dataclass
class FactHit:
    doc_id: str
    table_idx: int
    row_idx: int
    col_idx: int
    label: str
    label_col: str
    col_name: str
    value_raw: float
    unit_scale: float
    statement: str
    period_year: int
    is_begin: bool
    period_type: str
    score: float
    label_score: float


def _period_bonus(is_begin: bool, ptype: str, statement: str, want: str | None) -> float:
    if want == "begin":
        return 1.0 if is_begin else 0.0
    if want == "end":
        return 1.0 if (not is_begin and ptype == "stock") else 0.25
    if want == "period":
        return 1.0 if ptype == "flow" else 0.25
    if statement == "CDKT":
        return 1.0 if (not is_begin and ptype == "stock") else 0.15
    if statement in ("KQKD", "LCTT"):
        return 1.0 if ptype == "flow" else 0.5
    return 0.8 if not is_begin else 0.2


W_LABEL, W_PERIOD, W_STMT, W_STATEMENT = 1.0, 0.35, 0.30, 0.12


def lookup(ir: QueryIR, ticker: str | None = None, years: list[int] | None = None,
           top: int = 5, min_label: float = 0.42, query: str | None = None) -> list[FactHit]:
    """→ các fact khớp nhất cho (ticker × năm) trong QueryIR.

    `query` cho phép tra một cụm chỉ tiêu KHÁC câu hỏi gốc — cần cho tỷ số dạng
    "A trên B": hai vế phải tra riêng, mỗi vế là một truy vấn độc lập.
    """
    f, idx = _load()
    tks = [ticker] if ticker else ir.tickers
    yrs = years if years is not None else ir.years
    if not tks or not yrs:
        return []
    parts = [idx[(t, y)] for t in tks for y in yrs if (t, y) in idx]
    if not parts:
        return []
    rows = np.concatenate(parts)
    sub = f.iloc[rows]

    qt = tokens(query if query is not None else ir.retrieval_query)
    if not qt:
        return []

    # Chấm điểm trên NHÃN DUY NHẤT rồi map ngược — trong một (ticker, năm) có
    # hàng nghìn fact nhưng chỉ vài trăm nhãn khác nhau, chấm lại là lãng phí.
    labels = sub["label_norm"].to_numpy()
    uniq, inv = np.unique(labels, return_inverse=True)
    uscore = np.array([score_label(qt, u) for u in uniq], dtype=np.float32)
    lab = uscore[inv]

    keep = np.flatnonzero(lab >= min_label)
    if keep.size == 0:
        return []
    sub = sub.iloc[keep]
    lab = lab[keep]

    want_sep = ir.stmt_type
    stmt_bonus = np.where(
        sub["stmt_type"].to_numpy() == (want_sep or "consolidated"), 1.0,
        np.where(sub["stmt_type"].to_numpy() == "other", 0.7, 0.0 if want_sep else 0.35))
    # thuyết minh chi tiết hơn báo cáo chính; khi nhãn khớp ngang nhau thì báo cáo
    # chính đáng tin hơn vì nó là con số tổng hợp chính thức
    st = sub["statement"].to_numpy()
    stmt_kind = np.where(np.isin(st, ["CDKT", "KQKD", "LCTT"]), 1.0, 0.55)

    per = np.array([_period_bonus(bool(b), p, s, ir.time_point)
                    for b, p, s in zip(sub["is_begin"], sub["period_type"], st)],
                   dtype=np.float32)

    # Phạt cột sai đơn vị. Đã gặp thật (q503): câu hỏi "bao nhiêu tỷ đồng" nhưng
    # bảng có cột "Tỷ lệ %" bên cạnh cột giá trị, khớp nhãn dòng ngang nhau nên
    # rơi vào cột tỷ lệ rồi chia tiếp 1e9 → ra 1.8e-08.
    colpen = np.array([_col_unit_penalty(c, ir.unit_kind)
                       for c in sub["col_name"]], dtype=np.float32)

    total = (W_LABEL * lab + W_PERIOD * per + W_STMT * stmt_bonus
             + W_STATEMENT * stmt_kind + colpen)
    order = np.argsort(-total)[:top]

    out = []
    for i in order:
        r = sub.iloc[int(i)]
        out.append(FactHit(
            doc_id=r.doc_id, table_idx=int(r.table_idx), row_idx=int(r.row_idx),
            col_idx=int(r.col_idx), label=r.label, label_col=r.label_col,
            col_name=r.col_name, value_raw=float(r.value_raw),
            unit_scale=float(r.unit_scale), statement=r.statement,
            period_year=int(r.period_year), is_begin=bool(r.is_begin),
            period_type=r.period_type, score=round(float(total[i]), 4),
            label_score=round(float(lab[i]), 4)))
    return out


def lookup_per_target(ir: QueryIR, top: int = 3) -> dict[tuple[str, int], list[FactHit]]:
    """Một kết quả cho MỖI cặp (công ty, năm) — câu so sánh nhiều công ty cần đủ vế."""
    out = {}
    for t in ir.tickers:
        for y in ir.years:
            hits = lookup(ir, ticker=t, years=[y], top=top)
            if hits:
                out[(t, y)] = hits
    return out


def lookup_aligned(ir: QueryIR, top: int = 6) -> dict[tuple[str, int], FactHit]:
    """Cùng MỘT chỉ tiêu cho mọi (công ty, năm) — bắt buộc với câu so sánh.

    `lookup_per_target` chấm nhãn độc lập từng target nên rất dễ lấy "Doanh thu
    thuần" ở công ty này và "Doanh thu bán hàng" ở công ty kia — hiệu số giữa hai
    chỉ tiêu khác nhau là vô nghĩa. Ở đây chọn nhãn TỐT NHẤT TOÀN CỤC trước, rồi
    mới lấy đúng nhãn đó ở từng target.
    """
    cand = lookup_per_target(ir, top=top)
    if not cand:
        return {}
    n_targets = len({k for k in cand})
    # nhãn nào phủ được nhiều target nhất; hoà thì lấy nhãn điểm cao nhất
    agg: dict[str, list[float]] = {}
    for hits in cand.values():
        best_of_label: dict[str, float] = {}
        for h in hits:
            best_of_label[h.label] = max(best_of_label.get(h.label, 0.0), h.score)
        for lb, s in best_of_label.items():
            agg.setdefault(lb, []).append(s)
    label = max(agg, key=lambda lb: (len(agg[lb]), sum(agg[lb])))
    if len(agg[label]) < max(2, n_targets) and n_targets > 1:
        # không nhãn nào phủ đủ target → để tầng trên quyết định bỏ qua
        if len(agg[label]) < 2:
            return {}
    out = {}
    for key, hits in cand.items():
        pick = next((h for h in hits if h.label == label), None)
        if pick is not None:
            out[key] = pick
    return out


def lookup_concept(ir: QueryIR, ticker: str, year: int, concept: str) -> FactHit | None:
    """Tra theo concept kế toán chuẩn hoá (TONG_TAI_SAN, DOANH_THU_THUAN…).

    Dùng cho công thức tỷ số — ở đó chỉ tiêu do CÔNG THỨC quy định, không do câu
    hỏi nói ra ("ROE" không chứa chữ nào của "lợi nhuận sau thuế").
    """
    f, idx = _load()
    rows = idx.get((ticker, year))
    if rows is None:
        return None
    sub = f.iloc[rows]
    sub = sub[sub["concept"] == concept]
    if sub.empty:
        return None
    st = sub["statement"].to_numpy()
    per = np.array([_period_bonus(bool(b), p, s, ir.time_point)
                    for b, p, s in zip(sub["is_begin"], sub["period_type"], st)],
                   dtype=np.float32)
    want = ir.stmt_type or "consolidated"
    stmt = np.where(sub["stmt_type"].to_numpy() == want, 1.0,
                    np.where(sub["stmt_type"].to_numpy() == "other", 0.7, 0.2))
    # concept chỉ được gán trên 4 báo cáo chính là chính xác; bản trong thuyết
    # minh là số chi tiết, không phải số tổng hợp chính thức
    kind = np.where(np.isin(st, ["CDKT", "KQKD", "LCTT"]), 1.0, 0.3)
    total = per + 0.5 * stmt + 0.6 * kind
    r = sub.iloc[int(np.argmax(total))]
    return FactHit(
        doc_id=r.doc_id, table_idx=int(r.table_idx), row_idx=int(r.row_idx),
        col_idx=int(r.col_idx), label=r.label, label_col=r.label_col,
        col_name=r.col_name, value_raw=float(r.value_raw),
        unit_scale=float(r.unit_scale), statement=r.statement,
        period_year=int(r.period_year), is_begin=bool(r.is_begin),
        period_type=r.period_type, score=float(total.max()), label_score=1.0)
