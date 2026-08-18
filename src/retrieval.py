"""Stage D — Table retrieval.

    QueryIR → lọc metadata → BM25 + tín hiệu cấu trúc → rerank → top-k

Khác pipeline mentor một điểm quan trọng: mentor trả **top-10** cho cả LLM lẫn
bài nộp. Metric của BTC là **F₂ trên relevant_tables**, gold thường 1 bảng:

    k=1 → F₂ 1.000 | k=2 → 0.833 | k=3 → 0.714 | k=5 → 0.556 | k=10 → 0.357

Nộp top-10 tự cắt hơn 60% điểm retrieval. Nên ở đây tách hai con số:
  · `k_read`   (mặc định 10) — số bảng đưa cho LLM đọc, tối ưu RECALL
  · `k_submit` (adaptive 1-4) — số bảng ghi vào relevant_tables, tối ưu F₂
Stage G còn cắt tiếp `k_submit` xuống đúng những bảng mà pandas_query chạm tới.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from bm25 import Bm25Index
from common import ARTIFACTS, fold, tokens
from docfilter import filter_docs
from qparse import QueryIR

# trọng số các tín hiệu cộng thêm vào điểm BM25 (đã chuẩn hoá về [0,1])
W_PERIOD = 2.5      # bảng có đúng cột năm câu hỏi hỏi
W_PHRASE = 3.0      # khớp cụm từ liên tiếp ≥3 token (tên riêng trong thuyết minh)
W_COVER = 2.0       # tỷ lệ token nội dung của câu hỏi có mặt trong bảng
W_SECTION = 1.0     # section khớp với loại chỉ tiêu suy ra từ câu hỏi
W_DATA = 0.5        # bảng thật sự có dữ liệu số

# gợi ý section từ từ khoá chỉ tiêu — không quyết định, chỉ cộng điểm
_SECTION_HINTS = [
    ("KQKD", ("DOANH THU", "GIA VON", "LOI NHUAN GOP", "LOI NHUAN THUAN",
              "LOI NHUAN TRUOC THUE", "LOI NHUAN SAU THUE", "CHI PHI BAN HANG",
              "CHI PHI QUAN LY", "CHI PHI TAI CHINH", "THU NHAP LAI", "LAI THUAN",
              "CHI PHI THUE", "EPS", "LAI CO BAN TREN CO PHIEU")),
    ("LCTT", ("LUU CHUYEN TIEN", "DONG TIEN", "CFO", "TIEN THU TU", "TIEN CHI",
              "HOAT DONG DAU TU", "HOAT DONG TAI CHINH")),
    ("CDKT", ("TONG TAI SAN", "TAI SAN NGAN HAN", "TAI SAN DAI HAN", "NO PHAI TRA",
              "NO NGAN HAN", "NO DAI HAN", "VON CHU SO HUU", "HANG TON KHO",
              "PHAI THU", "PHAI TRA", "TIEN VA TUONG DUONG TIEN", "TAI SAN CO DINH",
              "CHO VAY KHACH HANG", "TIEN GUI CUA KHACH HANG", "VON DIEU LE")),
    ("VCSH", ("THAY DOI VON CHU SO HUU", "BIEN DONG VON")),
]


@lru_cache(maxsize=1)
def _load():
    meta = pd.read_parquet(ARTIFACTS / "table_meta.parquet")
    idx = Bm25Index.load(ARTIFACTS / "bm25")
    rows_by_doc: dict[str, np.ndarray] = {
        d: g["row"].to_numpy() for d, g in meta.groupby("doc_id", sort=False)
    }
    periods = [
        {p["year"] for p in json.loads(s) if p and p.get("year")} if s else set()
        for s in meta["col_periods"]
    ]
    folded = [fold(p) for p in meta["passage"]]
    return meta, idx, rows_by_doc, periods, folded


RRF_K = 60          # hằng số làm mềm của Reciprocal Rank Fusion


@lru_cache(maxsize=1)
def _rerank_cache():
    """`artifacts/rerank_cache.parquet` do notebook Kaggle sinh (dense + rerank).

    Không có file ⇒ chạy thuần lexical. Nhờ vậy máy không GPU vẫn chạy được toàn
    bộ pipeline, và khi có file thì chỉ cần copy vào artifacts/ là hybrid bật lên.
    """
    p = ARTIFACTS / "rerank_cache.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    out: dict[int, dict[int, tuple[int, float]]] = {}
    for qid, g in df.groupby("qid"):
        out[int(qid)] = {int(r.row): (int(r["rank"]), float(r.rerank_score))
                         for _, r in g.iterrows()}
    return out


def rrf(*rank_maps: dict[int, int], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σᵢ 1/(k + rankᵢ(d) + 1).

    Vì sao RRF chứ không cộng điểm thô: BM25 (0..~30), cosine (−1..1) và logit
    của cross-encoder (−10..10) ở ba thang hoàn toàn khác nhau — cộng thẳng là
    để một nguồn nuốt hai nguồn kia. RRF chỉ dùng THỨ HẠNG nên miễn nhiễm với
    thang đo và với outlier.
    """
    out: dict[int, float] = {}
    for rm in rank_maps:
        for row, rank in rm.items():
            out[row] = out.get(row, 0.0) + 1.0 / (k + rank + 1)
    return out


@dataclass
class Hit:
    row: int
    doc_id: str
    table_idx: int
    score: float
    section: str
    caption: str
    reasons: dict = field(default_factory=dict)
    # 4 ứng viên mã hoá <vị_trí_bảng> (Unknown #1) — mang theo để đổi convention
    # chỉ tốn một tham số, không phải chạy lại retrieval.
    line_no: object = None
    page_no: object = None
    char_start: object = None


def _section_hint(fq: str) -> set[str]:
    out = set()
    for sec, kws in _SECTION_HINTS:
        if any(k in fq for k in kws):
            out.add(sec)
    return out


def _phrase_score(q_toks: list[str], passage_folded: str) -> float:
    """Cụm ≥3 token liên tiếp của câu hỏi xuất hiện nguyên vẹn trong bảng.

    Đây là tín hiệu cứu 7,9% câu hỏi có tên riêng trong thuyết minh
    ("Quỹ Đầu tư Giá trị Bảo Việt") — embedding và BM25 rời rạc đều bỏ sót.
    """
    best = 0
    n = len(q_toks)
    for i in range(n):
        for L in range(min(9, n - i), 2, -1):
            if L <= best:
                break
            if " ".join(q_toks[i:i + L]) in passage_folded:
                best = L
                break
    return min(best / 6.0, 1.0)


def retrieve(ir: QueryIR, k_read: int = 10, k_submit: int | None = None,
             candidate_docs: list[str] | None = None) -> list[Hit]:
    meta, idx, rows_by_doc, periods, folded = _load()
    docs = candidate_docs if candidate_docs is not None else filter_docs(ir)
    if docs:
        rows = np.concatenate([rows_by_doc[d] for d in docs if d in rows_by_doc]) \
            if any(d in rows_by_doc for d in docs) else np.array([], dtype=np.int64)
    else:
        # câu hỏi không nêu công ty nào (vd q464 "trong các công ty có…") — không
        # được trả rỗng, relevant_tables rỗng là mất trắng cả 3 tiêu chí.
        rows = np.arange(len(meta), dtype=np.int64)
    if rows.size == 0:
        return []

    base = idx.score(ir.retrieval_query, rows)
    if rows.size > 3000:
        # các tín hiệu cấu trúc bên dưới chạy vòng lặp Python trên từng bảng,
        # nên lọc thô bằng BM25 trước rồi mới chấm kỹ.
        keep = np.argpartition(-base, 2000)[:2000]
        rows, base = rows[keep], base[keep]
    if base.max() > 0:
        base = base / base.max()

    fq = fold(ir.retrieval_query)
    q_toks = tokens(ir.retrieval_query)
    q_set = set(q_toks)
    # cover phải cân theo IDF: đếm đều thì "CUA/VA/TRONG" nặng ngang "KY PHIEU",
    # mà hư từ thì bảng nào cũng có nên cover mất hết khả năng phân biệt.
    q_w = {t: float(idx.idf[idx.vocab[t]]) if t in idx.vocab else 0.0 for t in q_set}
    q_wsum = sum(q_w.values()) or 1.0
    want_years = set(ir.years)
    want_sections = _section_hint(fq)

    sub = meta.iloc[rows]
    n_num = sub["n_numeric_cells"].to_numpy()
    secs = sub["section"].to_numpy()

    scored: list[Hit] = []
    for i, r in enumerate(rows):
        pf = folded[r]
        cover = sum(w for t, w in q_w.items() if t in pf) / q_wsum
        phrase = _phrase_score(q_toks, pf)
        per = 1.0 if (want_years and (periods[r] & want_years)) else 0.0
        sec = 1.0 if (want_sections and secs[i] in want_sections) else 0.0
        data = 1.0 if n_num[i] >= 4 else 0.0
        s = (float(base[i]) + W_PERIOD * per + W_PHRASE * phrase
             + W_COVER * cover + W_SECTION * sec + W_DATA * data)
        scored.append((s, i, r, cover, phrase, per, sec))

    scored.sort(key=lambda x: -x[0])

    # ── tầng hợp nhất: lexical ⊕ dense/rerank bằng RRF ──
    lex_rank = {int(t[2]): rank for rank, t in enumerate(scored)}
    rr = _rerank_cache()
    ce = (rr or {}).get(ir.qid) or {}
    if ce:
        ce_rank = {row: rank for row, (rank, _) in ce.items() if row in lex_rank}
        fused = rrf(lex_rank, ce_rank)
        order = sorted(fused, key=lambda r: -fused[r])
    else:
        fused = {}
        order = [t[2] for t in scored]

    by_row = {int(t[2]): t for t in scored}
    out = []
    for row in order[:max(k_read, k_submit or 0)]:
        s, i, r, cover, phrase, per, sec = by_row[int(row)]
        m = meta.iloc[r]
        ce_rank_v, ce_score_v = ce.get(int(row), (None, None))
        final = fused.get(int(row), float(s))
        out.append(Hit(
            row=int(r), doc_id=m.doc_id, table_idx=int(m.table_idx),
            score=round(float(final), 6), section=m.section, caption=m.caption,
            line_no=m.line_no, page_no=m.page_no, char_start=m.char_start,
            reasons={
                # thành phần điểm lexical (đã chuẩn hoá về [0,1] trước khi cân)
                "bm25": round(float(base[i]), 4),
                "cover": round(cover, 4),
                "phrase": round(phrase, 4),
                "period": per,
                # KHÔNG đặt tên "section": trace dict đã có khoá đó cho TÊN
                # section (CDKT/KQKD/…), `**reasons` sẽ ghi đè mất.
                "sec_hit": sec,
                "lex_score": round(float(s), 4),
                "lex_rank": lex_rank[int(row)],
                # tầng cross-encoder (None nếu chưa chạy notebook Kaggle)
                "ce_rank": ce_rank_v,
                "ce_score": None if ce_score_v is None else round(ce_score_v, 4),
                # hợp nhất
                "rrf_score": round(fused[int(row)], 6) if fused else None,
                "fusion": "rrf(lex,ce)" if ce else "lex_only",
            }))
    return out


def select_submit(hits: list[Hit], ir: QueryIR, k_max: int = 6,
                  adaptive: bool = False) -> list[Hit]:
    """Chọn các bảng ghi vào `relevant_tables` — tối ưu F₂.

    ## Công thức đúng (không xấp xỉ), suy từ định nghĩa F₂

    Gold G bảng, nộp k bảng, trúng h bảng:  P = h/k, R = h/G

        F₂ = 5PR/(4P+R) = 5h / (4G + k)

    k chỉ nằm ở MẪU và chỉ CỘNG — nộp thêm một bảng sai làm mẫu tăng 1, trong khi
    nộp thêm một bảng đúng làm tử tăng 5. Nới từ k lên k+1 có lãi khi

        p > h / (4G + k)        p = xác suất bảng thêm vào là gold

    ## Số đo thật (probe `line_no`, k=2, 17/08/2026)

        TABLES_PRECISION 0.3676 ⇒ h ≈ 0.735 bảng trúng/câu
        TABLES_RECALL    0.3417 ⇒ G ≈ 0.735/0.3417 ≈ 2.15 bảng gold/câu
        ⇒ ngưỡng tại k=2:  0.735/(8.6+2) = 0.069

    Bảng hạng 3 chỉ cần **6,9%** khả năng đúng là đã nên nộp; tỷ lệ trúng trung
    bình mỗi bảng hiện là 37%. Ngưỡng chỉ chạm 0.10 khi k ≈ 8.

    ## Vì sao hai lần trước tôi tính sai
    Cả hai lần đều giả định "nộp toàn bảng ĐÚNG" rồi kết luận k nhỏ là tối ưu.
    Giả định đó chỉ đúng khi precision ≈ 1. Precision thật là 0.37, nên chi phí
    precision của việc nới k nhỏ hơn nhiều so với lợi ích recall (trọng số ×4).
    """
    if not hits:
        return []
    k = adaptive_k(ir, k_max) if adaptive else k_max

    # Ưu tiên phủ doc trước: câu nhiều công ty/năm mà dồn hết slot vào một doc
    # thì recall chết hẳn, không cứu được bằng bất kỳ k nào.
    best_per_doc: dict[str, Hit] = {}
    for h in hits:
        best_per_doc.setdefault(h.doc_id, h)
    sel = sorted(best_per_doc.values(), key=lambda h: -h.score)[:k]

    # rồi lấp nốt slot còn lại theo điểm — kể cả trùng doc. Gold ~2,15 bảng
    # thường là hai mảnh của một báo cáo bị tách ("CĐKT" + "CĐKT (tiếp theo)"),
    # nên bảng thứ hai cùng doc có xác suất trúng cao.
    chosen = {(h.doc_id, h.table_idx) for h in sel}
    for h in hits:
        if len(sel) >= k:
            break
        key = (h.doc_id, h.table_idx)
        if key not in chosen:
            sel.append(h)
            chosen.add(key)
    return sel[:k]


def adaptive_k(ir: QueryIR, k_max: int = 12) -> int:
    """Số bảng nộp, TÍNH RIÊNG cho từng câu.

    `F₂ = 5h/(4G+k)` là công thức theo TỪNG CÂU, mà G thay đổi rất mạnh:
    câu 1 công ty 1 năm có G≈1-2, câu 10 công ty × 3 năm có G≈10+. Dùng một
    hằng số k cho cả hai là sai ở cả hai đầu — thừa với câu đơn (mất precision
    vô ích), thiếu với câu nhóm (recall bị chặn cứng).

    Neo vào số đo thật: quét k phẳng cho ra đỉnh ở **k ≈ 7,5** trên toàn bộ
    1.012 câu. Nên hàm này phải có TRUNG BÌNH ≈ 7 để không phá vỡ kết quả đã
    đo, chỉ phân bổ lại: câu đơn bớt đi, câu nhóm nhiều hơn.

    Tăng theo **log** chứ không tuyến tính, vì G không tỷ lệ thuận với số
    target: câu 10 công ty thường chỉ cần 1 bảng/công ty, còn câu 1 công ty vẫn
    cần ~2 bảng (CĐKT bị tách đôi — Mục 3 §14 CLAUDE.md).

        n_targets:  1 → k=6 · 2 → k=6 · 4 → k=8 · 8 → k=9 · 20 → k=11

    Đây vẫn là **giả thuyết chưa đo** — phải so với k=6 phẳng bằng một lượt nộp.
    """
    n_targets = max(1, len(ir.tickers)) * max(1, len(ir.years))
    k = 4.0 + 1.5 * math.log2(1 + n_targets)
    return int(max(3, min(k_max, round(k))))
