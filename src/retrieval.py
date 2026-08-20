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
import os
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


def hybrid_weight() -> float:
    """Trọng số của tầng dense/rerank trong RRF. 0 = tắt hẳn (mặc định).

    ĐỌC KỸ TRƯỚC KHI BẬT. Có file `rerank_cache.parquet` KHÔNG có nghĩa là nên
    dùng. Lần đo đầu tiên (19/08) cho thấy nó làm hại:

        lexical thuần   TABLES_F2 0,4137  ·  P 0,2796  ·  R 0,6099
        + rerank 50-50  TABLES_F2 0,3293  ·  P 0,2042  ·  R 0,4942

    Nguyên nhân: notebook Stage D+ cấp cho cross-encoder chuỗi `retrieval_query`
    (đã fold, bỏ dấu, bỏ tên công ty, bỏ năm) — thứ dành cho BM25, sai miền với
    model neural. Logit top-1 trung vị −4,09 tức "không liên quan".

    Vì vậy mặc định TẮT. Bật lại bằng `--ce-weight 0.5` (hoặc env VIFINQA_CE_WEIGHT)
    sau khi đã chạy lại Stage D+ với `QUERY_TEXT = "raw"`, và chỉ giữ nếu
    leaderboard nói nó hơn 0,4137.
    """
    try:
        return max(0.0, float(os.environ.get("VIFINQA_CE_WEIGHT", "0")))
    except ValueError:
        return 0.0


@lru_cache(maxsize=1)
def _rerank_cache():
    """`artifacts/rerank_cache.parquet` do notebook Kaggle sinh (dense + rerank).

    Không có file ⇒ chạy thuần lexical. Nhờ vậy máy không GPU vẫn chạy được toàn
    bộ pipeline.
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


def rrf(*rank_maps: dict[int, int], k: int = RRF_K,
        weights: tuple[float, ...] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion có trọng số: score(d) = Σᵢ wᵢ/(k + rankᵢ(d) + 1).

    Vì sao RRF chứ không cộng điểm thô: BM25 (0..~30), cosine (−1..1) và logit
    của cross-encoder (−10..10) ở ba thang hoàn toàn khác nhau — cộng thẳng là
    để một nguồn nuốt hai nguồn kia. RRF chỉ dùng THỨ HẠNG nên miễn nhiễm với
    thang đo và với outlier.

    Trọng số có mặt vì RRF cân bằng KHÔNG phải lúc nào cũng đúng: đo trên
    leaderboard 19/08, hợp nhất 50-50 với một cross-encoder được cấp câu truy vấn
    sai miền làm TABLES_F2 tụt 0,4137 → 0,3293. Một nguồn nhiễu trộn ngang hàng
    thì kéo cả kết quả xuống, nên nguồn chưa chứng minh được phải vào với trọng
    số nhỏ hơn.
    """
    out: dict[int, float] = {}
    for i, rm in enumerate(rank_maps):
        w = 1.0 if weights is None else weights[i]
        for row, rank in rm.items():
            out[row] = out.get(row, 0.0) + w / (k + rank + 1)
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


@lru_cache(maxsize=1)
def _pos_index() -> dict:
    """(doc_id, table_idx) → vị trí bảng, cho những bảng KHÔNG do retrieval trả về.

    Fact table và solver trỏ thẳng vào table_idx mà retrieval có thể không xếp
    hạng tới. Trước đây các bảng đó được dựng Hit rỗng, `line_no` = None, và
    `emit.table_position` biến None thành **0** ⇒ ref `doc|0` trỏ vào chỗ không
    tồn tại. Đo trên bài nộp đầy đủ: **119/6072 ref (1,96%) là `|0`** — mất
    precision thuần tuý, không đổi lại được gì.
    """
    meta, *_ = _load()
    out = {}
    for row, d, t, ln, pg, cs, sec, cap in zip(
            meta["row"], meta["doc_id"], meta["table_idx"], meta["line_no"],
            meta["page_no"], meta["char_start"], meta["section"], meta["caption"]):
        out[(d, int(t))] = (int(row), ln, pg, cs, sec, cap)
    return out


def hit_from_meta(doc_id: str, table_idx: int, score: float = 0.0,
                  section: str = "OTHER", reasons: dict | None = None) -> Hit:
    """Hit đầy đủ vị trí cho một bảng biết trước (fact-table / solver)."""
    got = _pos_index().get((doc_id, int(table_idx)))
    if got is None:
        return Hit(row=-1, doc_id=doc_id, table_idx=int(table_idx), score=score,
                   section=section, caption="", reasons=reasons or {})
    row, ln, pg, cs, sec, cap = got
    return Hit(row=row, doc_id=doc_id, table_idx=int(table_idx), score=score,
               section=sec or section, caption=cap or "", reasons=reasons or {},
               line_no=ln, page_no=pg, char_start=cs)


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
    if ce and hybrid_weight() > 0:
        ce_rank = {row: rank for row, (rank, _) in ce.items() if row in lex_rank}
        fused = rrf(lex_rank, ce_rank, weights=(1.0, hybrid_weight()))
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


def ce_extra_hits(ir: QueryIR, n: int, exclude: set[tuple[str, int]]) -> list[Hit]:
    """n bảng hạng cao nhất của cross-encoder mà tập nộp lexical CHƯA có.

    ## Vì sao CỘNG THÊM chứ không TRỘN

    Trộn RRF thì bảng CE **đẩy** bảng lexical ra khỏi tập nộp — đo 19/08: thay
    ~58% tập nộp, TABLES_F2 0,4137 → 0,3293. Cộng thêm thì tập lexical đã đo được
    giữ nguyên, chỉ nới k. Rủi ro chỉ còn ở precision, và F₂ tính rất rẻ:

        F₂ = 5h/(4G + k)   ⇒   nới k lên k+1 có lãi khi   p > h/(4G + k)

    Với h ≈ 1,7 · G ≈ 2,15 · k = 6:  p > 1,7/(8,6+6) = **11,6%**. Nghĩa là bảng
    thêm vào chỉ cần đúng 1 lần trong 9 là đã hoà. Một cross-encoder dù tầm thường
    cũng thường vượt ngưỡng đó, miễn là nó ĐỘC LẬP với lexical — mà Jaccard(lex,ce)
    = 0,215 nói đúng là độc lập.

    Khác biệt then chốt so với fusion: cách này **không bao giờ bỏ** bảng nào
    lexical đã chọn, nên RECALL chỉ có thể tăng. Nhưng F₂ thì KHÔNG có sàn:
    k nằm ở mẫu, nên nếu bảng thêm vào không trúng lần nào (p = 0) thì

        5h/(4G+8) vs 5h/(4G+6)  ⇒  tụt ~12% tương đối.

    Đây là một CƯỢC có ngưỡng rõ ràng (p > 11,6%), không phải bữa trưa miễn phí.
    """
    if n <= 0:
        return []
    rr = _rerank_cache()
    ce = (rr or {}).get(ir.qid) or {}
    if not ce:
        return []
    meta, _, _, _, _ = _load()
    out: list[Hit] = []
    for row, (rank, score) in sorted(ce.items(), key=lambda kv: kv[1][0]):
        m = meta.iloc[int(row)]
        key = (m.doc_id, int(m.table_idx))
        if key in exclude:
            continue
        out.append(Hit(
            row=int(row), doc_id=m.doc_id, table_idx=int(m.table_idx),
            score=0.0, section=m.section, caption=m.caption,
            line_no=m.line_no, page_no=m.page_no, char_start=m.char_start,
            reasons={"source": "ce_extra", "ce_rank": rank,
                     "ce_score": round(float(score), 4)}))
        exclude.add(key)
        if len(out) >= n:
            break
    return out


def select_submit(hits: list[Hit], ir: QueryIR, k: int = 6) -> list[Hit]:
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

    `k` do người gọi quyết định — xem `adaptive_k()`.
    """
    if not hits:
        return []
    k = max(1, int(k))

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


# Chỉ tiêu của hai vế thường nằm ở HAI báo cáo khác nhau (tỷ suất lợi nhuận =
# KQKD ÷ CĐKT; tăng trưởng = cùng chỉ tiêu ở hai kỳ). Câu như vậy cần nhiều bảng
# hơn trên mỗi document.
_MULTI_METRIC_OPS = {"RATIO", "DIFF", "GROWTH"}
TABLES_PER_DOC = 1.2            # đo gián tiếp: gold ≈ 2,7 bảng trên ≈ 2,27 doc
TABLES_PER_DOC_MULTI = 1.8
K_MULT = 2.5                    # k* = G/ρ với ρ = mật độ trúng ≈ 0,4 ở đầu bảng


def n_targets(ir: QueryIR, n_docs: int | None = None) -> int:
    """Số cặp (công ty × năm) mà câu hỏi thật sự chạm tới.

    Cắt theo số doc ứng viên vì tích công ty × năm hay phóng đại: câu hỏi nêu
    5 năm nhưng công ty chỉ có 3 báo cáo thì chỉ 3 cặp là có thật.
    """
    n = max(1, len(ir.tickers)) * max(1, len(ir.years))
    return min(n, max(1, n_docs)) if n_docs else n


def estimate_gold_tables(ir: QueryIR, n_docs: int | None = None) -> float:
    """Ước lượng G — số bảng gold của RIÊNG câu này."""
    per_doc = (TABLES_PER_DOC_MULTI if _MULTI_METRIC_OPS & set(ir.ops)
               else TABLES_PER_DOC)
    return per_doc * n_targets(ir, n_docs)


def adaptive_k(ir: QueryIR, k_max: int = 12, n_docs: int | None = None,
               mult: float = K_MULT) -> int:
    """Số bảng nộp, TÍNH RIÊNG cho từng câu.

    ## Vì sao không thể dùng một hằng số

    `F₂ = 5h/(4G+k)` là công thức theo TỪNG CÂU, mà G thay đổi rất mạnh. Đo trên
    1.012 câu: **43,4% câu chỉ có ĐÚNG MỘT cặp (công ty × năm)**, trong khi p90
    là 6 cặp và cao nhất là 18. Một hằng số k sai ở cả hai đầu:

        G=1, h→1:   k=3 ⇒ F₂ 0,71   ·  k=6 ⇒ 0,50  ·  k=10 ⇒ 0,36
        G=6, h≈0,3k: k=6 ⇒ F₂ 0,30  ·  k=12 ⇒ 0,50 ·  k=18 ⇒ 0,64

    ## k tối ưu nằm ở điểm BÃO HOÀ

    Mô hình h(k) = min(G, ρk) với ρ = mật độ trúng của bảng xếp hạng:

        F₂ = 5ρk/(4G+k)   khi ρk < G   → TĂNG theo k
        F₂ = 5G/(4G+k)    khi ρk ≥ G   → GIẢM theo k

    ⇒ đỉnh đúng tại **k\\* = G/ρ**. Đo được ρ theo dải hạng (probe 17/08):

        hạng 1-2: 0,368/bảng · hạng 3-6: 0,236 · hạng 7-10: 0,126

    Lấy ρ ≈ 0,4 ở đầu bảng ⇒ hệ số 2,5.

    ## Hiệu chuẩn: giữ nguyên TRUNG BÌNH đã đo, chỉ phân bổ lại

    Quét k phẳng cho đỉnh ở **k ≈ 7,5**. Công thức này chạy trên 1.012 câu cho
    **trung bình 7,43** — tức không phá vỡ điều duy nhất đã đo được, mà chuyển
    slot từ câu đơn sang câu nhóm:

        k=3: 367 câu · k=4: 72 · k=6: 28 · k=9: 191 · k=12: 354
        43,4% câu nộp ÍT hơn 6 · 53,9% nộp NHIỀU hơn 6

    Đây là **giả thuyết chưa đo trên leaderboard**. Nó thay đúng một biến so với
    cấu hình `k6` (F₂ 0,4137) nên quy được nguyên nhân.

    ## Cảnh báo cho lần sau
    `--adaptive-k` bản trước là **no-op**: công thức `4 + 1,5·log₂(1+n)` có sàn
    5,5 nên với `--k-max 6` mọi câu đều ra đúng 6. Cờ bật mà không đổi gì, không
    có lỗi nào báo. Vì vậy hàm này có test kiểm chính chuyện đó.
    """
    g = estimate_gold_tables(ir, n_docs)
    return int(max(2, min(k_max, round(mult * g))))


DOCS_SLACK = 1                  # đệm cho doc filter bỏ sót


def adaptive_docs(ir: QueryIR, cap: int = 8, n_docs: int | None = None) -> int:
    """Số doc ghi vào `relevant_docs` — chấm RIÊNG với `relevant_tables`.

    Khác hẳn bài toán bảng ở một điểm quyết định: **precision của doc rất cao**.
    Đo 17/08 khi nộp 2,06 doc/câu: P = 0,932, R = 0,845. Doc thứ n mà ta thêm
    vào gần như luôn là gold, nên ngưỡng đáng thêm

        p > h/(4G + n) = 1,92/(4·2,27 + 2,06) = 0,17

    bị vượt rất xa. Nghĩa là nộp thiếu doc là mất điểm gần như thuần tuý — chỉ
    cần đừng nộp doc mà mình không có lý do gì để tin.

    Nên: nộp theo số cặp (công ty × năm) cộng một đệm, chứ không phải hằng số 3.
    Câu 1 công ty 1 năm nộp 2; câu 8 cặp nộp 9 — hằng số 3 chặn cứng recall của
    nhóm sau, mà nhóm ấy là 29,8% bộ đề.
    """
    n = n_targets(ir, n_docs) + DOCS_SLACK
    if n_docs:
        n = min(n, n_docs)          # đừng bịa doc mà doc filter không đưa ra
    return int(max(1, min(cap, n)))
