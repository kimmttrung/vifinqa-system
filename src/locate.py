"""Cell grounding tất định — định vị (dòng, cột) trong một bảng.

Bài học lớn nhất từ error analysis của mentor: 54,7% lỗi end-to-end là
"numerical extraction" — đọc nhầm ô, chứ KHÔNG phải tính sai (0,9%).
"LLM không dốt toán, vấn đề là chọn sai ô số liệu."

Nên bước này làm bằng rule, đo được, và luôn được executor đối chiếu lại.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from common import fold, parse_number, tokens
from emit import flatten_header
from qparse import QueryIR

# hư từ tiếng Việt — có mặt ở mọi nhãn dòng nên không giúp phân biệt
_STOP = {"CUA", "VA", "CAC", "CHO", "TRONG", "VOI", "THEO", "LA", "MOT", "NHUNG",
         "DEN", "TU", "TAI", "VE", "SO", "BANG", "GIUA", "MOI", "DUOC", "TREN",
         "NAM", "KY", "CUOI", "DAU", "TONG"}


@dataclass
class Cell:
    row: int              # chỉ số dòng trong phần dữ liệu (0-based, sau header)
    col: int
    label: str            # nhãn dòng nguyên văn
    label_col: int
    col_name: str
    raw: str
    value: float
    row_score: float
    col_score: float

    @property
    def score(self) -> float:
        return self.row_score * 0.7 + self.col_score * 0.3


def _weights(toks: list[str]) -> dict[str, float]:
    """Token dài và không phải hư từ thì mang nhiều thông tin hơn."""
    return {t: (0.15 if t in _STOP else min(1.0, 0.35 + 0.12 * len(t)))
            for t in dict.fromkeys(toks)}


# Cặp từ TRÁI NGHĨA trong kế toán. Điểm khớp theo phủ token không tách được
# "đầu tư tài chính DÀI hạn" khỏi "đầu tư tài chính NGẮN hạn": chỉ lệch một
# token ngắn trên tổng 4-5 token nên vẫn đạt điểm cao. Phải veto cứng.
_ANTONYMS = [
    ("NGAN HAN", "DAI HAN"),
    ("HUU HINH", "VO HINH"),
    ("PHAI THU", "PHAI TRA"),
    ("TRONG NUOC", "NUOC NGOAI"),
]


def _contradicts(qf: str, lf: str) -> bool:
    for a, b in _ANTONYMS:
        if (a in qf and b in lf and a not in lf) or (b in qf and a in lf and b not in lf):
            return True
    return False


def score_label(query_toks: list[str], label: str) -> float:
    """Độ khớp giữa cụm chỉ tiêu trong câu hỏi và nhãn một dòng của bảng."""
    lf = fold(label)
    if not lf:
        return 0.0
    if _contradicts(" ".join(query_toks), lf):
        return 0.0
    lt = set(tokens(lf))
    w = _weights(query_toks)
    tot = sum(w.values()) or 1.0
    hit = sum(v for t, v in w.items() if t in lt)
    cov = hit / tot
    # nhãn ngắn gọn đúng trọng tâm hơn nhãn dài lan man
    brevity = len(lt & set(w)) / max(len(lt), 1)
    # cụm liên tiếp xuất hiện nguyên vẹn là bằng chứng mạnh nhất
    phrase = 0.0
    qt = [t for t in query_toks if t not in _STOP]
    for i in range(len(qt)):
        for L in range(min(8, len(qt) - i), 1, -1):
            if " ".join(qt[i:i + L]) in lf:
                phrase = max(phrase, L / 6.0)
                break
    return 0.55 * cov + 0.15 * brevity + 0.30 * min(phrase, 1.0)


def _period_score(p: dict | None, ir: QueryIR, section: str, relax: bool = False) -> float:
    if not p:
        # relax: bảng không nhận được cột kỳ (11,7% bảng CDKT/KQKD/LCTT) vẫn phải
        # lấy được số, nếu không cả câu hỏi mất trắng.
        return 0.10 if relax else 0.0
    s = 0.0
    if ir.years and p.get("year") in ir.years:
        s += 0.6
    elif ir.years:
        if not relax:
            return 0.0                   # sai năm là loại, không phải trừ điểm
        s += 0.05
    else:
        s += 0.2
    want = ir.time_point
    is_begin = bool(p.get("is_begin"))
    ptype = p.get("period_type")
    if want == "begin":
        s += 0.4 if is_begin else 0.0
    elif want == "end":
        s += 0.4 if (not is_begin and ptype == "stock") else 0.1
    elif want == "period":
        s += 0.4 if ptype == "flow" else 0.1
    else:
        # không nêu mốc: suy từ section — CĐKT là số dư, KQKD/LCTT là phát sinh
        if section == "CDKT":
            s += 0.4 if (not is_begin and ptype == "stock") else 0.05
        elif section in ("KQKD", "LCTT"):
            s += 0.4 if ptype == "flow" else 0.2
        else:
            s += 0.25 if not is_begin else 0.05
    return s


_MA_SO_RE = re.compile(r"^\d{2,3}[A-Za-z]?$")

# Cột "Số cổ phiếu" và cột "VND" hay đứng cạnh nhau dưới cùng một tầng header
# ("31/12/2024 và 1/1/2024 | Số cổ phiếu | VND"), cả hai đều khớp năm nên chỉ
# tín hiệu kỳ thì không tách được. Phải phạt theo ĐƠN VỊ mà câu hỏi yêu cầu.
_COL_SHARES = re.compile(r"SO CO PHIEU|SO LUONG CO PHIEU|CO PHAN|SO LUONG")
_COL_PCT = re.compile(r"TY LE|PHAN TRAM|%")
_COL_NOTE = re.compile(r"THUYET MINH|MA SO|GHI CHU")


def _col_unit_penalty(col_name: str, unit_kind: str) -> float:
    f = fold(col_name.replace("_", " "))
    if _COL_NOTE.search(f):
        return -0.5
    if unit_kind in ("vnd", "usd"):
        if _COL_SHARES.search(f):
            return -0.45
        if _COL_PCT.search(f):
            return -0.35
    elif unit_kind == "shares":
        return 0.25 if _COL_SHARES.search(f) else -0.25
    elif unit_kind in ("percent", "pp"):
        return 0.2 if _COL_PCT.search(f) else 0.0
    return 0.0


# Chi phí/lỗ trong BCTC hay ghi trong ngoặc = số âm ("(3.036.974)"), nhưng câu
# hỏi "Chi phí dự phòng ... là bao nhiêu" mong đợi độ lớn. Chỉ áp dụng cho tra
# cứu trực tiếp — hỏi chênh lệch/tăng trưởng thì dấu là có nghĩa, phải giữ.
_EXPENSE_RE = re.compile(r"CHI PHI|GIA VON|LO |KHAU HAO|DU PHONG|THUE TNDN|"
                         r"TRICH LAP|GIAM TRU")


def expense_magnitude(value: float, ir: QueryIR) -> float:
    if value >= 0 or {"DIFF", "GROWTH"} & set(ir.ops):
        return value
    return abs(value) if _EXPENSE_RE.search(fold(ir.retrieval_query)) else value


def locate(grid: list[list[str]], n_header: int, col_periods_json: str,
           ir: QueryIR, section: str, metric_query: str | None = None,
           relax: bool = False) -> Cell | None:
    """→ ô khớp nhất, hoặc None nếu bảng không có ứng viên nào."""
    if not grid or len(grid) <= n_header:
        return None
    n_header = max(1, min(n_header, len(grid) - 1))
    names = flatten_header(grid, n_header)
    n_cols = len(names)
    periods = json.loads(col_periods_json) if col_periods_json else []
    periods += [None] * (n_cols - len(periods))

    qt = tokens(metric_query or ir.retrieval_query)
    if not qt:
        return None

    col_scores = [_period_score(periods[c], ir, section, relax)
                  + _col_unit_penalty(names[c], ir.unit_kind) for c in range(n_cols)]
    if max(col_scores, default=0.0) <= 0.0:
        return None
    min_row = 0.15 if relax else 0.25

    body = grid[n_header:]
    best: Cell | None = None
    for ri, row in enumerate(body):
        # nhãn = ô text đầu tiên không phải mã số / không phải số
        label, label_col = "", -1
        for ci in range(min(3, n_cols)):
            cell = (row[ci] if ci < len(row) else "").strip()
            if len(cell) >= 3 and not _MA_SO_RE.match(cell) and parse_number(cell) is None:
                label, label_col = cell, ci
                break
        if label_col < 0:
            continue
        rs = score_label(qt, label)
        if rs < min_row:
            continue
        for ci in range(n_cols):
            if col_scores[ci] <= 0.0 or ci == label_col:
                continue
            raw = (row[ci] if ci < len(row) else "").strip()
            v = parse_number(raw)
            if v is None:
                continue
            cand = Cell(row=ri, col=ci, label=label, label_col=label_col,
                        col_name=names[ci], raw=raw, value=v,
                        row_score=round(rs, 4), col_score=round(col_scores[ci], 4))
            if best is None or cand.score > best.score:
                best = cand
    return best


def locate_best(tables, ir: QueryIR):
    """Thử lần lượt các bảng ứng viên → (chỉ số bảng, Cell) tốt nhất.

    Bảng hạng 1 của retrieval không phải lúc nào cũng chứa chỉ tiêu được hỏi
    (BM25 hay bị caption boilerplate kéo lệch). Thử tiếp bảng hạng 2, 3… rẻ hơn
    nhiều so với trả về 0.0. Hết ứng viên thì nới ràng buộc năm/cột trên bảng
    hạng 1 — thà lấy số hơi lệch kỳ còn hơn bỏ trắng cả câu.
    """
    best = None
    for i, td in enumerate(tables):
        if td is None:
            continue
        c = locate(td.grid, td.n_header_rows, td.col_periods, ir, td.section)
        if c is not None and (best is None or c.score > best[1].score):
            best = (i, c)
    if best is not None:
        return best
    for i, td in enumerate(tables):
        if td is None:
            continue
        c = locate(td.grid, td.n_header_rows, td.col_periods, ir, td.section, relax=True)
        if c is not None:
            return i, c
    return None


# ───────────────────────── sinh pandas_query ─────────────────────────

def _pystr(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _wrap(expr: str, multiplier: float, use_abs: bool) -> str:
    if use_abs:
        expr = f"abs({expr})"
    if not math.isclose(multiplier, 1.0, rel_tol=1e-12):
        expr = f"{expr} * {multiplier!r}"
    return expr


def codegen_lookup(var: str, cell: Cell, label_col_name: str, multiplier: float,
                   dup_rank: int = 0, use_abs: bool = False) -> str:
    """Query lọc theo NHÃN (đọc được, ổn định) — executor sẽ đối chiếu lại.

    dup_rank > 0 khi bảng có nhiều dòng trùng nhãn: lấy đúng lần xuất hiện thứ n.
    """
    sel = (f"{var}.loc[{var}[{_pystr(label_col_name)}].astype(str).str.strip() == "
           f"{_pystr(cell.label)}, {_pystr(cell.col_name)}]")
    return _wrap(f"float({sel}.iloc[{dup_rank}])", multiplier, use_abs)


def codegen_positional(var: str, cell: Cell, multiplier: float,
                       use_abs: bool = False) -> str:
    """Dự phòng khi lọc theo nhãn không ra đúng ô (nhãn trùng, OCR lệch)."""
    return _wrap(f"float({var}.iat[{cell.row}, {cell.col}])", multiplier, use_abs)
