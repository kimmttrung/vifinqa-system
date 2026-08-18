"""Bộ giải tất định ĐA Ô — nhắm thẳng vào ANSWER_ACCURACY.

## Vì sao cần

Đo trên leaderboard 17/08: ANSWER_ACCURACY = 0,166 trong khi TABLES_RECALL = 0,61.
Nghĩa là **dữ liệu đúng đã nằm trong tay**, chỉ là trả về sai. Nguyên nhân: tầng
tra cứu cũ luôn trả về ĐÚNG MỘT ô, còn ~355 câu hỏi đơn vị `%` / `lần` / `năm nào`
/ `bao nhiêu công ty` cần **hai ô trở lên rồi mới tính**.

## Phạm vi
Chỉ những dạng suy luận **quy được về công thức đóng**. Câu lọc nhóm / multi-hop
phụ thuộc (T4) vẫn để cho LLM planner ở Stage E — cố nhồi vào rule sẽ sinh đáp án
sai một cách tự tin, tệ hơn là bỏ trống.

    AGGREGATE  tổng nhiều công ty/năm
    AVERAGE    trung bình
    SUPERLATIVE  max/min — trả GIÁ TRỊ, hoặc trả NĂM nếu câu hỏi là "năm nào"
    DIFF       hiệu hai kỳ / hai công ty
    GROWTH     tăng trưởng (a−b)/|b|
    RATIO      tỷ số theo công thức tài chính (ROE, biên LN, D/E…)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from common import fold
from factlookup import FactHit, lookup, lookup_aligned, lookup_concept
from qparse import QueryIR

_NON_VND = {"percent", "pp", "times", "year", "count", "shares", "usd"}


@dataclass
class Solution:
    value: float
    code: str                       # biểu thức pandas dùng df1..dfN
    facts: list[FactHit] = field(default_factory=list)   # thứ tự == df1..dfN
    kind: str = ""


# ───────────────────── sinh biểu thức đọc một ô ─────────────────────

def _q(s: str) -> str:
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def cell_expr(var: str, f: FactHit) -> str:
    return (f"float({var}.loc[{var}[{_q(f.label_col)}].astype(str).str.strip() == "
            f"{_q(f.label)}, {_q(f.col_name)}].iloc[0])")


def _scaled(var: str, f: FactHit, magnitude: bool = False) -> tuple[str, float]:
    """Biểu thức đã quy về VND, và giá trị tương ứng.

    `magnitude=True` bọc abs(): chi phí và dự phòng trong BCTC ghi trong ngoặc
    (số âm), nên "tỷ lệ chi phí QLDN trên doanh thu" ra −7,87% thay vì +7,87%.
    Mẫu số của tỷ số cũng luôn là độ lớn (tổng tài sản, doanh thu, vốn CSH).
    """
    e = cell_expr(var, f)
    v = f.value_raw * f.unit_scale
    if not math.isclose(f.unit_scale, 1.0, rel_tol=1e-12):
        e = f"{e} * {f.unit_scale!r}"
    if magnitude:
        e, v = f"abs({e})", abs(v)
    return e, v


def _final(expr: str, value: float, ir: QueryIR) -> tuple[str, float]:
    """Quy đổi sang đơn vị câu hỏi yêu cầu ở BƯỚC CUỐI, không sớm hơn."""
    if ir.unit_kind in _NON_VND:
        return expr, value
    if math.isclose(ir.unit_scale, 1.0, rel_tol=1e-12):
        return expr, value
    return f"({expr}) / {ir.unit_scale!r}", value / ir.unit_scale


# ───────────────────── công thức tỷ số tài chính ─────────────────────
# (regex trên câu hỏi đã fold) → (tử, mẫu, nhân 100 hay không)
# Tử/mẫu là concept của Stage B, KHÔNG phải chữ trong câu hỏi — "ROE" không chứa
# chữ nào của "lợi nhuận sau thuế", nên khớp nhãn kiểu lexical vô dụng ở đây.
RATIO_FORMULAS: list[tuple[re.Pattern, str, str, bool]] = [
    (re.compile(r"\bROE\b|TY SUAT (LOI NHUAN TREN )?VON CHU SO HUU"),
     "LOI_NHUAN_SAU_THUE", "VON_CHU_SO_HUU", True),
    (re.compile(r"\bROA\b|TY SUAT (LOI NHUAN TREN )?TONG TAI SAN"),
     "LOI_NHUAN_SAU_THUE", "TONG_TAI_SAN", True),
    (re.compile(r"BIEN LOI NHUAN GOP|TY LE LOI NHUAN GOP TREN DOANH THU|"
                r"TY SUAT LOI NHUAN GOP"),
     "LOI_NHUAN_GOP", "DOANH_THU_THUAN", True),
    (re.compile(r"BIEN LOI NHUAN RONG|TY SUAT LOI NHUAN RONG|\bROS\b|"
                r"BIEN LOI NHUAN SAU THUE|TY SUAT LOI NHUAN SAU THUE"),
     "LOI_NHUAN_SAU_THUE", "DOANH_THU_THUAN", True),
    (re.compile(r"BIEN LOI NHUAN (HOAT DONG|THUAN)"),
     "LOI_NHUAN_THUAN_HDKD", "DOANH_THU_THUAN", True),
    (re.compile(r"HE SO (KHA NANG )?THANH TOAN HIEN HANH|TY SO THANH TOAN HIEN HANH|"
                r"TAI SAN NGAN HAN GAP BAO NHIEU LAN NO NGAN HAN"),
     "TAI_SAN_NGAN_HAN", "NO_NGAN_HAN", False),
    (re.compile(r"\bD/E\b|NO PHAI TRA TREN VON CHU SO HUU|HE SO NO TREN VON"),
     "NO_PHAI_TRA", "VON_CHU_SO_HUU", False),
    (re.compile(r"HE SO NO( PHAI TRA)? TREN TONG TAI SAN|TY LE NO TREN TONG TAI SAN"),
     "NO_PHAI_TRA", "TONG_TAI_SAN", True),
    (re.compile(r"VONG QUAY TONG TAI SAN|DOANH THU THUAN TREN TONG TAI SAN"),
     "DOANH_THU_THUAN", "TONG_TAI_SAN", False),
    (re.compile(r"VONG QUAY HANG TON KHO"),
     "GIA_VON_HANG_BAN", "HANG_TON_KHO", False),
    (re.compile(r"CFO ?/ ?LNST|LUU CHUYEN TIEN THUAN.*TREN LOI NHUAN SAU THUE|"
                r"HE SO CHUYEN DOI LOI NHUAN"),
     "LC_TIEN_THUAN_HDKD", "LOI_NHUAN_SAU_THUE", False),
    (re.compile(r"CFO TREN DOANH THU|LUU CHUYEN TIEN THUAN.*TREN DOANH THU"),
     "LC_TIEN_THUAN_HDKD", "DOANH_THU_THUAN", True),
    (re.compile(r"TY LE CHI PHI BAN HANG TREN DOANH THU"),
     "CHI_PHI_BAN_HANG", "DOANH_THU_THUAN", True),
    (re.compile(r"TY LE CHI PHI QUAN LY.*TREN DOANH THU"),
     "CHI_PHI_QUAN_LY_DOANH_NGHIEP", "DOANH_THU_THUAN", True),
    (re.compile(r"TY TRONG HANG TON KHO TREN TONG TAI SAN"),
     "HANG_TON_KHO", "TONG_TAI_SAN", True),
]

# ────────────────── chốt chặn: câu nào KHÔNG được đụng vào ──────────────────
#
# Nguy hiểm nhất của bộ giải rule không phải là bỏ sót, mà là bắn vào câu nó
# không hiểu. Ví dụ thật (q471): "tổng doanh thu thuần của các công ty CÓ TỶ LỆ
# lợi nhuận sau thuế trên doanh thu > X" — có HAI chỉ tiêu: một để LỌC, một để
# TRẢ LỜI. Rule chỉ thấy một, cộng bừa cả nhóm, ra số sai mà vẫn chạy trơn tru.
# Sai kiểu này còn tệ hơn bỏ trống vì nó không để lại dấu vết nào.
#
# Nên chuyển sang **gating dương**: chỉ giải khi chắc chắn câu đơn tầng.

_NESTED = re.compile(
    r"TRONG NHOM|XET CAC|XET NHUNG|XET NHOM|TRONG SO (CAC )?|TRONG CAC DOANH NGHIEP|"
    r"(CAC|NHUNG|BON|BA|NAM|SAU|BAY)\s+(CONG TY|DOANH NGHIEP|DON VI|MA|NAM)"
    r"[^,\.]{0,70}\bCO\b|"
    # \bCO\b(?! PHAN): fold("cổ phần") = "CO PHAN", nên "của công ty cổ phần"
    # khớp nhầm thành mệnh đề quan hệ "công ty CÓ …" và chặn oan câu đơn tầng.
    r"(TAI|O|VAO|CUA)\s+(DOANH NGHIEP|CONG TY|NAM|THOI DIEM|MA)[^,\.]{0,60}\bCO\b(?! PHAN)|"
    # "Tại năm MÀ Hoà Phát ĐẠT doanh thu cao nhất, hệ số… là bao nhiêu" — mệnh đề
    # lồng dùng "mà/đạt" thay vì "có", chốt chặn cũ để lọt hết nhóm này.
    r"(TAI|VAO|O|DEN)\s+(NAM|THOI DIEM|DOANH NGHIEP|CONG TY)\s+MA\b|"
    r"\bNAM\s+MA\b|\bDOANH NGHIEP\s+MA\b|\bCONG TY\s+MA\b|"
    # "trong NĂM CÓ số dư … lớn nhất" — mệnh đề quan hệ dính liền, không có
    # giới từ dẫn nên hai luật trên không bắt được. Chặn "Việt Nam có" bằng
    # lookbehind, và KHÔNG chặn "công ty cổ phần" (fold ra "CONG TY CO PHAN").
    r"(?<!VIET )\bNAM\s+CO\b(?! PHAN)|\bDOANH NGHIEP\s+CO\b(?! PHAN)|"
    r"TRUNG VI|DUY TRI|LIEN TIEP|DONG LOAT|THEO KICH BAN|"
    r"\bNEU\b|GIA SU|GIA DINH|SAU KHI (GIAM|TANG|LOC)|"
    r"CAO HON|THAP HON|LON HON|NHO HON|VUOT QUA|"
    r"NGAY SAU|KE TIEP|DAU TIEN|LIEN TRUOC|TU DO"
)

# Đầu mục chỉ tiêu tài chính — đếm số chỉ tiêu KHÁC NHAU trong câu.
# ≥2 chỉ tiêu ⇒ gần như chắc chắn có mệnh đề lọc lồng bên trong.
_METRIC_HEADS = [
    "DOANH THU", "GIA VON", "LOI NHUAN", "CHI PHI", "TAI SAN", "NO PHAI TRA",
    "NO NGAN HAN", "NO DAI HAN", "VON CHU SO HUU", "HANG TON KHO", "TIEN GUI",
    "LUU CHUYEN TIEN", "DONG TIEN", "VONG QUAY", "PHAI THU", "PHAI TRA",
    "CHO VAY", "VAY NGAN HAN", "VAY DAI HAN", "CO PHIEU", "KHAU HAO",
    "DU PHONG", "THUE", "\\bCFO\\b", "\\bROE\\b", "\\bROA\\b", "BIEN LOI NHUAN",
    "HE SO", "TY SUAT", "\\bD/E\\b",
]
_METRIC_RE = [re.compile(p) for p in _METRIC_HEADS]


def _n_metrics(fq: str) -> int:
    return sum(1 for rx in _METRIC_RE if rx.search(fq))


def too_complex(ir: QueryIR, max_metrics: int = 1) -> bool:
    """True ⇒ nhường Stage E, đừng đoán."""
    fq = fold(ir.question)
    if _NESTED.search(fq):
        return True
    if {"CONDITIONAL", "MEDIAN", "COUNT"} & set(ir.ops):
        return True
    return _n_metrics(fq) > max_metrics


_EXPENSE_LABEL = re.compile(r"CHI PHI|GIA VON|DU PHONG|TRICH LAP|KHAU HAO|"
                            r"GIAM TRU|LO |THUE TNDN")


def _is_expense(label: str) -> bool:
    return bool(_EXPENSE_LABEL.search(fold(label)))


_ABS_DIFF = re.compile(r"CHENH LECH|KHAC BIET|CHENH BAO NHIEU")
_SIGNED_DIFF = re.compile(r"HIEU SO|TRU DI|TANG BAO NHIEU|GIAM BAO NHIEU|"
                          r"BIEN DONG|THAY DOI")
_MIN_WORDS = re.compile(r"NHO NHAT|THAP NHAT|IT NHAT")


def _ratio_formula(ir: QueryIR):
    """Chỉ trả công thức khi khớp ĐÚNG MỘT — hai công thức cùng khớp nghĩa là
    câu hỏi dùng một tỷ số để lọc và hỏi một tỷ số khác (q363: D/E để chọn năm,
    rồi hỏi hệ số thanh toán lãi vay). Đó là multi-hop, không phải tra tỷ số."""
    fq = fold(ir.retrieval_query + " " + ir.question)
    found = [(num, den, pct) for rx, num, den, pct in RATIO_FORMULAS if rx.search(fq)]
    return found[0] if len(found) == 1 else None


# ───────────────────────────── các bộ giải ─────────────────────────────

def _solve_ratio(ir: QueryIR) -> Solution | None:
    spec = _ratio_formula(ir)
    if spec is None or not ir.tickers or not ir.years:
        return None
    num_c, den_c, is_pct = spec
    tk, yr = ir.tickers[0], ir.years[-1]
    fn = lookup_concept(ir, tk, yr, num_c)
    fd = lookup_concept(ir, tk, yr, den_c)
    if fn is None or fd is None:
        return None
    # Lợi nhuận giữ dấu (công ty lỗ thì ROE/ROS phải âm), chi phí lấy độ lớn.
    en, vn = _scaled("df1", fn, magnitude=_is_expense(fn.label))
    ed, vd = _scaled("df2", fd, magnitude=True)
    if vd == 0:
        return None
    mult = 100.0 if (is_pct and ir.unit_kind in ("percent", "pp")) else 1.0
    val = vn / vd * mult
    code = f"({en}) / ({ed})" + (f" * {mult!r}" if mult != 1.0 else "")
    return Solution(value=val, code=code, facts=[fn, fd], kind="RATIO")


# ───────────────── tỷ số dạng "A trên B" (không cần bảng công thức) ─────────────────
#
# Bảng RATIO_FORMULAS chỉ phủ tỷ số CÓ TÊN (ROE, D/E…) — mới bắn 5/291 câu.
# Phần lớn câu hỏi nói thẳng cả hai vế: "Tỷ trọng nguyên giá TSCĐ hữu hình TRÊN
# tổng tài sản". Không cần công thức, chỉ cần tách ở "trên" rồi tra từng vế.

_RATIO_SPLIT = re.compile(r"^(.*?)\s+(?:TREN|CHIA CHO)\s+(.*)$")
_TIMES_SPLIT = re.compile(r"^(.*?)\s+GAP(?:\s+BAO NHIEU)?\s+LAN\s+(.*)$")
# tiền tố "tỷ lệ/tỷ trọng/hệ số…" không phải tên chỉ tiêu, phải bỏ trước khi tra
_RATIO_LEAD = re.compile(r"^(TY LE|TY TRONG|TY SUAT|HE SO|BIEN|MUC|GIA TRI)\s+")
# Chỉ cắt cụm THỜI ĐIỂM/PHẠM VI ở đuôi. Bản đầu liệt kê token lẻ `TAI` nên nó
# cắt luôn "TAI CHINH": "đầu tư TÀI CHÍNH dài hạn" bị rút thành "đầu tư", mất
# chữ "dài hạn" ⇒ veto trái nghĩa không còn gì để so và khớp sang "ngắn hạn".
_TAIL_NOISE = re.compile(
    r"\s+(CUOI NAM|DAU NAM|CUOI KY|DAU KY|CUOI QUY|TRONG NAM|TRONG KY|"
    r"TAI NGAY|DEN NGAY|VAO NGAY|TAI THOI DIEM|CUA)\b.*$")


def _clean_side(s: str) -> str:
    s = _RATIO_LEAD.sub("", s.strip())
    s = _TAIL_NOISE.sub("", s).strip()
    return s


def _solve_generic_ratio(ir: QueryIR) -> Solution | None:
    """Tách "A trên B" / "A gấp bao nhiêu lần B" rồi tra hai vế độc lập."""
    if not ir.tickers or not ir.years:
        return None
    rq = ir.retrieval_query
    m = _RATIO_SPLIT.match(rq) or _TIMES_SPLIT.match(rq)
    if not m:
        return None
    num_q, den_q = _clean_side(m.group(1)), _clean_side(m.group(2))
    if len(num_q.split()) < 2 or len(den_q.split()) < 2:
        return None

    tk, yr = ir.tickers[0], ir.years[-1]
    hn = lookup(ir, ticker=tk, years=[yr], top=1, min_label=0.5, query=num_q)
    hd = lookup(ir, ticker=tk, years=[yr], top=1, min_label=0.5, query=den_q)
    if not hn or not hd:
        return None
    fn, fd = hn[0], hd[0]
    # hai vế trỏ vào ĐÚNG một ô ⇒ tách sai, tỷ số sẽ ra 1.0 một cách vô nghĩa
    if (fn.doc_id, fn.table_idx, fn.row_idx, fn.col_idx) == \
       (fd.doc_id, fd.table_idx, fd.row_idx, fd.col_idx):
        return None
    en, vn = _scaled("df1", fn, magnitude=_is_expense(fn.label))
    ed, vd = _scaled("df2", fd, magnitude=True)
    if vd == 0 or vn == vd:
        # hai vế ra ĐÚNG một giá trị ⇒ tách sai cụm, tỷ số 100% vô nghĩa (q670)
        return None
    mult = 100.0 if ir.unit_kind in ("percent", "pp") else 1.0
    code = f"({en}) / ({ed})" + (f" * {mult!r}" if mult != 1.0 else "")
    return Solution(value=vn / vd * mult, code=code, facts=[fn, fd], kind="RATIO_AB")


def _targets(ir: QueryIR) -> dict[tuple[str, int], FactHit]:
    return lookup_aligned(ir)


def _solve_growth(ir: QueryIR, hits: dict) -> Solution | None:
    """(sau − trước)/|trước| — cần đúng 2 kỳ của cùng một công ty."""
    if len(ir.tickers) != 1 or len(ir.years) < 2:
        return None
    y0, y1 = min(ir.years), max(ir.years)
    a, b = hits.get((ir.tickers[0], y1)), hits.get((ir.tickers[0], y0))
    if a is None or b is None:
        return None
    ea, va = _scaled("df1", a)
    eb, vb = _scaled("df2", b)
    if vb == 0:
        return None
    mult = 100.0 if ir.unit_kind in ("percent", "pp") else 1.0
    val = (va - vb) / abs(vb) * mult
    code = f"(({ea}) - ({eb})) / abs({eb})" + (f" * {mult!r}" if mult != 1.0 else "")
    return Solution(value=val, code=code, facts=[a, b], kind="GROWTH")


def _solve_diff(ir: QueryIR, hits: dict) -> Solution | None:
    """Hiệu hai vế. `chênh lệch` không có hướng → abs; `hiệu số`/`trừ đi` giữ dấu."""
    if len(hits) != 2:
        return None
    keys = sorted(hits)
    # câu hỏi nêu công ty/năm theo thứ tự nào thì trừ theo thứ tự đó
    if len(ir.tickers) == 2:
        keys = sorted(keys, key=lambda k: ir.tickers.index(k[0]))
    else:
        keys = sorted(keys, key=lambda k: -k[1])       # năm sau trừ năm trước
    a, b = hits[keys[0]], hits[keys[1]]
    ea, va = _scaled("df1", a)
    eb, vb = _scaled("df2", b)
    fq = fold(ir.question)
    use_abs = bool(_ABS_DIFF.search(fq)) and not _SIGNED_DIFF.search(fq)
    val = abs(va - vb) if use_abs else (va - vb)
    code = f"({ea}) - ({eb})"
    if use_abs:
        code = f"abs({code})"
    return Solution(value=val, code=code, facts=[a, b], kind="DIFF")


def _solve_reduce(ir: QueryIR, hits: dict, op: str) -> Solution | None:
    """tổng / trung bình / lớn nhất / nhỏ nhất trên nhiều (công ty, năm)."""
    if len(hits) < 2:
        return None
    keys = sorted(hits)
    exprs, vals, facts = [], [], []
    for i, k in enumerate(keys, 1):
        e, v = _scaled(f"df{i}", hits[k])
        exprs.append(e)
        vals.append(v)
        facts.append(hits[k])
    joined = ", ".join(exprs)
    if op == "SUM":
        val, code = sum(vals), f"sum([{joined}])"
    elif op == "AVG":
        val, code = sum(vals) / len(vals), f"(sum([{joined}]) / {len(vals)})"
    elif op == "MIN":
        val, code = min(vals), f"min([{joined}])"
    else:
        val, code = max(vals), f"max([{joined}])"
    return Solution(value=val, code=code, facts=facts, kind=op)


def _solve_argmax_year(ir: QueryIR, hits: dict) -> Solution | None:
    """"Năm nào ... lớn nhất" → trả về SỐ NĂM, không phải giá trị."""
    if len(hits) < 2 or len({k[1] for k in hits}) < 2:
        return None
    keys = sorted(hits, key=lambda k: k[1])
    pairs, vals = [], []
    for i, k in enumerate(keys, 1):
        e, v = _scaled(f"df{i}", hits[k])
        pairs.append(f"({e}, {float(k[1])!r})")
        vals.append((v, float(k[1])))
    pick_min = bool(_MIN_WORDS.search(fold(ir.question)))
    best = (min if pick_min else max)(vals)[1]
    fn = "min" if pick_min else "max"
    code = f"{fn}([{', '.join(pairs)}])[1]"
    return Solution(value=best, code=code, facts=[hits[k] for k in keys],
                    kind="ARGMAX_YEAR")


# ───────────────────────────── điều phối ─────────────────────────────

def solve(ir: QueryIR) -> Solution | None:
    """→ Solution, hoặc None nếu không dạng nào áp được (rơi về tra 1 ô)."""
    ops = set(ir.ops)

    # Tỷ số đi trước: nó tự tra concept nên không phụ thuộc khớp nhãn của câu hỏi.
    # Cho phép 3 "chỉ tiêu" vì bản thân tên tỷ số đã chứa 2 (vd "tỷ suất lợi
    # nhuận trên vốn chủ sở hữu" = TY SUAT + LOI NHUAN + VON CHU SO HUU).
    if ir.unit_kind in ("percent", "pp", "times") and not too_complex(ir, max_metrics=4):
        # tỷ số có tên đi trước (chính xác hơn vì dùng concept), rồi mới đến
        # dạng "A trên B" tách bằng chữ
        s = _solve_ratio(ir) or _solve_generic_ratio(ir)
        if s is not None:
            return _finish(s, ir)

    if too_complex(ir):
        return None

    hits = _targets(ir)
    if len(hits) < 2:
        return None

    if ir.unit_kind == "year":
        s = _solve_argmax_year(ir, hits)
        if s is not None:
            return _finish(s, ir, skip_unit=True)
        return None

    if "GROWTH" in ops:
        s = _solve_growth(ir, hits)
        if s is not None:
            return _finish(s, ir, skip_unit=(ir.unit_kind in ("percent", "pp", "times")))

    if "DIFF" in ops:
        s = _solve_diff(ir, hits)
        if s is not None:
            return _finish(s, ir)

    if "SUPERLATIVE" in ops:
        op = "MIN" if _MIN_WORDS.search(fold(ir.question)) else "MAX"
        s = _solve_reduce(ir, hits, op)
        if s is not None:
            return _finish(s, ir)

    if "AVERAGE" in ops:
        s = _solve_reduce(ir, hits, "AVG")
        if s is not None:
            return _finish(s, ir)

    if "AGGREGATE" in ops and len(hits) >= 2:
        s = _solve_reduce(ir, hits, "SUM")
        if s is not None:
            return _finish(s, ir)

    return None


def _finish(s: Solution, ir: QueryIR, skip_unit: bool = False) -> Solution | None:
    if skip_unit:
        code, val = s.code, s.value
    else:
        code, val = _final(s.code, s.value, ir)
    if not math.isfinite(val):
        return None
    s.code, s.value = code, val
    return s
