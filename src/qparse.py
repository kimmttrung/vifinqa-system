"""Stage C — Question parser tất định: câu hỏi → QueryIR.

Năm tín hiệu quyết định đáp án đúng/sai đều rút được bằng rule, KHÔNG cần LLM:
  ticker · năm · separate|consolidated · đơn vị đáp án · stock|flow

Sai bất kỳ tín hiệu nào là sai đáp án dù lấy đúng chỉ tiêu. Đây là lý do
chúng nằm ở tầng tất định chứ không giao cho mô hình.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from functools import lru_cache

from common import fold, load_code_stock, tokens
from companies import ALIASES, resolve_companies


@lru_cache(maxsize=256)
def _company_name_tokens(ticker: str) -> tuple[str, ...]:
    """Token trong tên chính thức + mọi alias của một mã — để loại khỏi câu truy vấn."""
    ws = set(tokens(load_code_stock().get(ticker, "")))
    for phrase, tk in ALIASES.items():
        if tk == ticker:
            ws |= set(phrase.split())
    # giữ lại từ vừa là tên công ty vừa là thuật ngữ kế toán, nếu không sẽ mất chỉ tiêu
    keep = {"DIEN", "LUC", "TAI", "CHINH", "DAU", "TU", "PHAT", "TRIEN", "XAY", "DUNG",
            "THUONG", "MAI", "DICH", "VU", "SAN", "XUAT", "VAN", "HANG", "TIEU", "DUNG"}
    return tuple(w for w in ws if len(w) >= 2 and w not in keep)

# ───────────────────────── đơn vị đáp án ─────────────────────────
# scale = số chia từ VND sang đơn vị câu hỏi yêu cầu.
# kind:  vnd | percent | pp | times | year | count | shares | usd
# Thứ tự QUAN TRỌNG: 'diem phan tram' phải đứng trước 'phan tram',
# 'nghin ty'/'tram ty' phải đứng trước 'ty'.
_UNIT_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"DIEM\s*(PHAN\s*TRAM|%)"), "pp", 1.0),
    (re.compile(r"NGHIN\s*TY(\s*(DONG|VND))?"), "vnd", 1e12),
    (re.compile(r"TRAM\s*TY(\s*(DONG|VND))?"), "vnd", 1e11),
    (re.compile(r"(?<!CONG )TY\s*(DONG|VND)"), "vnd", 1e9),
    (re.compile(r"TRIEU\s*(DONG|VND)"), "vnd", 1e6),
    (re.compile(r"(NGHIN|NGAN)\s*(DONG|VND)"), "vnd", 1e3),
    (re.compile(r"\bUSD\b|\bDO\s*LA\b"), "usd", 1.0),
    (re.compile(r"CO\s*PHIEU"), "shares", 1.0),
    (re.compile(r"PHAN\s*TRAM|%"), "percent", 1.0),
    (re.compile(r"\bLAN\b"), "times", 1.0),
    (re.compile(r"\b(DONG|VND)\b"), "vnd", 1.0),
]

# Vị trí "câu hỏi thật sự nằm ở đâu" — đơn vị phải lấy ở mệnh đề nghi vấn,
# không phải ở mệnh đề giả định ("nếu doanh thu giảm 10%").
_ASK_RE = re.compile(r"BAO NHIEU|\bMAY\b|TINH THEO DON VI|TINH BANG|DON VI (TINH )?LA|KET QUA")

_YEAR_ANSWER = re.compile(r"NAM NAO|VAO NAM NAO|LA NAM MAY")
_COUNT_ANSWER = re.compile(r"CO BAO NHIEU (CONG TY|DOANH NGHIEP|DON VI|NGAN HANG|MA)")

# ───────────────────────── năm ─────────────────────────
_Y = r"(20[0-2]\d)"
_RANGE_RES = [
    re.compile(rf"GIAI DOAN {_Y}\s*[-–—]\s*{_Y}"),
    re.compile(rf"GIAI DOAN (?:TU )?(?:NAM )?{_Y} (?:DEN|TOI|-|–) (?:NAM )?{_Y}"),
    re.compile(rf"TU (?:NAM )?{_Y} (?:DEN|TOI) (?:NAM )?{_Y}"),
    re.compile(rf"(?:CAC )?NAM (?:TAI CHINH )?TU {_Y} DEN {_Y}"),
    re.compile(rf"{_Y}\s*[-–—]\s*{_Y}(?!\d)"),
]
_ANY_YEAR = re.compile(rf"(?<![0-9/.]){_Y}(?![0-9])")
# "31/12/2023", "31.12.2023", "31 tháng 12 năm 2019"
_DATE_YEAR = re.compile(rf"(\d{{1,2}})\s*[/.\-]\s*(\d{{1,2}})\s*[/.\-]\s*{_Y}")
_DATE_TEXT = re.compile(rf"(\d{{1,2}}) THANG (\d{{1,2}}) NAM {_Y}")

# ───────────────────────── mốc thời gian ─────────────────────────
_END_RE = re.compile(r"CUOI NAM|CUOI KY|CUOI QUY|DEN NGAY|TAI NGAY|VAO NGAY|"
                     r"TAI THOI DIEM|TINH DEN|THOI DIEM CUOI|SO CUOI")
_BEGIN_RE = re.compile(r"DAU NAM|DAU KY|SO DAU|01/01|1/1/")
_PERIOD_RE = re.compile(r"TRONG NAM|TRONG KY|CA NAM|PHAT SINH TRONG|TRONG GIAI DOAN")

# ───────────────────────── loại báo cáo ─────────────────────────
_SEPARATE_RE = re.compile(r"CONG TY ME|RIENG LE|BAO CAO RIENG|CO SO RIENG|"
                          r"PHAM VI CONG TY ME|CO SO CONG TY ME|BAN THAN NGAN HANG")
_CONSOL_RE = re.compile(r"HOP NHAT|TOAN TAP DOAN|CA TAP DOAN")

# ───────────────────────── phép suy luận ─────────────────────────
_OPS: list[tuple[str, re.Pattern]] = [
    ("GROWTH", re.compile(r"TANG TRUONG|TOC DO TANG|TANG BAO NHIEU|GIAM BAO NHIEU|"
                          r"SO VOI NAM|SO VOI CUNG|BIEN DONG|THAY DOI BAO NHIEU|"
                          r"MUC TANG|MUC GIAM|TANG\s*/\s*GIAM")),
    ("DIFF", re.compile(r"CHENH LECH|HIEU SO|TRU DI|CHENH BAO NHIEU|KHAC BIET|"
                        r"BIEN DONG|GIUA (CUOI )?NAM")),
    ("RATIO", re.compile(r"TY LE|TY TRONG|TY SUAT|HE SO|BIEN LOI NHUAN|BIEN [A-Z]+ NHUAN|"
                         r"\bROE\b|\bROA\b|\bROS\b|\bCFO\b|\bD/E\b|GAP BAO NHIEU LAN|"
                         r"CHIEM BAO NHIEU|VONG QUAY|TREN DOANH THU|TREN TONG TAI SAN|"
                         r"KHA NANG THANH TOAN|CHIA CHO")),
    ("AGGREGATE", re.compile(r"\bTONG\b|CONG LAI|TONG CONG|TONG SO")),
    ("AVERAGE", re.compile(r"TRUNG BINH|BINH QUAN")),
    ("MEDIAN", re.compile(r"TRUNG VI")),
    ("SUPERLATIVE", re.compile(r"LON NHAT|CAO NHAT|NHO NHAT|THAP NHAT|NHIEU NHAT|"
                               r"IT NHAT LA|DAT GIA TRI LON|NAM NAO|DUNG DAU")),
    ("COUNT", _COUNT_ANSWER),
    # "CO <chi tieu> DUONG" phai neu DICH DANH chi tieu, khong dung [A-Z ]+ chung
    # chung: sau khi fold, "Co phan Duong Quang Ngai" ra "CO PHAN DUONG ..." va khop
    # nham thanh menh de loc "co ... duong". Cung ho bay voi TY thuoc CONG TY (Muc 3 s8).
    ("CONDITIONAL", re.compile(r"TRONG NHOM CO|XET CAC|CHI XET|NEU |GIA SU|GIA DINH|"
                               r"THEO KICH BAN|SAU KHI LOC|"
                               r"CO (LOI NHUAN|DOANH THU|DONG TIEN|LUU CHUYEN|CFO|"
                               r"BIEN|TY LE|TY SUAT|GIA TRI)[A-Z ]*DUONG|"
                               r"CAO HON TRUNG VI|THAP HON TRUNG VI")),
]


@dataclass
class QueryIR:
    qid: int
    question: str
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    year_span: tuple[int, int] | None = None
    stmt_type: str | None = None          # separate | consolidated | None(=ưu tiên consolidated)
    stmt_explicit: bool = False
    unit_kind: str = "vnd"
    unit_scale: float = 1.0
    time_point: str | None = None         # end | begin | period
    ops: list[str] = field(default_factory=list)
    tier: int = 1
    retrieval_query: str = ""
    debug: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["year_span"] = list(self.year_span) if self.year_span else None
        return d


def _extract_years(fq: str) -> tuple[list[int], tuple[int, int] | None]:
    years: list[int] = []
    span = None
    for rx in _RANGE_RES:
        for m in rx.finditer(fq):
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b <= a + 15:
                years.extend(range(a, b + 1))
                span = (a, b) if span is None else (min(span[0], a), max(span[1], b))
    for rx in (_DATE_YEAR, _DATE_TEXT):
        years.extend(int(m.group(3)) for m in rx.finditer(fq))
    years.extend(int(m.group(1)) for m in _ANY_YEAR.finditer(fq))
    # 2 chữ số đầu của mã tài khoản không bao giờ ra 20xx nên không cần lọc thêm
    out = sorted({y for y in years if 2010 <= y <= 2026})
    return out, span


def _extract_unit(fq: str) -> tuple[str, float]:
    if _YEAR_ANSWER.search(fq):
        return "year", 1.0
    if _COUNT_ANSWER.search(fq):
        return "count", 1.0
    hits = [(m.start(), kind, scale)
            for rx, kind, scale in _UNIT_PATTERNS for m in rx.finditer(fq)]
    if not hits:
        return "vnd", 1.0
    asks = [m.start() for m in _ASK_RE.finditer(fq)]
    if asks:
        # đơn vị đúng là cái xuất hiện ngay sau mệnh đề nghi vấn CUỐI CÙNG
        for a in reversed(asks):
            after = [h for h in hits if h[0] >= a]
            if after:
                after.sort(key=lambda h: h[0])
                return after[0][1], after[0][2]
    hits.sort(key=lambda h: h[0])
    return hits[-1][1], hits[-1][2]


def _extract_time_point(fq: str) -> str | None:
    m_end = _END_RE.search(fq)
    m_beg = _BEGIN_RE.search(fq)
    m_per = _PERIOD_RE.search(fq)
    cands = [(m.start(), k) for m, k in
             ((m_end, "end"), (m_beg, "begin"), (m_per, "period")) if m]
    if not cands:
        return None
    # "đầu năm" hiếm (1,4%) nhưng khi có thì luôn thắng — nó là tín hiệu rất riêng
    for _, k in cands:
        if k == "begin":
            return "begin"
    cands.sort()
    return cands[0][1]


def _extract_stmt(fq: str) -> tuple[str | None, bool]:
    if _SEPARATE_RE.search(fq):
        return "separate", True
    if _CONSOL_RE.search(fq):
        return "consolidated", True
    return None, False


def _extract_ops(fq: str) -> list[str]:
    return [name for name, rx in _OPS if rx.search(fq)]


_STRIP_FOR_QUERY = re.compile(
    r"LA BAO NHIEU|BAO NHIEU|CUA CONG TY ME|CONG TY ME|DEN NGAY|VAO CUOI|"
    r"CTCP|CONG TY CO PHAN|TONG CONG TY|NGAN HANG TMCP|TAP DOAN|"
    r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*20[0-2]\d|20[0-2]\d|"
    r"DIEM PHAN TRAM|PHAN TRAM|TRIEU DONG|NGHIN TY DONG|TRAM TY DONG|TY DONG|"
    r"\bLAN\b|[?%(),;]"
)


def _retrieval_query(question: str, tickers: list[str]) -> str:
    """Câu truy vấn cho BM25/dense: bỏ mã CK, TÊN CÔNG TY, năm, đuôi đơn vị, khuôn mẫu.

    Bỏ tên công ty là bắt buộc, không phải làm đẹp: tên công ty nằm trong caption
    của gần như MỌI bảng thuộc doc đó, nên nó không phân biệt được bảng nào với
    bảng nào — chỉ làm loãng tín hiệu của tên chỉ tiêu.
    Cái còn lại đúng là *tên chỉ tiêu kế toán*, thứ duy nhất cần khớp nội dung bảng.
    """
    fq = fold(question)
    for tk in tickers:
        fq = re.sub(rf"\b{re.escape(tk)}\b", " ", fq)
    for tk in tickers:
        for w in _company_name_tokens(tk):
            fq = re.sub(rf"\b{re.escape(w)}\b", " ", fq)
    fq = _STRIP_FOR_QUERY.sub(" ", fq)
    return re.sub(r"\s+", " ", fq).strip()


def _tier(ir: QueryIR) -> int:
    hard = {"CONDITIONAL", "MEDIAN"}
    n_ops = len([o for o in ir.ops if o != "AGGREGATE"])
    if hard & set(ir.ops) or n_ops >= 3:
        return 4
    if n_ops >= 2 or len(ir.tickers) >= 3 or "SUPERLATIVE" in ir.ops:
        return 3
    if n_ops == 1 or len(ir.tickers) >= 2 or len(ir.years) >= 2:
        return 2
    return 1


def parse_question(qid: int, question: str) -> QueryIR:
    fq = fold(question)
    tickers, dbg = resolve_companies(question)
    years, span = _extract_years(fq)
    kind, scale = _extract_unit(fq)
    stmt, explicit = _extract_stmt(fq)
    ops = _extract_ops(fq)

    ir = QueryIR(
        qid=qid,
        question=question,
        tickers=tickers,
        years=years,
        year_span=span,
        stmt_type=stmt,
        stmt_explicit=explicit,
        unit_kind=kind,
        unit_scale=scale,
        time_point=_extract_time_point(fq),
        ops=ops,
        retrieval_query=_retrieval_query(question, tickers),
        debug=dbg,
    )
    ir.tier = _tier(ir)
    return ir


def parse_all(questions: list[dict]) -> list[QueryIR]:
    return [parse_question(q["id"], q["question"]) for q in questions]
