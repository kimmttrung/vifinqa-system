# %% [markdown]
# # ViFinQA — Stage A: Corpus Normalization
#
# **Input**  : 1.973 file `financial_statements/TICKER/YEAR/DOC/DOC_extracted.txt` (~363 MiB)
# **Output** : `artifacts/docs.parquet` (1.973 dòng) · `artifacts/tables.parquet` (~146.246 dòng) · `artifacts/qc_report.md`
#
# Notebook này **thuần CPU** — regex + HTMLParser, GPU không tăng tốc gì cả.
# Chọn Accelerator = **None** để tiết kiệm quota T4 (30h/tuần). T4 chỉ cần ở notebook
# `02_index` (embedding) và `03_infer` (LLM). Phần phụ lục cuối file dùng T4 nếu bạn đã bật sẵn.
#
# Chỉ dùng stdlib + pandas/pyarrow (Kaggle cài sẵn) — không cần `pip install`, chạy được khi tắt internet.

# %%
import os
import re
import json
import time
import unicodedata
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter, defaultdict

IN_KAGGLE = Path("/kaggle/input").exists()


def find_data_root() -> Path:
    """Tìm thư mục chứa financial_statements/ — không phụ thuộc tên dataset hay số tầng lồng.

    Kaggle mount ở nhiều dạng khác nhau tuỳ cách tạo dataset:
        /kaggle/input/<slug>/financial_statements                      (2 tầng)
        /kaggle/input/<slug>/data/financial_statements                 (3 tầng)
        /kaggle/input/datasets/<user>/<slug>/data/financial_statements (5 tầng)  ← đã gặp thật
    Nên quét tới 6 tầng thay vì hard-code.
    """
    if IN_KAGGLE:
        base = Path("/kaggle/input")
        for depth in range(1, 7):
            pattern = "/".join(["*"] * depth) + "/financial_statements"
            hits = [p for p in base.glob(pattern) if p.is_dir()]
            if hits:
                if len(hits) > 1:
                    print("Có nhiều bản, dùng bản đầu:", [str(h) for h in hits])
                return hits[0].parent
        print("KHÔNG tìm thấy. Cây thật của /kaggle/input:")
        for p in sorted(base.rglob("*"))[:60]:
            print("   ", p)
        raise FileNotFoundError(
            "Không thấy financial_statements/ trong /kaggle/input. "
            "Xem cây ở trên rồi gán tay: DATA_ROOT = Path('/kaggle/input/.../data')"
        )
    env = os.environ.get("VIFINQA_DATA")
    if env:
        return Path(env)
    here = Path.cwd()
    for cand in (here, here.parent, here.parent.parent):
        if (cand / "financial_statements").is_dir():
            return cand
    raise FileNotFoundError("Không thấy financial_statements/ — set env VIFINQA_DATA")


DATA_ROOT = find_data_root()
OUT_DIR = Path("/kaggle/working/artifacts") if IN_KAGGLE else Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_ROOT :", DATA_ROOT)
print("OUT_DIR   :", OUT_DIR)

# %% [markdown]
# ## A2 — Chuẩn hoá text (`fold`)
#
# OCR sai dấu tiếng Việt liên tục: `KẾT QUÀ`, `LƯU CHUYÊN`, `NỘ PHẢI TRẢ`, `THUYÊT MINH`.
# Bỏ dấu rồi UPPER thì tất cả các biến thể đó **trùng nhau** → mọi so khớp keyword đều đi qua `fold()`.

# %%
_DMAP = str.maketrans({"đ": "d", "Đ": "D"})
_WS = re.compile(r"\s+")


def strip_diacritics(s: str) -> str:
    s = s.translate(_DMAP)
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def fold(s) -> str:
    """Bỏ dấu + gộp whitespace + UPPER. Dùng cho MỌI so khớp keyword tiếng Việt."""
    if not s:
        return ""
    return _WS.sub(" ", strip_diacritics(str(s))).strip().upper()


assert fold("KẾT QUÀ") == fold("KẾT QUẢ") == "KET QUA"
assert fold("LƯU CHUYÊN TIỀN TỆ") == fold("LƯU CHUYỂN TIỀN TỆ") == "LUU CHUYEN TIEN TE"
assert fold("THUYÊT MINH") == "THUYET MINH"
print("A2 ok")

# %% [markdown]
# ## A5 — Parser số kiểu Việt Nam
#
# `1.234.567` → 1234567 · `(162.105.381)` → -162105381 · `-` → None · `1.234,56` → 1234.56
#
# Quy tắc tách dấu: nếu có cả `.` và `,` thì dấu **xuất hiện sau** là thập phân.
# Nếu chỉ một loại dấu và mọi nhóm sau dấu đầu đều đúng 3 chữ số → dấu ngăn nghìn (chuẩn VN).

# %%
_NULLS = {"", "-", "--", "---", "–", "—", ".", ",", "x", "X", "n/a", "N/A", "NA", "*"}
_DATE_LIKE = re.compile(r"^\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*\d{2,4}$")
_NUMCHARS = re.compile(r"^[0-9.,]+$")


def parse_number(s):
    """Trả float, hoặc None nếu ô không phải số."""
    if s is None:
        return None
    raw = str(s).strip()
    if raw in _NULLS:
        return None
    if _DATE_LIKE.match(raw):          # 31/12/2015 là mốc thời gian, không phải số liệu
        return None
    t = raw.replace(" ", " ")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1].strip()
    t = t.rstrip("%").strip()
    t = re.sub(r"\s", "", t)
    if t.startswith(("-", "−")):
        neg, t = True, t[1:]
    elif t.startswith("+"):
        t = t[1:]
    if not t or not _NUMCHARS.match(t):
        return None

    last_dot, last_com = t.rfind("."), t.rfind(",")
    if last_dot >= 0 and last_com >= 0:
        cut = max(last_dot, last_com)
        int_part = re.sub(r"[.,]", "", t[:cut])
        frac = t[cut + 1:]
    else:
        sep = "." if last_dot >= 0 else ("," if last_com >= 0 else None)
        if sep is None:
            int_part, frac = t, ""
        else:
            parts = t.split(sep)
            groups_ok = all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3
            if groups_ok:
                int_part, frac = "".join(parts), ""      # dấu ngăn nghìn
            else:
                int_part, frac = parts[0], "".join(parts[1:])  # dấu thập phân
    if not int_part.isdigit() or (frac and not frac.isdigit()):
        return None
    val = float(f"{int_part}.{frac}") if frac else float(int_part)
    return -val if neg else val


assert parse_number("1.234.567") == 1234567.0
assert parse_number("(162.105.381)") == -162105381.0
assert parse_number("-") is None
assert parse_number("1.234,56") == 1234.56
assert parse_number("642") == 642.0
assert parse_number("411a") is None
assert parse_number("31/12/2015") is None
assert parse_number("31.12.2022Triệu VND") is None
assert parse_number("1.954.764.678.040") == 1954764678040.0
print("A5 ok")

# %% [markdown]
# ## A4 — HTML `<table>` → lưới 2-D (expand rowspan/colspan)
#
# Báo cáo ngân hàng dùng `rowspan`/`colspan` rất nhiều (header 2 tầng).
# Ô bị span được **nhân bản** ra mọi vị trí nó chiếm — nhờ vậy nối header nhiều tầng theo cột rất dễ.

# %%
_MAX_SPAN = 60  # chặn colspan/rowspan rác do OCR (vd colspan="9999")


def _to_int(v, default=1):
    try:
        n = int(str(v).strip())
        return n if 1 <= n <= _MAX_SPAN else default
    except Exception:
        return default


class _GridParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self._row = None
        self._cell = None
        self._rs = 1
        self._cs = 1

    def _flush_cell(self):
        if self._cell is not None:
            txt = _WS.sub(" ", "".join(self._cell)).strip()
            if self._row is None:
                self._row = []
            self._row.append((txt, self._rs, self._cs))
            self._cell, self._rs, self._cs = None, 1, 1

    def _flush_row(self):
        self._flush_cell()
        if self._row:
            self.rows.append(self._row)
        self._row = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._flush_row()
            self._row = []
        elif tag in ("td", "th"):
            self._flush_cell()
            if self._row is None:
                self._row = []
            self._cell = []
            self._rs = _to_int(a.get("rowspan"))
            self._cs = _to_int(a.get("colspan"))
        elif tag == "br":
            if self._cell is not None:
                self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._flush_cell()
        elif tag in ("tr", "table"):
            self._flush_row()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_to_grid(raw_html: str):
    """HTML → list[list[str]] hình chữ nhật, rowspan/colspan đã expand."""
    p = _GridParser()
    try:
        p.feed(raw_html)
        p.close()
    except Exception:
        pass
    p._flush_row()
    occupied = {}
    max_c = 0
    for r, row in enumerate(p.rows):
        c = 0
        for txt, rs, cs in row:
            while (r, c) in occupied:
                c += 1
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, c + dc)] = txt
            c += cs
            max_c = max(max_c, c)
    if not occupied:
        return []
    n_rows = max(r for r, _ in occupied) + 1
    return [[occupied.get((r, c), "") for c in range(max_c)] for r in range(n_rows)]


_ACB = ('<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Tại ngày</td></tr>'
        '<tr><td>Thuyết minh</td><td>31.12.2022 Triệu VND</td><td>31.12.2021 Triệu VND</td></tr>'
        '<tr><td>I</td><td>Tiền mặt, vàng bạc, đá quý</td><td>4</td><td>8.460.883</td>'
        '<td>7.509.867</td></tr></table>')
_g = html_to_grid(_ACB)
assert len(_g) == 3 and {len(r) for r in _g} == {5}, _g
assert _g[0] == ["", "", "Tại ngày", "Tại ngày", "Tại ngày"]
assert _g[1][:2] == ["", ""] and _g[1][2] == "Thuyết minh"
assert _g[2][3] == "8.460.883"
print("A4 ok — lưới ACB:", _g[0], "|", _g[1])

# %% [markdown]
# ## A6 — Suy đơn vị
#
# Đơn vị nằm ở 2 nơi: dòng phía trên bảng (`Đơn vị: Triệu VND`) hoặc **trong ô header**
# (`31.12.2022 Triệu VND` — kiểu ngân hàng).
#
# OCR làm hỏng cả chữ "Đơn vị" (đã gặp thật: `Dom vi: VND`) nên **không dò nhãn**,
# chỉ dò thẳng token đơn vị. Ưu tiên header cell nếu nó cho scale > 1.

# %%
# Dùng (?<![A-Z]) thay cho \b ở đầu: OCR hay dính liền số với đơn vị ("31.12.2022Triệu VND"),
# lúc đó giữa '2' và 'T' KHÔNG có word boundary nên \bTRIEU sẽ trượt và rơi xuống match 'VND' = 1.0.
# (?<!CONG ) chặn "Công ty VND" bị đọc thành đơn vị tỷ — sau khi bỏ dấu thì "tỷ" và "ty" trùng nhau.
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


def detect_unit(caption: str, header_text: str):
    """→ (unit_scale, unit_source ∈ {header_cell, caption, default})"""
    h = _detect_unit(fold(header_text))
    c = _detect_unit(fold(caption))
    if h is not None and h > 1:
        return h, "header_cell"
    if c is not None:
        return c, "caption"
    if h is not None:
        return h, "header_cell"
    return 1.0, "default"


assert detect_unit("Đơn vị: VND", "TÀI SẢN Mã số 31/12/2015") == (1.0, "caption")
assert detect_unit("Dom vi: VND", "") == (1.0, "caption")          # OCR hỏng nhãn vẫn bắt được
assert detect_unit("", "Thuyết minh 31.12.2022 Triệu VND") == (1e6, "header_cell")
assert detect_unit("", "Thuyết minh 31.12.2022Triệu VND") == (1e6, "header_cell")  # dính liền
assert detect_unit("Công ty VND", "") == (1.0, "caption")          # không đọc nhầm thành tỷ
print("A6 ok")

# %% [markdown]
# ## A7 — Ngữ nghĩa cột (stock / flow / đầu kỳ)
#
# Bước ROI cao nhất Stage A: 45,6% câu hỏi hỏi **cuối năm**, 24,4% hỏi **trong năm**, 1,4% hỏi **đầu năm**.
# Lấy đúng chỉ tiêu nhưng sai cột ⇒ sai đáp án.
#
# `31/12/2015`→(stock, 2015, end) · `01/01/2015`→(stock, 2015, **begin**) · `Năm 2015`→(flow, 2015)

# %%
# Không dùng \b quanh năm: OCR hay dính năm với đơn vị ("2018VND", "2023VND") — giữa '8' và 'V'
# không có word boundary nên \b(20\d{2})\b sẽ trượt và cả cột mất mốc thời gian.
_DATE_IN = re.compile(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(20\d{2})")
_NAM_YYYY = re.compile(r"(?<![A-Z])NAM\s*(20\d{2})(?![0-9])")
_BARE_YYYY = re.compile(r"(?<![0-9])(20\d{2})(?![0-9])")


def classify_period(header_text: str, doc_year, section: str):
    """→ dict{period_type, year, is_begin} hoặc None nếu cột không mang mốc thời gian."""
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


assert classify_period("31/12/2015", 2015, "CDKT") == {"period_type": "stock", "year": 2015, "is_begin": False}
assert classify_period("01/01/2015", 2015, "CDKT") == {"period_type": "stock", "year": 2015, "is_begin": True}
assert classify_period("Năm 2015", 2015, "KQKD") == {"period_type": "flow", "year": 2015, "is_begin": False}
assert classify_period("Tại ngày 31.12.2022 Triệu VND", 2022, "CDKT")["year"] == 2022
assert classify_period("Số đầu năm", 2019, "CDKT")["is_begin"] is True
assert classify_period("Mã số", 2015, "CDKT") is None
assert classify_period("2018VND", 2018, "TM")["year"] == 2018      # năm dính đơn vị
assert classify_period("Năm2023VND", 2023, "KQKD") == {"period_type": "flow", "year": 2023,
                                                       "is_begin": False}
print("A7 ok")

# %% [markdown]
# ## A8 — Phân loại section bằng **mã biểu mẫu**
#
# Đo trên corpus: 1.613/1.973 doc (81,8%) có mã biểu mẫu ngay trên bảng.
# Đây là tín hiệu mạnh hơn hẳn text heading. **Ngân hàng lệch một bậc so với doanh nghiệp:**
#
# | | CĐKT | KQKD | LCTT | Thuyết minh |
# |---|---|---|---|---|
# | DN (TT200) | B01 | B02 | B03 | B09 |
# | TCTD (ngân hàng) | **B02** | **B03** | **B04** | **B05** |
# | CTCK (chứng khoán) | B01 | B02 | B03 | B09 |
#
# 18% doc còn lại fallback sang keyword đã fold (nên `LƯU CHUYÊN` OCR hỏng vẫn khớp).

# %%
FORM_RE = re.compile(r"\bB\s*0?(\d{1,2})\s*[-/]\s*(DN|TCTD|CTCK|DNBH)", re.I)

_SECTION_BY_FORM = {
    ("DN", 1): "CDKT", ("DN", 2): "KQKD", ("DN", 3): "LCTT", ("DN", 4): "VCSH", ("DN", 9): "TM",
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


def section_from_form(line: str):
    m = FORM_RE.search(line)
    if not m:
        return None
    return _SECTION_BY_FORM.get((m.group(2).upper(), int(m.group(1))))


def section_from_heading(folded_line: str):
    for kw, sec in _HEADING_RULES:
        if kw in folded_line:
            return sec
    return None


assert section_from_form("MÃU B 01-DN/HN") == "CDKT"
assert section_from_form("MẦU B 02-DN/HN") == "KQKD"
assert section_from_form("Mẫu B02/TCTD") == "CDKT"      # ngân hàng: B02 = CĐKT
assert section_from_form("Mẫu B05/TCTD-HN") == "TM"
assert section_from_heading(fold("BẢO CẢO LƯU CHUYÊN TIỀN TỆ HỢP NHẤT")) == "LCTT"
assert section_from_heading(fold("THUYÊT MINH BÁO CÁO TÀI CHÍNH")) == "TM"
print("A8 ok")

# %% [markdown]
# ## A1 + A3 — Quét document & định vị bảng
#
# `table_idx` (1-based), `char_start`, `line_no`, `page_no` — **lưu cả 4** vì format
# `<vị_trí_bảng>` mà BTC yêu cầu chưa xác định được (Unknown #1 trong CLAUDE.md).
# Probe leaderboard xong chỉ cần đổi 1 dòng, không phải parse lại 363 MiB.

# %%
PAGE_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$")
TABLE_RE = re.compile(r"<table\b.*?</table>", re.I | re.S)
STMT_TYPES = ("consolidated", "separate", "aggregated")


def parse_doc_meta(txt_path: Path, data_root: Path) -> dict:
    doc_dir = txt_path.parent
    doc_id = doc_dir.name
    year_dir = doc_dir.parent.name
    ticker = doc_dir.parent.parent.name
    low = doc_id.lower()
    stmt_type = next((s for s in STMT_TYPES if s in low), "other")
    m = re.search(r"(20\d{2})", doc_id)
    year = int(m.group(1)) if m else (int(year_dir) if year_dir.isdigit() else None)
    return {
        "doc_id": doc_id,
        "file_stem": txt_path.name[:-4],          # bỏ '.txt', vẫn còn '_extracted'
        "ticker": ticker,
        "year": year,
        "year_dir": int(year_dir) if year_dir.isdigit() else None,
        "stmt_type": stmt_type,
        "rel_path": str(txt_path.relative_to(data_root)).replace("\\", "/"),
    }


def _line_index(text: str):
    """→ (danh sách offset đầu dòng, danh sách nội dung dòng)."""
    starts, lines, pos = [], [], 0
    for line in text.split("\n"):
        starts.append(pos)
        lines.append(line)
        pos += len(line) + 1
    return starts, lines


def _bisect(sorted_vals, x):
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def _header_row_count(grid, max_header=3):
    """Số hàng header = số hàng đầu chưa phải hàng dữ liệu.

    Dừng ở hàng banner (một ô colspan phủ HẾT chiều rộng, vd 'I. LƯU CHUYỂN TIỀN TỆ...').
    Lưu ý: banner phải phủ hết n_cols — nếu không, header 2 tầng kiểu ngân hàng
    ('Tại ngày' colspan=3 trên tổng 5 cột) sẽ bị cắt mất tầng chứa đơn vị 'Triệu VND'.
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
            break                                   # banner phủ toàn bộ chiều rộng
        has_num = any(parse_number(c) is not None for c in row)
        if has_num and len(cells) >= 2:
            break                                   # đã sang hàng dữ liệu
        if n >= 1 and n_cols >= 4 and len(cells) <= 2:
            break                                   # hàng nhãn kiểu ['A','TÀI SẢN','','','']
        n += 1
    return max(n, 1)


def process_document(txt_path: Path, data_root: Path):
    """→ (doc_record, [table_record...])"""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    meta = parse_doc_meta(txt_path, data_root)
    doc_year = meta["year"]
    starts, lines = _line_index(text)

    # Sự kiện theo dòng: đổi trang, đổi section (mã biểu mẫu ưu tiên hơn heading)
    page_offsets, page_nums = [], []
    sec_offsets, sec_values = [], []
    for i, line in enumerate(lines):
        if len(line) > 300:            # dòng dài = chính là bảng, bỏ qua cho nhanh
            continue
        stripped = line.strip()
        if not stripped:
            continue
        m = PAGE_RE.match(stripped)
        if m:
            page_offsets.append(starts[i])
            page_nums.append(int(m.group(1)))
            continue
        sec = section_from_form(stripped)
        if sec is None:
            sec = section_from_heading(fold(stripped))
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
        header_cells = [c for r in grid[:n_head] for c in r]
        header_text = " ".join(header_cells)
        has_ma_so = "MA SO" in fold(header_text)

        section = sec_values[si] if si >= 0 else "OTHER"
        unit_scale, unit_source = detect_unit(caption, header_text)

        col_periods = []
        for c in range(n_cols):
            col_head = " ".join(grid[r][c] for r in range(n_head))
            col_periods.append(classify_period(col_head, doc_year, section))

        tables.append({
            "doc_id": meta["doc_id"],
            "ticker": meta["ticker"],
            "year": doc_year,
            "stmt_type": meta["stmt_type"],
            "table_idx": idx,                 # 1-based, ứng viên #1 cho <vị_trí_bảng>
            "char_start": cs,                 # ứng viên #2
            "char_end": ce,
            "line_no": li + 1,                # 1-based, ứng viên #3
            "page_no": page_nums[pi] if pi >= 0 else None,   # ứng viên #4
            "caption": caption[:500],
            "section": section,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_header_rows": n_head,
            "has_ma_so": has_ma_so,
            "unit_scale": unit_scale,
            "unit_source": unit_source,
            "n_numeric_cells": sum(1 for r in grid for c in r if parse_number(c) is not None),
            "col_periods": json.dumps(col_periods, ensure_ascii=False),
            "grid": json.dumps(grid, ensure_ascii=False),
            "raw_html": raw_html,
        })

    meta.update({
        "n_chars": len(text),
        "n_pages": len(page_offsets),
        "n_tables": len(tables),
    })
    return meta, tables


# %% [markdown]
# ### Kiểm tra trên 2 document thật trước khi chạy full
# Ground truth đã đo sẵn: `AAA/2015/consolidated` có **47 bảng**, bảng bắt đầu ở dòng
# 19, 214, 239, 286, 333… và file có 43 trang.

# %%
_aaa = DATA_ROOT / ("financial_statements/AAA/2015/AAA_financial_statements_2015_consolidated/"
                    "AAA_financial_statements_2015_consolidated_extracted.txt")
_meta, _tabs = process_document(_aaa, DATA_ROOT)

assert _meta["n_tables"] == 47, _meta["n_tables"]
assert _meta["n_pages"] == 43, _meta["n_pages"]
assert _meta["ticker"] == "AAA" and _meta["year"] == 2015 and _meta["stmt_type"] == "consolidated"
assert [t["line_no"] for t in _tabs[:5]] == [19, 214, 239, 286, 333]

_cdkt = _tabs[1]          # bảng ở dòng 214 = Bảng cân đối kế toán
assert _cdkt["section"] == "CDKT" and _cdkt["has_ma_so"] and _cdkt["unit_scale"] == 1.0
_g = json.loads(_cdkt["grid"])
_periods = json.loads(_cdkt["col_periods"])
assert _periods[3] == {"period_type": "stock", "year": 2015, "is_begin": False}
assert _periods[4] == {"period_type": "stock", "year": 2015, "is_begin": True}

# Ràng buộc kế toán: 100 + 200 == 270 (đây là bài test thật cho parser số)
_by_code = {r[1]: parse_number(r[3]) for r in _g if len(r) > 3}
assert _by_code["100"] + _by_code["200"] == _by_code["270"] == 1954764678040.0
print("AAA ok — section/đơn vị/cột/số đều đúng, 100+200==270 ==", _by_code["270"])

_acb = DATA_ROOT / ("financial_statements/ACB/2022/ACB_financial_statements_2022_separate/"
                    "ACB_financial_statements_2022_separate_extracted.txt")
_m2, _t2 = process_document(_acb, DATA_ROOT)
_bank_cdkt = next(t for t in _t2 if t["section"] == "CDKT" and t["n_numeric_cells"] > 20)
assert _bank_cdkt["unit_scale"] == 1e6, _bank_cdkt["unit_scale"]     # Triệu VND từ header cell
assert not _bank_cdkt["has_ma_so"]                                   # ngân hàng KHÔNG có Mã số
print("ACB ok — unit", _bank_cdkt["unit_scale"], "| has_ma_so", _bank_cdkt["has_ma_so"],
      "| header rows", _bank_cdkt["n_header_rows"])

# %% [markdown]
# ## A9 — Chạy toàn corpus → parquet
#
# Ghi theo lô (mỗi 150 doc một `write_table`) để RAM không phình:
# `raw_html` + `grid` của 146k bảng gộp lại xấp xỉ kích thước corpus gốc.

# %%
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ALL_TXT = sorted((DATA_ROOT / "financial_statements").glob("*/*/*/*_extracted.txt"))
print("Số file:", len(ALL_TXT))
assert len(ALL_TXT) == 1973, f"Kỳ vọng 1973 file, thấy {len(ALL_TXT)}"

TABLES_PQ = OUT_DIR / "tables.parquet"
DOCS_PQ = OUT_DIR / "docs.parquet"
SHARD = 150

t0 = time.time()
doc_rows, buf, writer, n_tables = [], [], None, 0
failures = []

try:
    for i, p in enumerate(ALL_TXT, 1):
        try:
            meta, tabs = process_document(p, DATA_ROOT)
        except Exception as e:                      # không để 1 file hỏng chặn cả pipeline
            failures.append({"path": str(p), "error": repr(e)[:300]})
            continue
        doc_rows.append(meta)
        buf.extend(tabs)
        n_tables += len(tabs)
        if len(buf) >= 4000 or i == len(ALL_TXT):
            tbl = pa.Table.from_pylist(buf)
            if writer is None:
                writer = pq.ParquetWriter(TABLES_PQ, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            buf = []
        if i % SHARD == 0 or i == len(ALL_TXT):
            print(f"  {i:4d}/{len(ALL_TXT)} doc · {n_tables:6d} bảng · {time.time()-t0:5.1f}s",
                  flush=True)
finally:
    if buf:
        tbl = pa.Table.from_pylist(buf)
        if writer is None:
            writer = pq.ParquetWriter(TABLES_PQ, tbl.schema, compression="zstd")
        writer.write_table(tbl)
    if writer is not None:
        writer.close()

docs = pd.DataFrame(doc_rows)
docs.to_parquet(DOCS_PQ, compression="zstd", index=False)

print(f"\nXong sau {time.time()-t0:.1f}s")
print("docs.parquet  :", len(docs), "dòng ·", DOCS_PQ.stat().st_size / 1e6, "MB")
print("tables.parquet:", n_tables, "dòng ·", TABLES_PQ.stat().st_size / 1e6, "MB")
if failures:
    print("FILE LỖI:", len(failures))

# %% [markdown]
# ## Nghiệm thu Stage A
# Đối chiếu với các con số **đã đo độc lập** trên corpus trước khi viết code.

# %%
tables = pd.read_parquet(TABLES_PQ)

checks = {
    "docs == 1973": len(docs) == 1973,
    "tables == 146246": len(tables) == 146246,
    "tickers == 100": docs.ticker.nunique() == 100,
    # 957/954/7/55 là con số chính thức trong dataset card của HuggingFace
    "consolidated == 957": (docs.stmt_type == "consolidated").sum() == 957,
    "separate == 954": (docs.stmt_type == "separate").sum() == 954,
    "aggregated == 7": (docs.stmt_type == "aggregated").sum() == 7,
    "other == 55": (docs.stmt_type == "other").sum() == 55,
    "max bảng/doc == 248": docs.n_tables.max() == 248,
    "lưới hợp lệ >= 99%": (tables.n_cols > 0).mean() >= 0.99,
    "không file lỗi": len(failures) == 0,
}
_fin = tables[tables.section.isin(["CDKT", "KQKD", "LCTT"])]
_hp = _fin.col_periods.map(lambda s: any(p is not None for p in json.loads(s)))
checks["cột kỳ trên bảng dữ liệu >= 85%"] = _hp[_fin.n_numeric_cells >= 6].mean() >= 0.85
for k, v in checks.items():
    print(("  PASS  " if v else "  FAIL  "), k)

print("\nSection:\n", tables.section.value_counts())
print("\nĐơn vị:\n", tables.groupby(["unit_scale", "unit_source"]).size())
print("\nhas_ma_so:", f"{tables.has_ma_so.mean():.1%}")

fin = tables[tables.section.isin(["CDKT", "KQKD", "LCTT"])]
has_period = fin.col_periods.map(lambda s: any(p is not None for p in json.loads(s)))
cov_all = has_period.mean()
cov_data = has_period[fin.n_numeric_cells >= 6].mean()
# Đo trên mẫu 120 doc: 82,5% mọi bảng · 89,3% bảng có dữ liệu số.
# Phần trượt chủ yếu là bảng KHÔNG có cột kỳ thật (danh sách HĐQT, bảng tỷ lệ khấu hao)
# bị section "dính" từ mã biểu mẫu phía trên — không phải lỗi parser kỳ.
print(f"Cột kỳ — mọi bảng CDKT/KQKD/LCTT   : {cov_all:.1%}")
print(f"Cột kỳ — bảng có dữ liệu số (≥6 ô) : {cov_data:.1%}")

# %%
lines = [
    "# Stage A — QC report",
    "",
    f"- docs: **{len(docs)}** · tables: **{len(tables)}** · file lỗi: **{len(failures)}**",
    f"- lưới hợp lệ: {(tables.n_cols>0).mean():.2%} · has_ma_so: {tables.has_ma_so.mean():.1%}",
    f"- cột kỳ (CDKT/KQKD/LCTT): mọi bảng {cov_all:.1%} · bảng có dữ liệu số {cov_data:.1%}",
    "",
    "## Nghiệm thu",
    *[f"- {'PASS' if v else 'FAIL'} — {k}" for k, v in checks.items()],
    "",
    "## Section",
    tables.section.value_counts().to_markdown(),
    "",
    "## Đơn vị",
    tables.groupby(["unit_scale", "unit_source"]).size().to_markdown(),
    "",
    "## 50 bảng đáng ngờ (nhiều ô nhưng không có ô số nào)",
    (tables[(tables.n_numeric_cells == 0) & (tables.n_rows >= 4)]
     .nlargest(50, "n_rows")[["doc_id", "table_idx", "line_no", "page_no", "section",
                              "n_rows", "n_cols", "caption"]].to_markdown(index=False)),
]
(OUT_DIR / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")
print("Đã ghi", OUT_DIR / "qc_report.md")

# %% [markdown]
# ## Ví dụ dùng output ở stage sau
# Đây chính là thứ Stage B/D sẽ query — không bao giờ phải mở lại file .txt gốc.

# %%
q = tables[(tables.ticker == "HPG") & (tables.year == 2023) &
           (tables.stmt_type == "consolidated") & (tables.section == "CDKT")]
print(q[["doc_id", "table_idx", "line_no", "page_no", "n_rows", "n_cols",
         "unit_scale", "has_ma_so"]].head())

if len(q):
    g = json.loads(q.iloc[0].grid)
    for row in g[:6]:
        print(row[:5])

# %% [markdown]
# ## Phụ lục (tuỳ chọn) — dùng T4 nếu bạn đã bật GPU
#
# Stage A không cần GPU. Nhưng nếu session đang có T4 sẵn thì build luôn embedding index
# cho Stage D — 146k bảng, khoảng 40–50 phút trên 1×T4.
#
# Passage embed **không phải toàn bộ ô** (vô ích, gây nhiễu) mà là:
# caption + header + 25 nhãn cột đầu. Cần bật Internet để tải model, hoặc attach model
# dưới dạng Kaggle Dataset khi chạy offline.

# %%
BUILD_INDEX = False        # đổi True nếu muốn chạy

if BUILD_INDEX:
    import torch
    from sentence_transformers import SentenceTransformer

    def to_passage(row):
        g = json.loads(row.grid)
        head = " | ".join(g[0]) if g else ""
        labels = " ; ".join(r[0] for r in g[1:26] if r and r[0].strip())
        return (f"[{row.doc_id}] trang {row.page_no} mục {row.section}\n"
                f"{row.caption}\n{head}\n{labels}")

    passages = [to_passage(r) for r in tables.itertuples()]
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding", device="cuda")
    model.max_seq_length = 256
    emb = model.encode(passages, batch_size=128, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    import numpy as np
    np.save(OUT_DIR / "table_emb.npy", emb.astype("float16"))
    print("embeddings:", emb.shape)
