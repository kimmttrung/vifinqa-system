"""Primitive dùng chung — trích từ Stage A để Stage B..G không phải copy-paste.

Mọi so khớp text tiếng Việt PHẢI đi qua fold(): OCR sai dấu liên tục
(KẾT QUÀ / LƯU CHUYÊN / NỘ PHẢI TRẢ), bỏ dấu thì các biến thể trùng nhau.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

# ─────────────────────────── đường dẫn ───────────────────────────

_IN_KAGGLE = Path("/kaggle/input").exists()


def _search(marker: str, max_depth: int = 6) -> Path | None:
    """Tìm thư mục CHỨA `marker` (vd 'artifacts/tables.parquet'), trả None nếu không có.

    Kaggle mount dataset ở độ sâu không đoán trước được:
        /kaggle/input/<slug>/artifacts/…
        /kaggle/input/<slug>/vifinqa/artifacts/…
        /kaggle/input/datasets/<user>/<slug>/data/artifacts/…   (đã gặp thật)
    nên phải quét thay vì hard-code.
    """
    roots = [Path("/kaggle/input")] if _IN_KAGGLE else []
    here = Path(__file__).resolve().parent
    roots += [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent]
    for root in roots:
        if not root.exists():
            continue
        if (root / marker).exists():
            return root
        for depth in range(1, max_depth + 1):
            hits = list(root.glob("/".join(["*"] * depth) + "/" + marker))
            if hits:
                return hits[0].parents[len(Path(marker).parts) - 1]
    return None


def find_data_root() -> Path | None:
    """Thư mục chứa financial_statements/. **Trả None thay vì raise.**

    Stage A cần corpus gốc, nhưng Stage B..G chỉ đọc artifacts/*.parquet. Trên
    Kaggle ta KHÔNG upload 363 MiB corpus — nếu hàm này raise ở tầng module thì
    `import common` chết ngay và không stage nào chạy được.
    """
    env = os.environ.get("VIFINQA_DATA")
    if env and Path(env).exists():
        return Path(env)
    return _search("financial_statements")


def _resolve_artifacts() -> Path:
    """artifacts/ mặc định nằm TRONG repo, không nằm cạnh dataset.

    Thứ tự: env → artifacts/ của repo (nếu đã có dữ liệu) → dò trong /kaggle/input
    → artifacts/ của repo (kể cả rỗng, để build_corpus.py ghi vào).
    """
    env = os.environ.get("VIFINQA_ARTIFACTS")
    if env:
        return Path(env)
    local = REPO_ROOT / "artifacts"
    # Thu muc ton tai la du, khong doi phai co parquet: neu doi thi lan chay dau
    # tien (artifacts/ con rong) se ghi ra ngoai repo, vao bat ky artifacts/ nao
    # _search() tim thay truoc.  Tren Kaggle repo clone ve KHONG co artifacts/
    # (bi gitignore) nen tu dong roi xuong _search -> /kaggle/input.  Dung y do.
    if local.is_dir():
        return local
    root = _search("artifacts/tables.parquet") or _search("artifacts/table_meta.parquet")
    if root:
        return root / "artifacts"
    return local


def _resolve_questions() -> Path | None:
    env = os.environ.get("VIFINQA_QUESTIONS")
    if env and Path(env).exists():
        return Path(env)
    root = _search("questions/questions.jsonl")
    return (root / "questions" / "questions.jsonl") if root else None


# src/common.py → parents[1] là gốc repo
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = find_data_root()
ARTIFACTS = _resolve_artifacts()
OUT_DIR = Path(os.environ.get(
    "VIFINQA_OUT",
    "/kaggle/working/out" if _IN_KAGGLE else REPO_ROOT / "out"))

# ─────────────────────────── A2: fold ───────────────────────────

_DMAP = str.maketrans({"đ": "d", "Đ": "D"})
_WS = re.compile(r"\s+")


def strip_diacritics(s: str) -> str:
    s = s.translate(_DMAP)
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def fold(s) -> str:
    """Bỏ dấu + gộp whitespace + UPPER."""
    if not s:
        return ""
    return _WS.sub(" ", strip_diacritics(str(s))).strip().upper()


_TOKEN_RE = re.compile(r"[A-Z0-9]+")


def tokens(s) -> list[str]:
    """fold rồi tách token chữ-số. Dùng cho BM25 và so khớp tên công ty."""
    return _TOKEN_RE.findall(fold(s))


# ─────────────────────── A5: parser số kiểu VN ───────────────────────

_NULLS = {"", "-", "--", "---", "–", "—", ".", ",", "x", "X", "n/a", "N/A", "NA", "*"}
_DATE_LIKE = re.compile(r"^\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*\d{2,4}$")
_NUMCHARS = re.compile(r"^[0-9.,]+$")


def parse_number(s):
    """'1.234.567'→1234567 · '(162.105.381)'→-162105381 · '-'→None · '1.234,56'→1234.56"""
    if s is None:
        return None
    raw = str(s).strip()
    if raw in _NULLS:
        return None
    if _DATE_LIKE.match(raw):
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
                int_part, frac = "".join(parts), ""
            else:
                int_part, frac = parts[0], "".join(parts[1:])
    if not int_part.isdigit() or (frac and not frac.isdigit()):
        return None
    val = float(f"{int_part}.{frac}") if frac else float(int_part)
    return -val if neg else val


# ─────────────────── A4: HTML <table> → lưới 2-D ───────────────────

_MAX_SPAN = 60


def _to_int(v, default=1):
    try:
        n = int(str(v).strip())
        return n if 1 <= n <= _MAX_SPAN else default
    except Exception:
        return default


class _GridParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self._row, self._cell = [], None, None
        self._rs = self._cs = 1

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
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._flush_cell()
        elif tag in ("tr", "table"):
            self._flush_row()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_to_grid(raw_html: str) -> list[list[str]]:
    """HTML → list[list[str]] hình chữ nhật, rowspan/colspan đã expand."""
    p = _GridParser()
    try:
        p.feed(raw_html)
        p.close()
    except Exception:
        pass
    p._flush_row()
    occupied, max_c = {}, 0
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


# ─────────────────────────── I/O artifacts ───────────────────────────

def load_docs():
    import pandas as pd
    return pd.read_parquet(ARTIFACTS / "docs.parquet")


def load_tables(columns=None):
    """tables.parquet nặng 62 MB (raw_html + grid). Truyền columns= để đọc nhẹ."""
    import pandas as pd
    return pd.read_parquet(ARTIFACTS / "tables.parquet", columns=columns)


TABLE_LIGHT_COLS = [
    "doc_id", "ticker", "year", "stmt_type", "table_idx", "line_no", "page_no",
    "caption", "section", "n_rows", "n_cols", "n_header_rows", "has_ma_so",
    "unit_scale", "unit_source", "n_numeric_cells", "col_periods",
]


def load_questions() -> list[dict]:
    p = _resolve_questions()
    if p is None:
        raise FileNotFoundError(
            "Không thấy questions/questions.jsonl. Set env VIFINQA_QUESTIONS trỏ tới file.")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@lru_cache(maxsize=1)
def load_code_stock() -> dict[str, str]:
    """{ticker: tên công ty} — NGUỒN CHÂN LÝ, không sửa file."""
    import csv
    root = _search("code_stock.csv")
    path = (root / "code_stock.csv") if root else (DATA_ROOT or Path.cwd()) / "code_stock.csv"
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vals = list(row.values())
            if len(vals) >= 2 and vals[0]:
                out[vals[0].strip()] = (vals[1] or "").strip()
    return out
