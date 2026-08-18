"""Stage B — Fact table: bảng phẳng (chỉ tiêu × kỳ) → giá trị.

Vì sao fact table quan trọng hơn nó có vẻ:

  1. Đây là **cell grounding tất định**. Error analysis của mentor: 54,7% lỗi
     end-to-end là đọc nhầm ô, 0,9% là tính sai. Chỗ nào tra được bằng fact table
     thì tỷ lệ đọc nhầm ô về gần 0.
  2. Nó cho **table_idx chính xác** ⇒ nộp k=1 ⇒ F₂ = 1.000, thay vì top-10 của
     pipeline retrieval (F₂ = 0.357).
  3. Nó là **bộ đo chất lượng duy nhất ta có**. Cuộc thi không phát nhãn gold cho
     relevant_tables, nên không tự đo được recall. Nhưng ràng buộc kế toán
     (270 = 440, 100+200 = 270) và đối chiếu liên năm thì tự kiểm được — sai
     ở đâu là biết parser hỏng ở đó.

Bốn adapter, không phải hai:

  | Adapter | Đối tượng            | Khoá nhận diện        |
  |---------|----------------------|-----------------------|
  | TT200   | ~70 doanh nghiệp     | cột "Mã số" (100/270/440…) |
  | TCTD    | 21 ngân hàng + EVF   | **nhãn** — ngân hàng KHÔNG có Mã số |
  | TT334   | CTCK (SSI/MBS/FTS)   | Mã số riêng của CTCK  |
  | TT232   | Bảo hiểm (BVH)       | Mã số riêng           |
"""
from __future__ import annotations

import json
import re

from common import fold, parse_number

# ───────────────────────── phân loại adapter ─────────────────────────

BANK_TICKERS = {
    "VCB", "MBB", "VPB", "CTG", "BID", "ACB", "HDB", "SHB", "SSB", "VIB", "MSB",
    "OCB", "NAB", "NVB", "STB", "EIB", "ABB", "BAB", "VAB", "KLB", "SGB", "EVF",
}
BROKER_TICKERS = {"SSI", "MBS", "FTS"}
INSURER_TICKERS = {"BVH"}


def adapter_for(ticker: str) -> str:
    if ticker in BANK_TICKERS:
        return "TCTD"
    if ticker in BROKER_TICKERS:
        return "TT334"
    if ticker in INSURER_TICKERS:
        return "TT232"
    return "TT200"


# ───────────────────────── chuẩn hoá nhãn ─────────────────────────

# "13. Chi phí khác" · "A. TÀI SẢN" · "I. Tiền mặt" · "(a) Phải thu" · "- Trong đó"
_NUMBERING = re.compile(r"^\s*[\(\[]?\s*(?:[0-9]{1,2}|[IVXLC]{1,5}|[A-Ea-e])\s*[\)\].:-]\s*")
_TRAILING_NOTE = re.compile(r"\s*\((?:THUYET MINH|TM)[^)]*\)\s*$")


def norm_label(label: str) -> str:
    s = fold(label)
    for _ in range(3):                     # "A. I. 1. Tiền" lồng nhiều tầng
        s2 = _NUMBERING.sub("", s)
        if s2 == s:
            break
        s = s2
    s = _TRAILING_NOTE.sub("", s)
    return s.strip(" .:-")


# ───────────────── concept: khoá kế toán chuẩn hoá ─────────────────
# Mã số là NGUỒN CHÂN LÝ cho TT200 — nó bền với mọi lỗi OCR dấu tiếng Việt.

CONCEPT_BY_MA_SO_CDKT = {
    "100": "TAI_SAN_NGAN_HAN", "110": "TIEN_VA_TUONG_DUONG_TIEN",
    "111": "TIEN", "112": "TUONG_DUONG_TIEN", "120": "DTTC_NGAN_HAN",
    "130": "PHAI_THU_NGAN_HAN", "131": "PHAI_THU_KHACH_HANG_NGAN_HAN",
    "132": "TRA_TRUOC_NGUOI_BAN_NGAN_HAN", "140": "HANG_TON_KHO",
    "150": "TAI_SAN_NGAN_HAN_KHAC", "200": "TAI_SAN_DAI_HAN",
    "210": "PHAI_THU_DAI_HAN", "220": "TAI_SAN_CO_DINH",
    "221": "TSCD_HUU_HINH", "227": "TSCD_VO_HINH", "230": "BAT_DONG_SAN_DAU_TU",
    "240": "TAI_SAN_DO_DANG_DAI_HAN", "242": "XAY_DUNG_CO_BAN_DO_DANG",
    "250": "DTTC_DAI_HAN", "252": "DAU_TU_VAO_CONG_TY_LIEN_KET",
    "260": "TAI_SAN_DAI_HAN_KHAC", "261": "CHI_PHI_TRA_TRUOC_DAI_HAN",
    "270": "TONG_TAI_SAN",
    "300": "NO_PHAI_TRA", "310": "NO_NGAN_HAN",
    "311": "PHAI_TRA_NGUOI_BAN_NGAN_HAN", "312": "NGUOI_MUA_TRA_TIEN_TRUOC",
    "313": "THUE_PHAI_NOP", "320": "VAY_NGAN_HAN", "322": "QUY_KHEN_THUONG_PHUC_LOI",
    "330": "NO_DAI_HAN", "338": "VAY_DAI_HAN",
    "400": "VON_CHU_SO_HUU", "410": "VON_CHU_SO_HUU",
    "411": "VON_GOP_CUA_CHU_SO_HUU", "421": "LNST_CHUA_PHAN_PHOI",
    "440": "TONG_NGUON_VON",
}
CONCEPT_BY_MA_SO_KQKD = {
    "01": "DOANH_THU_BAN_HANG", "02": "CAC_KHOAN_GIAM_TRU",
    "10": "DOANH_THU_THUAN", "11": "GIA_VON_HANG_BAN", "20": "LOI_NHUAN_GOP",
    "21": "DOANH_THU_TAI_CHINH", "22": "CHI_PHI_TAI_CHINH", "23": "CHI_PHI_LAI_VAY",
    "24": "LAI_LO_CONG_TY_LIEN_KET", "25": "CHI_PHI_BAN_HANG",
    "26": "CHI_PHI_QUAN_LY_DOANH_NGHIEP", "30": "LOI_NHUAN_THUAN_HDKD",
    "31": "THU_NHAP_KHAC", "32": "CHI_PHI_KHAC", "40": "LOI_NHUAN_KHAC",
    "50": "LOI_NHUAN_TRUOC_THUE", "51": "CHI_PHI_THUE_TNDN_HIEN_HANH",
    "52": "CHI_PHI_THUE_TNDN_HOAN_LAI", "60": "LOI_NHUAN_SAU_THUE",
    "61": "LNST_CO_DONG_CONG_TY_ME", "62": "LNST_CO_DONG_KHONG_KIEM_SOAT",
    "70": "LAI_CO_BAN_TREN_CO_PHIEU",
}
CONCEPT_BY_MA_SO_LCTT = {
    "20": "LC_TIEN_THUAN_HDKD", "30": "LC_TIEN_THUAN_HDDT",
    "40": "LC_TIEN_THUAN_HDTC", "50": "LC_TIEN_THUAN_TRONG_KY",
    "60": "TIEN_DAU_KY", "70": "TIEN_CUOI_KY",
}

# Ngân hàng không có Mã số → phải khớp nhãn. Nhãn đã fold nên OCR sai dấu vẫn khớp.
CONCEPT_BY_LABEL_TCTD: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^TONG (CONG )?TAI SAN"), "TONG_TAI_SAN"),
    (re.compile(r"^TONG (CONG )?NGUON VON"), "TONG_NGUON_VON"),
    (re.compile(r"^TONG NO PHAI TRA"), "NO_PHAI_TRA"),
    (re.compile(r"^(TONG )?VON CHU SO HUU"), "VON_CHU_SO_HUU"),
    (re.compile(r"^CHO VAY KHACH HANG$"), "CHO_VAY_KHACH_HANG"),
    (re.compile(r"^TIEN GUI CUA KHACH HANG"), "TIEN_GUI_KHACH_HANG"),
    (re.compile(r"^TIEN GUI (VA CHO VAY )?(TAI )?CAC TO CHUC TIN DUNG"),
     "TIEN_GUI_TAI_TCTD_KHAC"),
    (re.compile(r"^TIEN MAT"), "TIEN_MAT"),
    (re.compile(r"^TIEN GUI TAI NGAN HANG NHA NUOC"), "TIEN_GUI_TAI_NHNN"),
    (re.compile(r"^CHUNG KHOAN (KINH DOANH|DAU TU)"), "CHUNG_KHOAN"),
    (re.compile(r"^THU NHAP LAI (VA CAC KHOAN )?THUAN|^THU NHAP LAI THUAN"),
     "THU_NHAP_LAI_THUAN"),
    (re.compile(r"^LAI( |/)?(LO)? THUAN TU HOAT DONG DICH VU"), "LAI_THUAN_DICH_VU"),
    (re.compile(r"^(TONG )?CHI PHI HOAT DONG"), "CHI_PHI_HOAT_DONG"),
    (re.compile(r"^CHI PHI DU PHONG RUI RO TIN DUNG"), "CHI_PHI_DU_PHONG_TIN_DUNG"),
    (re.compile(r"^(TONG )?LOI NHUAN TRUOC THUE"), "LOI_NHUAN_TRUOC_THUE"),
    (re.compile(r"^LOI NHUAN SAU THUE"), "LOI_NHUAN_SAU_THUE"),
    (re.compile(r"^VON DIEU LE"), "VON_DIEU_LE"),
]

# Doanh nghiệp thiếu Mã số (OCR mất cột) — cứu bằng nhãn
CONCEPT_BY_LABEL_TT200: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^TONG CONG TAI SAN"), "TONG_TAI_SAN"),
    (re.compile(r"^TONG CONG NGUON VON"), "TONG_NGUON_VON"),
    (re.compile(r"^TAI SAN NGAN HAN$"), "TAI_SAN_NGAN_HAN"),
    (re.compile(r"^TAI SAN DAI HAN$"), "TAI_SAN_DAI_HAN"),
    (re.compile(r"^NO PHAI TRA$"), "NO_PHAI_TRA"),
    (re.compile(r"^NO NGAN HAN$"), "NO_NGAN_HAN"),
    (re.compile(r"^VON CHU SO HUU$"), "VON_CHU_SO_HUU"),
    (re.compile(r"^HANG TON KHO$"), "HANG_TON_KHO"),
    (re.compile(r"^DOANH THU THUAN"), "DOANH_THU_THUAN"),
    (re.compile(r"^GIA VON HANG BAN"), "GIA_VON_HANG_BAN"),
    (re.compile(r"^LOI NHUAN GOP"), "LOI_NHUAN_GOP"),
    (re.compile(r"^CHI PHI LAI VAY"), "CHI_PHI_LAI_VAY"),
    (re.compile(r"^CHI PHI BAN HANG"), "CHI_PHI_BAN_HANG"),
    (re.compile(r"^CHI PHI QUAN LY DOANH NGHIEP"), "CHI_PHI_QUAN_LY_DOANH_NGHIEP"),
    (re.compile(r"^(TONG )?LOI NHUAN (KE TOAN )?TRUOC THUE"), "LOI_NHUAN_TRUOC_THUE"),
    (re.compile(r"^LOI NHUAN SAU THUE( THU NHAP DOANH NGHIEP)?$"), "LOI_NHUAN_SAU_THUE"),
    (re.compile(r"^LUU CHUYEN TIEN THUAN TU HOAT DONG KINH DOANH"), "LC_TIEN_THUAN_HDKD"),
    (re.compile(r"^LUU CHUYEN TIEN THUAN TU HOAT DONG DAU TU"), "LC_TIEN_THUAN_HDDT"),
    (re.compile(r"^LUU CHUYEN TIEN THUAN TU HOAT DONG TAI CHINH"), "LC_TIEN_THUAN_HDTC"),
]

_MA_SO_BY_SECTION = {
    "CDKT": CONCEPT_BY_MA_SO_CDKT,
    "KQKD": CONCEPT_BY_MA_SO_KQKD,
    "LCTT": CONCEPT_BY_MA_SO_LCTT,
}


def concept_of(ma_so: str | None, label_norm: str, section: str, adapter: str) -> str | None:
    if adapter in ("TT200", "TT334", "TT232") and ma_so:
        c = _MA_SO_BY_SECTION.get(section, {}).get(ma_so.lstrip("0") or ma_so) \
            or _MA_SO_BY_SECTION.get(section, {}).get(ma_so)
        if c:
            return c
    rules = CONCEPT_BY_LABEL_TCTD if adapter == "TCTD" else CONCEPT_BY_LABEL_TT200
    for rx, c in rules:
        if rx.match(label_norm):
            return c
    return None


# ───────────────────────── trích fact từ một bảng ─────────────────────────

_MA_SO_CELL = re.compile(r"^\d{2,3}[a-zA-Z]?$")


def _find_ma_so_col(grid, n_header: int, names: list[str]) -> int:
    """Cột Mã số: ưu tiên header ghi 'Mã số', nếu OCR hỏng thì dò nội dung."""
    for i, nm in enumerate(names):
        if "MA SO" in fold(nm.replace("_", " ")):
            return i
    body = grid[n_header:]
    best, best_hits = -1, 0
    for c in range(min(len(names), 4)):
        vals = [(r[c] if c < len(r) else "").strip() for r in body]
        hits = sum(1 for v in vals if _MA_SO_CELL.match(v))
        if hits > best_hits and hits >= max(4, 0.4 * sum(1 for v in vals if v)):
            best, best_hits = c, hits
    return best


def extract_table(td, ticker: str, year: int, stmt_type: str) -> list[dict]:
    """td: grids.TableData → danh sách fact."""
    from emit import flatten_header

    grid = td.grid
    if not grid or len(grid) <= td.n_header_rows:
        return []
    n_header = max(1, min(td.n_header_rows, len(grid) - 1))
    names = flatten_header(grid, n_header)
    n_cols = len(names)
    periods = json.loads(td.col_periods) if td.col_periods else []
    periods += [None] * (n_cols - len(periods))
    per_cols = [c for c in range(n_cols) if periods[c] and periods[c].get("year")]
    if not per_cols:
        return []

    adapter = adapter_for(ticker)
    ma_col = _find_ma_so_col(grid, n_header, names)

    out = []
    for ri, row in enumerate(grid[n_header:]):
        label, label_col = "", -1
        for ci in range(min(3, n_cols)):
            if ci == ma_col:
                continue
            cell = (row[ci] if ci < len(row) else "").strip()
            if len(cell) >= 3 and parse_number(cell) is None and not _MA_SO_CELL.match(cell):
                label, label_col = cell, ci
                break
        if label_col < 0:
            continue
        ln = norm_label(label)
        if not ln:
            continue
        ma_so = (row[ma_col].strip() if 0 <= ma_col < len(row) else "") or None
        if ma_so and not _MA_SO_CELL.match(ma_so):
            ma_so = None
        concept = concept_of(ma_so, ln, td.section, adapter)

        for ci in per_cols:
            v = parse_number((row[ci] if ci < len(row) else "").strip())
            if v is None:
                continue
            p = periods[ci]
            out.append({
                "ticker": ticker, "year": year, "stmt_type": stmt_type,
                "doc_id": td.doc_id, "table_idx": td.table_idx,
                "statement": td.section, "adapter": adapter,
                "ma_so": ma_so, "label": label[:200], "label_norm": ln[:200],
                "concept": concept,
                "period_year": int(p["year"]), "period_type": p.get("period_type"),
                "is_begin": bool(p.get("is_begin")),
                "unit_scale": float(td.unit_scale),
                "value_raw": float(v),
                "value": float(v) * float(td.unit_scale),
                "row_idx": ri, "col_idx": ci,
                "label_col": names[label_col] if label_col < len(names) else "col_0",
                "col_name": names[ci],
            })
    return out
