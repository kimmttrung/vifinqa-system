"""Định danh công ty trong câu hỏi → set ticker.

Ba nguồn tín hiệu, hợp nhất lại:
  1. Mã CK viết hoa trong nguyên văn ("(VJC)", "của HPG")     — 64,0% câu
  2. Tên chính thức trong code_stock.csv, khớp bằng IDF-coverage — 34,1% câu
  3. Bảng alias thủ công (Vinamilk, Hoà Phát, Sacombank…)      — 1,9% câu

Vì sao không dùng substring cho tên chính thức: bốn công ty họ Masan
(MSN/MCH/MML/MSR) và hai công ty họ Gelex (GEX/GEE) đều chứa nhau.
IDF-coverage trong cửa sổ token giải quyết được, substring thì không.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from common import fold, load_code_stock, tokens

# ───────────────────── alias thủ công ─────────────────────
# Khoá đã fold sẵn (bỏ dấu + UPPER). Khớp bằng substring trên câu hỏi đã fold.
# Chỉ đưa vào đây những tên KHÔNG suy ra được từ code_stock.csv:
# tên thương hiệu, tên cũ, tên viết tắt quốc tế.
ALIASES: dict[str, str] = {
    # ngân hàng — tên thương hiệu
    "VIETCOMBANK": "VCB", "VIETINBANK": "CTG", "BIDV": "BID",
    "MBBANK": "MBB", "MB BANK": "MBB", "NGAN HANG QUAN DOI": "MBB",
    "VPBANK": "VPB", "SEABANK": "SSB", "HDBANK": "HDB", "OCEANBANK": "OGC",
    "EXIMBANK": "EIB", "SACOMBANK": "STB", "NAM A BANK": "NAB", "NAMABANK": "NAB",
    "KIENLONGBANK": "KLB", "KIEN LONG BANK": "KLB", "SAIGONBANK": "SGB",
    "VIETABANK": "VAB", "VIET A BANK": "VAB", "BAC A BANK": "BAB", "BACABANK": "BAB",
    "ABBANK": "ABB", "AN BINH BANK": "ABB", "MARITIME BANK": "MSB",
    "NCB": "NVB", "NGAN HANG QUOC DAN": "NVB", "SHBANK": "SHB",
    "PHUONG DONG": "OCB", "NGAN HANG A CHAU": "ACB",
    "EVN FINANCE": "EVF", "EVNFINANCE": "EVF",
    # tập đoàn / tên thương hiệu
    "VINAMILK": "VNM", "SUA VIET NAM": "VNM",
    "HOA PHAT": "HPG", "VINGROUP": "VIC", "VINCOM RETAIL": "VRE",
    "NOVALAND": "NVL", "DIA OC NO VA": "NVL",
    "VIETJET": "VJC", "PETROLIMEX": "PLX", "XANG DAU VIET NAM": "PLX",
    "SABECO": "SAB", "BAO VIET": "BVH", "MASAN CONSUMER": "MCH",
    "MASAN MEATLIFE": "MML", "MASAN HIGH-TECH": "MSR", "MASAN HIGH TECH": "MSR",
    "THE GIOI DI DONG": "MWG", "DIEN MAY XANH": "MWG",
    "VIGLACERA": "VGC", "GELEX": "GEX", "DIEN LUC GELEX": "GEE",
    "VINATEX": "VGT", "DET MAY VIET NAM": "VGT",
    "VINAFOR": "VIF", "LAM NGHIEP VIET NAM": "VIF",
    "VINAFOOD": "VSF", "LUONG THUC MIEN NAM": "VSF",
    "CAO SU VIET NAM": "GVR", "BINH SON": "BSR", "LOC HOA DAU": "BSR",
    "PV GAS": "GAS", "KHI VIET NAM": "GAS", "PV POWER": "POW",
    "DIEN LUC DAU KHI": "POW", "PVTRANS": "PVT", "VAN TAI DAU KHI": "PVT",
    "DAM PHU MY": "DPM", "DAM CA MAU": "DCM",
    "CANG HANG KHONG": "ACV", "DEO CA": "HHV",
    "SONADEZI": "SNZ", "BECAMEX IJC": "IJC", "HA TANG KY THUAT": "IJC",
    "KINH BAC": "KBC", "HOA SEN": "HSG", "NAM KIM": "NKG", "MINH PHU": "MPC",
    "DABACO": "DBC", "SAO MAI": "ASM", "HA DO": "HDG", "SAM HOLDINGS": "SAM",
    "TASCO": "HUT", "PHAT DAT": "PDR", "NAM LONG": "NLG", "VAN PHU": "VPI",
    "HAI PHAT": "HPX", "OCEAN GROUP": "OGC", "TAP DOAN DAI DUONG": "OGC",
    "SONG DA": "SJG", "KHAI HOAN LAND": "KHG", "SUNSHINE HOMES": "SSH",
    "CEN LAND": "CRE", "BAT DONG SAN THE KY": "CRE",
    "DUONG QUANG NGAI": "QNS", "VICEM HA TIEN": "HT1", "XI MANG HA TIEN": "HT1",
    "TRUONG THANH": "TTF", "AN PHAT XANH": "AAA", "NHUA AN PHAT": "AAA",
    "VICONSHIP": "VSC", "CONTAINER VIET NAM": "VSC",
    "DIEN GIA LAI": "GEG", "PROTRADE": "PRT", "BINH DUONG": "PRT",
    "HOANG ANH GIA LAI": "HAG", "HAGL AGRICO": "HNG",
    "DUC LONG GIA LAI": "DLG", "XAY DUNG HOA BINH": "HBC", "HOA BINH CORP": "HBC",
    "DAT XANH SERVICES": "DXS", "DICH VU BAT DONG SAN DAT XANH": "DXS",
    "DAT XANH GROUP": "DXG", "TAP DOAN DAT XANH": "DXG", "BLUEMARQ": "DXG",
    "PHU NHUAN": "PNJ", "HOANG HUY": "HHS", "DA NHIM": "DNH",
    "NHIET DIEN HAI PHONG": "HND", "DIEN LUC TKV": "DTK",
    "VIEN THONG FPT": "FOX", "FPT TELECOM": "FOX",
    "CHUNG KHOAN FPT": "FTS", "CHUNG KHOAN MB": "MBS", "CHUNG KHOAN SSI": "SSI",
    "NONG NGHIEP BAF": "BAF", "TAP DOAN F.I.T": "FIT", "TAP DOAN FIT": "FIT",
    "TAP DOAN C.E.O": "CEO", "DIA OC SAI GON THUONG TIN": "SCR", "TTC LAND": "SCR",
}

# Từ nền trong tên doanh nghiệp — không mang thông tin định danh.
_FORM_WORDS = {
    "CTCP", "CT", "CP", "CONG", "TY", "CO", "PHAN", "TONG", "TNHH",
    "NGAN", "HANG", "TMCP", "VA", "CUA", "MTV",
}


@lru_cache(maxsize=1)
def _name_index():
    """→ (tickers, {ticker: [token…]}, {token: idf})"""
    code = load_code_stock()
    per = {t: [w for w in dict.fromkeys(tokens(n)) if w not in _FORM_WORDS]
           for t, n in code.items()}
    per = {t: (w or tokens(code[t])) for t, w in per.items()}   # phòng tên toàn từ nền
    n = len(per)
    df: dict[str, int] = {}
    for ws in per.values():
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + n / c) for w, c in df.items()}
    return tuple(per), per, idf


_TICKER_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,3}\b")
_MAX_GAP = 3          # khoảng cách token tối đa giữa 2 từ liên tiếp của tên
_MIN_COVERAGE = 0.75  # tỷ lệ IDF của tên phải khớp được


def explicit_tickers(question: str) -> list[str]:
    """Mã CK viết hoa trong nguyên văn; chấp nhận cả chuỗi viết thường nếu có ≥2 mã."""
    valid = set(load_code_stock())
    found = [t for t in _TICKER_RE.findall(question) if t in valid]
    if not found:
        # q431: 'hpx,kbc,nvl,vic,vpi,vre' — chỉ nhận khi thấy ≥2 mã liền nhau,
        # tránh nuốt từ tiếng Việt trùng mã (vd 'gas', 'ceo', 'fit').
        low = [w.upper() for w in re.findall(r"\b[a-z][a-z0-9]{2,3}\b", question)]
        hits = [w for w in low if w in valid]
        if len(set(hits)) >= 2:
            found = hits
    return list(dict.fromkeys(found))


def alias_tickers(folded_q: str) -> list[str]:
    out = []
    for phrase, tk in ALIASES.items():
        if phrase in folded_q:
            out.append(tk)
    return list(dict.fromkeys(out))


def _best_ordered_cov(pos: dict[str, list[int]], ws: list[str], idf: dict[str, float]):
    """Khớp tên như một *cụm từ có lỗ*: các từ phải xuất hiện ĐÚNG THỨ TỰ và
    cách nhau ≤ _MAX_GAP token. Bag-of-words trong cửa sổ không đủ — "Năm 2024,
    khi rà soát" từng khớp nhầm GAS ("Khí Việt Nam") vì "khi"/"năm" là hư từ.

    → (coverage theo IDF, set chỉ số từ đã khớp)
    """
    occ = sorted((p, wi) for wi, w in enumerate(ws) for p in pos.get(w, []))
    if not occ:
        return 0.0, set()
    total = sum(idf.get(w, 0.0) for w in ws) or 1.0
    best_cov, best_set = 0.0, set()
    for si, (p0, wi0) in enumerate(occ):
        matched = {wi0}
        qi, wi = p0, wi0
        for qj, wj in occ[si + 1:]:
            if qj <= qi:
                continue
            if qj - qi > _MAX_GAP:
                break
            if wj > wi:
                qi, wi = qj, wj
                matched.add(wj)
        cov = sum(idf.get(ws[i], 0.0) for i in matched) / total
        if cov > best_cov:
            best_cov, best_set = cov, matched
    return best_cov, best_set


def name_tickers(question: str, min_coverage: float = _MIN_COVERAGE) -> dict[str, float]:
    """Khớp tên chính thức trong code_stock.csv → {ticker: coverage}."""
    _, per, idf = _name_index()
    qt = tokens(question)
    if not qt:
        return {}
    pos: dict[str, list[int]] = {}
    for i, w in enumerate(qt):
        pos.setdefault(w, []).append(i)

    out: dict[str, float] = {}
    for tk, ws in per.items():
        cov, matched = _best_ordered_cov(pos, ws, idf)
        if cov < min_coverage:
            continue
        # từ hiếm nhất của tên là từ định danh — thiếu nó thì đây là tên khác.
        # ("Bất động sản Đất Xanh" không được khớp thành "Phát triển BĐS Phát Đạt")
        rarest = max(range(len(ws)), key=lambda i: idf.get(ws[i], 0.0))
        if rarest not in matched:
            continue
        out[tk] = round(cov, 3)
    return out


def resolve_companies(question: str) -> tuple[list[str], dict]:
    """→ (danh sách ticker theo thứ tự xuất hiện, debug dict)."""
    fq = fold(question)
    ex = explicit_tickers(question)
    al = alias_tickers(fq)
    nm = name_tickers(question)

    # "CTCP Chứng khoán FPT" khớp FTS (đúng) nhưng đồng thời kích hoạt FPT hai
    # đường: _TICKER_RE thấy token FPT viết hoa, và tên chính thức của FPT sau khi
    # bỏ từ nền cũng chỉ còn đúng {FPT}. Một mã mà TOÀN BỘ từ định danh của nó
    # nằm lọt trong tên một công ty khác đã khớp chắc thì nó không phải chủ thể
    # riêng — đó là phần tên của công ty kia. (FTS⊃FPT, FOX⊃FPT)
    per = _name_index()[1]
    strong = {u: set(per[u]) for u, cov in nm.items() if cov >= 0.9}
    cands = set(ex) | set(nm) | set(al)
    swallowed = {
        tk for tk in cands
        if any(u != tk and set(per[tk]) < uws for u, uws in strong.items())
    }
    ex = [t for t in ex if t not in swallowed]
    nm = {t: c for t, c in nm.items() if t not in swallowed}
    al = [t for t in al if t not in swallowed]
    merged = list(dict.fromkeys([*ex, *nm.keys(), *al]))
    # sắp theo vị trí xuất hiện đầu tiên trong câu để giữ thứ tự tự nhiên
    def first_pos(tk: str) -> int:
        p = fq.find(tk)
        if p >= 0:
            return p
        best = len(fq)
        for w in _name_index()[1][tk]:
            q = fq.find(w)
            if q >= 0:
                best = min(best, q)
        for phrase, t in ALIASES.items():
            if t == tk:
                q = fq.find(phrase)
                if q >= 0:
                    best = min(best, q)
        return best

    merged.sort(key=first_pos)
    return merged, {"explicit": ex, "by_name": nm, "by_alias": al}
