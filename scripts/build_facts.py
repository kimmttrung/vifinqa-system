"""Stage B — quét tables.parquet → artifacts/facts.parquet + QC report.

    python scripts/build_facts.py                # chỉ 4 báo cáo chính (mặc định)
    python scripts/build_facts.py --notes        # kèm cả bảng thuyết minh

Auto-QC ở cuối chính là bộ đo chất lượng thay cho nhãn gold mà cuộc thi không phát.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import ARTIFACTS
from facts import extract_table
from grids import TableData

STATEMENTS = ("CDKT", "KQKD", "LCTT", "VCSH")
_COLS = ["doc_id", "ticker", "year", "stmt_type", "table_idx", "grid",
         "n_header_rows", "unit_scale", "section", "col_periods", "caption",
         "n_numeric_cells"]


def build(include_notes: bool) -> pd.DataFrame:
    keep = set(STATEMENTS) | ({"TM", "OTHER"} if include_notes else set())
    pf = pq.ParquetFile(ARTIFACTS / "tables.parquet")
    rows, t0, n_tables = [], time.time(), 0
    for batch in pf.iter_batches(batch_size=6000, columns=_COLS):
        cols = {n: batch.column(n).to_pylist() for n in _COLS}
        for i in range(batch.num_rows):
            if cols["section"][i] not in keep or (cols["n_numeric_cells"][i] or 0) < 4:
                continue
            td = TableData(
                doc_id=cols["doc_id"][i], table_idx=int(cols["table_idx"][i]),
                grid=json.loads(cols["grid"][i]) if cols["grid"][i] else [],
                n_header_rows=int(cols["n_header_rows"][i] or 1),
                unit_scale=float(cols["unit_scale"][i] or 1.0),
                section=cols["section"][i], col_periods=cols["col_periods"][i] or "[]",
                caption=cols["caption"][i] or "")
            n_tables += 1
            rows.extend(extract_table(td, cols["ticker"][i],
                                      int(cols["year"][i] or 0), cols["stmt_type"][i]))
        print(f"  {n_tables:6d} bảng · {len(rows):8d} fact · {time.time()-t0:5.1f}s",
              flush=True)
    return pd.DataFrame(rows)


# ───────────────────────── auto-QC ─────────────────────────

IDENTITIES = [
    ("270 == 440", "TONG_TAI_SAN", ["TONG_NGUON_VON"], None),
    ("100+200 == 270", "TONG_TAI_SAN", ["TAI_SAN_NGAN_HAN", "TAI_SAN_DAI_HAN"], "sum"),
    ("300+400 == 440", "TONG_NGUON_VON", ["NO_PHAI_TRA", "VON_CHU_SO_HUU"], "sum"),
    ("10-11 == 20", "LOI_NHUAN_GOP", ["DOANH_THU_THUAN", "GIA_VON_HANG_BAN"], "diff"),
]


def qc_identities(f: pd.DataFrame, rtol: float = 0.005) -> list[dict]:
    """Ràng buộc kế toán — lệch nghĩa là parser hoặc OCR hỏng ở đúng chỗ đó."""
    # Group theo DOC chứ không theo table_idx: bảng CĐKT gần như luôn bị tách đôi
    # ("BẢNG CÂN ĐỐI KẾ TOÁN" + "(tiếp theo)"), phần Tài sản và phần Nguồn vốn
    # nằm ở hai bảng khác nhau nên 270 và 440 không bao giờ cùng một table_idx.
    # Chỉ kiểm trên 4 báo cáo chính. Bảng thuyết minh cũng có dòng "Doanh thu
    # thuần"/"Giá vốn" (chi tiết theo bộ phận) được gán cùng concept — gộp vào
    # thì median bị kéo lệch và tưởng nhầm là parser sai.
    f = f[f.statement.isin(STATEMENTS)]
    piv = (f[f.concept.notna()]
           .groupby(["doc_id", "period_year", "is_begin", "concept"])
           ["value"].median().unstack("concept"))
    out = []
    for name, target, parts, mode in IDENTITIES:
        if target not in piv.columns or any(p not in piv.columns for p in parts):
            out.append({"rule": name, "n": 0, "pass": 0, "rate": None})
            continue
        sub = piv[[target, *parts]].dropna()
        if sub.empty:
            out.append({"rule": name, "n": 0, "pass": 0, "rate": None})
            continue
        if mode == "sum":
            lhs = sub[parts].sum(axis=1)
        elif mode == "diff":
            lhs = sub[parts[0]] - sub[parts[1]].abs()
        else:
            lhs = sub[parts[0]]
        rhs = sub[target]
        ok = ((lhs - rhs).abs() <= rtol * rhs.abs().clip(lower=1.0))
        out.append({"rule": name, "n": int(len(sub)), "pass": int(ok.sum()),
                    "rate": round(float(ok.mean()), 4)})
    return out


def qc_cross_year(f: pd.DataFrame, rtol: float = 0.01) -> dict:
    """Cột 01/01/Y của báo cáo năm Y phải bằng cột 31/12 của báo cáo năm Y−1.

    Đây là phép kiểm mạnh nhất với lỗi OCR số: hai con số đến từ HAI file khác
    nhau, OCR sai một bên là lộ ngay.
    """
    d = f[f.concept.notna() & (f.statement == "CDKT")]
    end = (d[~d.is_begin].groupby(["ticker", "stmt_type", "period_year", "concept"])
           ["value"].median().rename("end"))
    beg = (d[d.is_begin].groupby(["ticker", "stmt_type", "period_year", "concept"])
           ["value"].median().rename("beg"))
    beg = beg.reset_index()
    beg["period_year"] = beg["period_year"] - 1          # đầu năm Y ≡ cuối năm Y−1
    m = beg.set_index(["ticker", "stmt_type", "period_year", "concept"]).join(end, how="inner")
    if m.empty:
        return {"n": 0, "pass": 0, "rate": None}
    ok = (m["beg"] - m["end"]).abs() <= rtol * m["end"].abs().clip(lower=1.0)
    worst = (m[~ok].assign(dev=(m["beg"] - m["end"]).abs())
             .nlargest(20, "dev").reset_index().to_dict("records"))
    return {"n": int(len(m)), "pass": int(ok.sum()), "rate": round(float(ok.mean()), 4),
            "worst": worst}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", action="store_true", help="kèm bảng thuyết minh (TM)")
    args = ap.parse_args()

    t0 = time.time()
    f = build(args.notes)
    if f.empty:
        print("Không trích được fact nào.")
        return 1

    out = ARTIFACTS / "facts.parquet"
    f.to_parquet(out, compression="zstd", index=False)
    print(f"\nfacts.parquet: {len(f):,} dòng · {out.stat().st_size/1e6:.1f} MB "
          f"· {time.time()-t0:.0f}s")

    cov = f.concept.notna().mean()
    print(f"concept coverage: {cov:.1%} ({f.concept.notna().sum():,} fact có concept)")
    print("\ntheo adapter:\n", f.adapter.value_counts().to_string())
    print("\ntheo statement:\n", f.statement.value_counts().to_string())

    ids = qc_identities(f)
    print("\n=== ràng buộc kế toán ===")
    for r in ids:
        rate = "n/a" if r["rate"] is None else f"{r['rate']:.1%}"
        print(f"  {r['rule']:18s} {r['pass']:6d}/{r['n']:6d}  {rate}")

    cy = qc_cross_year(f)
    rate = "n/a" if cy["rate"] is None else f"{cy['rate']:.1%}"
    print(f"\n=== cross-year (đầu năm Y == cuối năm Y−1) ===\n  "
          f"{cy['pass']}/{cy['n']}  {rate}")

    # công ty × năm × concept có mặt — đo độ phủ thật của fast-path
    key = f[f.concept.notna()].groupby(["ticker", "period_year"]).concept.nunique()
    print(f"\nđộ phủ: {len(key)} cặp (ticker, năm) có fact; "
          f"trung vị {int(key.median())} concept/cặp")

    lines = [
        "# Stage B — QC report", "",
        f"- fact: **{len(f):,}** · concept coverage **{cov:.1%}**",
        f"- adapter: {f.adapter.value_counts().to_dict()}",
        f"- statement: {f.statement.value_counts().to_dict()}", "",
        "## Ràng buộc kế toán",
        pd.DataFrame(ids).to_markdown(index=False), "",
        "## Cross-year",
        f"- {cy['pass']}/{cy['n']} = {rate}", "",
        "## 20 lệch cross-year lớn nhất (ứng viên OCR sai số)",
        pd.DataFrame(cy.get("worst", [])).to_markdown(index=False)
        if cy.get("worst") else "_không có_",
    ]
    (ARTIFACTS / "qc_facts.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nĐã ghi {ARTIFACTS / 'qc_facts.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
