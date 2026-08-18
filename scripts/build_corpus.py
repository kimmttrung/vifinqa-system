"""Stage A — quét corpus gốc → artifacts/{docs,tables}.parquet + qc_report.md

    python scripts/build_corpus.py                      # ~5 phút, CPU thuần
    VIFINQA_DATA=/path/to/ViFinQA python scripts/build_corpus.py

Nghiệm thu bằng các con số đã ĐO ĐỘC LẬP trước khi viết code (xem CLAUDE.md §3).
Script trả exit code 1 nếu bất kỳ check nào trượt — dùng được trong CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import ARTIFACTS, DATA_ROOT
from corpus import process_document

# Con số kỳ vọng — đo trực tiếp trên dataset, không phải giả định.
# 957/954/7/55 là số chính thức trong dataset card của HuggingFace.
EXPECTED = {
    "docs": 1973, "tables": 146246, "tickers": 100,
    "consolidated": 957, "separate": 954, "aggregated": 7, "other": 55,
    "max_tables_per_doc": 248,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="mặc định: artifacts/")
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N doc (để thử)")
    ap.add_argument("--no-check", action="store_true", help="bỏ qua nghiệm thu")
    args = ap.parse_args()

    if DATA_ROOT is None:
        print("Không thấy corpus. Set env VIFINQA_DATA trỏ tới thư mục chứa "
              "financial_statements/")
        return 1
    out_dir = Path(args.out) if args.out else ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)

    all_txt = sorted((DATA_ROOT / "financial_statements").glob("*/*/*/*_extracted.txt"))
    if args.limit:
        all_txt = all_txt[:args.limit]
    print(f"corpus : {DATA_ROOT}")
    print(f"file   : {len(all_txt)}")
    if not all_txt:
        print("Không thấy file _extracted.txt nào.")
        return 1

    tables_pq, docs_pq = out_dir / "tables.parquet", out_dir / "docs.parquet"
    t0 = time.time()
    doc_rows, buf, writer, n_tables, failures = [], [], None, 0, []

    try:
        for i, p in enumerate(all_txt, 1):
            try:
                meta, tabs = process_document(p, DATA_ROOT)
            except Exception as e:            # noqa: BLE001 — 1 file hỏng không được chặn cả run
                failures.append({"path": str(p), "error": repr(e)[:300]})
                continue
            doc_rows.append(meta)
            buf.extend(tabs)
            n_tables += len(tabs)
            # ghi theo lô: grid + raw_html của 146k bảng xấp xỉ kích thước corpus gốc
            if len(buf) >= 4000 or i == len(all_txt):
                tbl = pa.Table.from_pylist(buf)
                if writer is None:
                    writer = pq.ParquetWriter(tables_pq, tbl.schema, compression="zstd")
                writer.write_table(tbl)
                buf = []
            if i % 200 == 0 or i == len(all_txt):
                print(f"  {i:5d}/{len(all_txt)} doc · {n_tables:7d} bảng · "
                      f"{time.time()-t0:5.1f}s", flush=True)
    finally:
        if buf:
            tbl = pa.Table.from_pylist(buf)
            if writer is None:
                writer = pq.ParquetWriter(tables_pq, tbl.schema, compression="zstd")
            writer.write_table(tbl)
        if writer is not None:
            writer.close()

    docs = pd.DataFrame(doc_rows)
    docs.to_parquet(docs_pq, compression="zstd", index=False)
    tables = pd.read_parquet(tables_pq, columns=[
        "doc_id", "section", "n_cols", "has_ma_so", "unit_scale", "unit_source",
        "n_numeric_cells", "col_periods"])

    print(f"\nxong sau {time.time()-t0:.1f}s")
    print(f"docs.parquet  : {len(docs):,} dòng · {docs_pq.stat().st_size/1e6:.2f} MB")
    print(f"tables.parquet: {len(tables):,} dòng · {tables_pq.stat().st_size/1e6:.1f} MB")

    fin = tables[tables.section.isin(["CDKT", "KQKD", "LCTT"])]
    has_period = fin.col_periods.map(lambda s: any(p is not None for p in json.loads(s)))
    cov_data = has_period[fin.n_numeric_cells >= 6].mean() if len(fin) else 0.0

    checks = {
        f"docs == {EXPECTED['docs']}": len(docs) == EXPECTED["docs"],
        f"tables == {EXPECTED['tables']}": len(tables) == EXPECTED["tables"],
        f"tickers == {EXPECTED['tickers']}": docs.ticker.nunique() == EXPECTED["tickers"],
        "consolidated == 957": (docs.stmt_type == "consolidated").sum() == 957,
        "separate == 954": (docs.stmt_type == "separate").sum() == 954,
        "aggregated == 7": (docs.stmt_type == "aggregated").sum() == 7,
        "other == 55": (docs.stmt_type == "other").sum() == 55,
        "max bảng/doc == 248": docs.n_tables.max() == EXPECTED["max_tables_per_doc"],
        "lưới hợp lệ >= 99%": (tables.n_cols > 0).mean() >= 0.99,
        "không file lỗi": len(failures) == 0,
        "cột kỳ trên bảng dữ liệu >= 85%": cov_data >= 0.85,
    }
    if args.limit or args.no_check:
        checks = {k: v for k, v in checks.items() if k.startswith("lưới") or "lỗi" in k}

    print("\n=== nghiệm thu ===")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    report = [
        "# Stage A — QC report", "",
        f"- docs **{len(docs):,}** · tables **{len(tables):,}** · file lỗi **{len(failures)}**",
        f"- lưới hợp lệ {(tables.n_cols>0).mean():.2%} · has_ma_so {tables.has_ma_so.mean():.1%}",
        f"- cột kỳ (bảng CDKT/KQKD/LCTT có ≥6 ô số): {cov_data:.1%}", "",
        "## Nghiệm thu",
        *[f"- {'PASS' if v else 'FAIL'} — {k}" for k, v in checks.items()], "",
        "## Section", tables.section.value_counts().to_string(), "",
        "## Đơn vị", tables.groupby(["unit_scale", "unit_source"]).size().to_string(),
    ]
    (out_dir / "qc_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nĐã ghi {out_dir/'qc_report.md'}")

    if failures:
        print(f"\n{len(failures)} FILE LỖI:")
        for f in failures[:5]:
            print("   ", f)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
