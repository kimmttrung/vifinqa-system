"""Xem log hybrid của một câu hỏi.

    python scripts/show_trace.py --trace out/v4/trace.jsonl --qid 64
    python scripts/show_trace.py --trace out/v4/trace.jsonl --stats

Cột trong bảng ứng viên:
    bm25    điểm BM25 đã chuẩn hoá về [0,1] trong tập ứng viên của câu này
    cover   tỷ lệ IDF của token câu hỏi có mặt trong bảng
    phrase  cụm ≥3 token liên tiếp khớp nguyên vẹn (bắt tên riêng thuyết minh)
    per     bảng có cột đúng năm câu hỏi hỏi (0/1)
    sec     section khớp loại chỉ tiêu suy từ câu hỏi (0/1)
    LEX     điểm lexical tổng = bm25 + 2.5·per + 3.0·phrase + 2.0·cover + 1.0·sec + 0.5·data
    ce      hạng/điểm cross-encoder reranker (trống nếu chưa chạy notebook Kaggle)
    RRF     điểm hợp nhất 1/(60+rank_lex+1) + 1/(60+rank_ce+1)
    ✓       bảng này ĐƯỢC nộp vào relevant_tables
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def show(row: dict, top: int) -> None:
    ir = row["ir"]
    print("=" * 100)
    print(f"[{row['qid']}] {row['question']}")
    print("-" * 100)
    print(f"IR      : ticker={ir['tickers']} năm={ir['years']} loại={ir['stmt_type']} "
          f"đơn vị={ir['unit_kind']}/{ir['unit_scale']:g} mốc={ir['time_point']} "
          f"T{ir['tier']} ops={ir['ops']}")
    print(f"truy vấn: {ir['retrieval_query']}")
    print(f"doc lọc : {row['doc_filter']['n']} doc → {row['doc_filter']['docs'][:4]}")
    print(f"k dùng  : {row['k_used']}")

    print(f"\n{'':2} {'doc|line':46} {'sec':5} {'bm25':>6} {'cover':>6} {'phrase':>6} "
          f"{'per':>4} {'sec':>4} {'LEX':>7} {'ce':>5} {'RRF':>9}")
    for c in row["candidates"][:top]:
        mark = "✓" if c["submitted"] else " "
        ref = f"{c['doc_id']}|{c['line_no']}"
        ce = "" if c.get("ce_rank") is None else f"#{c['ce_rank']}"
        rrf = "" if c.get("rrf_score") is None else f"{c['rrf_score']:.6f}"
        print(f"{mark:2} {ref[:46]:46} {c['section']:5} {c['bm25']:6.3f} {c['cover']:6.3f} "
              f"{c['phrase']:6.3f} {c['period']:4.0f} {c['sec_hit']:4.0f} "
              f"{c['lex_score']:7.3f} {ce:>5} {rrf:>9}")

    if row["fact_hits"]:
        print("\nfact-table (tra thẳng theo nhãn, không qua xếp hạng bảng):")
        for g in row["fact_hits"][:5]:
            print(f"   {g['label_score']:.3f}  {g['label'][:58]:58} | {g['col_name'][:26]:26} "
                  f"| {g['statement']:5} | {g['value_raw']:>18,.0f} ×{g['unit_scale']:g}")

    if row["solver"]:
        s = row["solver"]
        print(f"\nsolver  : {s['kind']} → {s['value']!r}")
        for i, g in enumerate(s["facts"], 1):
            print(f"   df{i}: {g['label'][:58]:58} | {g['col_name'][:26]}")

    sb = row["submitted"]
    if sb:
        print(f"\nĐÁP ÁN  : {sb['answer']!r}")
        print(f"query   : {sb['pandas_query'][:180]}")
        print(f"nộp     : {len(sb['relevant_tables'])} bảng · {len(sb['relevant_docs'])} doc")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--qid", type=int, action="append")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.trace).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.stats:
        print(f"câu: {len(rows)}")
        print("k dùng    :", dict(sorted(Counter(r["k_used"] for r in rows).items())))
        print("nguồn giải:", dict(Counter(
            (r["solver"]["kind"] if r["solver"] else
             ("fact" if r["fact_hits"] else "bm25")) for r in rows)))
        print("fusion    :", dict(Counter(
            (r["candidates"][0]["fusion"] if r["candidates"] else "none") for r in rows)))
        print("doc lọc/câu:", round(sum(r["doc_filter"]["n"] for r in rows) / len(rows), 2))
        return 0

    want = set(args.qid or [])
    for r in rows:
        if not want or r["qid"] in want:
            show(r, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
