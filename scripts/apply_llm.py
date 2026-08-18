"""Merge `llm_patch.jsonl` (sinh trên Kaggle) vào một submission ở local.

    python scripts/apply_llm.py --sub out/run --patch out/llm_patch.jsonl

Mỗi câu vá vào đều được **chạy lại trong sandbox ở local** trước khi chấp nhận.
Kaggle chạy đúng không có nghĩa là ở đây chạy đúng: khác phiên bản pandas, khác
đường dẫn CSV. Câu nào chạy lỗi thì giữ nguyên đáp án tất định — thà kém chính
xác còn hơn mất điểm Execution Accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execute import verify
from submit import Record, Submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="thư mục submission (có submission.json)")
    ap.add_argument("--patch", required=True, help="llm_patch.jsonl")
    ap.add_argument("--out", default=None, help="thư mục đích (mặc định: ghi đè --sub)")
    args = ap.parse_args()

    sub_dir = Path(args.sub)
    recs = json.loads((sub_dir / "submission.json").read_text(encoding="utf-8"))
    patch = {}
    for line in Path(args.patch).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            patch[int(d["id"])] = d

    applied = rejected = 0
    conflicts: list[tuple[int, float, float]] = []
    for r in recs:
        p = patch.get(r["id"])
        if not p:
            continue
        paths = {e["variable"]: e["csv_path"] for e in r["evidence"]}
        res = verify(p["pandas_query"], paths, sub_dir)
        if not res.ok:
            rejected += 1
            continue
        r["answer"] = float(res.value)
        r["pandas_query"] = p["pandas_query"]
        applied += 1
        # Cờ đối chiếu: nhóm công thức đóng có đáp án tất định làm chuẩn. Rule
        # không hallucinate, nên LLM lệch xa ở đó là dấu hiệu LLM sai, không
        # phải rule sai — in ra để soi, KHÔNG tự động ghi đè (LLM cũng có thể
        # đang sửa một chỗ rule chọn nhầm ô).
        if p.get("rule_answer") is not None:
            ra = float(p["rule_answer"])
            if ra != 0 and abs(res.value - ra) > 0.02 * abs(ra):
                conflicts.append((r["id"], ra, res.value))

    out_dir = Path(args.out) if args.out else sub_dir
    if out_dir != sub_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        if (out_dir / "data").exists():
            shutil.rmtree(out_dir / "data")
        shutil.copytree(sub_dir / "data", out_dir / "data")

    s = Submission(out_dir)
    for r in recs:
        s.add(Record(id=r["id"], question=r["question"], answer=r["answer"],
                     relevant_docs=r["relevant_docs"], relevant_tables=r["relevant_tables"],
                     evidence=r["evidence"], pandas_query=r["pandas_query"]))
    errs = s.validate()
    z = s.zip()
    print(f"vá {applied} câu · từ chối {rejected} (chạy lỗi ở local) · "
          f"lỗi format {len(errs)} · lệch rule {len(conflicts)}")
    for qid, ra, la in conflicts[:15]:
        print(f"   q{qid}: rule={ra!r} vs llm={la!r}")
    for e in errs[:10]:
        print("   ", e)
    print("ZIP:", z)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
