"""Kiểm chứng độc lập một bài nộp — chạy được bởi BẤT KỲ AI, không cần pipeline.

    python scripts/verify_submission.py out/v4/submission.zip
    python scripts/verify_submission.py out/v4 --report out/v4/VERIFY.md

BTC nói sẽ chạy lại pipeline để đối chiếu, phòng trường hợp thí sinh bịa số.
Script này làm đúng việc đó, và làm được vì kiến trúc đã thiết kế cho nó:

    LLM/rule sinh CODE  →  executor sinh SỐ

Không con số nào trong `answer` được ai "viết ra". Mỗi số đều rút từ CSV đi kèm
bằng đúng `pandas_query` đi kèm. Muốn kiểm chỉ cần chạy lại — đó là toàn bộ ý
nghĩa của việc bắt buộc `evidence` + `pandas_query` phải tự chứa.

Bốn tầng kiểm, độc lập nhau:
  1. Hình thức     — đủ trường, đúng kiểu, id không trùng, CSV tồn tại
  2. Thực thi      — `pandas_query` chạy được trên CSV đi kèm
  3. Nhất quán     — kết quả query KHỚP `answer` (đây là tầng chống bịa)
  4. Truy nguyên   — mỗi `doc|line_no` trỏ đúng một thẻ <table> có thật trong corpus
                     (chỉ chạy khi có artifacts/tables.parquet)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execute import run_query
from common import ARTIFACTS

RTOL = 1e-6


def _load_dir(target: Path) -> tuple[Path, Path | None]:
    """→ (thư mục có submission.json, thư mục tạm cần xoá)"""
    if target.is_dir():
        return target, None
    tmp = Path(tempfile.mkdtemp(prefix="vifinqa_verify_"))
    with zipfile.ZipFile(target) as z:
        z.extractall(tmp)
    return tmp, tmp


def check_traceability(records: list[dict]) -> dict:
    """Mỗi `doc_id|line_no` có thật là dòng bắt đầu một thẻ <table> không?

    Đây là tầng khó giả nhất: `line_no` phải khớp với vị trí thật trong file
    _extracted.txt của corpus gốc. Bịa ra một số là lộ ngay.
    """
    import pandas as pd
    p = ARTIFACTS / "tables.parquet"
    if not p.exists():
        return {"skipped": "không có artifacts/tables.parquet"}
    t = pd.read_parquet(p, columns=["doc_id", "line_no"])
    real = set(zip(t.doc_id, t.line_no.astype(int)))
    docs = set(t.doc_id)

    n_ref = n_bad_fmt = n_bad_doc = n_bad_line = 0
    bad_examples = []
    for r in records:
        for ref in r["relevant_tables"]:
            n_ref += 1
            if "|" not in ref:
                n_bad_fmt += 1
                continue
            d, _, pos = ref.rpartition("|")
            if d not in docs:
                n_bad_doc += 1
                if len(bad_examples) < 5:
                    bad_examples.append((r["id"], ref, "doc không tồn tại"))
                continue
            try:
                if (d, int(pos)) not in real:
                    n_bad_line += 1
                    if len(bad_examples) < 5:
                        bad_examples.append((r["id"], ref, "line_no không phải đầu <table>"))
            except ValueError:
                n_bad_fmt += 1
    return {"n_ref": n_ref, "bad_format": n_bad_fmt, "bad_doc": n_bad_doc,
            "bad_line": n_bad_line, "examples": bad_examples,
            "ok_rate": (n_ref - n_bad_fmt - n_bad_doc - n_bad_line) / max(n_ref, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="submission.zip hoặc thư mục chứa submission.json")
    ap.add_argument("--report", default=None)
    ap.add_argument("--rtol", type=float, default=RTOL)
    args = ap.parse_args()

    base, tmp = _load_dir(Path(args.target))
    try:
        sj = base / "submission.json"
        if not sj.exists():
            print(f"KHÔNG thấy {sj}")
            return 1
        records = json.loads(sj.read_text(encoding="utf-8"))

        # ── tầng 1: hình thức ──
        form: list[str] = []
        ids = [r.get("id") for r in records]
        if len(set(ids)) != len(ids):
            form.append(f"id trùng: {[k for k, v in Counter(ids).items() if v > 1][:5]}")
        need = {"id", "question", "answer", "relevant_docs", "relevant_tables",
                "evidence", "pandas_query"}
        for r in records:
            miss = need - set(r)
            if miss:
                form.append(f"q{r.get('id')}: thiếu trường {sorted(miss)}")
            elif not isinstance(r["answer"], (int, float)) or \
                    not math.isfinite(float(r["answer"])):
                form.append(f"q{r['id']}: answer không phải số hữu hạn: {r['answer']!r}")

        # ── tầng 2+3: thực thi & nhất quán ──
        n_exec = n_match = n_no_ev = 0
        fails: list[tuple[int, str]] = []
        mismatches: list[tuple[int, float, float]] = []
        for r in records:
            paths = {e["variable"]: e["csv_path"] for e in r.get("evidence", [])}
            if not paths:
                n_no_ev += 1
                fails.append((r["id"], "không có evidence"))
                continue
            try:
                import pandas as pd
                frames = {v: pd.read_csv(base / p) for v, p in paths.items()}
            except Exception as e:                                  # noqa: BLE001
                fails.append((r["id"], f"đọc CSV lỗi: {type(e).__name__}: {e}"[:160]))
                continue
            res = run_query(r["pandas_query"], frames, timeout=10.0)
            if not res.ok:
                fails.append((r["id"], str(res.error)[:160]))
                continue
            n_exec += 1
            if math.isclose(res.value, float(r["answer"]), rel_tol=args.rtol, abs_tol=1e-9):
                n_match += 1
            else:
                mismatches.append((r["id"], float(r["answer"]), res.value))

        # ── tầng 4: truy nguyên ──
        trace = check_traceability(records)

        n = len(records)
        lines = [
            "# Kiểm chứng bài nộp", "",
            f"Nguồn: `{args.target}`  ·  **{n}** câu", "",
            "| Tầng kiểm | Kết quả |",
            "|---|---|",
            f"| 1. Hình thức hợp lệ | {'PASS' if not form else f'{len(form)} LỖI'} |",
            f"| 2. `pandas_query` chạy được | **{n_exec}/{n}** = {n_exec/n:.2%} |",
            f"| 3. Kết quả khớp `answer` | **{n_match}/{n}** = {n_match/n:.2%} |",
        ]
        if "skipped" in trace:
            lines.append(f"| 4. Truy nguyên `line_no` | bỏ qua ({trace['skipped']}) |")
        else:
            lines.append(f"| 4. `doc\\|line_no` trỏ đúng thẻ `<table>` thật | "
                         f"**{trace['ok_rate']:.2%}** trên {trace['n_ref']} tham chiếu |")
        lines += ["",
                  "Tầng 3 là tầng chống bịa: mọi `answer` đều phải tái lập được bằng cách",
                  "chạy `pandas_query` đi kèm trên `data/*.csv` đi kèm. Không con số nào",
                  "được viết tay.", ""]

        if form:
            lines += ["## Lỗi hình thức", ""] + [f"- {e}" for e in form[:30]] + [""]
        if fails:
            lines += [f"## {len(fails)} câu không thực thi được", "",
                      "| id | lỗi |", "|---|---|"]
            lines += [f"| {i} | `{e}` |" for i, e in fails[:30]] + [""]
        if mismatches:
            lines += [f"## {len(mismatches)} câu LỆCH đáp án (nghiêm trọng)", "",
                      "| id | answer khai báo | query trả về |", "|---|---|---|"]
            lines += [f"| {i} | {a!r} | {b!r} |" for i, a, b in mismatches[:30]] + [""]
        if trace.get("examples"):
            lines += ["## Tham chiếu bảng đáng ngờ", "",
                      "| id | ref | vấn đề |", "|---|---|---|"]
            lines += [f"| {i} | `{r}` | {w} |" for i, r, w in trace["examples"]] + [""]

        out = "\n".join(lines)
        print(out)
        if args.report:
            Path(args.report).write_text(out, encoding="utf-8")
            print(f"\nĐã ghi {args.report}")

        return 0 if (not form and not fails and not mismatches) else 1
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
