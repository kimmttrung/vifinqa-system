"""Đóng gói mọi thứ Kaggle cần vào một thư mục để upload làm Dataset.

    python scripts/package_kaggle.py --sub out/v4

Sinh ra `kaggle_bundle/` gồm:
    artifacts/table_meta.parquet   index bảng + passage       (~16 MB)
    artifacts/facts.parquet        fact table                 (~22 MB)
    artifacts/bm25.*               index BM25                 (~9 MB)
    artifacts/docs.parquet         metadata document
    questions/questions.jsonl
    code_stock.csv
    src/*.py                       toàn bộ module (layout phẳng)
    submission.zip                 bài nộp local (Stage E cần data/*.csv trong đó)

KHÔNG kèm `tables.parquet` (59 MB, chỉ Stage A/B cần) và KHÔNG kèm corpus gốc
(363 MiB). Tổng bundle ≈ 55 MB.

Đóng `data/*.csv` dưới dạng submission.zip thay vì ~4.900 file lẻ: Kaggle Dataset
xử lý rất chậm với dataset nhiều file nhỏ, còn notebook giải nén chỉ mất vài giây.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import ARTIFACTS, DATA_ROOT, REPO_ROOT

NEEDED_ARTIFACTS = [
    "table_meta.parquet", "facts.parquet", "docs.parquet",
    "bm25.tf.npz", "bm25.meta.npz", "bm25.vocab.json",
]
OPTIONAL_ARTIFACTS = ["rerank_cache.parquet", "table_emb.f16.npy"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="out/v4", help="thư mục submission để Stage E dùng")
    ap.add_argument("--out", default="kaggle_bundle")
    args = ap.parse_args()

    root = Path(DATA_ROOT or Path.cwd())
    dst = Path(args.out)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "artifacts").mkdir(parents=True)
    (dst / "questions").mkdir(parents=True)

    missing = []
    for name in NEEDED_ARTIFACTS:
        src = ARTIFACTS / name
        if src.exists():
            shutil.copy2(src, dst / "artifacts" / name)
        else:
            missing.append(name)
    for name in OPTIONAL_ARTIFACTS:
        src = ARTIFACTS / name
        if src.exists():
            shutil.copy2(src, dst / "artifacts" / name)
            print(f"  (kèm thêm {name})")

    shutil.copy2(root / "questions" / "questions.jsonl", dst / "questions")
    shutil.copy2(root / "code_stock.csv", dst / "code_stock.csv")
    # Layout phẳng: bundle mang nguyên thư mục src/, notebook thêm nó vào
    # sys.path rồi `from qparse import ...` (không còn package `pipeline`).
    shutil.copytree(REPO_ROOT / "src", dst / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))

    sub_zip = Path(args.sub) / "submission.zip"
    if sub_zip.exists():
        shutil.copy2(sub_zip, dst / "submission.zip")
    else:
        missing.append(str(sub_zip))

    total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    n = sum(1 for f in dst.rglob("*") if f.is_file())
    print(f"\n{dst}/  ·  {n} file  ·  {total/1e6:.0f} MB")
    for f in sorted(dst.rglob("*")):
        if f.is_file() and f.stat().st_size > 1e6:
            print(f"   {f.relative_to(dst).as_posix():40} {f.stat().st_size/1e6:7.1f} MB")
    if missing:
        print("\nTHIẾU (chạy build_index.py / build_facts.py / run_pipeline.py trước):")
        for m in missing:
            print("   ", m)
        return 1
    print("\nUpload thư mục này lên kaggle.com/datasets → New Dataset → tên 'vifinqa-bundle'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
