"""Sinh bộ probe giải mã `relevant_tables` (Unknown #1 + #2 trong CLAUDE.md).

    python scripts/make_probes.py            # 2 lượt: chốt doc_id
    python scripts/make_probes.py --stage 2  # 4 lượt: chốt vị trí bảng

## Vì sao phải probe

Ví dụ BTC đưa (`AAA_..._2015_consolidated|350`) đã bị LOẠI TRỪ bằng đo đạc: doc
đó chỉ có 47 bảng, 43 trang, và dòng 350 là text thường. Ví dụ đó là dummy
(hỏi VNM nhưng doc là AAA). Nếu đoán sai convention thì **F₂ = 0 toàn bộ**,
mọi cải tiến retrieval đều vô nghĩa. Phải đo, không được đoán.

## Thiết kế 2 tầng — tiết kiệm lượt nộp

Giai đoạn 1 (2 lượt): mỗi câu nộp CẢ 4 cách mã hoá vị trí, chỉ đổi `doc_id`.
    · nếu cả hai đều 0 → sai ở chỗ khác (định dạng ref, hoặc cả 4 giả thuyết sai)
    · lượt nào > 0 → chốt được doc_id, VÀ biết 1 trong 4 vị trí là đúng
    Trần điểm mỗi lượt: gold 1 bảng, nộp 4 ref ⇒ F₂ = 0.625

Giai đoạn 2 (4 lượt): giữ doc_id đã chốt, dò từng cách mã hoá một.
    · lượt nào nhảy lên ≈1.6× giai đoạn 1 → đó là convention thật

Tổng 6/10 lượt trong một ngày. Sau đó `--position-scheme` chốt cứng.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# Giai đoạn 1: k_max=1 bắt buộc — mỗi câu nộp ĐÚNG 1 bảng để 4 cách mã hoá ra
# đúng 4 ref. Để k_max=4 thì thành 16 ref (F₂ trần chỉ 0.25), tín hiệu loãng.
STAGE1 = [
    ("probe1_all_folder",
     ["--position-scheme", "all", "--doc-id-scheme", "folder", "--k-max", "1"]),
    ("probe2_all_stem",
     ["--position-scheme", "all", "--doc-id-scheme", "file_stem", "--k-max", "1"]),
]

# Giai đoạn 2: k_max=2 — mức tối ưu suy ra từ số đo giai đoạn 1 (gold ≈ 2,1
# bảng/câu). Nhờ vậy lượt nào trúng convention cũng ĐỒNG THỜI là bài nộp tốt
# nhất hiện có, không phí lượt. Ba lượt còn lại vẫn về ~0 nên vẫn phân biệt được.
STAGE2 = [
    (f"probe_{s}", ["--position-scheme", s, "--k-max", "2"])
    for s in ("table_idx", "line_no", "page_no", "char_start")
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--doc-id-scheme", default="folder",
                    help="giai đoạn 2: doc_id đã chốt ở giai đoạn 1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    runs = STAGE1 if args.stage == 1 else [
        (tag, [*flags, "--doc-id-scheme", args.doc_id_scheme]) for tag, flags in STAGE2
    ]
    for tag, flags in runs:
        cmd = [PY, str(HERE / "run_pipeline.py"), "--tag", tag, *flags]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n{'='*70}\n{tag}\n{'='*70}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"!! {tag} lỗi (rc={rc})")
            return rc

    print("\nXong. Nộp lần lượt các file out/<tag>/submission.zip rồi ghi lại F₂:")
    for tag, _ in runs:
        print(f"  out/{tag}/submission.zip   → F₂ = ______")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
