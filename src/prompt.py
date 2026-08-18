"""Dựng prompt PoT cho Stage E — đặt ở đây để TEST được ở local, không cần GPU.

Nếu để trong notebook thì chỉ phát hiện lỗi sau khi đã nạp model 14B và chạy
20 phút. Ở đây chạy được ngay trên CSV của bất kỳ bài nộp nào.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import tokens
from locate import score_label
from qparse import QueryIR

MAX_ROWS_PREVIEW = 30
MAX_COLS_PREVIEW = 9
MAX_CELL = 60

SYSTEM = """Bạn là chuyên gia phân tích báo cáo tài chính Việt Nam, viết mã pandas.

RUNTIME CONTRACT
- KHÔNG import gì. Có sẵn: pd, np, math và các DataFrame df1, df2, ...
- Gán kết quả cuối vào biến `result`. `result` phải là MỘT số thực duy nhất.
- Không print, không đọc file, không truy cập mạng.

DATA CONTRACT
- Mọi ô số trong CSV ĐÃ được chuẩn hoá thành float chuẩn Python. Không cần xử lý
  dấu chấm ngăn nghìn, dấu phẩy thập phân hay ngoặc đơn âm.
- Ô trống = NaN. Kiểm tra trước khi tính.
- Tên cột đã làm phẳng từ header nhiều tầng, giữ nguyên mốc thời gian và đơn vị.
  Ví dụ 'tai_ngay_31_12_2022trieu_vnd' = cột "Tại ngày 31/12/2022, đơn vị Triệu VND".
- Tên cột chứa 'trieu' ⇒ giá trị đang ở TRIỆU đồng, 'ty' ⇒ TỶ đồng, 'nghin' ⇒
  NGHÌN đồng. Quy về đồng trước khi so sánh giữa các bảng khác đơn vị.
- Preview chỉ hiện một phần dòng. DataFrame trong runtime luôn ĐỦ MỌI DÒNG —
  nếu không thấy chỉ tiêu ở preview, dùng .str.contains() trên toàn bộ DataFrame.

SEMANTIC RULES
- "chênh lệch" / "khác biệt" (không hướng) → abs().
- "tăng bao nhiêu" / "giảm bao nhiêu" / "A trừ B" (có hướng) → giữ dấu.
- "phần trăm" / "%" → nhân 100.   "lần" → KHÔNG nhân 100.
- "điểm phần trăm" → hiệu của hai tỷ lệ đã nhân 100.
- Chi phí, dự phòng, giá vốn thường ghi âm trong báo cáo; nếu câu hỏi hỏi độ lớn
  thì abs(). Lợi nhuận giữ nguyên dấu (công ty lỗ thì kết quả âm là đúng).
- "công ty mẹ" = báo cáo riêng, "hợp nhất" = báo cáo hợp nhất.
- "cuối năm N" = số dư 31/12/N. "đầu năm N" = số dư 01/01/N. "trong năm N" = phát sinh.

RELIABILITY RULES
- Lọc dòng theo NHÃN đầy đủ (str.strip() == '...' hoặc .str.contains), không theo
  vị trí số. Nếu nhãn có thể trùng, thêm điều kiện để chọn đúng một dòng.
- Không làm tròn ở bước trung gian. Chỉ quy đổi đơn vị ở bước cuối.
- Nếu dữ liệu cần thiết KHÔNG có trong các DataFrame được cấp, gán result = None.
  Đừng bịa số.

Trả lời DUY NHẤT một khối mã:
```python
result = ...
```"""

_UNIT_TXT = {"vnd": "đồng (VND)", "percent": "phần trăm (%)", "pp": "điểm phần trăm",
             "times": "lần", "year": "năm — trả về số năm, vd 2023.0",
             "count": "số lượng doanh nghiệp", "shares": "số cổ phiếu", "usd": "USD"}


def pick_rows(df: pd.DataFrame, ir: QueryIR, budget: int = MAX_ROWS_PREVIEW) -> list[int]:
    """Chọn dòng preview THEO CÂU HỎI, không phải `head(n)`.

    Bảng thuyết minh thường 50-200 dòng và dòng cần thiết hay nằm ngoài 30 dòng
    đầu. `head(30)` làm mô hình không thấy dữ liệu rồi trả None — trông giống lỗi
    retrieval (nhóm "insufficient evidence" 34,7% của mentor) nhưng nguyên nhân
    thật là CẮT PREVIEW, còn DataFrame đầy đủ vẫn nằm trong tay mô hình.

    Giữ 8 dòng đầu (tiêu đề mục lớn — cần cho ngữ cảnh phân cấp kế toán) + các
    dòng nhãn khớp câu hỏi nhất, rồi sắp lại theo THỨ TỰ GỐC để mô hình vẫn thấy
    đúng quan hệ trên–dưới của bảng.
    """
    if len(df) <= budget:
        return list(range(len(df)))
    qt = tokens(ir.retrieval_query)
    labels = df[df.columns[0]].astype(str).tolist()
    keep = set(range(min(8, len(df))))
    for i in sorted(range(len(df)), key=lambda j: -score_label(qt, labels[j])):
        if len(keep) >= budget:
            break
        keep.add(i)
    return sorted(keep)


def pick_cols(df: pd.DataFrame, ir: QueryIR, budget: int = MAX_COLS_PREVIEW) -> list[str]:
    """Giữ 2 cột đầu (nhãn), rồi ưu tiên cột khớp NĂM câu hỏi hỏi.

    Bảng ngân hàng hay có 6-10 cột kỳ; cắt `[:9]` mù có thể mất đúng cột cần.
    """
    cols = list(df.columns)
    if len(cols) <= budget:
        return cols
    years = {str(y) for y in ir.years}
    head, rest = cols[:2], []
    rest += [c for c in cols[2:] if any(y in c for y in years)]
    rest += [c for c in cols[2:] if c not in rest]
    return (head + rest)[:budget]


def preview(csv_path: Path, ir: QueryIR, var: str = "df1",
            max_rows: int = MAX_ROWS_PREVIEW) -> str:
    df = pd.read_csv(csv_path)
    cols = pick_cols(df, ir)
    idx = pick_rows(df, ir, max_rows)
    body = df.iloc[idx][cols].copy()
    for c in body.columns:
        if body[c].dtype == object:
            body[c] = body[c].astype(str).str.slice(0, MAX_CELL)
    txt = body.to_csv(index=False)
    if len(idx) < len(df):
        txt += (f"… bảng có {len(df)} dòng, đang hiện {len(idx)} dòng liên quan nhất. "
                f"{var} trong runtime chứa ĐỦ mọi dòng — nếu không thấy chỉ tiêu ở "
                f"trên, dùng {var}[{var}.iloc[:,0].astype(str).str.contains('…')].\n")
    if len(cols) < len(df.columns):
        others = [c for c in df.columns if c not in cols]
        txt += f"… còn {len(others)} cột khác: {others}\n"
    return txt


def build_prompt(rec: dict, ir: QueryIR, base_dir: Path,
                 max_dfs: int | None = None, max_rows: int = MAX_ROWS_PREVIEW) -> str:
    p = [f"CÂU HỎI: {rec['question']}", ""]
    if ir.unit_kind == "vnd" and ir.unit_scale != 1.0:
        p.append(f"ĐƠN VỊ ĐÁP ÁN: VND chia cho {ir.unit_scale:g} "
                 f"(kết quả cuối phải là số đã chia)")
    else:
        p.append(f"ĐƠN VỊ ĐÁP ÁN: {_UNIT_TXT.get(ir.unit_kind, ir.unit_kind)}")
    if ir.tickers:
        p.append(f"CÔNG TY: {', '.join(ir.tickers)}")
    if ir.years:
        p.append(f"NĂM: {', '.join(map(str, ir.years))}")
    if ir.stmt_type:
        p.append("PHẠM VI: " + ("báo cáo riêng (công ty mẹ)" if ir.stmt_type == "separate"
                                else "báo cáo hợp nhất"))
    p.append("")
    ev = rec["evidence"][:max_dfs] if max_dfs else rec["evidence"]
    for e in ev:
        f = base_dir / e["csv_path"]
        if not f.exists():
            continue
        p += [f"### {e['variable']}   ({f.name})", "```csv",
              preview(f, ir, e["variable"], max_rows), "```"]
    if max_dfs and len(rec["evidence"]) > max_dfs:
        p.append(f"(Còn {len(rec['evidence'])-max_dfs} bảng nữa không hiện ở đây "
                 f"vì giới hạn độ dài; chúng KHÔNG có trong runtime.)")
    return "\n".join(p)


def build_prompt_fitted(rec: dict, ir: QueryIR, base_dir: Path,
                        n_tokens, budget: int) -> tuple[str, list[dict], str]:
    """Thu nhỏ prompt cho vừa `budget` token → (prompt, evidence còn lại, cách thu).

    Đo trên 1.012 câu: p90 = 4.354 token nhưng **max = 8.825**, vượt `max_model_len`
    8.192. Nếu chỉ bỏ qua câu quá dài thì mất trắng những câu nhiều bảng nhất —
    mà đó thường là câu nhiều công ty, tức câu khó và đáng điểm.

    Thu theo thứ tự ít thiệt hại nhất: giảm số dòng preview trước (chỉ mất ngữ
    cảnh), rồi mới bớt bảng (mất dữ liệu thật). `evidence` trả về phải được dùng
    làm frames cho executor, nếu không code sẽ trỏ vào dfN không tồn tại.
    """
    for max_rows in (MAX_ROWS_PREVIEW, 20, 14):
        pr = build_prompt(rec, ir, base_dir, max_rows=max_rows)
        if n_tokens(pr) <= budget:
            how = "full" if max_rows == MAX_ROWS_PREVIEW else f"rows={max_rows}"
            return pr, rec["evidence"], how
    for n in range(len(rec["evidence"]) - 1, 0, -1):
        pr = build_prompt(rec, ir, base_dir, max_dfs=n, max_rows=14)
        if n_tokens(pr) <= budget:
            return pr, rec["evidence"][:n], f"dfs={n},rows=14"
    pr = build_prompt(rec, ir, base_dir, max_dfs=1, max_rows=10)
    return pr, rec["evidence"][:1], "dfs=1,rows=10"
