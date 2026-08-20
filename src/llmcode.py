"""Kiểm và chuẩn hoá code do LLM sinh, TRƯỚC khi cho chạy.

Hai lỗi thật đã bắt được trên bài nộp thử 20/08 (`out/mini`, 3 câu, 1 câu vá):

    "pandas_query": "result = (9651580686.0 + 208253201298.0) / 1e6"

Một dòng, hai lỗi khác nhau, và **không tầng kiểm nào cũ bắt được**:

1. **Số viết tay.** Model đọc số từ preview trong prompt rồi gõ lại vào code,
   không chạm vào `df1`. Đây đúng là thứ cả kiến trúc dựng lên để ngăn ("LLM
   sinh KẾ HOẠCH, executor sinh SỐ"). `verify_submission.py` tầng 3 hỏi "chạy
   query có ra đúng answer không" — mà query hard-code thì LUÔN khớp, nên nó
   báo PASS và còn in ra "Không con số nào được viết tay".

2. **Dạng `result = ...`.** Ví dụ `pandas_query` của BTC là một BIỂU THỨC, và
   mọi query do rule sinh cũng là biểu thức — các lượt nộp đó cho
   EXECUTION == ANSWER khớp nhau chính xác, tức grader chạy được biểu thức. Ta
   KHÔNG có bằng chứng nào rằng nó chạy được lệnh gán. Nếu grader dùng `eval()`
   thì `result = ...` là SyntaxError ⇒ mất trắng Execution của mọi câu LLM vá.
   Ta không thấy vì `execute.run_query` tự fallback sang `exec` — chốt chặn của
   mình che mất rủi ro của grader.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

# Hằng số hợp lệ trong code tính toán: hệ số quy đổi đơn vị, phần trăm, chỉ số
# dòng/cột, năm. Mọi số khác có nhiều hơn ngần này chữ số nghĩa đều đáng ngờ —
# giá trị tài chính thật luôn dài (9.651.580.686), hệ số quy đổi thì không.
MAX_SIGNIFICANT_DIGITS = 3
YEAR_MIN, YEAR_MAX = 2000, 2035


@dataclass
class CodeCheck:
    ok: bool
    code: str                 # code đã chuẩn hoá (biểu thức nếu rút được)
    form: str                 # "expr" | "assign" | "multi" | "unparsable"
    reason: str | None = None
    literals: tuple = ()      # các số viết tay bắt được


def _significant_digits(v: float | int) -> int:
    """Số chữ số nghĩa: bỏ dấu, bỏ số 0 dẫn đầu và số 0 đuôi.

    1e6 → 1 · 100 → 1 · 0.02 → 1 · 1.5 → 2 · 2018 → 4 · 9651580686 → 9
    """
    s = f"{abs(v):.10g}"
    if "e" in s or "E" in s:                      # 1e+06 → phần định trị
        s = s.split("e")[0].split("E")[0]
    s = s.replace(".", "").replace("-", "")
    return len(s.strip("0")) or 1


def _is_benign(v: float | int) -> bool:
    if v == 0:
        return True
    if float(v).is_integer() and YEAR_MIN <= abs(v) <= YEAR_MAX:
        return True                                # năm dùng để lọc cột/dòng
    return _significant_digits(v) <= MAX_SIGNIFICANT_DIGITS


def hardcoded_literals(tree: ast.AST) -> list[float]:
    """Các hằng số trong code trông như GIÁ TRỊ lấy từ bảng.

    Duyệt AST chứ không regex: số nằm trong chuỗi (tên cột
    `tai_ngay_31_12_2022trieu_vnd`, nhãn `'31/12/2018'`) không được tính, nếu
    không thì gần như mọi query đều bị chặn oan.
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool) and not _is_benign(node.value):
            out.append(node.value)
    return out


def to_expression(tree: ast.Module) -> tuple[str | None, str]:
    """→ (biểu thức, dạng). Rút `result = EXPR` về `EXPR` khi rút được.

    Chỉ rút khi thân hàm đúng MỘT lệnh. Nhiều lệnh thì không có phép biến đổi
    tổng quát nào về một biểu thức, nên trả nguyên trạng và để tầng trên đếm.
    """
    body = tree.body
    if len(body) == 1:
        st = body[0]
        if isinstance(st, ast.Expr):
            return ast.unparse(st.value), "expr"
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name) and st.targets[0].id == "result":
            return ast.unparse(st.value), "assign"
    return None, "multi"


def check(code: str, strict: bool = True) -> CodeCheck:
    """Kiểm một đoạn code LLM sinh.

    strict=True (lượt đầu) — từ chối cả code nhiều lệnh, để vòng retry có cơ hội
    sinh lại một biểu thức. strict=False (lượt cuối) — nhận code nhiều lệnh, vì
    có còn hơn không: đằng nào cũng chỉ so với việc giữ đáp án tất định.
    """
    code = (code or "").strip()
    if not code:
        return CodeCheck(False, code, "unparsable", "code rỗng")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CodeCheck(False, code, "unparsable", f"SyntaxError: {e.msg}")

    lits = hardcoded_literals(tree)
    if lits:
        return CodeCheck(False, code, "expr", literals=tuple(lits),
                         reason=f"số viết tay trong code: {lits[:4]} — phải đọc từ dfN")

    expr, form = to_expression(tree)
    if expr is not None:
        return CodeCheck(True, expr, form)
    if strict:
        return CodeCheck(False, code, "multi",
                         reason="code nhiều lệnh — cần MỘT biểu thức duy nhất")
    return CodeCheck(True, code, "multi")
