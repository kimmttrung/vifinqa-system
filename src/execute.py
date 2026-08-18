"""Stage F — Sandbox thực thi pandas_query.

Hai vai trò:
  1. Tự kiểm tra trước khi nộp — Execution Accuracy là 1 trong 3 tiêu chí chấm,
     câu nào chạy lỗi thì mất điểm trắng, phải phát hiện ở nhà chứ không phải
     trên leaderboard.
  2. Chốt chặn chống hallucination: LLM chỉ được sinh KẾ HOẠCH/CODE, con số
     luôn do executor sinh ra từ CSV. LLM không bao giờ tự viết đáp án.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# builtins tối thiểu — đủ cho code pandas, không đủ để đụng vào hệ thống
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "round": round,
    "sorted": sorted, "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "range": range,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "any": any, "all": all, "print": print,
}


@dataclass
class ExecResult:
    ok: bool
    value: float | None
    error: str | None = None
    raw: object = None


def _coerce(v) -> float | None:
    """Kết quả pandas → một float duy nhất, hoặc None nếu không quy về được."""
    if v is None:
        return None
    if isinstance(v, (pd.Series, pd.Index)):
        if len(v) != 1:
            return None
        v = v.iloc[0] if isinstance(v, pd.Series) else v[0]
    if isinstance(v, pd.DataFrame):
        if v.shape != (1, 1):
            return None
        v = v.iat[0, 0]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return None


def load_frames(csv_paths: dict[str, str | Path], base_dir: Path) -> dict[str, pd.DataFrame]:
    """{'df1': 'data/x.csv'} → {'df1': DataFrame}"""
    out = {}
    for var, rel in csv_paths.items():
        p = Path(rel)
        if not p.is_absolute():
            p = base_dir / rel
        out[var] = pd.read_csv(p, dtype=None, keep_default_na=True)
    return out


def run_query(code: str, frames: dict[str, pd.DataFrame], timeout: float = 5.0) -> ExecResult:
    """Chạy `code`; lấy biến `result` nếu có, nếu không thì giá trị của biểu thức cuối."""
    env = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd, "np": np, "math": math,
        "dfs": frames, **frames,
    }

    def _run():
        body = code.strip()
        try:
            val = eval(compile(body, "<query>", "eval"), env)        # noqa: S307
        except SyntaxError:
            exec(compile(body, "<query>", "exec"), env)              # noqa: S102
            val = env.get("result")
        return val

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            val = ex.submit(_run).result(timeout=timeout)
    except FTimeout:
        return ExecResult(False, None, f"timeout > {timeout}s")
    except Exception as e:                                            # noqa: BLE001
        return ExecResult(False, None, f"{type(e).__name__}: {e}"[:300])

    num = _coerce(val)
    if num is None:
        return ExecResult(False, None, f"không quy được về 1 số: {type(val).__name__}", raw=val)
    return ExecResult(True, num, None, raw=val)


def verify(code: str, csv_paths: dict[str, str], base_dir: Path,
           expected: float | None = None, rtol: float = 1e-6) -> ExecResult:
    """Chạy query và (nếu có `expected`) đối chiếu — cross-check trước khi nộp."""
    try:
        frames = load_frames(csv_paths, base_dir)
    except Exception as e:                                            # noqa: BLE001
        return ExecResult(False, None, f"đọc CSV lỗi: {type(e).__name__}: {e}"[:300])
    res = run_query(code, frames)
    if res.ok and expected is not None:
        if not math.isclose(res.value, expected, rel_tol=rtol, abs_tol=1e-9):
            return ExecResult(False, res.value,
                              f"lệch đáp án: query={res.value!r} vs expected={expected!r}")
    return res
