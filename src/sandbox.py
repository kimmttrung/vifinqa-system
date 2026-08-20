"""Chạy code do LLM sinh trong TIẾN TRÌNH CON có trần bộ nhớ.

## Vì sao phải tách tiến trình

`execute.run_query()` dùng `eval`/`exec` thẳng trong tiến trình gọi nó, không có
trần nào. Code do model sinh là **dữ liệu không tin được**: một
`df1.merge(df2, how='cross')` trên hai bảng vài trăm dòng, hay
`reindex(range(10**9))`, cấp phát hết RAM trong vài giây và giết cả session.

Đã xảy ra thật trên Kaggle 20/08/2026: kernel chết ở giây 15.273 (giờ thứ 4,24),
ngay sau các warning phát ra từ `<query>` — tức đúng lúc exec, không phải lúc
sinh token. Mất trọn 4 giờ GPU.

Trần bộ nhớ chỉ đặt được ở mức **tiến trình** (`RLIMIT_AS`), không đặt được ở
mức thread, nên buộc phải có tiến trình con. Vượt trần thì Python nhận
`MemoryError` bắt được, thay vì bị OOM killer của kernel bắn hạ cả session.

## Vì sao chia LÔ chứ không mỗi câu một tiến trình

Khởi động Python + import pandas mất ~0,4s; nhân 1.012 câu là ~7 phút mỗi lượt.
Chạy theo lô thì chi phí đó chia đều còn ~3ms/câu.

Con chết giữa lô cũng không mất cả lô: nó ghi **từng kết quả ra file ngay khi
có** và flush, nên cha đọc lại biết chính xác câu nào giết nó — đánh dấu câu đó
hỏng rồi chạy tiếp phần còn lại. Vòng lặp luôn tiến: mỗi vòng hoặc có thêm kết
quả, hoặc câu đầu hàng đợi bị đánh dấu hỏng.

Cũng nhờ đó mà `timeout` mới thật sự là timeout: bản trong `execute.py` dùng
`ThreadPoolExecutor` mà `__exit__` gọi `shutdown(wait=True)`, nên code chạy vô
tận vẫn treo tiến trình cha mãi mãi dù đã báo timeout. Ở đây cha giết hẳn con.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEFAULT_MEM_GB = 4.0
DEFAULT_TIMEOUT_S = 8.0
# Khởi động interpreter + import pandas. Cộng vào timeout tổng của lô, nếu không
# thì lô đầu tiên gần như chắc chắn bị giết oan.
_STARTUP_S = 90.0


@dataclass
class JobResult:
    ok: bool
    value: float | None
    error: str | None = None


def _run_child(jobs: list[dict], base_dir: Path, mem_gb: float, timeout_s: float,
               work: Path, python_exe: str) -> dict[int, JobResult]:
    """Một lượt: giao `jobs` cho tiến trình con, đọc lại những gì nó kịp ghi."""
    jobs_p, out_p = work / "jobs.jsonl", work / "results.jsonl"
    with jobs_p.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    out_p.unlink(missing_ok=True)

    budget = _STARTUP_S + timeout_s * len(jobs)
    try:
        subprocess.run(
            [python_exe, str(_HERE / "sandbox.py"),
             "--jobs", str(jobs_p), "--out", str(out_p), "--base", str(base_dir),
             "--mem-gb", str(mem_gb), "--timeout", str(timeout_s)],
            capture_output=True, text=True, timeout=budget)
    except subprocess.TimeoutExpired:
        pass          # con bị giết; phần nó đã flush vẫn đọc được ở dưới

    out: dict[int, JobResult] = {}
    if out_p.exists():
        for line in out_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue      # dòng cuối bị cắt giữa chừng vì con chết đột ngột
            out[int(d["id"])] = JobResult(bool(d["ok"]), d.get("value"), d.get("error"))
    return out


def run_jobs(jobs: list[dict], base_dir: Path, mem_gb: float = DEFAULT_MEM_GB,
             timeout_s: float = DEFAULT_TIMEOUT_S, python_exe: str | None = None,
             log=None) -> dict[int, JobResult]:
    """Chạy nhiều query trong tiến trình con → {id: JobResult}.

    jobs: [{'id': int, 'code': str, 'frames': {'df1': 'data/x.csv', ...}}]

    Mọi id trong `jobs` đều có mặt trong kết quả — câu làm chết tiến trình con
    được đánh dấu hỏng chứ không biến mất, nếu không tầng trên sẽ retry vô hạn.
    """
    if not jobs:
        return {}
    python_exe = python_exe or sys.executable
    base_dir = Path(base_dir)
    out: dict[int, JobResult] = {}
    n_crash = 0

    with tempfile.TemporaryDirectory(prefix="vifinqa_sbx_") as tmp:
        work = Path(tmp)
        while True:
            todo = [j for j in jobs if int(j["id"]) not in out]
            if not todo:
                break
            out.update(_run_child(todo, base_dir, mem_gb, timeout_s, work, python_exe))
            head = int(todo[0]["id"])
            if head not in out:
                # con chết/treo ngay ở câu này ⇒ chính nó là thủ phạm
                n_crash += 1
                out[head] = JobResult(
                    False, None,
                    "tiến trình con chết (vượt trần RAM hoặc treo) — code bị loại")
                if log:
                    log(f"      ⚠ q{head}: code giết tiến trình con, đã cô lập và chạy tiếp")
    if n_crash and log:
        log(f"      sandbox: {n_crash} câu làm chết tiến trình con "
            f"(kernel notebook không hề hấn gì)")
    return out


# ───────────────────────── phía tiến trình con ─────────────────────────

def _worker(jobs_path: Path, out_path: Path, base_dir: Path,
            mem_gb: float, timeout_s: float) -> int:
    # import TRƯỚC khi đặt trần: pandas/numpy nạp xong đã chiếm vài trăm MB địa
    # chỉ ảo, đặt trần trước thì chính import sẽ chết.
    import pandas as pd

    sys.path.insert(0, str(_HERE))
    from execute import _SAFE_BUILTINS, _coerce      # noqa: PLC2701

    try:                                   # resource là Unix-only (Kaggle có)
        import resource
        lim = int(mem_gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except (ImportError, ValueError, OSError):
        pass                               # Windows/local: không có trần, cha vẫn canh timeout

    try:                                   # đồng hồ ngắt mỗi câu (Unix)
        import signal

        def _alarm(_sig, _frm):
            raise TimeoutError(f"quá {timeout_s}s")

        signal.signal(signal.SIGALRM, _alarm)
        _set_alarm = lambda s: signal.setitimer(signal.ITIMER_REAL, s)   # noqa: E731
    except (ImportError, AttributeError, ValueError):
        _set_alarm = lambda s: None                                      # noqa: E731

    jobs = [json.loads(l) for l in
            jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    cache: dict[str, "pd.DataFrame"] = {}
    CACHE_MAX = 128

    with out_path.open("a", encoding="utf-8") as fout:
        for job in jobs:
            ok, value, err = False, None, None
            try:
                frames = {}
                for var, rel in job["frames"].items():
                    p = Path(rel)
                    if not p.is_absolute():
                        p = base_dir / rel
                    key = str(p)
                    if key not in cache:
                        if len(cache) >= CACHE_MAX:
                            cache.pop(next(iter(cache)))
                        cache[key] = pd.read_csv(p)
                    frames[var] = cache[key].copy()

                env = {"__builtins__": _SAFE_BUILTINS, "pd": pd,
                       "np": __import__("numpy"), "math": __import__("math"),
                       "dfs": frames, **frames}
                body = str(job["code"]).strip()
                _set_alarm(timeout_s)
                try:
                    try:
                        val = eval(compile(body, "<query>", "eval"), env)   # noqa: S307
                    except SyntaxError:
                        exec(compile(body, "<query>", "exec"), env)         # noqa: S102
                        val = env.get("result")
                finally:
                    _set_alarm(0)
                num = _coerce(val)
                if num is None:
                    err = f"không quy được về 1 số: {type(val).__name__}"
                else:
                    ok, value = True, num
            except MemoryError:
                err = "MemoryError: code vượt trần RAM của sandbox"
            except BaseException as e:            # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]

            fout.write(json.dumps({"id": int(job["id"]), "ok": ok, "value": value,
                                   "error": err}, ensure_ascii=False) + "\n")
            fout.flush()      # flush sau TỪNG câu: con chết thì cha vẫn đọc được
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--mem-gb", type=float, default=DEFAULT_MEM_GB)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    a = ap.parse_args(argv)
    return _worker(Path(a.jobs), Path(a.out), Path(a.base), a.mem_gb, a.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
