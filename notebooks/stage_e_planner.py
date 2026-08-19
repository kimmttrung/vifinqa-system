# %% [markdown]
# # ViFinQA — Stage E : LLM Planner (Program-of-Thought)
#
# **Accelerator : GPU T4 ×2** · **Internet : ON** · thời gian ~2,5–4 giờ
#
# ## INPUT
#
# | Thứ | Lấy từ đâu |
# |---|---|
# | code (`src/*.py`, `scripts/run_pipeline.py`) | `git clone` repo |
# | `artifacts/*` · `questions/questions.jsonl` | repo, **hoặc** Kaggle Dataset đính kèm |
# | `artifacts/rerank_cache.parquet` | tuỳ chọn — có thì bài nộp nền dùng hybrid RRF |
# | model 14B AWQ | tải qua Internet |
#
# **Không phải upload `submission.zip` sẵn.** Notebook tự chạy `run_pipeline.py`
# ngay trong session để dựng bài nộp nền (~4–8 phút CPU), rồi LLM vá lên trên đó.
# Nhờ vậy bài nền LUÔN khớp với code trong repo — không còn cảnh vá `llm_patch`
# của hôm nay lên `submission.zip` dựng từ code của tuần trước.
#
# ## OUTPUT — tab Output, `/kaggle/working/`
#
# | File | Làm gì với nó |
# |---|---|
# | `llm_patch.jsonl` | **kết quả chính** — mang về local chạy `apply_llm.py` |
# | `llm_log.jsonl` | mọi lần sinh code kể cả lỗi, để soi câu nào hỏng vì sao |
# | `out/base/` | bài nộp nền dựng tại chỗ (không cần mang về) |
#
# Mang `llm_patch.jsonl` về local rồi:
# `python scripts/apply_llm.py --sub out/v4 --patch llm_patch.jsonl --out out/v5`
#
# ---
# ## Nguyên tắc bất di bất dịch: LLM sinh CODE, executor sinh SỐ
# LLM không bao giờ được viết con số vào `answer`. Mọi giá trị đều do sandbox
# chạy `pandas_query` trên CSV mà ra. Đây vừa là lá chắn chống hallucination,
# vừa là điều kiện để Execution Accuracy không bao giờ tụt.
#
# ## Vì sao đưa TẤT CẢ 1.012 câu vào LLM
#
# | | Answer Accuracy |
# |---|---|
# | Mentor · LLM + retrieved evidence (k=10) | **64,0 %** |
# | Mentor · LLM + oracle evidence | 87,0 % |
# | **Ta · rule thuần (đo trên leaderboard 17/08)** | **24,1 %** |
#
# Retrieval của ta có recall **0,605** — không kém mentor bao nhiêu. Nên khoảng
# cách 24 % vs 64 % **không phải do retrieval**, mà do rule chỉ mã hoá được công
# thức đóng. Đường tất định đang là cái **trần**, không phải cái nền.
#
# Rule không bị bỏ, nó đổi vai:
# * CSV đã chuẩn hoá số sẵn → triệt nhóm lỗi parse (mentor: 9,3 % Technical +
#   phần lớn 54,7 % Numerical Extraction)
# * Sandbox verify → trọng tài duy nhất quyết định code có được nhận hay không
# * `rule_answer` đi kèm mỗi bản vá → `apply_llm.py` báo cờ khi LLM lệch >2 %
#
# ## Vì sao AWQ 4-bit là bắt buộc trên T4×2
# fp16 14B = 28 GB; T4×2 = 30 GB tổng ⇒ TP=2 còn ~2 GB cho KV cache, nghẹt ngay.
# AWQ 4-bit ≈ 9 GB ⇒ mỗi T4 giữ ~4,5 GB, còn ~10 GB KV cache.
# **sm75 không có Marlin kernel** nên vLLM tự rơi về AWQ GEMM thường: chậm hơn
# nhưng chạy được. `enforce_eager=True` để tránh lỗi CUDA-graph capture trên Turing.

# %%
import gc
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/kimmttrung/vifinqa-system.git"
REPO_BRANCH = "main"
BASE_TAG = "base"
# Cấu hình dựng bài nộp nền. PHẢI KHỚP với bài bạn sẽ vá ở local, vì LLM viết
# code dựa trên df1..dfN của bài nền này, còn apply_llm.py lại chạy code đó trên
# df1..dfN của bài ở local. Lệch nhau thì code vẫn chạy, chỉ là đọc nhầm bảng —
# sai mà không có lỗi nào báo. (Riêng `--ce-extra` chỉ NỐI vào cuối nên df1..df6
# giữ nguyên, vá chéo được; `--ce-weight` thì đảo cả thứ tự, không vá chéo được.)
BASE_ARGS = ["--k-max", "6", "--adaptive-k"]

T_START = time.time()
IN_KAGGLE = Path("/kaggle/working").exists()
KAGGLE_INPUT = Path("/kaggle/input")


def log(msg: str) -> None:
    print(f"[{time.time()-T_START:7.1f}s] {msg}", flush=True)


def get_repo() -> Path:
    """Clone repo trên Kaggle · dò ngược lên cây thư mục khi chạy local.

    `--depth 1` là bắt buộc khi repo mang theo artifacts/: clone đủ lịch sử sẽ
    tải lại mọi phiên bản artifacts của mọi commit.
    """
    if not IN_KAGGLE:
        here = Path.cwd().resolve()
        for cand in [here, *here.parents]:
            if (cand / "src" / "qparse.py").exists():
                return cand
        raise FileNotFoundError("Không thấy src/qparse.py — chạy notebook từ trong repo")

    dst = Path("/kaggle/working/vifinqa-system")
    if not (dst / "src" / "qparse.py").exists():
        log(f"clone {REPO_URL} …")
        rc = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(dst)],
            capture_output=True, text=True)
        if rc.returncode:
            raise RuntimeError(
                f"git clone hỏng (rc={rc.returncode}): {rc.stderr.strip()[:400]}\n"
                "Kiểm tra Settings → Internet: On (cần xác thực số điện thoại).")
    return dst


def find_under(root: Path, name: str, max_depth: int = 4) -> Path | None:
    """File `name` nông nhất dưới `root`. Kaggle mount dataset ở độ sâu không đoán trước."""
    if not root.exists():
        return None
    for d in range(max_depth + 1):
        hits = sorted(root.glob("/".join(["*"] * d + [name])))
        if hits:
            return hits[0]
    return None


def resolve_artifacts(repo: Path) -> tuple[Path, str]:
    """artifacts/ lấy từ repo, KHÔNG có thì lấy từ Kaggle Dataset đính kèm.

    Hai nguồn vì repo ~106 MB có thể không push được (mạng, hoặc GitHub chặn file
    >100 MB). Khi đó vào Kaggle → + Add Input → Dataset chứa artifacts là chạy tiếp
    được, không phải sửa notebook. Trả về cả nguồn để in ra cho biết đang dùng cái nào.
    """
    local = repo / "artifacts"
    if (local / "table_meta.parquet").exists():
        return local, "repo (đã commit)"
    hit = find_under(KAGGLE_INPUT, "table_meta.parquet")
    if hit:
        return hit.parent, f"Kaggle Dataset ({hit.parent})"
    raise FileNotFoundError(
        "Không tìm thấy table_meta.parquet ở đâu cả.\n"
        "  Cách 1: commit artifacts/ lên GitHub rồi push.\n"
        "  Cách 2: tạo Kaggle Dataset chứa artifacts/ rồi + Add Input vào notebook.")


def resolve_questions(repo: Path, arti: Path) -> Path:
    for c in (repo / "questions" / "questions.jsonl",
              arti / "questions.jsonl",
              arti.parent / "questions" / "questions.jsonl"):
        if c.exists():
            return c
    hit = find_under(KAGGLE_INPUT, "questions.jsonl")
    if hit:
        return hit
    raise FileNotFoundError("Không tìm thấy questions.jsonl (repo hoặc Kaggle Dataset)")


REPO = get_repo()
sys.path.insert(0, str(REPO / "src"))
ARTI, ARTI_SRC = resolve_artifacts(REPO)
QUES = resolve_questions(REPO, ARTI)
# Trỏ THẲNG bằng biến môi trường thay vì để common.py tự dò: hàm dò quét
# /kaggle/input TRƯỚC, nên còn Dataset cũ đính kèm là nó nhặt nhầm bản cũ mà
# không báo gì.
os.environ["VIFINQA_ARTIFACTS"] = str(ARTI)
os.environ["VIFINQA_QUESTIONS"] = str(QUES)

NEED = ["table_meta.parquet", "docs.parquet", "facts.parquet",
        "bm25.tf.npz", "bm25.meta.npz", "bm25.vocab.json"]
miss = [n for n in NEED if not (ARTI / n).exists()]
if miss:
    raise FileNotFoundError(f"artifacts thiếu file: {miss}  (đang đọc {ARTI})")

from execute import run_query                # noqa: E402
from qparse import parse_question            # noqa: E402
from common import ARTIFACTS                 # noqa: E402

OUT = Path("/kaggle/working") if IN_KAGGLE else Path("out/llm")
OUT.mkdir(parents=True, exist_ok=True)
log(f"repo      : {REPO}")
log(f"artifacts : {ARTIFACTS}   ← {ARTI_SRC}")

# ── bài nộp nền: dựng TẠI CHỖ bằng chính code vừa clone ──
# Trước đây bước này giải nén một submission.zip upload sẵn. Bỏ đi vì bài nền
# và code rất dễ lệch phiên bản, mà lệch thì `evidence`/`csv_path` trong prompt
# trỏ sang bảng khác, hỏng âm thầm.
BASE = (Path("/kaggle/working/out") if IN_KAGGLE else Path("out")) / BASE_TAG
if not (BASE / "submission.json").exists():
    log(f"dựng bài nộp nền bằng run_pipeline.py (~4–8 phút) → {BASE}")
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_pipeline.py"),
         "--tag", BASE_TAG, *BASE_ARGS],
        cwd=str(REPO), capture_output=True, text=True)
    print(rc.stdout[-2500:])
    if rc.returncode or not (BASE / "submission.json").exists():
        raise RuntimeError(f"run_pipeline.py hỏng: {rc.stderr.strip()[-800:]}")

records = json.loads((BASE / "submission.json").read_text(encoding="utf-8"))
log(f"repo {REPO}  ·  base {BASE}  ·  {len(records)} câu")
_ce = [a for a in BASE_ARGS if a.startswith("--ce-")]
log(f"bài nền dựng bằng: {' '.join(BASE_ARGS)}"
    + (f"  ⇒ có dùng cross-encoder ({', '.join(_ce)})" if _ce
       else "  ⇒ lexical thuần (cross-encoder TẮT — đó là mặc định đã đo tốt nhất)"))

# %% [markdown]
# ## 1 — Chọn phạm vi

# %%
LLM_SCOPE = "all"        # "all" | "hard"  ("hard" để chạy ablation so sánh)

_HARD_OPS = {"RATIO", "GROWTH", "MEDIAN", "CONDITIONAL", "COUNT", "AVERAGE", "SUPERLATIVE"}
_HARD_UNITS = {"percent", "pp", "times", "year", "count"}

targets = []
for r in records:
    ir = parse_question(r["id"], r["question"])
    hard = bool(_HARD_OPS & set(ir.ops)) or ir.unit_kind in _HARD_UNITS or ir.tier >= 3
    if (LLM_SCOPE == "all" or hard) and r["evidence"]:
        targets.append((r, ir))

log(f"đưa vào LLM: {len(targets)}/{len(records)}  (scope={LLM_SCOPE})")
log(f"  theo tier : {dict(sorted(Counter(ir.tier for _, ir in targets).items()))}")
log(f"  theo đơn vị: {dict(Counter(ir.unit_kind for _, ir in targets).most_common())}")
log(f"  df/câu     : trung bình "
    f"{sum(len(r['evidence']) for r, _ in targets)/max(len(targets),1):.1f}")

# %% [markdown]
# ## 2 — Prompt PoT
#
# Bốn hợp đồng theo mentor, nhưng **Data Contract đã được đơn giản hoá**: Stage G
# của ta chuẩn hoá số ngay khi ghi CSV (`1.234.567`→`1234567.0`, `(162.105)`→
# `-162105.0`). Mô hình không còn phải tự parse dấu chấm/ngoặc kiểu Việt.
#
# Preview dùng **nhãn dòng + tên cột đã làm phẳng** (`tai_ngay_31_12_2022trieu_vnd`).
# Đây chính là "cell grounding" mà mentor gọi là trọng tâm: cho mô hình thấy rõ
# tiền tố hàng/cột thay vì bắt nó đoán.

# %% [markdown]
# Prompt sống trong `src/prompt.py` chứ không phải trong notebook, để **test
# được ở local mà không cần GPU**. Nếu để trong notebook thì lỗi prompt chỉ lộ ra
# sau khi đã nạp model 14B và chạy 20 phút.
#
# Đã đo ở local trên cả 1.012 câu:
# * dòng chứa đáp án vào được preview: `head(30)` **97,0 %** → chọn theo câu hỏi **98,2 %**
# * độ dài prompt: trung vị **2.241** token · p90 **4.354** · **max 8.825**
#
# Max vượt `max_model_len` 8.192 ⇒ `build_prompt_fitted()` thu nhỏ dần (giảm số
# dòng preview trước, mới bớt bảng sau) thay vì bỏ trắng câu.

# %%
from prompt import SYSTEM, build_prompt, build_prompt_fitted  # noqa: E402

_rec0, _ir0 = targets[0]
_demo = build_prompt(_rec0, _ir0, BASE)
log(f"prompt mẫu ({len(_demo)} ký tự):\n{'-'*72}\n{_demo[:1800]}\n{'-'*72}")

# %% [markdown]
# ## 3a — Cài vLLM
#
# **Kaggle KHÔNG cài sẵn vLLM** — image chỉ có torch/transformers. Cell này cài
# (~5–10 phút, ~2 GB) rồi kiểm ngay ba thứ hay hỏng, để bạn biết trong 1 phút thay
# vì phát hiện sau khi đã chờ nạp model.
#
# Hai biến môi trường phải đặt **trước** khi `import vllm`, đặt sau là vô tác dụng:
#
# | Biến | Vì sao |
# |---|---|
# | `VLLM_WORKER_MULTIPROC_METHOD=spawn` | TP=2 phải fork worker; trong notebook, `fork` mặc định hay treo cứng vì tiến trình cha đã giữ CUDA context |
# | `VLLM_USE_V1=0` | engine V1 nhắm vào sm80+; trên Turing (sm75) dùng V0 ổn định hơn |
#
# Nếu cài xong mà vẫn `ModuleNotFoundError`: **Run → Restart Session** rồi Run All
# lại — pip đã thay gói ngay dưới chân tiến trình đang chạy.

# %%
VLLM_SPEC = os.environ.get("VIFINQA_VLLM_SPEC", "vllm")

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V1", "0")

try:
    import vllm                                          # noqa: F401
except ModuleNotFoundError:
    log(f"cài {VLLM_SPEC} … (5–10 phút, ~2 GB — bình thường)")
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", VLLM_SPEC],
                        capture_output=True, text=True)
    print(rc.stdout[-1200:])
    print(rc.stderr[-1200:])
    try:
        import vllm                                      # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Cài xong vẫn không import được vllm.\n"
            "→ Run → Restart Session, rồi Run All lại (pip vừa thay gói dưới chân "
            "tiến trình đang chạy nên nó chưa thấy)."
        ) from e

import torch                                             # noqa: E402

cap = torch.cuda.get_device_capability(0)
log(f"vllm {vllm.__version__} · torch {torch.__version__} · "
    f"GPU sm{cap[0]}{cap[1]} ×{torch.cuda.device_count()}")

if torch.cuda.device_count() < 2:
    raise RuntimeError(
        f"Chỉ thấy {torch.cuda.device_count()} GPU. 14B AWQ cần T4 ×2 "
        "(Settings → Accelerator → GPU T4 x2). Muốn chạy 1 GPU thì đổi MODEL sang "
        "bản 7B AWQ và TP=1 ở cell dưới.")
if cap[0] < 8:
    log("  sm<80: không bf16, không Marlin ⇒ bắt buộc dtype='half' + AWQ GEMM thường "
        "(chậm hơn nhưng chạy được). Cell dưới đã đặt đúng.")

# pip vừa có thể nâng/hạ numpy — mà phần sau còn cần pandas để đọc CSV bằng chứng.
# Vỡ ở đây thì biết ngay, thay vì vỡ sau 2 giờ sinh code.
import pandas as _pd                                     # noqa: E402
_ = _pd.DataFrame({"x": [1.0]}).x.sum()
log(f"pandas vẫn chạy sau khi cài: {_pd.__version__}")

# %% [markdown]
# ## 3b — vLLM: Qwen 14B AWQ, tensor-parallel 2

# %%
MODEL = os.environ.get("VIFINQA_LLM", "Qwen/Qwen2.5-14B-Instruct-AWQ")
MAX_LEN = 8192

from vllm import LLM, SamplingParams      # noqa: E402

log(f"nạp {MODEL} (AWQ, TP=2, dtype=half) …")
llm = LLM(model=MODEL, dtype="half", quantization="awq", tensor_parallel_size=2,
          max_model_len=MAX_LEN, gpu_memory_utilization=0.90,
          enforce_eager=True, trust_remote_code=True)
tok = llm.get_tokenizer()
log("mô hình sẵn sàng")

# %% [markdown]
# ## 4 — Sinh code → thực thi → chỉ giữ câu chạy ra số
#
# `temperature=0` lượt 1. Câu nào code lỗi thì thử lại 2 lượt `temperature=0.35`.
# Self-consistency rẻ hơn nhiều so với mất trắng câu đó, và **executor là trọng
# tài** nên retry không bao giờ làm hỏng một đáp án đã đúng.

# %%
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _CODE_RE.search(text)
    return (m.group(1) if m else text).strip()


# Cache CÓ TRẦN. Bản đầu là dict không giới hạn: nó giữ mọi DataFrame đã đọc,
# mà bài nộp nền có ~5.000 CSV và mỗi câu dùng ~5,3 cái ⇒ tới cuối lượt 0 nó ôm
# gần như toàn bộ. Cộng với vLLM là hết RAM host — đã làm kernel chết ở giây
# 17.007 (4,7 giờ) trong lần chạy 19/08. LRU 192 phần tử đủ để các câu cùng một
# công ty dùng lại nhau, mà bộ nhớ thì có trần.
FRAME_CACHE_MAX = 192
_frame_cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()


def frames_for(rec: dict) -> dict:
    out = {}
    for e in rec["evidence"]:
        f = BASE / e["csv_path"]
        if not f.exists():
            continue
        k = e["csv_path"]
        if k in _frame_cache:
            _frame_cache.move_to_end(k)
        else:
            _frame_cache[k] = pd.read_csv(f)
            while len(_frame_cache) > FRAME_CACHE_MAX:
                _frame_cache.popitem(last=False)
        out[e["variable"]] = _frame_cache[k].copy()
    return out


def ram_gb() -> float:
    """RAM tiến trình đang dùng (GB). Kaggle T4×2 cho ~29 GB, vLLM ăn ~4–6 GB."""
    try:
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except Exception:                                       # noqa: BLE001
        pass
    return float("nan")


def append_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def n_tokens(s: str) -> int:
    return len(tok(s).input_ids)


def build_chat(rec, ir) -> tuple[str, list[dict], str]:
    """→ (chat prompt, evidence THỰC SỰ được đưa vào, cách thu nhỏ).

    `evidence` trả về phải dùng làm frames cho executor: nếu prompt bị bớt bảng
    mà runtime vẫn cấp đủ df thì mô hình có thể tham chiếu dfN nó chưa thấy;
    ngược lại nếu runtime thiếu df mà prompt có thì code chạy lỗi KeyError.
    """
    body, ev, how = build_prompt_fitted(rec, ir, BASE, n_tokens, MAX_LEN - 950)
    chat = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True)
    return chat, ev, how


# Chia lô + ghi dần. Bản đầu gom cả 1.012 prompt vào một lời gọi generate() rồi
# chỉ ghi file ở cell cuối — kernel chết giữa chừng là mất sạch 4,7 giờ. Giờ mỗi
# lô ghi ngay, và chạy lại thì đọc file cũ để bỏ qua câu đã xong.
CHUNK = 128
PATCH_PATH = OUT / "llm_patch.jsonl"
LOG_PATH = OUT / "llm_log.jsonl"

patched: dict[int, dict] = {}
if PATCH_PATH.exists():
    for line in PATCH_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            patched[int(d["id"])] = d
    log(f"chạy tiếp từ lần trước: đã có sẵn {len(patched)} câu trong {PATCH_PATH.name}")

pending = [(rec, ir) for rec, ir in targets if rec["id"] not in patched]
journal: list[dict] = []
shrunk = Counter()

for attempt, temp in enumerate([0.0, 0.35, 0.35]):
    if not pending:
        break
    log(f"── lượt {attempt} (T={temp}) trên {len(pending)} câu ──")
    sp = SamplingParams(temperature=temp, top_p=0.95 if temp else 1.0,
                        max_tokens=800, seed=attempt,
                        stop=["```\n\n", "\n\n\n"])
    still, n_ok, n_none, n_err = [], 0, 0, 0

    for start in range(0, len(pending), CHUNK):
        batch = pending[start:start + CHUNK]
        prompts, keep = [], []
        for rec, ir in batch:
            pr, ev, how = build_chat(rec, ir)
            shrunk[how] += 1
            prompts.append(pr)
            keep.append(({**rec, "evidence": ev}, ir))

        outs = llm.generate(prompts, sp)

        new_patch, new_log = [], []
        for (rec, ir), o in zip(keep, outs):
            code = extract_code(o.outputs[0].text)
            res = run_query(code, frames_for(rec), timeout=8.0)
            if res.ok:
                row = {
                    "id": rec["id"], "answer": res.value, "pandas_query": code,
                    "attempt": attempt, "rule_answer": rec["answer"],
                    # Danh sách CSV ĐÚNG THỨ TỰ mà mô hình đã nhìn thấy khi viết
                    # code. apply_llm.py đối chiếu với bài ở local rồi mới dám
                    # chạy — chốt chặn cho kiểu sai "df1 trỏ sang bảng khác".
                    "evidence": [e["csv_path"] for e in rec["evidence"]],
                }
                patched[rec["id"]] = row
                new_patch.append(row)
                n_ok += 1
                new_log.append({"id": rec["id"], "attempt": attempt, "status": "ok",
                                "answer": res.value, "rule_answer": rec["answer"],
                                "code": code})
            else:
                still.append((rec, ir))
                if "None" in str(res.error) or res.raw is None:
                    n_none += 1
                else:
                    n_err += 1
                new_log.append({"id": rec["id"], "attempt": attempt, "status": "fail",
                                "error": res.error, "code": code[:400]})

        append_jsonl(PATCH_PATH, new_patch)      # ← ghi NGAY, không đợi cell cuối
        append_jsonl(LOG_PATH, new_log)
        journal.extend(new_log)
        del outs, prompts, keep, new_patch, new_log
        gc.collect()
        log(f"   lô {start//CHUNK + 1}/{-(-len(pending)//CHUNK)}: "
            f"nhận {n_ok} · lỗi {n_err} · None {n_none} · RAM {ram_gb():.1f} GB")

    log(f"   lượt {attempt} xong: nhận {n_ok} · lỗi {n_err} · trả None {n_none} "
        f"· còn lại {len(still)}")
    if attempt == 0:
        log(f"   thu nhỏ prompt: {dict(shrunk.most_common())}")
    pending = still

# %% [markdown]
# ## 5 — Ghi kết quả + log toàn hệ thống

# %%
# Ghi đè bằng bản đã gộp: file .jsonl ở trên được nối dần theo lô, nên nếu bạn
# chạy tiếp một session dở thì nó có thể chứa nhiều dòng cho cùng một id. Ở đây
# dedupe theo id (bản mới nhất thắng) để bài nộp gọn.
with (OUT / "llm_patch.jsonl").open("w", encoding="utf-8") as f:
    for v in patched.values():
        f.write(json.dumps(v, ensure_ascii=False) + "\n")

conflict = [(i, v["rule_answer"], v["answer"]) for i, v in patched.items()
            if v["rule_answer"] and abs(v["answer"] - v["rule_answer"])
            > 0.02 * abs(v["rule_answer"])]

log("=" * 72)
log(f"LLM giải được      : {len(patched)}/{len(targets)}  "
    f"({len(patched)/max(len(targets),1):.1%})")
log(f"theo lượt          : {dict(sorted(Counter(v['attempt'] for v in patched.values()).items()))}")
log(f"không giải được    : {len(pending)} (giữ nguyên đáp án tất định)")
log(f"lệch >2% so với rule: {len(conflict)}  ← soi tay nhóm này trước")
for i, ra, la in conflict[:20]:
    log(f"    q{i}: rule={ra!r}  llm={la!r}")
log(f"→ {OUT/'llm_patch.jsonl'}   (mang về local)")
log(f"→ {OUT/'llm_log.jsonl'}     (mọi lần sinh code, kể cả lỗi)")
# %% [markdown]
# ## 6 — Đóng gói `submission.zip` NGAY TẠI ĐÂY
#
# Repo đã clone sẵn trong session, bài nộp nền cũng dựng tại chỗ, nên không có lý
# do gì phải mang `llm_patch.jsonl` về local rồi mới ráp. Cell này làm đúng việc
# `scripts/apply_llm.py` làm, chỉ khác là chạy luôn trên Kaggle:
#
# 1. lấy `submission.json` của bài nền (đầy đủ `relevant_docs`/`relevant_tables`/`evidence`)
# 2. với mỗi câu LLM giải được: **chạy lại code trong sandbox** rồi mới nhận
# 3. câu nào lỗi thì giữ nguyên đáp án tất định
# 4. ghi `submission.json` + `data/*.csv` + zip
#
# Bước 2 là chỗ không được bỏ: LLM chỉ sinh **code**, con số luôn do executor
# chạy ra. Đó là lá chắn chống bịa số, và cũng là thứ giữ Execution Accuracy.
#
# Tải mỗi `submission.zip` ở tab Output là nộp được. Vẫn nên tải kèm
# `llm_log.jsonl` để soi câu hỏng.

# %%
from submit import Record, Submission          # noqa: E402
from execute import verify                     # noqa: E402

SUB_DIR = OUT / "submission"
records = json.loads((BASE / "submission.json").read_text(encoding="utf-8"))

applied = rejected = mismatch = 0
conflicts: list[tuple[int, float, float]] = []
for r in records:
    pt = patched.get(r["id"])
    if not pt:
        continue
    local_ev = [e["csv_path"] for e in r["evidence"]]
    seen_ev = pt.get("evidence")
    if seen_ev is not None and local_ev[:len(seen_ev)] != list(seen_ev):
        mismatch += 1                     # bài nền đã đổi so với lúc sinh code
        continue
    res = verify(pt["pandas_query"], {e["variable"]: e["csv_path"] for e in r["evidence"]},
                 BASE)
    if not res.ok:
        rejected += 1
        continue
    ra = r["answer"]
    r["answer"] = float(res.value)
    r["pandas_query"] = pt["pandas_query"]
    applied += 1
    if ra and abs(res.value - ra) > 0.02 * abs(ra):
        conflicts.append((r["id"], float(ra), float(res.value)))

# Thư mục data/ phải đi cùng: pandas_query chạy trên chính các CSV này.
if SUB_DIR.exists():
    shutil.rmtree(SUB_DIR)
SUB_DIR.mkdir(parents=True)
shutil.copytree(BASE / "data", SUB_DIR / "data")

s = Submission(SUB_DIR)
for r in records:
    s.add(Record(id=r["id"], question=r["question"], answer=r["answer"],
                 relevant_docs=r["relevant_docs"], relevant_tables=r["relevant_tables"],
                 evidence=r["evidence"], pandas_query=r["pandas_query"]))
errs = s.validate()
zpath = s.zip()

log("=" * 72)
log(f"vá {applied} câu · từ chối {rejected} (code chạy lỗi) · "
    f"bỏ qua {mismatch} (evidence lệch bài nền) · lỗi format {len(errs)}")
log(f"lệch >2% so với rule: {len(conflicts)}  ← soi tay nhóm này trước khi nộp")
for qid, ra, la in conflicts[:15]:
    log(f"    q{qid}: rule={ra!r}  llm={la!r}")
for e in errs[:10]:
    log(f"    {e}")
log(f"→ {zpath}   ({zpath.stat().st_size/1e6:.1f} MB)  ← TẢI CÁI NÀY RỒI NỘP")

# %%
log("=" * 72)
log("Ở máy local chạy:")
log("   (chỉ cần nếu bạn muốn vá lên một bài nộp DỰNG Ở LOCAL — ví dụ bài có")
log("    --ce-extra mà bài nền trên Kaggle không có. Còn nếu nộp thẳng zip ở")
log("    cell 6 thì bỏ qua bước này.)")
log(f"   python scripts/apply_llm.py --sub out/<bài dựng bằng {' '.join(BASE_ARGS)}>"
    f" --patch llm_patch.jsonl --out out/v6")
log("   → out/v5/submission.zip   (mỗi câu vá vào đều được chạy lại sandbox ở local)")

# %% [markdown]
# ## Dự phòng nếu vLLM lỗi kernel trên Turing
#
# ```bash
# pip install -q llama-cpp-python \
#   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
# ```
# ```python
# from llama_cpp import Llama
# llm = Llama.from_pretrained("Qwen/Qwen2.5-14B-Instruct-GGUF",
#                             filename="*q4_k_m.gguf",
#                             n_gpu_layers=-1, n_ctx=8192, split_mode=1)
# out = llm.create_chat_completion(
#     [{"role":"system","content":SYSTEM},{"role":"user","content":build_prompt(rec, ir)}],
#     temperature=0.0, max_tokens=800)
# code = extract_code(out["choices"][0]["message"]["content"])
# ```
# Chậm hơn ~3× nhưng cực ổn định. Giữ nguyên SYSTEM và vòng lặp retry ở trên.
#
# **Mô hình thay thế hợp lệ** (đều ≤14B, mã nguồn mở, phát hành trước 01/06/2026):
# `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` · `Qwen/Qwen3-14B-AWQ` ·
# `Qwen/Qwen2.5-7B-Instruct-AWQ` (nhẹ, chạy 1×T4).
