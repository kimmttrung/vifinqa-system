# %% [markdown]
# # ViFinQA — Stage D+ : Dense retrieval + Cross-encoder Reranker
#
# **Accelerator : GPU T4 ×1** · **Internet : ON** · một lần chạy ~2–3 giờ
#
# ## INPUT — notebook tự lấy, bạn không phải chuẩn bị gì
#
# | Thứ | Lấy từ đâu | Bắt buộc |
# |---|---|---|
# | code (`src/*.py`) | `git clone` repo | có |
# | `artifacts/table_meta.parquet` (146.246 bảng + passage) | repo, **hoặc** Kaggle Dataset đính kèm | có |
# | `artifacts/{docs,facts}.parquet` · `bm25.*` | cùng nguồn với trên | có |
# | `questions/questions.jsonl` (1.012 câu) | repo, hoặc Kaggle Dataset | có |
# | `artifacts/table_emb.f16.npy` | lần chạy trước | không — có thì bỏ qua khâu embed |
# | 2 model HuggingFace | tải qua Internet | có |
#
# **Artifacts có hai nguồn.** Ưu tiên `artifacts/` trong repo. Nếu repo không kèm
# được (106 MB, có thể push hỏng) thì vào **+ Add Input** → chọn Dataset chứa
# artifacts; notebook tự dò trong `/kaggle/input` và in ra nó đang đọc nguồn nào.
# Không phải sửa dòng code nào.
#
# ## OUTPUT — nằm ở tab Output, `/kaggle/working/`
#
# | File | Kích thước | Làm gì với nó |
# |---|---|---|
# | `rerank_cache.parquet` | ~1 MB | **kết quả chính**: bỏ vào `artifacts/` rồi commit ⇒ `src/retrieval.py` tự bật hybrid RRF ở mọi lần chạy sau |
# | `table_emb.f16.npy` | ~300 MB | tuỳ chọn: giữ lại để lần sau khỏi embed. **Đừng commit** — GitHub chặn file >100 MB |
#
# Notebook **không** sinh submission. Bài nộp vẫn dựng ở local (hoặc ở Stage E)
# bằng `run_pipeline.py` sau khi đã có `rerank_cache.parquet`.
#
# Chỉ cần bật **Internet: On** (để `git clone` + tải model từ HuggingFace).
#
# ---
# ## Hai model, cố định — không dò, không fallback
#
# | Vai trò | Model | VRAM fp16 |
# |---|---|---|
# | Bi-encoder | `Qwen/Qwen3-Embedding-4B` | ~8,0 GB |
# | Cross-encoder | `BAAI/bge-reranker-v2-m3` | ~1,2 GB |
#
# Cả hai nạp đồng thời ≈ 9,2 GB, vừa 1×T4 (≈15 GB khả dụng).
#
# **4B chứ không phải 8B**: mentor đo 4B + reranker = 80,19 % Recall@10 · 8B +
# reranker = 80,80 %. Chênh 0,61 %, nhưng 8B fp16 = 16 GB **không vừa 1×T4**.
#
# **Muốn chạy nhanh gấp ~5 lần**: đổi `EMB_MODEL` sang `Qwen/Qwen3-Embedding-0.6B`
# ở cell hằng số bên dưới — cùng họ nên pooling và prefix giữ nguyên, chỉ đổi một
# dòng. Đánh đổi chất lượng chưa đo trên corpus này.
#
# ## Vì sao KHÔNG dùng FAISS
# Pipeline luôn lọc metadata trước: 1.973 doc → trung vị 2 doc → vài trăm bảng.
# So vector với vài trăm hàng là một phép nhân ma trận nhỏ. FAISS chỉ đáng khi
# phải quét cả 146k — điều không bao giờ xảy ra ở đây.
#
# ## Ràng buộc T4 (sm75) — đọc trước khi đổi tham số
# Không bf16 · không FlashAttention-2 · không Marlin kernel. Bắt buộc `torch.float16`.

# %%
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_URL = "https://github.com/kimmttrung/vifinqa-system.git"
REPO_BRANCH = "main"

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

from docfilter import filter_docs           # noqa: E402
from qparse import parse_question           # noqa: E402
from common import ARTIFACTS, load_code_stock, load_questions  # noqa: E402

OUT = Path("/kaggle/working") if IN_KAGGLE else ARTIFACTS
OUT.mkdir(parents=True, exist_ok=True)

log(f"repo      : {REPO}")
log(f"artifacts : {ARTIFACTS}   ← {ARTI_SRC}")
log(f"questions : {QUES}")
log(f"output    : {OUT}")
log(f"GPU       : {torch.cuda.get_device_name(0)} ×{torch.cuda.device_count()}  "
    f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

# Cột `row` ĐỌC TỪ parquet, không tự đánh lại số. retrieval.py khoá rerank_cache
# theo đúng cột này; tự đánh lại thì hôm nào build_index đổi thứ tự là lệch âm
# thầm — cache vẫn nạp được, chỉ trỏ sang bảng khác.
meta = pd.read_parquet(ARTIFACTS / "table_meta.parquet",
                       columns=["doc_id", "table_idx", "section", "passage", "row"])
assert (meta["row"].to_numpy() == np.arange(len(meta))).all(), \
    "cột row trong table_meta.parquet không còn khớp thứ tự dòng"
questions = load_questions()
log(f"bảng      : {len(meta):,}   câu hỏi: {len(questions)}   "
    f"mã CK: {len(load_code_stock())}")

# %% [markdown]
# ## 1 — Hằng số model
#
# `EMB_POOL` và `EMB_PREFIX` đi LIỀN với `EMB_MODEL`, đổi model thì phải đổi kèm:
#
# | Họ model | pooling | padding | instruct prefix ở query |
# |---|---|---|---|
# | Qwen3-Embedding | `last` | left | **có** |
# | BGE-M3 / Vietnamese_Embedding | `cls` | right | không |
#
# Dùng sai kiểu pooling **không báo lỗi** — nó chỉ làm chất lượng tụt âm thầm.
#
# ### `QUERY_TEXT = "raw"` — câu hỏi nguyên văn, không phải `retrieval_query`
#
# `ir.retrieval_query` được thiết kế cho **BM25**: `fold()` bỏ dấu, viết hoa, bỏ mã
# CK, bỏ tên công ty, bỏ năm. Với BM25 thì đúng (OCR sai dấu liên tục, tên công ty
# có trong caption của mọi bảng nên không phân biệt được gì). Nhưng đưa chuỗi đó
# cho model neural là **sai miền dữ liệu**:
#
# | | Văn bản đưa vào |
# |---|---|
# | Câu hỏi gốc | `Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?` |
# | `retrieval_query` | `LAI TIEN GUI NAM HANG` |
#
# Mất dấu, mất năm, `Hàng không` teo còn `HANG`. Đo trên `rerank_cache.parquet` của
# lần chạy đầu: logit top-1 trung vị **−4,09** — tức cross-encoder cho rằng ứng viên
# TỐT NHẤT vẫn là không liên quan. RRF chỉ dùng thứ hạng nên không sập, nhưng thứ
# hạng đó kém hơn nhiều so với khả năng thật của model.
#
# Tầng lexical trong `src/retrieval.py` **vẫn dùng `retrieval_query`** — không đổi.
# Chỉ hai tầng neural ở notebook này đổi sang câu hỏi thô.
#
# `EMB_DIM = 1024`: với Qwen3-Embedding đây là Matryoshka (cắt 2560 → 1024, gần
# như không mất chất lượng); index còn 300 MB thay vì 748 MB, vừa một Kaggle Dataset.

# %%
EMB_MODEL = "Qwen/Qwen3-Embedding-4B"    # ← đổi sang "Qwen/Qwen3-Embedding-0.6B" nếu cần nhanh
EMB_POOL = "last"                        # Qwen3-Embedding: last-token pooling
EMB_PREFIX = True                        # Qwen3-Embedding: instruct prefix ở PHÍA QUERY
EMB_DIM = 1024                           # Matryoshka: cắt từ hidden_size gốc

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # 568M, đa ngữ, chạy ổn trên sm75

# "raw"  = nguyên văn câu hỏi, CÒN DẤU        ← khuyên dùng cho model neural
# "bm25" = ir.retrieval_query (đã fold, bỏ dấu, bỏ tên công ty & năm)
QUERY_TEXT = "raw"

BATCH, MAXLEN = 24, 384
EMB_PATH = OUT / "table_emb.f16.npy"
EMB_SIDECAR = OUT / "table_emb.meta.json"

log(f"embedder  : {EMB_MODEL}  (pool={EMB_POOL}, dim={EMB_DIM})")
log(f"reranker  : {RERANK_MODEL}")
log(f"query     : {QUERY_TEXT}")

# %% [markdown]
# ## 1a — PREFLIGHT: kiểm tra mạng + hai model TRƯỚC khi làm việc nặng
#
# **Chạy cell này trước.** Nó tải vài KB `config.json` (không phải weights), mất
# ~30 giây, và cứu bạn khỏi việc phát hiện lỗi sau 20 phút chờ.
#
# > ### Bẫy Kaggle phải biết
# > Bật `Internet: On` trong Settings **đòi xác thực số điện thoại**. Chưa xác
# > thực thì công tắc bật được nhưng mạng vẫn chặn, và `from_pretrained` báo
# > `OSError: Can't load the configuration of '...'` — thông báo này gây hiểu
# > nhầm là sai tên model, thực ra là không có mạng.
# > Xác thực tại: Kaggle → Settings → Phone Verification.

# %%
import socket
import urllib.request

from transformers import AutoConfig


def check_net(host="huggingface.co", timeout=8) -> bool:
    try:
        socket.gethostbyname(host)
        urllib.request.urlopen(f"https://{host}", timeout=timeout).read(64)
        return True
    except Exception as e:                                      # noqa: BLE001
        log(f"KHÔNG có mạng tới {host}: {type(e).__name__}: {e}")
        return False


if not check_net():
    raise RuntimeError(
        "Không có internet. Settings → Internet: On, và phải đã xác thực số "
        "điện thoại Kaggle. Bật xong nhớ Restart Session.")
log("internet  : OK")

for name in (EMB_MODEL, RERANK_MODEL):
    # In lỗi THẬT: transformers bọc mọi thứ thành OSError chung chung, đọc vào
    # tưởng sai tên model trong khi thực ra là mạng hoặc transformers quá cũ.
    try:
        cfg = AutoConfig.from_pretrained(name)
        log(f"  OK   {name}  (hidden={getattr(cfg, 'hidden_size', '?')})")
    except Exception as e:                                      # noqa: BLE001
        raise RuntimeError(
            f"Không đọc được config của {name}: {type(e).__name__}: {e}\n"
            "Nếu lỗi nhắc tới kiến trúc lạ: !pip install -q -U transformers "
            "rồi Restart Session.") from e

# %% [markdown]
# ## 2 — Embed 146.246 passage
#
# Passage do `src/passages.py` dựng sẵn: metadata + caption + header + nhãn dòng.
# **Không** embed toàn bộ ô — con số không mang thông tin truy xuất, chỉ gây nhiễu.
#
# Model nạp **một lần** rồi giữ nguyên trong VRAM để dùng tiếp cho phía query ở
# cell 3 — nạp lại 8 GB weights lần nữa là phí vài phút mà không được gì.

# %%
from transformers import AutoModel, AutoTokenizer  # noqa: E402

_emb_bundle = None


def load_embedder():
    """Nạp bi-encoder một lần duy nhất cho cả passage lẫn query."""
    global _emb_bundle
    if _emb_bundle is None:
        log(f"tải {EMB_MODEL} …")
        tok = AutoTokenizer.from_pretrained(
            EMB_MODEL, padding_side="left" if EMB_POOL == "last" else "right")
        model = AutoModel.from_pretrained(
            EMB_MODEL, torch_dtype=torch.float16).cuda().eval()
        _emb_bundle = (tok, model)
        log(f"đã tải · VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return _emb_bundle


def pool_hidden(h, mask, how: str):
    """`last` cho họ Qwen3-Embedding · `cls` cho họ BGE. Không hoán đổi được."""
    if how == "cls":
        return h[:, 0]
    if mask[:, -1].all():
        return h[:, -1]
    idx = mask.sum(dim=1) - 1
    return h[torch.arange(h.size(0), device=h.device), idx]


def find_cached_emb() -> Path | None:
    """table_emb.f16.npy của lần chạy trước: /kaggle/working → artifacts/ → Dataset.

    Kaggle xoá sạch /kaggle/working giữa các session, nên nếu chỉ nhìn ở đó thì
    LẦN NÀO CŨNG phải embed lại 2–3 giờ. File 285 MB không commit lên GitHub được
    (giới hạn 100 MB), nên đường tái dùng thật sự là up nó thành Kaggle Dataset
    rồi + Add Input.
    """
    for c in (EMB_PATH, ARTIFACTS / "table_emb.f16.npy"):
        if c.exists():
            return c
    return find_under(KAGGLE_INPUT, "table_emb.f16.npy") if IN_KAGGLE else None


def emb_is_usable(path: Path) -> bool:
    """Kiểm shape + model đã sinh ra nó. Dùng nhầm embedding của model khác thì
    cosine vẫn tính được, không lỗi gì cả — chỉ là kết quả rác."""
    shape = np.load(path, mmap_mode="r").shape
    if shape != (len(meta), EMB_DIM):
        log(f"  bỏ qua {path}: shape {shape} ≠ {(len(meta), EMB_DIM)}")
        return False
    side = path.with_name("table_emb.meta.json")
    if side.exists():
        info = json.loads(side.read_text(encoding="utf-8"))
        if info.get("model") != EMB_MODEL:
            log(f"  bỏ qua {path}: sinh bởi {info.get('model')}, không phải {EMB_MODEL}")
            return False
        return True
    log(f"  {path} không có sidecar → không kiểm được model, tin theo shape")
    return True


_cached = find_cached_emb()
if _cached and emb_is_usable(_cached):
    emb = np.load(_cached)
    log(f"dùng lại embedding có sẵn: {emb.shape}  ({_cached})")
else:
    tok, model = load_embedder()
    passages = meta["passage"].tolist()
    emb = np.zeros((len(passages), EMB_DIM), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(passages), BATCH):
            b = tok(passages[i:i + BATCH], padding=True, truncation=True,
                    max_length=MAXLEN, return_tensors="pt").to("cuda")
            v = pool_hidden(model(**b).last_hidden_state, b["attention_mask"], EMB_POOL)
            # Cắt Matryoshka TRƯỚC rồi mới chuẩn hoá — đảo thứ tự là vector không
            # còn norm 1 và cosine sai lệch.
            v = torch.nn.functional.normalize(v[:, :EMB_DIM].float(), dim=-1)
            emb[i:i + BATCH] = v.half().cpu().numpy()
            if (i // BATCH) % 250 == 0 and i:
                rate = i / (time.time() - t0)
                log(f"  embed {i:,}/{len(passages):,}  ({rate:.0f} passage/s, "
                    f"ETA {(len(passages)-i)/rate/60:.0f} phút)")
    np.save(EMB_PATH, emb)
    EMB_SIDECAR.write_text(json.dumps(
        {"model": EMB_MODEL, "dim": EMB_DIM, "rows": len(emb), "maxlen": MAXLEN},
        ensure_ascii=False), encoding="utf-8")
    log(f"embed xong: {emb.shape} → {EMB_PATH.stat().st_size/1e6:.0f} MB")
    log(f"  sidecar {EMB_SIDECAR.name}: tải VỀ CÙNG file .npy, nó là thứ chặn việc"
        f" dùng nhầm embedding của model khác")

# %% [markdown]
# ## 3 — Cross-encoder reranker
#
# Bước ROI cao nhất của cả khâu retrieval: mentor đo **63,90 % → 80,19 %** chỉ nhờ
# thêm rerank. Lý do: bi-encoder nén cả bảng vào MỘT vector, còn cross-encoder đọc
# **đồng thời** câu hỏi và bảng nên bắt được tương tác từ–từ. Đúng thứ cần cho
# "cho vay khách hàng **ngành Thương mại**": phải phân biệt dòng ngành với dòng tổng.

# %%
from transformers import AutoModelForSequenceClassification  # noqa: E402

log(f"tải {RERANK_MODEL} …")
rr_tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
rr_model = (AutoModelForSequenceClassification
            .from_pretrained(RERANK_MODEL, torch_dtype=torch.float16).cuda().eval())

q_tok, q_model = load_embedder()      # đã nằm sẵn trong VRAM nếu vừa embed xong
log(f"hai model đã sẵn sàng · VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")


@torch.inference_mode()
def embed_query(q: str) -> np.ndarray:
    # Họ Qwen3-Embedding yêu cầu instruct prefix ở PHÍA QUERY (không phải phía
    # passage). Họ BGE thì không dùng prefix — thêm vào sẽ làm lệch phân bố.
    text = (("Instruct: Tìm bảng biểu báo cáo tài chính chứa chỉ tiêu được hỏi\n"
             f"Query: {q}") if EMB_PREFIX else q)
    b = q_tok([text], truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
    h = pool_hidden(q_model(**b).last_hidden_state, b["attention_mask"], EMB_POOL)
    return torch.nn.functional.normalize(h[:, :EMB_DIM].float(), dim=-1)[0].cpu().numpy()


@torch.inference_mode()
def rerank(query: str, passages: list[str], batch: int = 12) -> np.ndarray:
    out = []
    for i in range(0, len(passages), batch):
        chunk = passages[i:i + batch]
        b = rr_tok([query] * len(chunk), chunk, padding=True, truncation=True,
                   max_length=512, return_tensors="pt").to("cuda")
        out.append(rr_model(**b).logits.view(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

# %% [markdown]
# ## 4 — Chạy 1.012 câu → `rerank_cache.parquet`
#
# Luồng: **lọc metadata → dense top-50 → rerank → top-10**.
# Lọc metadata trước là bắt buộc — nó thu 146k bảng xuống vài trăm, nếu không
# cross-encoder sẽ phải chấm 146k cặp/câu (bất khả thi).

# %%
K_DENSE, K_KEEP = 50, 10
rows_by_doc = {d: g["row"].to_numpy() for d, g in meta.groupby("doc_id", sort=False)}
passages_all = meta["passage"].tolist()
emb_f32 = emb.astype(np.float32)

records, t0 = [], time.time()
n_global = 0
for n, q in enumerate(questions, 1):
    ir = parse_question(q["id"], q["question"])
    docs = filter_docs(ir)
    parts = [rows_by_doc[d] for d in docs if d in rows_by_doc]
    if parts:
        rows = np.concatenate(parts)
    else:
        rows = np.arange(len(meta))       # câu không nêu công ty nào
        n_global += 1

    qtext = q["question"] if QUERY_TEXT == "raw" else ir.retrieval_query
    qv = embed_query(qtext)
    sims = emb_f32[rows] @ qv
    take = np.argsort(-sims)[:K_DENSE]
    cand = [int(rows[i]) for i in take]
    dense_score = {int(rows[i]): float(sims[i]) for i in take}

    scores = rerank(qtext, [passages_all[r] for r in cand])
    order = np.argsort(-scores)[:K_KEEP]
    for rank, i in enumerate(order):
        r = cand[int(i)]
        records.append({
            "qid": q["id"], "row": r, "rank": rank,
            "rerank_score": float(scores[int(i)]),
            "dense_score": dense_score[r],
            "dense_rank": cand.index(r),
            "doc_id": meta.doc_id.iloc[r], "table_idx": int(meta.table_idx.iloc[r]),
            "section": meta.section.iloc[r],
        })
    if n % 25 == 0:
        rate = n / (time.time() - t0)
        log(f"  rerank {n}/{len(questions)}  ({rate:.2f} câu/s, "
            f"ETA {(len(questions)-n)/rate/60:.0f} phút)")

cache = pd.DataFrame(records)
cache.to_parquet(OUT / "rerank_cache.parquet", compression="zstd", index=False)
log(f"ghi {OUT/'rerank_cache.parquet'}  ({len(cache):,} dòng)")

# %% [markdown]
# ## 5 — Log tổng kết & đối chiếu với tầng lexical
#
# Không có nhãn gold nên không đo được recall trực tiếp. Nhưng đo được **mức độ
# đồng thuận** giữa hai tầng: nếu dense/rerank trùng BM25 gần hết thì nó không
# thêm thông tin gì và không đáng chạy.

# %%
log("=" * 72)
log(f"embedder             : {EMB_MODEL}")
log(f"reranker             : {RERANK_MODEL}")
log(f"câu xử lý            : {cache.qid.nunique()}/{len(questions)}")
log(f"câu không lọc được doc: {n_global}")
log(f"bảng/câu sau rerank   : {len(cache)/max(cache.qid.nunique(),1):.1f}")
log(f"section top-1         : "
    f"{cache[cache['rank']==0].section.value_counts().to_dict()}")
_med = cache[cache['rank'] == 0].rerank_score.median()
log(f"rerank_score top-1    : trung vị {_med:.3f} "
    f"· p10 {cache[cache['rank']==0].rerank_score.quantile(.1):.3f}")
log("   logit của bge-reranker: >0 nghĩa là 'liên quan'. Trung vị top-1 mà ÂM SÂU"
    " (< −3) là dấu hiệu câu truy vấn sai miền — kiểm lại QUERY_TEXT."
    if _med < -3 else "   trung vị top-1 dương ⇒ câu truy vấn đúng miền dữ liệu")

# Nhả VRAM trước khi đối chiếu — phần dưới chỉ chạy CPU. Viết theo kiểu chạy lại
# được nhiều lần: `del` thẳng sẽ NameError ở lần chạy thứ hai, còn đặt lại
# _emb_bundle = None để load_embedder() nạp lại sạch nếu cần quay lên cell trên.
globals().pop("rr_model", None)
globals().pop("q_model", None)
_emb_bundle = None
gc.collect()
torch.cuda.empty_cache()

try:
    from retrieval import retrieve
    agree = []
    for q in questions[:150]:
        ir = parse_question(q["id"], q["question"])
        lex = {(h.doc_id, h.table_idx) for h in retrieve(ir, k_read=10)}
        ce = {(r.doc_id, int(r.table_idx))
              for r in cache[cache.qid == q["id"]].itertuples()}
        if lex and ce:
            agree.append(len(lex & ce) / len(lex | ce))
    log(f"Jaccard(lexical top10, rerank top10) trên 150 câu: {np.mean(agree):.3f}")
    log("  <0.35 ⇒ hai tầng bổ sung nhau mạnh, RRF sẽ có lãi lớn")
    log("  >0.70 ⇒ trùng lặp nhiều, lợi ích của rerank hạn chế")
except Exception as e:                                          # noqa: BLE001
    log(f"bỏ qua đối chiếu lexical: {type(e).__name__}: {e}")

log("=" * 72)
log("XONG. Tải file ở tab Output về máy local:")
log("   rerank_cache.parquet → artifacts/ rồi COMMIT ⇒ hybrid RRF bật ở mọi nơi")
log("   table_emb.f16.npy    → artifacts/ nhưng ĐỪNG commit (300 MB, GitHub chặn")
log("                          file >100 MB; .gitignore đã chặn sẵn). Chỉ để chạy")
log("                          lại notebook này mà không phải embed lần nữa.")
