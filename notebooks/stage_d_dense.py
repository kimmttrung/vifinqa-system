# %% [markdown]
# # ViFinQA — Stage D+ : Dense retrieval + Cross-encoder Reranker
#
# **Accelerator : GPU T4 ×1** · **Internet : ON** · một lần chạy ~2–3 giờ
#
# | Vào | Ra |
# |---|---|
# | `git clone` repo (đã kèm `artifacts/` + `questions/`) | `table_emb.f16.npy` · `rerank_cache.parquet` |
#
# **Không cần Kaggle Dataset nào.** Repo đã mang sẵn `artifacts/` nên clone xong là
# chạy được. Chỉ cần bật **Internet: On** (để clone + tải model từ HuggingFace).
#
# Chạy xong: tải `rerank_cache.parquet` ở tab Output, bỏ vào `artifacts/` của repo
# rồi commit ⇒ `src/retrieval.py` tự bật tầng hybrid RRF ở mọi lần chạy sau, cả
# local lẫn Kaggle, không phải sửa dòng code nào.
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
OUT = Path("/kaggle/working") if IN_KAGGLE else Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.time()-T_START:7.1f}s] {msg}", flush=True)


def get_repo() -> Path:
    """Clone repo trên Kaggle · dò ngược lên cây thư mục khi chạy local.

    `--depth 1` là bắt buộc: repo mang theo artifacts/ (~106 MB/bản), clone đủ
    lịch sử sẽ tải mọi phiên bản artifacts từng commit.
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


REPO = get_repo()
sys.path.insert(0, str(REPO / "src"))
# Trỏ THẲNG vào repo thay vì để common.py tự dò: trên Kaggle có thể còn Dataset cũ
# đính kèm, và hàm dò quét /kaggle/input TRƯỚC nên sẽ nhặt nhầm artifacts cũ.
os.environ["VIFINQA_ARTIFACTS"] = str(REPO / "artifacts")
os.environ["VIFINQA_QUESTIONS"] = str(REPO / "questions" / "questions.jsonl")

from docfilter import filter_docs           # noqa: E402
from qparse import parse_question           # noqa: E402
from common import ARTIFACTS, load_questions  # noqa: E402

need = ["table_meta.parquet", "docs.parquet", "facts.parquet",
        "bm25.tf.npz", "bm25.meta.npz", "bm25.vocab.json"]
miss = [n for n in need if not (ARTIFACTS / n).exists()]
if miss:
    raise FileNotFoundError(
        f"Thiếu artifacts trong repo: {miss}\n"
        "artifacts/ phải được commit lên GitHub (xem README §Chạy trên Kaggle).")

log(f"repo      : {REPO}")
log(f"artifacts : {ARTIFACTS}")
log(f"GPU       : {torch.cuda.get_device_name(0)} ×{torch.cuda.device_count()}  "
    f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

meta = pd.read_parquet(ARTIFACTS / "table_meta.parquet",
                       columns=["doc_id", "table_idx", "section", "passage"])
meta["row"] = range(len(meta))
questions = load_questions()
log(f"bảng      : {len(meta):,}   câu hỏi: {len(questions)}")

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
# `EMB_DIM = 1024`: với Qwen3-Embedding đây là Matryoshka (cắt 2560 → 1024, gần
# như không mất chất lượng); index còn 300 MB thay vì 748 MB, vừa một Kaggle Dataset.

# %%
EMB_MODEL = "Qwen/Qwen3-Embedding-4B"    # ← đổi sang "Qwen/Qwen3-Embedding-0.6B" nếu cần nhanh
EMB_POOL = "last"                        # Qwen3-Embedding: last-token pooling
EMB_PREFIX = True                        # Qwen3-Embedding: instruct prefix ở PHÍA QUERY
EMB_DIM = 1024                           # Matryoshka: cắt từ hidden_size gốc

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # 568M, đa ngữ, chạy ổn trên sm75

BATCH, MAXLEN = 24, 384
EMB_PATH = OUT / "table_emb.f16.npy"

log(f"embedder  : {EMB_MODEL}  (pool={EMB_POOL}, dim={EMB_DIM})")
log(f"reranker  : {RERANK_MODEL}")

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


if EMB_PATH.exists():
    emb = np.load(EMB_PATH)
    log(f"dùng lại embedding có sẵn: {emb.shape}")
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
    log(f"embed xong: {emb.shape} → {EMB_PATH.stat().st_size/1e6:.0f} MB")

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

    qv = embed_query(ir.retrieval_query)
    sims = emb_f32[rows] @ qv
    take = np.argsort(-sims)[:K_DENSE]
    cand = [int(rows[i]) for i in take]
    dense_score = {int(rows[i]): float(sims[i]) for i in take}

    scores = rerank(ir.retrieval_query, [passages_all[r] for r in cand])
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
log(f"rerank_score top-1    : trung vị {cache[cache['rank']==0].rerank_score.median():.3f} "
    f"· p10 {cache[cache['rank']==0].rerank_score.quantile(.1):.3f}")

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
