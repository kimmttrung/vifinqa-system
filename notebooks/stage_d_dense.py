# %% [markdown]
# # ViFinQA — Stage D+ : Dense retrieval + Cross-encoder Reranker
#
# **Accelerator : GPU T4 ×1** · **Internet : ON** · thời gian ~60–80 phút
#
# | Vào | Ra |
# |---|---|
# | Kaggle Dataset `vifinqa-bundle` (artifacts + questions + src) | `table_emb.f16.npy` · `rerank_cache.parquet` |
#
# Tải `rerank_cache.parquet` về bỏ vào `artifacts/` ở máy local ⇒ `src/retrieval.py`
# tự bật tầng hybrid RRF, không phải sửa dòng code nào.
#
# ---
# ## Vì sao Qwen3-Embedding-**4B** chứ không phải 8B
# Mentor đo: 4B + reranker = **80,19 %** Recall@10 · 8B + reranker = **80,80 %**.
# Chênh **0,61 %**. Nhưng 8B fp16 = 16 GB, **không vừa 1×T4 (≈15 GB khả dụng)**.
# Trả 2× chi phí cho 0,61 % là lỗ.
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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

T_START = time.time()
IN_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if IN_KAGGLE else Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.time()-T_START:7.1f}s] {msg}", flush=True)


# ── định vị bundle & nạp module (layout phẳng src/) ──
def find(marker: str) -> Path:
    roots = [Path("/kaggle/input")] if IN_KAGGLE else [Path.cwd(), Path.cwd().parent]
    for root in roots:
        if not root.exists():
            continue
        if (root / marker).exists():
            return root
        for d in range(1, 6):
            hits = list(root.glob("/".join(["*"] * d) + "/" + marker))
            if hits:
                return hits[0].parents[len(Path(marker).parts) - 1]
    raise FileNotFoundError(f"Không thấy {marker} trong {roots}")


BUNDLE = find("src/qparse.py")
sys.path.insert(0, str(BUNDLE / "src"))
os.environ.setdefault("VIFINQA_ARTIFACTS", str(find("artifacts/table_meta.parquet") / "artifacts"))

from docfilter import filter_docs           # noqa: E402
from qparse import parse_question           # noqa: E402
from common import ARTIFACTS, load_questions  # noqa: E402

log(f"bundle    : {BUNDLE}")
log(f"artifacts : {ARTIFACTS}")
log(f"GPU       : {torch.cuda.get_device_name(0)} ×{torch.cuda.device_count()}  "
    f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

meta = pd.read_parquet(ARTIFACTS / "table_meta.parquet",
                       columns=["doc_id", "table_idx", "section", "passage"])
meta["row"] = range(len(meta))
questions = load_questions()
log(f"bảng      : {len(meta):,}   câu hỏi: {len(questions)}")

# %% [markdown]
# ## 1a — PREFLIGHT: kiểm tra mạng + model TRƯỚC khi làm việc nặng
#
# **Chạy cell này trước.** Nó mất 30 giây và cứu bạn khỏi việc phát hiện lỗi sau
# 20 phút chờ.
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

def check_net(host="huggingface.co", timeout=8) -> bool:
    try:
        socket.gethostbyname(host)
        urllib.request.urlopen(f"https://{host}", timeout=timeout).read(64)
        return True
    except Exception as e:                                      # noqa: BLE001
        log(f"KHÔNG có mạng tới {host}: {type(e).__name__}: {e}")
        return False


HAS_NET = check_net()
log(f"internet  : {'OK' if HAS_NET else 'CHẶN — bật Internet:On + xác thực SĐT'}")

# Danh sách model theo thứ tự ưu tiên. Pooling KHÁC NHAU giữa các họ — dùng sai
# kiểu pooling làm hỏng toàn bộ chất lượng mà không có dấu hiệu gì.
EMB_CANDIDATES = [
    # (tên, số chiều dùng, kiểu pooling, có instruct prefix ở phía query?)
    ("Qwen/Qwen3-Embedding-4B",      1024, "last",  True),
    ("Qwen/Qwen3-Embedding-0.6B",    1024, "last",  True),
    ("BAAI/bge-m3",                  1024, "cls",   False),
    ("AITeamVN/Vietnamese_Embedding", 1024, "cls",  False),
]


def resolve_model():
    """Thử từng ứng viên, in LỖI THẬT (transformers bọc lại thành OSError chung chung)."""
    from transformers import AutoConfig
    for name, dim, pool, pref in EMB_CANDIDATES:
        try:
            cfg = AutoConfig.from_pretrained(name)
            log(f"  OK   {name}  (hidden={getattr(cfg, 'hidden_size', '?')}, "
                f"pool={pool})")
            return name, dim, pool, pref
        except Exception as e:                                  # noqa: BLE001
            log(f"  FAIL {name}: {type(e).__name__}: {str(e)[:160]}")
    raise RuntimeError(
        "Không tải được model nào. Kiểm tra: (1) Internet: On trong Settings, "
        "(2) đã xác thực số điện thoại Kaggle, (3) transformers đủ mới "
        "— chạy `!pip install -q -U transformers` rồi Restart Session.")


EMB_MODEL, EMB_DIM, EMB_POOL, EMB_PREFIX = resolve_model()
log(f"dùng      : {EMB_MODEL}  dim={EMB_DIM}  pool={EMB_POOL}")

# %% [markdown]
# ## 1b — Embed 146.246 passage
#
# `EMB_DIM = 1024`: với Qwen3-Embedding đây là Matryoshka (cắt 2560 → 1024, gần
# như không mất chất lượng); với bge-m3 thì 1024 đã là chiều gốc. Index 300 MB
# thay vì 748 MB, vừa một Kaggle Dataset.
#
# Passage do `src/passages.py` dựng sẵn: metadata + caption + header + nhãn dòng.
# **Không** embed toàn bộ ô — con số không mang thông tin truy xuất, chỉ gây nhiễu.

# %%
BATCH, MAXLEN = 24, 384
EMB_PATH = OUT / "table_emb.f16.npy"


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
    from transformers import AutoModel, AutoTokenizer

    log(f"tải {EMB_MODEL} …")
    tok = AutoTokenizer.from_pretrained(
        EMB_MODEL, padding_side="left" if EMB_POOL == "last" else "right")
    model = AutoModel.from_pretrained(EMB_MODEL, torch_dtype=torch.float16).cuda().eval()
    log(f"đã tải · VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

    passages = meta["passage"].tolist()
    emb = np.zeros((len(passages), EMB_DIM), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(passages), BATCH):
            b = tok(passages[i:i + BATCH], padding=True, truncation=True,
                    max_length=MAXLEN, return_tensors="pt").to("cuda")
            v = pool_hidden(model(**b).last_hidden_state, b["attention_mask"], EMB_POOL)
            v = torch.nn.functional.normalize(v[:, :EMB_DIM].float(), dim=-1)
            emb[i:i + BATCH] = v.half().cpu().numpy()
            if (i // BATCH) % 250 == 0 and i:
                rate = i / (time.time() - t0)
                log(f"  embed {i:,}/{len(passages):,}  ({rate:.0f} passage/s, "
                    f"ETA {(len(passages)-i)/rate/60:.0f} phút)")
    np.save(EMB_PATH, emb)
    log(f"embed xong: {emb.shape} → {EMB_PATH.stat().st_size/1e6:.0f} MB")
    del model
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 2 — Cross-encoder reranker
#
# Bước ROI cao nhất của cả khâu retrieval: mentor đo **63,90 % → 80,19 %** chỉ nhờ
# thêm rerank. Lý do: bi-encoder nén cả bảng vào MỘT vector, còn cross-encoder đọc
# **đồng thời** câu hỏi và bảng nên bắt được tương tác từ–từ. Đúng thứ cần cho
# "cho vay khách hàng **ngành Thương mại**": phải phân biệt dòng ngành với dòng tổng.

# %%
from transformers import (AutoModel, AutoModelForSequenceClassification,  # noqa: E402
                          AutoTokenizer)

RERANK_CANDIDATES = ["BAAI/bge-reranker-v2-m3",       # 568M, đa ngữ, ổn định sm75
                     "AITeamVN/Vietnamese_Reranker",
                     "BAAI/bge-reranker-base"]

rr_tok = rr_model = None
for name in RERANK_CANDIDATES:
    try:
        log(f"tải {name} …")
        rr_tok = AutoTokenizer.from_pretrained(name)
        rr_model = (AutoModelForSequenceClassification
                    .from_pretrained(name, torch_dtype=torch.float16).cuda().eval())
        RERANK_MODEL = name
        break
    except Exception as e:                                      # noqa: BLE001
        log(f"  FAIL {name}: {type(e).__name__}: {str(e)[:140]}")
if rr_model is None:
    raise RuntimeError("Không tải được reranker nào — xem lại Internet/SĐT ở cell 1a")
log(f"reranker  : {RERANK_MODEL}")

log(f"tải lại {EMB_MODEL} cho phía query …")
q_tok = AutoTokenizer.from_pretrained(
    EMB_MODEL, padding_side="left" if EMB_POOL == "last" else "right")
q_model = AutoModel.from_pretrained(EMB_MODEL, torch_dtype=torch.float16).cuda().eval()
log(f"VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")


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
# ## 3 — Chạy 1.012 câu → `rerank_cache.parquet`
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
# ## 4 — Log tổng kết & đối chiếu với tầng lexical
#
# Không có nhãn gold nên không đo được recall trực tiếp. Nhưng đo được **mức độ
# đồng thuận** giữa hai tầng: nếu dense/rerank trùng BM25 gần hết thì nó không
# thêm thông tin gì và không đáng chạy.

# %%
log("=" * 72)
log(f"câu xử lý            : {cache.qid.nunique()}/{len(questions)}")
log(f"câu không lọc được doc: {n_global}")
log(f"bảng/câu sau rerank   : {len(cache)/max(cache.qid.nunique(),1):.1f}")
log(f"section top-1         : "
    f"{cache[cache['rank']==0].section.value_counts().to_dict()}")
log(f"rerank_score top-1    : trung vị {cache[cache['rank']==0].rerank_score.median():.3f} "
    f"· p10 {cache[cache['rank']==0].rerank_score.quantile(.1):.3f}")

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
log("XONG. Tải 2 file ở Output rồi bỏ vào artifacts/ ở máy local:")
log("   rerank_cache.parquet   ← bắt buộc, bật tầng hybrid RRF")
log("   table_emb.f16.npy      ← tuỳ chọn, để chạy lại không phải embed lần nữa")
