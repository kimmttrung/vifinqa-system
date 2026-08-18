# ViFinQA — Financial Table Retrieval & Text-to-Pandas Query Generation

Hệ thống trả lời 1.012 câu hỏi tài chính tiếng Việt trên kho **1.973 báo cáo tài chính
OCR** của 100 công ty niêm yết Việt Nam (2015–2025), sinh ra `submission.json` gồm
đáp án số, tài liệu/bảng chứng cứ và câu lệnh `pandas` tái lập được đáp án đó.

Cuộc thi: **Financial Table Retrieval & Text-to-Pandas Query Generation (ViFinQA)** —
AI Guru / CTCP Tập đoàn Dagoras Group.

---

## Nguyên tắc thiết kế

> **LLM sinh CODE, executor sinh SỐ.**

Không con số nào trong `answer` được mô hình "viết ra". Mọi giá trị đều do sandbox
rút từ CSV đi kèm bằng đúng `pandas_query` đi kèm. Nhờ vậy **bất kỳ ai cũng kiểm
chứng lại được toàn bộ bài nộp** mà không cần chạy pipeline:

```bash
python scripts/verify_submission.py out/v4/submission.zip
```

| Tầng kiểm | Kết quả đo trên bài nộp `v4` |
|---|---|
| Hình thức hợp lệ | PASS |
| `pandas_query` chạy được | **1012/1012 = 100%** |
| Kết quả khớp `answer` | **1012/1012 = 100%** |
| `doc\|line_no` trỏ đúng thẻ `<table>` có thật | **98,53%** trên 7.879 tham chiếu |

---

## Kết quả trên leaderboard công khai

| Bài nộp | Thay đổi | TABLES_F₂ | DOCS_F₂ | ANSWER | EXEC |
|---|---|---|---|---|---|
| `probe1_all_folder` | 4 cách mã hoá vị trí, k=1 | 0,1331 | 0,5929 | 0,1660 | 0,1660 |
| `probe2_all_stem` | `doc_id` có `_extracted` | 0,0 | 0,0 | 0,1660 | 0,1660 |
| `probe_table_idx` | vị trí = số thứ tự bảng | 0,0 | 0,8486 | 0,1660 | 0,1660 |
| **`probe_line_no`** | **vị trí = số dòng** | **0,3147** | 0,8486 | 0,1660 | 0,1660 |
| `probe_page_no` | vị trí = số trang | 0,0 | 0,8486 | 0,1660 | 0,1660 |
| `k6` | k=6 | 0,4137 | 0,8486 | 0,1660 | 0,1660 |
| `k10` | k=10 | 0,3999 | 0,8486 | 0,1660 | 0,1660 |
| `k6_solver` | + bộ giải đa ô | 0,4100 | 0,8394 | 0,2352 | 0,2352 |
| `v3` | + tỷ số "A trên B", veto trái nghĩa | 0,4098 | 0,8376 | 0,2411 | 0,2411 |
| **`v4`** | **+ k thích ứng theo câu** | **0,4267** | 0,8376 | 0,2411 | 0,2411 |

Chi tiết từng lượt và lý do thay đổi: [CLAUDE.md](CLAUDE.md) §10.

---

## Cài đặt

```bash
git clone <repo-url> && cd vifinqa-system
python -m venv venv && venv/Scripts/activate      # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Yêu cầu **Python ≥ 3.11**. Đường tất định (Stage A→G) chạy **CPU thuần** — không cần
GPU, không cần internet, không tải model nào.

Dataset ViFinQA (363 MiB) nằm **ngoài** repo. Trỏ tới nó bằng biến môi trường:

```bash
export VIFINQA_DATA=/duong/dan/toi/ViFinQA     # thư mục chứa financial_statements/
```

Nếu dataset là thư mục anh em của repo thì hệ thống tự tìm ra, không cần đặt biến.

---

## Chạy end-to-end

```bash
python scripts/build_corpus.py        # Stage A   ~5 phút  → artifacts/tables.parquet
python scripts/build_index.py         # Stage D   ~45 giây → artifacts/bm25.*
python scripts/build_facts.py --notes # Stage B   ~30 giây → artifacts/facts.parquet
python scripts/run_pipeline.py --tag v4 --k-max 6 --adaptive-k --trace
```

Tổng ~12 phút trên một core, ra `out/v4/submission.zip`.

Ba bước đầu chỉ chạy một lần; sau đó vòng lặp thử–sai chỉ còn bước cuối (~6 phút).

### Kiểm chứng trước khi nộp

```bash
python scripts/verify_submission.py out/v4/submission.zip --report out/v4/VERIFY.md
```

### Soi một câu hỏi cụ thể

```bash
python scripts/show_trace.py --trace out/v4/trace.jsonl --qid 2
python scripts/show_trace.py --trace out/v4/trace.jsonl --stats
```

In ra toàn bộ tín hiệu: `bm25 · cover · phrase · period · section · lex_score ·
ce_rank · rerank_score · rrf_score`, đánh dấu bảng nào được nộp và vì sao.

### Chạy test

```bash
pip install -r requirements-dev.txt
pytest                    # 103 test; test cần corpus tự bỏ qua nếu thiếu
```

---

## Kiến trúc

```
questions.jsonl
      │
      ├─[C] qparse ────► QueryIR {ticker, năm, riêng/hợp nhất, đơn vị, mốc, phép}
      │    companies       5 tín hiệu quyết định đúng/sai, đều rút bằng rule
      │
      ├─[B'] docfilter ─► 1.973 doc → trung vị 2 doc  (metadata thuần, không embedding)
      │
      ├─[D] retrieval ──► BM25 + 5 tín hiệu cấu trúc ─┐
      │    (+ dense/rerank từ Kaggle, hợp nhất bằng RRF)
      │                                               │
      ├─[B] factlookup ─► tra thẳng 1,54 M fact ──────┤
      │                                               ▼
      ├─ solve ─────────► RATIO · DIFF · GROWTH · ARGMAX_YEAR · SUM · AVG · MAX · MIN
      │
      ├─[F] execute ────► sandbox: chạy code, đối chiếu số
      │
      └─[G] emit/submit ► data/*.csv + submission.json + ZIP
```

| Stage | Vai trò | GPU | Trạng thái |
|---|---|---|---|
| **A** `corpus` | 1.973 file .txt → 146.246 bảng | — | ✅ |
| **B** `facts` | 1.536.188 fact, 4 adapter kế toán | — | ✅ |
| **C** `qparse` | câu hỏi → QueryIR | — | ✅ |
| **D** `bm25`/`retrieval` | truy xuất bảng | — | ✅ |
| **D+** dense + reranker | Qwen3-Embedding-4B + bge-reranker | T4 ×1 | notebook |
| **E** planner PoT | Qwen2.5-14B-AWQ + vLLM | T4 ×2 | notebook |
| **F** `execute` | sandbox | — | ✅ |
| **G** `emit`/`submit` | đóng gói bài nộp | — | ✅ |

Phần GPU: xem [KAGGLE.md](KAGGLE.md).

### Bốn adapter kế toán, không phải hai

| Adapter | Đối tượng | Khoá nhận diện |
|---|---|---|
| TT200 | ~70 doanh nghiệp | cột **Mã số** (100/270/440; 10/20/60) |
| TCTD | 21 ngân hàng + EVF | **nhãn dòng** — ngân hàng KHÔNG có cột Mã số |
| TT334 | CTCK (SSI/MBS/FTS) | Mã số riêng |
| TT232 | bảo hiểm (BVH) | Mã số riêng |

Auto-QC bằng ràng buộc kế toán — đây là bộ đo chất lượng duy nhất có được, vì cuộc
thi không phát nhãn gold:

| Ràng buộc | Đạt | n |
|---|---|---|
| `270 == 440` | 99,3% | 2.815 |
| `100 + 200 == 270` | 99,1% | 2.959 |
| `300 + 400 == 440` | 98,6% | 2.802 |
| đầu năm Y == cuối năm Y−1 | **94,4%** | 39.132 |

---

## Bố cục

```
src/     17 module top-level, phẳng — không có package con
  common.py             fold() · parse_number() · html_to_grid() · đường dẫn
  corpus.py             Stage A: .txt → lưới bảng + metadata
  companies.py          câu hỏi → mã CK (IDF-coverage CÓ THỨ TỰ + ~120 alias)
  qparse.py             Stage C: QueryIR
  docfilter.py          lọc doc bằng metadata
  bm25.py               BM25 thưa (scipy CSR) tự viết
  passages.py           bảng → đoạn text để index
  retrieval.py          Stage D + RRF + select_submit tối ưu F₂
  facts.py              Stage B: 4 adapter → fact table
  factlookup.py         fast-path tra cứu trên fact table
  locate.py             cell grounding một ô + sinh pandas_query
  solve.py              bộ giải đa ô + gating dương
  prompt.py             prompt PoT (tách khỏi notebook để test được ở local)
  execute.py            Stage F sandbox
  emit.py submit.py grids.py   Stage G

scripts/                CLI — chỉ 4 lệnh cần dùng thường xuyên
tests/                  103 test, fixture cắt từ corpus thật
notebooks/              Kaggle: stage_d_dense.py · stage_e_planner.py
```

---

## Những bẫy đã đo được và cách xử lý

Toàn bộ ghi chi tiết trong [CLAUDE.md](CLAUDE.md) §3. Bốn cái nguy hiểm nhất:

**1. `pd.read_csv` đọc `713.942` thành `713,942`** — sai **1000 lần** và hoàn toàn
im lặng, không exception. Chuẩn hoá số phải làm theo **từng ô**, lọc theo cột vẫn
để lọt. Có test hồi quy: `tests/test_emit.py::test_713942_khong_bi_pandas_doc_thanh_713_phay_942`.

**2. Sau khi bỏ dấu, tiếng Việt sinh ra hàng loạt va chạm.** Token lẻ gần như luôn
là một âm tiết của từ ghép khác:

| Va chạm | Hậu quả | Cách chặn |
|---|---|---|
| `TY` ⊂ `CONG TY` | "Công ty VND" đọc thành đơn vị tỷ | `(?<!CONG )` |
| `TAI` ⊂ `TAI CHINH` | "đầu tư tài chính dài hạn" bị cắt còn "đầu tư" | liệt kê **cụm**, không liệt kê token lẻ |
| `KHI`/`NAM` là hư từ | "khi rà soát" khớp nhầm GAS ("Khí Việt Nam") | khớp cụm **có thứ tự**, khoảng cách ≤3 token |
| `CO PHAN DUONG` ⊃ `DUONG` | "Cổ phần Đường" khớp thành "có … dương" | nêu đích danh chỉ tiêu |

**3. Không có word boundary giữa chữ số và chữ cái.** OCR dính `31.12.2022Triệu VND`,
`2018VND` — dùng `\b` là mất mốc thời gian của cả cột (từng làm tụt coverage 89%→72%).
Phải dùng `(?<![A-Z])` và `(?<![0-9])…(?![0-9])`.

**4. Bảng CĐKT gần như luôn bị tách đôi** ("BẢNG CÂN ĐỐI KẾ TOÁN" + "(tiếp theo)"),
Tài sản và Nguồn vốn nằm ở **hai bảng khác nhau**. Mọi kiểm tra ràng buộc kế toán
phải group theo `doc_id`, không theo `table_idx` — group sai làm `270 == 440` chỉ
còn 9 mẫu thay vì 2.815.

---

## Hai điều đã đo được và làm thay đổi thiết kế

### `<vị_trí_bảng>` = **số dòng**, không phải số thứ tự bảng

Đề bài chỉ cho một ví dụ, và ví dụ đó không nhất quán (hỏi VNM nhưng doc là AAA).
Đoán sai thì **F₂ = 0 toàn bài**, nên phải đo chứ không đoán. Thiết kế probe 2 tầng
tốn 5 lượt: tầng 1 nộp cả 4 cách mã hoá cùng lúc để tách biến `doc_id` khỏi biến vị
trí; tầng 2 dò từng cách. Giả thuyết ưu tiên ban đầu (số thứ tự bảng) **sai hoàn toàn** — 0,0.

### Công thức F₂ và số bảng nên nộp

Gold `G` bảng, nộp `k` bảng, trúng `h` bảng ⇒ `F₂ = 5h/(4G + k)`, **đúng từng câu**.
`k` chỉ nằm ở mẫu và chỉ cộng: thêm một bảng sai làm mẫu +1, thêm một bảng đúng làm
tử +5. Nới `k` có lãi khi `p > h/(4G + k)`.

Số đo thật ở `k=2`: `h ≈ 0,735`, `G ≈ 2,15` ⇒ ngưỡng **0,069**. Bảng hạng 3 chỉ cần
6,9% khả năng đúng là đã nên nộp, trong khi tỷ lệ trúng trung bình đang là 37%.

Tôi tính sai hai lần trước đó, cùng một gốc: giả định **nộp toàn bảng đúng**
(precision ≈ 1). Precision thật 0,37 ⇒ chi phí precision nhỏ hơn nhiều lợi ích
recall, mà F₂ cân recall gấp 4. Quét thực nghiệm xác nhận: k=2 → 0,3147 ·
**k=6 → 0,4137** · k=10 → 0,3999; `k` thích ứng theo số công ty×năm → **0,4267**.

---

## Giới hạn hiện tại

`ANSWER_ACCURACY = 0,2411`. Đường tất định chỉ mã hoá được **công thức đóng** —
424/1.012 câu là multi-hop, trong đó đường suy luận **do dữ liệu quyết định**
("tại năm mà HPG có doanh thu cao nhất, hệ số thanh toán hiện hành là bao nhiêu"),
không quy về công thức cố định được.

Bộ giải rule dùng **gating dương** — chỉ giải khi chắc chắn câu đơn tầng. Bản đầu
giải 372 câu nhưng nhiều câu sai vì bắn vào multi-hop nó không hiểu: code chạy trơn
tru, sandbox verify pass, nhưng số sai và **không để lại dấu vết**. Siết còn 204 câu
thì đúng. Nguyên tắc: *thà bỏ sót còn hơn đoán sai tự tin* — executor bắt được lỗi
cú pháp, không bắt được lỗi ngữ nghĩa.

Phần còn lại thuộc Stage E (LLM planner). Đối chiếu với số của mentor
(LLM + evidence retrieved = 64,0%) trong khi retrieval của hệ thống này đã đạt
recall 0,665 — khoảng cách nằm ở khâu suy luận, không ở khâu truy xuất.

---

## Ràng buộc cuộc thi

- **Cấm LLM đóng** (GPT-4o, Gemini, Claude…). Chỉ Open LLM/PLM trên HuggingFace,
  **≤ 14B tham số**, phát hành **trước 01/06/2026**.
  Model dùng: `Qwen/Qwen2.5-14B-Instruct-AWQ` (14B, 09/2024),
  `Qwen/Qwen3-Embedding-4B`, `BAAI/bge-reranker-v2-m3` — đều hợp lệ.
- Public test ≤ 10 lượt/ngày, hạn 31/08/2026. Private test ≤ 5 lượt, 01–03/09/2026.
- Bắt buộc nộp Working Notes Paper.

## Nguồn dữ liệu ngoài

Không dùng dữ liệu ngoài nào. Toàn bộ đáp án rút từ corpus do BTC cung cấp.
`code_stock.csv` (100 mã → tên công ty) đi kèm dataset và là nguồn chân lý cho việc
định danh công ty.

## Giấy phép

[MIT](LICENSE).
