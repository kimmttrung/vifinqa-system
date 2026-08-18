"""BM25 thưa trên toàn bộ 146k bảng — chạy CPU, không cần GPU.

Vì sao tự viết thay vì rank_bm25: rank_bm25 chấm điểm bằng vòng lặp Python trên
TOÀN corpus cho mỗi query (146k × 1.012 = quá chậm), còn ở đây ta luôn chấm trên
một *tập con vài trăm bảng* đã lọc metadata. Ma trận CSR + slice hàng cho phép
làm đúng như vậy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from common import tokens

K1 = 1.5
B = 0.75


@dataclass
class Bm25Index:
    tf: sparse.csr_matrix        # (n_tables, n_terms) tần suất thô
    vocab: dict[str, int]
    doc_len: np.ndarray
    idf: np.ndarray

    @property
    def n_docs(self) -> int:
        return self.tf.shape[0]

    def score(self, query: str, rows: np.ndarray | None = None) -> np.ndarray:
        """→ mảng điểm cho các hàng `rows` (mặc định: toàn bộ)."""
        qt = [self.vocab[t] for t in tokens(query) if t in self.vocab]
        if not qt:
            return np.zeros(len(rows) if rows is not None else self.n_docs, dtype=np.float32)
        sub = self.tf[rows] if rows is not None else self.tf
        dl = self.doc_len[rows] if rows is not None else self.doc_len
        avgdl = float(self.doc_len.mean()) or 1.0
        cols = np.unique(np.asarray(qt))
        m = sub[:, cols].toarray().astype(np.float32)          # (n_sub, |q|)
        denom = m + K1 * (1 - B + B * (dl[:, None] / avgdl))
        w = (m * (K1 + 1)) / np.maximum(denom, 1e-9)
        return (w * self.idf[cols][None, :]).sum(axis=1)

    def save(self, path: Path):
        path = Path(path)
        sparse.save_npz(path.with_suffix(".tf.npz"), self.tf)
        np.savez_compressed(path.with_suffix(".meta.npz"),
                            doc_len=self.doc_len, idf=self.idf)
        path.with_suffix(".vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Bm25Index":
        path = Path(path)
        tf = sparse.load_npz(path.with_suffix(".tf.npz")).tocsr()
        meta = np.load(path.with_suffix(".meta.npz"))
        vocab = json.loads(path.with_suffix(".vocab.json").read_text(encoding="utf-8"))
        return cls(tf=tf, vocab=vocab, doc_len=meta["doc_len"], idf=meta["idf"])


def build(passages, min_df: int = 2, max_df_ratio: float = 0.6) -> Bm25Index:
    """passages: iterable[str]. Bỏ token quá hiếm (rác OCR) và quá phổ biến."""
    tokenized = [tokens(p) for p in passages]
    n = len(tokenized)
    df: dict[str, int] = {}
    for ts in tokenized:
        for t in set(ts):
            df[t] = df.get(t, 0) + 1
    keep = {t for t, c in df.items() if c >= min_df and c <= max_df_ratio * n}
    vocab = {t: i for i, t in enumerate(sorted(keep))}

    indptr = np.zeros(n + 1, dtype=np.int64)
    indices, data, doc_len = [], [], np.zeros(n, dtype=np.float32)
    for i, ts in enumerate(tokenized):
        counts: dict[int, int] = {}
        for t in ts:
            j = vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        doc_len[i] = len(ts)
        for j, c in sorted(counts.items()):
            indices.append(j)
            data.append(c)
        indptr[i + 1] = len(indices)

    tf = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), indptr),
        shape=(n, len(vocab)))
    dfv = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log(1.0 + (n - dfv + 0.5) / (dfv + 0.5)).astype(np.float32)
    return Bm25Index(tf=tf, vocab=vocab, doc_len=doc_len, idf=idf)
