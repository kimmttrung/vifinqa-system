"""Stage G-2 — Đóng gói submission.

Bài nộp là 1 file ZIP có `submission.json` + thư mục `data/*.csv` ở cấp NGOÀI CÙNG.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Record:
    id: int
    question: str
    answer: float
    relevant_docs: list[str] = field(default_factory=list)
    relevant_tables: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    pandas_query: str = ""

    def to_json(self) -> dict:
        return {
            "id": int(self.id),
            "question": self.question,
            "answer": float(self.answer),
            "relevant_docs": list(dict.fromkeys(self.relevant_docs)),
            "relevant_tables": list(dict.fromkeys(self.relevant_tables)),
            "evidence": self.evidence,
            "pandas_query": self.pandas_query,
        }


class Submission:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.data_dir = self.out_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[Record] = []

    def add(self, rec: Record):
        self.records.append(rec)

    def write_json(self) -> Path:
        p = self.out_dir / "submission.json"
        payload = [r.to_json() for r in sorted(self.records, key=lambda r: r.id)]
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def used_csvs(self) -> set[str]:
        return {e["csv_path"] for r in self.records for e in r.evidence}

    def prune_unused_csvs(self) -> int:
        """Xoá CSV không được evidence nào trỏ tới — ZIP nhỏ lại, tránh nhiễu."""
        keep = {Path(p).name for p in self.used_csvs()}
        n = 0
        for f in self.data_dir.glob("*.csv"):
            if f.name not in keep:
                f.unlink()
                n += 1
        return n

    def zip(self, zip_path: Path | None = None) -> Path:
        zip_path = Path(zip_path or self.out_dir / "submission.zip")
        json_path = self.write_json()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(json_path, "submission.json")
            for f in sorted(self.data_dir.glob("*.csv")):
                z.write(f, f"data/{f.name}")
        return zip_path

    def validate(self) -> list[str]:
        """Kiểm tra hình thức trước khi nộp — sai format là mất trắng cả lượt."""
        errs = []
        ids = [r.id for r in self.records]
        if len(set(ids)) != len(ids):
            errs.append("có id trùng")
        for r in self.records:
            if not isinstance(r.answer, float) or r.answer != r.answer:
                errs.append(f"q{r.id}: answer không phải float hữu hạn")
            if not r.relevant_tables:
                errs.append(f"q{r.id}: relevant_tables rỗng")
            for t in r.relevant_tables:
                if "|" not in t:
                    errs.append(f"q{r.id}: relevant_tables sai định dạng: {t!r}")
            for e in r.evidence:
                if not e.get("csv_path", "").startswith("data/"):
                    errs.append(f"q{r.id}: csv_path phải bắt đầu bằng 'data/': {e}")
                if not (self.out_dir / e["csv_path"]).exists():
                    errs.append(f"q{r.id}: thiếu file {e['csv_path']}")
            if not r.pandas_query.strip():
                errs.append(f"q{r.id}: pandas_query rỗng")
        return errs
