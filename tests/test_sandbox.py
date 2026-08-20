"""Sandbox chạy code LLM trong tiến trình con có trần bộ nhớ."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sandbox  # noqa: E402
from sandbox import JobResult, run_jobs  # noqa: E402


@pytest.fixture
def base(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "t.csv").write_text(
        "nhan,nam_2023\nDoanh thu thuan,1500.0\nGia von hang ban,900.0\n",
        encoding="utf-8")
    return tmp_path


def _job(qid: int, code: str) -> dict:
    return {"id": qid, "code": code, "frames": {"df1": "data/t.csv"}}


def test_chay_duoc_va_tra_ve_dung_so(base: Path):
    out = run_jobs([
        _job(1, "float(df1.loc[df1.nhan == 'Doanh thu thuan', 'nam_2023'].iloc[0])"),
        _job(2, "result = float(df1.nam_2023.sum())"),
    ], base)
    assert out[1] == JobResult(True, 1500.0, None)
    assert out[2].ok and out[2].value == 2400.0


def test_code_hong_khong_lam_chet_ca_lo(base: Path):
    """Câu lỗi chỉ hỏng một mình; các câu khác trong cùng lô vẫn có kết quả."""
    out = run_jobs([
        _job(1, "float(df1.nam_2023.iloc[0])"),
        _job(2, "df1['khong_co_cot_nay'].sum()"),
        _job(3, "float(df1.nam_2023.iloc[1])"),
    ], base)
    assert out[1].ok and out[3].ok
    assert not out[2].ok and "KeyError" in out[2].error


def test_khong_quy_duoc_ve_mot_so_thi_bao_loi(base: Path):
    out = run_jobs([_job(1, "df1")], base)
    assert not out[1].ok and "không quy được về 1 số" in out[1].error


def test_thieu_csv_bao_loi_chu_khong_treo(base: Path):
    out = run_jobs([{"id": 1, "code": "float(df9.x.iloc[0])",
                     "frames": {"df9": "data/khong_ton_tai.csv"}}], base)
    assert not out[1].ok


def test_moi_id_deu_co_ket_qua_du_tien_trinh_con_chet(base: Path, monkeypatch):
    """Con chết ⇒ câu đầu hàng đợi bị đánh dấu hỏng, phần còn lại chạy tiếp.

    Không có bất biến này thì `run_jobs` lặp vô hạn: câu độc hại không bao giờ
    vào được kết quả nên vòng while luôn thấy nó ở hàng đợi.
    """
    real = sandbox._run_child
    calls = {"n": 0}

    def fake(jobs, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {}                       # con chết ngay câu đầu, không kịp ghi gì
        return real(jobs, *a, **kw)

    monkeypatch.setattr(sandbox, "_run_child", fake)
    out = run_jobs([_job(1, "float(df1.nam_2023.iloc[0])"),
                    _job(2, "float(df1.nam_2023.iloc[1])")], base)
    assert set(out) == {1, 2}
    assert not out[1].ok and "chết" in out[1].error
    assert out[2].ok and out[2].value == 900.0
