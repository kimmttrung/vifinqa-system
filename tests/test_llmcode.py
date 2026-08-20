"""Chốt chặn code LLM sinh. Fixture là code THẬT lấy từ bài nộp thử 20/08."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmcode import check, _significant_digits  # noqa: E402

# Nguyên văn q1 của out/mini/submission.json — model đọc số từ preview rồi gõ
# lại vào code, đúng thứ cả kiến trúc dựng lên để ngăn.
Q1_HARDCODED = "result = (9651580686.0 + 208253201298.0) / 1e6"
# Cùng câu đó nhưng viết đúng: đọc từ df4, dòng "Lãi tiền gửi" của bảng 50.
Q1_DUNG = ("result = float(df4.loc[df4['col_0'].astype(str).str.strip() == "
           "'Lãi tiền gửi', '2018vnd'].iloc[0]) / 1e6")


class TestChanSoVietTay:
    def test_bat_duoc_q1(self):
        r = check(Q1_HARDCODED)
        assert not r.ok
        assert r.literals == (9651580686.0, 208253201298.0)

    def test_bat_duoc_ca_khi_chi_mot_so(self):
        assert not check("result = 405441889.0").ok

    def test_khong_chan_he_so_quy_doi(self):
        for code in (f"{Q1_DUNG}",
                     "float(df1.iat[3, 4]) * 100",
                     "float(df1.x.iloc[0]) / 1e9",
                     "float(df1.x.iloc[0]) * 0.001",
                     "float(df1.x.iloc[0]) / 1000000"):
            assert check(code).ok, code

    def test_so_trong_chuoi_khong_tinh(self):
        """Tên cột và nhãn đầy số — regex sẽ chặn oan, AST thì không."""
        code = ("float(df1.loc[df1['col_0'] == 'Vay 500.000.000 đồng', "
                "'tai_ngay_31_12_2022trieu_vnd'].iloc[0])")
        assert check(code).ok

    def test_nam_duoc_phep(self):
        assert check("float(df1[df1.nam == 2018]['gia_tri'].iloc[0])").ok


class TestRutVeBieuThuc:
    def test_bo_result_gan(self):
        r = check("result = float(df1.x.iloc[0]) / 1e6")
        assert r.ok and r.form == "assign"
        assert "result" not in r.code
        # ast.unparse chuẩn hoá 1e6 → 1000000.0; giá trị không đổi nên không sao,
        # nhưng đừng khẳng định code khớp từng ký tự.
        assert r.code.startswith("float(df1.x.iloc[0]) / 1000000")

    def test_bieu_thuc_giu_nguyen(self):
        r = check("float(df1.x.iloc[0])")
        assert r.ok and r.form == "expr" and r.code == "float(df1.x.iloc[0])"

    def test_nhieu_lenh_bi_chan_o_luot_dau(self):
        code = "a = df1.x.sum()\nresult = a / 1e9"
        assert not check(code, strict=True).ok

    def test_nhieu_lenh_duoc_nhan_o_luot_cuoi(self):
        """Hết lượt retry thì có còn hơn không — nhưng phải nhận diện được."""
        code = "a = df1.x.sum()\nresult = a / 1e9"
        r = check(code, strict=False)
        assert r.ok and r.form == "multi"

    def test_code_hong_cu_phap(self):
        r = check("result = float(df1.x.iloc[0]")
        assert not r.ok and r.form == "unparsable"


def test_significant_digits():
    assert _significant_digits(1e6) == 1
    assert _significant_digits(100) == 1
    assert _significant_digits(0.02) == 1        # số 0 dẫn đầu không phải chữ số nghĩa
    assert _significant_digits(1.5) == 2
    assert _significant_digits(9651580686.0) == 10
