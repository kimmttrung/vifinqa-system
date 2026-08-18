"""Bộ giải đa ô — tập trung vào GATING.

Nguy hiểm nhất của rule không phải bỏ sót mà là bắn vào câu nó không hiểu:
code chạy trơn tru, sandbox verify pass, nhưng số sai. Không để lại dấu vết.
Nên phần lớn test ở đây kiểm rằng solver **từ chối** đúng chỗ.
"""
import pytest

from execute import run_query
from qparse import parse_question
from solve import too_complex


def ir(q: str):
    return parse_question(1, q)


class TestGatingTuChoiCauDaTang:
    @pytest.mark.parametrize("q", [
        # q471 thật: "tổng doanh thu thuần" để TRẢ LỜI, "tỷ lệ LNST/DTT" để LỌC
        "Năm 2016, trong bốn mã cổ phiếu AAA, DCM, DPM và GVR, tổng doanh thu thuần "
        "của các công ty có tỷ lệ lợi nhuận sau thuế trên doanh thu lớn hơn 5%?",
        # mệnh đề lồng dùng "mà/đạt" thay vì "có"
        "Tại năm mà Hoà Phát đạt doanh thu thuần cao nhất trong giai đoạn 2020-2024, "
        "hệ số thanh toán hiện hành là bao nhiêu lần?",
        # mệnh đề quan hệ dính liền "năm có"
        "Chi phí vận chuyển của MPC trong năm có số dư hàng tồn kho lớn nhất?",
        # lọc theo trung vị
        "Trong nhóm có tỷ lệ nợ cao hơn trung vị, tổng nợ phải trả là bao nhiêu?",
        # kịch bản giả định
        "Nếu doanh thu thuần giảm 10%, có bao nhiêu đơn vị có hệ số dưới 1,5 lần?",
        # hai tỷ số: một để lọc, một để hỏi
        "Trong giai đoạn 2016-2020, vào năm KBC có tỷ số D/E cao nhất, hệ số khả năng "
        "thanh toán lãi vay là bao nhiêu lần?",
    ])
    def test_phai_tu_choi(self, q):
        assert too_complex(ir(q)) is True, "solver không được đụng vào câu multi-hop"

    def test_viet_nam_co_khong_bi_nham_la_menh_de_long(self):
        """'Việt Nam có…' chứa 'NAM CO' nhưng không phải mệnh đề quan hệ."""
        q = "Tổng tài sản của Ngân hàng TMCP Ngoại thương Việt Nam cuối năm 2020?"
        assert too_complex(ir(q)) is False

    def test_cong_ty_co_phan_khong_bi_nham(self):
        """fold('công ty cổ phần') = 'CONG TY CO PHAN' — chứa 'CONG TY CO'."""
        q = "Doanh thu thuần của Công ty Cổ phần Đường Quảng Ngãi năm 2023?"
        assert too_complex(ir(q)) is False


class TestGatingChapNhanCauDonTang:
    @pytest.mark.parametrize("q", [
        "Tăng trưởng khoản vay ngắn hạn của MCH từ cuối năm 2021 đến cuối năm 2023 "
        "là bao nhiêu %?",
        "Năm nào có số dư Xây dựng cơ bản dở dang lớn nhất của SAM vào cuối các năm "
        "2020, 2021 và 2023?",
        "Tỷ lệ chi phí bán hàng trên doanh thu thuần của IJC năm 2020 là bao nhiêu %?",
    ])
    def test_phai_chap_nhan(self, q):
        assert too_complex(ir(q), max_metrics=4) is False


class TestSandbox:
    def test_tra_ve_mot_so(self):
        r = run_query("result = 1 + 1", {})
        assert r.ok and r.value == 2.0

    def test_bieu_thuc_khong_can_gan_result(self):
        assert run_query("40 + 2", {}).value == 42.0

    def test_code_loi_khong_lam_sap_pipeline(self):
        r = run_query("result = df9['khong_ton_tai'][0]", {})
        assert not r.ok and r.error

    def test_khong_quy_duoc_ve_mot_so_thi_that_bai(self):
        """Chặn LLM trả về Series/DataFrame thay vì một con số."""
        assert not run_query("result = [1, 2, 3]", {}).ok

    def test_nan_va_inf_bi_tu_choi(self):
        assert not run_query("result = float('nan')", {}).ok
        assert not run_query("result = float('inf')", {}).ok

    def test_khong_import_duoc(self):
        """Sandbox chỉ có builtins tối thiểu — LLM không đụng được vào hệ thống."""
        assert not run_query("import os\nresult = 1", {}).ok
