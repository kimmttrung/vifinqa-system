"""Định danh công ty — mọi ca ở đây là BẪY THẬT đã gặp trên corpus."""
import pytest

from companies import resolve_companies


def tks(q: str) -> set[str]:
    return set(resolve_companies(q)[0])


class TestBayDaGap:
    def test_khi_va_nam_la_hu_tu_khong_phai_KHI_VIET_NAM(self):
        """'Năm 2024, khi rà soát…' từng khớp nhầm GAS ('Khí Việt Nam').

        Bag-of-words trong cửa sổ không tách được, phải khớp như cụm từ có lỗ
        (đúng thứ tự, cách nhau ≤3 token).
        """
        q = ("Năm 2024, khi rà soát hiệu suất tài sản cố định của các doanh nghiệp "
             "thực phẩm (Tập đoàn Sao Mai; Tập đoàn DABACO; Masan; Tập đoàn Đại Dương; "
             "Vinamilk), doanh nghiệp nào lớn nhất?")
        got = tks(q)
        assert "GAS" not in got
        assert got == {"ASM", "DBC", "MSN", "OGC", "VNM"}

    def test_ho_masan_bon_ma_long_nhau(self):
        q = ("Trong số CTCP Hàng tiêu dùng Masan, CTCP Masan MeatLife, CTCP Sữa Việt Nam "
             "và CTCP Tập đoàn Sao Mai, doanh nghiệp nào cao nhất?")
        assert tks(q) == {"MCH", "MML", "VNM", "ASM"}, "MSN không được lọt vào"

    def test_ho_gelex_hai_ma_long_nhau(self):
        q = "Trong năm 2023, nhóm gồm Điện lực Gelex và Tập đoàn Gelex."
        assert tks(q) == {"GEE", "GEX"}

    def test_ma_ck_lot_trong_ten_cong_ty_khac(self):
        """'CTCP Chứng khoán FPT' = FTS. Nhưng tên chính thức của FPT sau khi bỏ
        từ nền chỉ còn đúng {FPT} nên nó khớp cả hai đường."""
        assert tks("Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023?") == {"FTS"}

    def test_dat_xanh_group_khac_dat_xanh_services(self):
        assert tks("Doanh thu của CTCP Dịch vụ Bất động sản Đất Xanh năm 2023?") == {"DXS"}

    def test_ma_viet_thuong_trong_danh_sach(self):
        """q431 thật: mã CK viết thường, chỉ nhận khi có ≥2 mã liền nhau."""
        q = "Trong ngành BĐS (gồm các công ty hpx,kbc,nvl,vic,vpi,vre), ROE năm 2024?"
        assert tks(q) == {"HPX", "KBC", "NVL", "VIC", "VPI", "VRE"}

    def test_ten_chinh_thuc_la_nguon_chan_ly(self):
        """code_stock.csv ghi STB = 'Ngân hàng TMCP Sài Gòn Tài Lộc'.
        Lạ so với thực tế nhưng file là nguồn chân lý — và SGB ('Sài Gòn Công
        Thương') không được khớp lây."""
        got = tks("Chi phí dự phòng của Ngân hàng TMCP Sài Gòn Tài Lộc năm 2020?")
        assert got == {"STB"}


class TestKhopTenDayDu:
    @pytest.mark.parametrize("q,expect", [
        ("Tổng tài sản của STB cuối năm 2021?", {"STB"}),
        ("Lãi tiền gửi của CTCP Hàng không Vietjet (VJC) năm 2018?", {"VJC"}),
        ("Tập đoàn Dệt May Việt Nam công ty mẹ năm 2017?", {"VGT"}),
        ("CTCP Tập đoàn Khải Hoàn Land ở phạm vi công ty mẹ?", {"KHG"}),
        ("Ngân hàng TMCP Phương Đông và Ngân hàng TMCP Nam Á", {"OCB", "NAB"}),
    ])
    def test_cac_dang_ten(self, q, expect):
        assert tks(q) == expect

    def test_cau_khong_neu_cong_ty_thi_tra_rong(self):
        q = "Trong các công ty có hàng tồn kho năm 2016 giảm ít nhất 10%?"
        assert tks(q) == set()
