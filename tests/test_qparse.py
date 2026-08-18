"""Stage C — question parser. Câu hỏi trong đây là nguyên văn từ questions.jsonl."""
import pytest

from qparse import parse_question


def ir(q: str, qid: int = 1):
    return parse_question(qid, q)


class TestDonViDapAn:
    @pytest.mark.parametrize("q,kind,scale", [
        ("Tổng tài sản của STB cuối năm 2021 là bao nhiêu triệu đồng?", "vnd", 1e6),
        ("Tiền của NKG cuối năm 2022 là bao nhiêu tỷ đồng?", "vnd", 1e9),
        ("LNTT của công ty mẹ QNS năm 2023 là bao nhiêu trăm tỷ đồng?", "vnd", 1e11),
        ("Vốn cổ phần của VGT là bao nhiêu nghìn tỷ đồng vào cuối năm 2024?", "vnd", 1e12),
        ("Tỷ suất lợi nhuận ròng của HBC năm 2016 là bao nhiêu %?", "percent", 1.0),
        ("Chênh lệch là bao nhiêu điểm phần trăm?", "pp", 1.0),
        ("Tài sản ngắn hạn gấp bao nhiêu lần nợ ngắn hạn?", "times", 1.0),
        ("Năm nào có số dư lớn nhất của SAM?", "year", 1.0),
    ])
    def test_don_vi(self, q, kind, scale):
        got = ir(q)
        assert (got.unit_kind, got.unit_scale) == (kind, scale)

    def test_nghin_ty_va_tram_ty_khong_bi_doc_thanh_ty(self):
        # 151 câu dùng hai đơn vị này — nhầm là sai 100× hoặc 1000×
        assert ir("… bao nhiêu nghìn tỷ đồng?").unit_scale == 1e12
        assert ir("… bao nhiêu trăm tỷ đồng?").unit_scale == 1e11
        assert ir("… bao nhiêu tỷ đồng?").unit_scale == 1e9

    def test_ma_co_phieu_khong_phai_don_vi_co_phieu(self):
        """'doanh nghiệp có MÃ CỔ PHIẾU DIG…' không phải hỏi số cổ phiếu."""
        q = ("Xét các doanh nghiệp có mã cổ phiếu DIG, KBC và VRE trong năm 2024, "
             "tài sản ngắn hạn gấp bao nhiêu lần nợ ngắn hạn?")
        assert ir(q).unit_kind == "times"

    def test_don_vi_lay_o_menh_de_nghi_van_khong_phai_menh_de_gia_dinh(self):
        """'nếu doanh thu giảm 10%' là giả định, không phải đơn vị đáp án."""
        q = ("Trong năm 2023, nếu doanh thu thuần giảm 10%, có bao nhiêu đơn vị có "
             "hệ số khả năng thanh toán lãi vay rơi xuống dưới 1,5 lần?")
        assert ir(q).unit_kind == "count"

    def test_don_vi_o_cuoi_cau_sau_dau_phay(self):
        q = "Chênh lệch số dư giữa hai mốc là bao nhiêu, tính bằng triệu đồng?"
        assert (ir(q).unit_kind, ir(q).unit_scale) == ("vnd", 1e6)


class TestLoaiBaoCao:
    @pytest.mark.parametrize("q,expect", [
        ("Lãi tiền gửi của công ty mẹ VJC năm 2018?", "separate"),
        ("KHG ở phạm vi công ty mẹ ghi nhận chi phí?", "separate"),
        ("KLB trên cơ sở công ty mẹ trong các năm?", "separate"),
        ("Tổng doanh thu bộ phận – thuần hợp nhất năm 2022?", "consolidated"),
        ("Tổng tài sản của STB cuối năm 2021?", None),
    ])
    def test_stmt_type(self, q, expect):
        assert ir(q).stmt_type == expect


class TestNam:
    def test_giai_doan_gach_ngang_va_en_dash(self):
        assert ir("Trong giai đoạn 2021-2023 của HPG").years == [2021, 2022, 2023]
        assert ir("Trong giai đoạn 2021–2023 của HPG").years == [2021, 2022, 2023]

    def test_tu_nam_den_nam(self):
        assert ir("từ năm 2019 đến năm 2022").years == [2019, 2020, 2021, 2022]

    def test_liet_ke_roi_rac(self):
        assert ir("trong các năm 2017, 2018, 2021 và 2023").years == [2017, 2018, 2021, 2023]

    def test_ngay_thang_day_du(self):
        assert ir("đến ngày 31/12/2023").years == [2023]
        assert ir("đến ngày 31 tháng 12 năm 2019").years == [2019]


class TestMocThoiGian:
    @pytest.mark.parametrize("q,expect", [
        ("Tổng tài sản cuối năm 2021", "end"),
        ("đến ngày 31/12/2023", "end"),
        ("Số thuế phải nộp đầu năm 2021", "begin"),
        ("Chi phí dự phòng trong năm 2020", "period"),
    ])
    def test_time_point(self, q, expect):
        assert ir(q).time_point == expect


class TestTruyVanRetrieval:
    def test_bo_ma_ck_va_ten_cong_ty(self):
        """Tên công ty nằm trong caption của MỌI bảng thuộc doc đó nên không
        phân biệt được bảng nào với bảng nào — chỉ làm loãng tín hiệu."""
        q = ir("Lợi nhuận trước thuế của công ty mẹ QNS năm 2023 là bao nhiêu trăm tỷ đồng?")
        rq = q.retrieval_query
        assert "QNS" not in rq and "2023" not in rq
        assert "LOI NHUAN TRUOC THUE" in rq

    def test_khong_vo_thanh_tung_ky_tu(self):
        """Regex kết thúc bằng '|' từng khớp chuỗi rỗng ở mọi vị trí."""
        rq = ir("Tổng tài sản của STB cuối năm 2021 là bao nhiêu triệu đồng?").retrieval_query
        assert " T A I " not in rq
        assert "TAI SAN" in rq
