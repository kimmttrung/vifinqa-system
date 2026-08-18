"""Stage A. Test không cần corpus chạy luôn; test cần corpus thì skip nếu thiếu."""
import json

import pytest

from common import DATA_ROOT, parse_number
from corpus import (classify_period, detect_unit, process_document,
                                   section_from_form, section_from_heading)

needs_corpus = pytest.mark.skipif(DATA_ROOT is None,
                                  reason="chưa có corpus (set VIFINQA_DATA)")


class TestDetectUnit:
    def test_don_vi_trong_caption(self):
        assert detect_unit("Đơn vị: VND", "TÀI SẢN Mã số 31/12/2015") == (1.0, "caption")

    def test_ocr_hong_nhan_don_vi_van_bat_duoc(self):
        # đã gặp thật: "Dom vi: VND" — nên KHÔNG dò nhãn, chỉ dò token đơn vị
        assert detect_unit("Dom vi: VND", "") == (1.0, "caption")

    def test_don_vi_trong_o_header_kieu_ngan_hang(self):
        assert detect_unit("", "Thuyết minh 31.12.2022 Triệu VND") == (1e6, "header_cell")

    def test_so_dinh_lien_don_vi_do_ocr(self):
        """\\b không tồn tại giữa '2' và 'T' — bug làm tụt coverage 89%→72%."""
        assert detect_unit("", "31.12.2022Triệu VND") == (1e6, "header_cell")

    def test_cong_ty_khong_bi_doc_thanh_ty(self):
        # sau khi bỏ dấu, "tỷ" và "ty" (công ty) trùng nhau
        assert detect_unit("Công ty VND", "") == (1.0, "caption")


class TestClassifyPeriod:
    def test_cuoi_ky_va_dau_ky(self):
        assert classify_period("31/12/2015", 2015, "CDKT") == \
            {"period_type": "stock", "year": 2015, "is_begin": False}
        assert classify_period("01/01/2015", 2015, "CDKT") == \
            {"period_type": "stock", "year": 2015, "is_begin": True}

    def test_ky_phat_sinh(self):
        assert classify_period("Năm 2015", 2015, "KQKD") == \
            {"period_type": "flow", "year": 2015, "is_begin": False}

    def test_nam_dinh_lien_don_vi(self):
        assert classify_period("2018VND", 2018, "TM")["year"] == 2018
        assert classify_period("Năm2023VND", 2023, "KQKD")["year"] == 2023

    def test_cot_khong_mang_moc_thoi_gian(self):
        assert classify_period("Mã số", 2015, "CDKT") is None
        assert classify_period("", 2015, "CDKT") is None

    def test_so_dau_nam(self):
        assert classify_period("Số đầu năm", 2019, "CDKT")["is_begin"] is True


class TestSection:
    def test_ngan_hang_lech_mot_bac_so_voi_doanh_nghiep(self):
        """B02 là KQKD với doanh nghiệp nhưng là CĐKT với ngân hàng."""
        assert section_from_form("MÃU B 02-DN/HN") == "KQKD"
        assert section_from_form("Mẫu B02/TCTD") == "CDKT"
        assert section_from_form("Mẫu B05/TCTD-HN") == "TM"

    def test_ma_bieu_mau_doanh_nghiep(self):
        assert section_from_form("MÃU B 01-DN/HN") == "CDKT"
        assert section_from_form("Mẫu số B 03 – DN") == "LCTT"

    def test_fallback_keyword_chiu_duoc_ocr_sai_dau(self):
        from common import fold
        assert section_from_heading(fold("BẢO CẢO LƯU CHUYÊN TIỀN TỆ HỢP NHẤT")) == "LCTT"
        assert section_from_heading(fold("THUYÊT MINH BÁO CÁO TÀI CHÍNH")) == "TM"


@needs_corpus
class TestTrenCorpusThat:
    """Ground truth đo độc lập trước khi viết code (CLAUDE.md §3)."""

    @pytest.fixture(scope="class")
    def aaa(self):
        p = (DATA_ROOT / "financial_statements/AAA/2015/"
             "AAA_financial_statements_2015_consolidated/"
             "AAA_financial_statements_2015_consolidated_extracted.txt")
        if not p.exists():
            pytest.skip("không có AAA/2015")
        return process_document(p, DATA_ROOT)

    def test_dung_47_bang_43_trang(self, aaa):
        meta, _ = aaa
        assert meta["n_tables"] == 47
        assert meta["n_pages"] == 43

    def test_vi_tri_bang_dung_tung_dong(self, aaa):
        _, tabs = aaa
        assert [t["line_no"] for t in tabs[:5]] == [19, 214, 239, 286, 333]

    def test_metadata_tu_duong_dan(self, aaa):
        meta, _ = aaa
        assert (meta["ticker"], meta["year"], meta["stmt_type"]) == \
            ("AAA", 2015, "consolidated")

    def test_rang_buoc_ke_toan_100_cong_200_bang_270(self, aaa):
        """Bài test thật cho parser số: nếu parse sai một ô là đẳng thức vỡ."""
        _, tabs = aaa
        cdkt = tabs[1]
        by_code = {r[1]: parse_number(r[3])
                   for r in json.loads(cdkt["grid"]) if len(r) > 3}
        assert by_code["100"] + by_code["200"] == by_code["270"] == 1954764678040.0

    def test_cot_ky_cua_bang_cdkt(self, aaa):
        _, tabs = aaa
        periods = json.loads(tabs[1]["col_periods"])
        assert periods[3] == {"period_type": "stock", "year": 2015, "is_begin": False}
        assert periods[4] == {"period_type": "stock", "year": 2015, "is_begin": True}

    def test_bang_ngan_hang_khong_co_ma_so_va_don_vi_trieu(self):
        p = (DATA_ROOT / "financial_statements/ACB/2022/"
             "ACB_financial_statements_2022_separate/"
             "ACB_financial_statements_2022_separate_extracted.txt")
        if not p.exists():
            pytest.skip("không có ACB/2022")
        _, tabs = process_document(p, DATA_ROOT)
        cdkt = next(t for t in tabs if t["section"] == "CDKT" and t["n_numeric_cells"] > 20)
        assert cdkt["unit_scale"] == 1e6, "đơn vị lấy từ ô header"
        assert not cdkt["has_ma_so"], "ngân hàng KHÔNG có cột Mã số"
