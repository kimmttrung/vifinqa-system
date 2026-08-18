"""Stage G — ghi CSV.

Chứa bài test hồi quy quan trọng nhất của cả dự án: lỗi `713.942`.
"""
import pandas as pd

from emit import (POSITION_SCHEMES, doc_ref, flatten_header,
                                 grid_to_csv, table_ref, table_refs)


class _Row:
    """Giả một dòng của table_meta.parquet."""
    doc_id = "AAA_financial_statements_2015_consolidated"
    table_idx = 2
    line_no = 214
    page_no = 8
    char_start = 14955


class TestChuanHoaSo:
    def test_713942_khong_bi_pandas_doc_thanh_713_phay_942(self, tmp_path):
        """Bug nguy hiểm nhất đã gặp — sai 1000 lần và HOÀN TOÀN IM LẶNG.

        Để nguyên chuỗi '713.942' trong CSV thì `pd.read_csv` suy ra 713.942
        (dấu chấm thập phân kiểu Anh) thay vì 713942 (dấu ngăn nghìn kiểu Việt).
        Không exception, không cảnh báo — chỉ là đáp án sai 1000 lần.
        """
        grid = [["Chỉ tiêu", "Năm nay"],
                ["Lãi thuần từ hoạt động dịch vụ", "713.942"]]
        out = tmp_path / "t.csv"
        grid_to_csv(grid, n_header=1, path=out)

        df = pd.read_csv(out)
        assert float(df.iloc[0, 1]) == 713942.0, "pandas đọc nhầm dấu ngăn nghìn"

    def test_chuan_hoa_theo_tung_o_khong_theo_cot(self, tmp_path):
        """Cột lẫn text vẫn phải chuẩn hoá được các ô số của nó.

        Lọc theo cột thì cột hỗn hợp bị bỏ qua nguyên vẹn và pandas lại tự suy
        kiểu sai — đúng cái bẫy ở trên.
        """
        grid = [["Chỉ tiêu", "Giá trị"],
                ["Tiền mặt", "1.234.567"],
                ["Ghi chú", "không có số"],
                ["Dự phòng", "(50.000)"]]
        out = tmp_path / "t.csv"
        grid_to_csv(grid, n_header=1, path=out)
        df = pd.read_csv(out)
        assert float(df.iloc[0, 1]) == 1234567.0
        assert df.iloc[1, 1] == "không có số"
        assert float(df.iloc[2, 1]) == -50000.0, "ngoặc đơn phải thành số âm"

    def test_o_trong_thanh_rong_khong_thanh_0(self, tmp_path):
        grid = [["Chỉ tiêu", "Giá trị"], ["Không có", "-"]]
        out = tmp_path / "t.csv"
        grid_to_csv(grid, n_header=1, path=out)
        df = pd.read_csv(out)
        assert pd.isna(df.iloc[0, 1]), "'-' phải là rỗng, không phải 0"


class TestFlattenHeader:
    def test_header_hai_tang_giu_ca_hai(self):
        grid = [["", "", "Tại ngày", "Tại ngày"],
                ["", "Thuyết minh", "31.12.2022 Triệu VND", "31.12.2021 Triệu VND"],
                ["I", "4", "8460883", "7509867"]]
        names = flatten_header(grid, n_header=2)
        assert names[2] == "tai_ngay_31_12_2022_trieu_vnd"
        assert names[3] == "tai_ngay_31_12_2021_trieu_vnd"

    def test_cot_rong_duoc_dat_ten_theo_chi_so(self):
        assert flatten_header([["", "A"], ["x", "1"]], 1) == ["col_0", "a"]

    def test_ten_trung_duoc_them_hau_to(self):
        names = flatten_header([["A", "A"], ["1", "2"]], 1)
        assert names == ["a", "a_2"]


class TestTableRef:
    def test_line_no_la_convention_da_chot(self):
        # Probe 17/08: line_no → F₂ 0,3147 · table_idx và page_no → 0,0
        assert table_ref(_Row(), "line_no") == \
            "AAA_financial_statements_2015_consolidated|214"

    def test_doc_id_la_ten_thu_muc_khong_co_extracted(self):
        # Probe 17/08: bản _extracted cho 0,0 trên MỌI chỉ số
        assert doc_ref("X_2015_consolidated", "folder") == "X_2015_consolidated"
        assert doc_ref("X_2015_consolidated", "file_stem") == "X_2015_consolidated_extracted"

    def test_scheme_all_phat_ca_4_cach(self):
        refs = table_refs(_Row(), "all")
        assert len(refs) == 4
        assert all("|" in r for r in refs)

    def test_scheme_la_ma_khong_hop_le_thi_bao_loi(self):
        import pytest
        with pytest.raises(ValueError):
            table_ref(_Row(), "khong_ton_tai")

    def test_du_4_ung_vien_vi_tri(self):
        assert set(POSITION_SCHEMES) == {"table_idx", "line_no", "page_no", "char_start"}
