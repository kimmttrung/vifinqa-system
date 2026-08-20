"""Primitive dùng chung. Fixture cắt từ corpus thật, không bịa."""
import json

import common
from common import fold, html_to_grid, load_questions, parse_number, tokens


class TestFold:
    def test_ocr_sai_dau_van_khop(self):
        # OCR sai dấu liên tục; bỏ dấu thì mọi biến thể trùng nhau.
        assert fold("KẾT QUÀ") == fold("KẾT QUẢ") == "KET QUA"
        assert fold("LƯU CHUYÊN TIỀN TỆ") == fold("LƯU CHUYỂN TIỀN TỆ")
        assert fold("THUYÊT MINH") == fold("THUYẾT MINH") == "THUYET MINH"
        assert fold("NỘ PHẢI TRẢ") == fold("NỢ PHẢI TRẢ")

    def test_d_gach_ngang(self):
        assert fold("Đơn vị") == "DON VI"
        assert fold("đồng") == "DONG"

    def test_gop_whitespace(self):
        assert fold("  Tổng   cộng \n tài sản ") == "TONG CONG TAI SAN"

    def test_rong(self):
        assert fold(None) == "" and fold("") == ""


class TestParseNumber:
    def test_dinh_dang_viet_nam(self):
        assert parse_number("1.234.567") == 1234567.0
        assert parse_number("1.954.764.678.040") == 1954764678040.0

    def test_ngoac_don_la_so_am(self):
        assert parse_number("(162.105.381)") == -162105381.0

    def test_phay_thap_phan(self):
        assert parse_number("1.234,56") == 1234.56
        assert parse_number("4,89") == 4.89

    def test_o_trong_va_gach_ngang(self):
        for s in ("-", "", "--", "n/a", None):
            assert parse_number(s) is None

    def test_ngay_thang_khong_phai_so(self):
        # 31/12/2015 là mốc thời gian; đọc nhầm thành số làm hỏng cả cột
        assert parse_number("31/12/2015") is None

    def test_don_vi_dinh_lien_do_ocr(self):
        assert parse_number("31.12.2022Triệu VND") is None

    def test_khong_phai_so(self):
        assert parse_number("411a") is None


class TestHtmlToGrid:
    # Header 2 tầng kiểu ngân hàng — cắt từ ACB/2022/separate.
    ACB = ('<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Tại ngày</td></tr>'
           '<tr><td>Thuyết minh</td><td>31.12.2022 Triệu VND</td>'
           '<td>31.12.2021 Triệu VND</td></tr>'
           '<tr><td>I</td><td>Tiền mặt, vàng bạc, đá quý</td><td>4</td>'
           '<td>8.460.883</td><td>7.509.867</td></tr></table>')

    def test_expand_rowspan_colspan_thanh_luoi_chu_nhat(self):
        g = html_to_grid(self.ACB)
        assert len(g) == 3
        assert {len(r) for r in g} == {5}, "mọi hàng phải cùng số cột"

    def test_colspan_duoc_nhan_ban(self):
        g = html_to_grid(self.ACB)
        assert g[0] == ["", "", "Tại ngày", "Tại ngày", "Tại ngày"]

    def test_gia_tri_dung_vi_tri(self):
        g = html_to_grid(self.ACB)
        assert g[1][2] == "Thuyết minh"
        assert g[2][3] == "8.460.883"

    def test_html_hong_khong_crash(self):
        assert html_to_grid("<table><tr><td>a</td></table>") == [["a"]]
        assert html_to_grid("") == []


def test_tokens():
    assert tokens("Lợi nhuận sau thuế (60)") == ["LOI", "NHUAN", "SAU", "THUE", "60"]


class TestLoadQuestions:
    """Bộ đề rút gọn để chạy thử hay được xuất bằng json.dump(list) rồi đẩy lên
    Kaggle Dataset, nên phải đọc được cả mảng JSON lẫn JSONL."""

    ROWS = [{"id": 1, "question": "Doanh thu thuần năm 2023?"},
            {"id": 2, "question": "Tổng tài sản cuối năm 2022?"}]

    def _load(self, tmp_path, monkeypatch, name, text):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        monkeypatch.setattr(common, "_resolve_questions", lambda: p)
        return load_questions()

    def test_jsonl(self, tmp_path, monkeypatch):
        txt = "\n".join(json.dumps(r, ensure_ascii=False) for r in self.ROWS)
        assert self._load(tmp_path, monkeypatch, "questions.jsonl", txt) == self.ROWS

    def test_mang_json(self, tmp_path, monkeypatch):
        txt = json.dumps(self.ROWS, ensure_ascii=False, indent=1)
        assert self._load(tmp_path, monkeypatch, "questions.json", txt) == self.ROWS

    def test_jsonl_co_dong_trong(self, tmp_path, monkeypatch):
        txt = "\n\n".join(json.dumps(r, ensure_ascii=False) for r in self.ROWS) + "\n"
        assert self._load(tmp_path, monkeypatch, "questions.jsonl", txt) == self.ROWS
