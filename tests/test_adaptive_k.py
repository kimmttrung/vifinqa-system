"""k và số doc tính riêng cho từng câu — thay cho hằng số 6/3."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qparse import QueryIR  # noqa: E402
from retrieval import adaptive_docs, adaptive_k, n_targets  # noqa: E402


def ir(tickers=1, years=1, ops=()):
    return QueryIR(qid=1, question="", tickers=[f"T{i}" for i in range(tickers)],
                   years=[2020 + i for i in range(years)], ops=list(ops))


class TestAdaptiveK:
    def test_cau_don_gian_nop_it_hon_6(self):
        """43,4% bộ đề là câu một cặp (công ty × năm). Với G≈1,2 thì k=6 tự cắt
        hơn 30% F₂ so với k=3 (0,50 vs 0,71)."""
        assert adaptive_k(ir(1, 1), k_max=12) == 3

    def test_cau_nhieu_muc_tieu_nop_nhieu_hon_6(self):
        assert adaptive_k(ir(4, 1), k_max=12) > 6
        assert adaptive_k(ir(4, 3), k_max=12) == 12

    def test_ty_so_can_nhieu_bang_hon_tra_cuu(self):
        """Tử và mẫu thường nằm ở hai báo cáo khác nhau (KQKD ÷ CĐKT)."""
        assert adaptive_k(ir(1, 1, ["RATIO"])) > adaptive_k(ir(1, 1))

    def test_ton_trong_tran(self):
        assert adaptive_k(ir(11, 3), k_max=12) == 12
        assert adaptive_k(ir(11, 3), k_max=6) == 6

    def test_khong_bao_gio_ra_0(self):
        assert adaptive_k(QueryIR(qid=1, question=""), k_max=12) >= 2

    def test_khong_phai_no_op(self):
        """Bản trước là NO-OP và không ai phát hiện: công thức 4+1,5·log₂(1+n)
        có sàn 5,5 nên với `--k-max 6` MỌI câu đều ra đúng 6 — cờ `--adaptive-k`
        bật mà không đổi gì, không lỗi nào báo ra.

        Điều kiện bắt buộc: ngay cả khi trần THẤP, câu đơn và câu nhóm vẫn phải
        ra hai giá trị khác nhau."""
        assert adaptive_k(ir(1, 1), k_max=6) != adaptive_k(ir(4, 3), k_max=6)
        ks = {adaptive_k(ir(t, y), k_max=12)
              for t, y in [(1, 1), (1, 2), (2, 2), (4, 3)]}
        assert len(ks) >= 3, f"k gần như không đổi theo câu hỏi: {ks}"

    def test_bao_hoa_o_tran(self):
        """Giới hạn thiết kế, ghi ra để biết: từ ~4 cặp trở lên đều chạm trần 12.
        Trên toàn bộ đề là 354/1012 câu (35%). Trần 12 là lựa chọn giữ TRUNG BÌNH
        7,43 ≈ đỉnh 7,5 đã đo của k phẳng; nới trần là biến thứ hai, đo riêng."""
        assert adaptive_k(ir(4, 1), k_max=12) == adaptive_k(ir(11, 3), k_max=12) == 12
        assert adaptive_k(ir(4, 1), k_max=16) < adaptive_k(ir(11, 3), k_max=16)

    def test_cat_theo_so_doc_that_su_co(self):
        """Câu nêu 5 năm nhưng công ty chỉ có 2 báo cáo thì chỉ 2 cặp là có thật."""
        assert n_targets(ir(1, 5), n_docs=2) == 2
        assert adaptive_k(ir(1, 5), k_max=12, n_docs=2) < adaptive_k(ir(1, 5), k_max=12)


class TestAdaptiveDocs:
    def test_cau_don_nop_it(self):
        assert adaptive_docs(ir(1, 1), cap=8) == 2

    def test_cau_nhom_vuot_tran_cu_la_3(self):
        """Hằng số 3 chặn cứng recall của 29,8% bộ đề (câu ≥2 công ty)."""
        assert adaptive_docs(ir(8, 1), cap=8) > 3

    def test_khong_nop_qua_so_doc_ung_vien(self):
        """Đừng bịa doc mà doc filter không hề đưa ra."""
        assert adaptive_docs(ir(8, 1), cap=8, n_docs=2) == 2

    def test_ton_trong_tran(self):
        assert adaptive_docs(ir(20, 3), cap=8) == 8
