# Stage B — QC report

- fact: **1,536,188** · concept coverage **14.8%**
- adapter: {'TT200': 1120648, 'TCTD': 345321, 'TT334': 50156, 'TT232': 20063}
- statement: {'TM': 983675, 'CDKT': 311812, 'LCTT': 122157, 'KQKD': 105252, 'VCSH': 12339, 'OTHER': 953}

## Ràng buộc kế toán
| rule           |    n |   pass |   rate |
|:---------------|-----:|-------:|-------:|
| 270 == 440     | 2812 |   2792 | 0.9929 |
| 100+200 == 270 | 2954 |   2933 | 0.9929 |
| 300+400 == 440 | 2805 |   2769 | 0.9872 |
| 10-11 == 20    | 2757 |   2554 | 0.9264 |

## Cross-year
- 36932/39132 = 94.4%

## 20 lệch cross-year lớn nhất (ứng viên OCR sai số)
| ticker   | stmt_type    |   period_year | concept                      |         beg |         end |         dev |
|:---------|:-------------|--------------:|:-----------------------------|------------:|------------:|------------:|
| PC1      | consolidated |          2023 | TSCD_HUU_HINH                | 4.81976e+18 | 9.5192e+12  | 4.81975e+18 |
| PC1      | consolidated |          2023 | PHAI_THU_KHACH_HANG_NGAN_HAN | 9.38985e+17 | 1.92817e+12 | 9.38984e+17 |
| PC1      | consolidated |          2023 | LNST_CHUA_PHAN_PHOI          | 4.26497e+17 | 9.56799e+11 | 4.26496e+17 |
| PC1      | consolidated |          2023 | XAY_DUNG_CO_BAN_DO_DANG      | 5.63433e+16 | 1.07818e+11 | 5.63432e+16 |
| PC1      | consolidated |          2023 | HANG_TON_KHO                 | 1.37887e+16 | 9.4408e+11  | 1.37877e+16 |
| BID      | consolidated |          2020 | TONG_TAI_SAN                 | 1.51669e+15 | 1.51669e+09 | 1.51668e+15 |
| BID      | consolidated |          2019 | TONG_TAI_SAN                 | 1.48996e+09 | 1.48996e+15 | 1.48996e+15 |
| BID      | consolidated |          2020 | NO_PHAI_TRA                  | 1.47686e+15 | 1.47686e+09 | 1.47686e+15 |
| BID      | consolidated |          2019 | NO_PHAI_TRA                  | 1.45113e+09 | 1.45113e+15 | 1.45113e+15 |
| BID      | consolidated |          2020 | CHO_VAY_KHACH_HANG           | 1.2143e+15  | 1.2143e+09  | 1.21429e+15 |
| BID      | consolidated |          2019 | CHO_VAY_KHACH_HANG           | 1.117e+09   | 1.117e+15   | 1.117e+15   |
| BID      | consolidated |          2016 | NO_PHAI_TRA                  | 9.84323e+14 | 9.84332e+08 | 9.84322e+14 |
| BID      | consolidated |          2015 | NO_PHAI_TRA                  | 8.29339e+08 | 8.29502e+14 | 8.29501e+14 |
| SHB      | consolidated |          2024 | TONG_TAI_SAN                 | 7.47478e+14 | 7.47478e+08 | 7.47477e+14 |
| SHB      | consolidated |          2023 | TONG_TAI_SAN                 | 6.30501e+08 | 6.30501e+14 | 6.305e+14   |
| SHB      | consolidated |          2024 | CHO_VAY_KHACH_HANG           | 5.22557e+14 | 5.15552e+08 | 5.22557e+14 |
| SHB      | consolidated |          2023 | CHO_VAY_KHACH_HANG           | 4.33913e+08 | 4.38464e+14 | 4.38464e+14 |
| KLB      | separate     |          2024 | NO_PHAI_TRA                  | 8.92044e+07 | 8.92044e+13 | 8.92043e+13 |
| KLB      | consolidated |          2024 | NO_PHAI_TRA                  | 8.88738e+07 | 8.88738e+13 | 8.88737e+13 |
| BID      | consolidated |          2020 | VON_CHU_SO_HUU               | 7.96466e+13 | 7.96466e+07 | 7.96465e+13 |