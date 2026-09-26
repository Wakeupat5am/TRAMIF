# Validation metadata audit

Nguồn: `data/raw/bodmas_metadata.csv` và danh sách họ đã cố định trong
`docs/train_family_cohort.csv`.

Giai đoạn validation: 01/02/2020–31/03/2020.

- Tổng số dòng validation: 16.539.
- Dòng có nhãn `family`: 9.259.
- Dòng thuộc 51 họ đã chọn từ train: 8.263.
- SHA duy nhất trong nhóm validation thuộc cohort: 8.263.
- Họ trong cohort không xuất hiện ở validation: `allaple`, `tescrypt`.
- SHA trùng giữa mọi dòng train và nhóm validation thuộc cohort: 0.

Danh sách 51 họ vẫn giữ theo train; không thêm hoặc loại họ dựa trên kết quả validation.
Chưa sử dụng nhãn future test trong bước kiểm tra này.

## Số SHA validation theo họ

Mười họ có ít SHA validation nhất trong cohort:

| Họ | SHA validation |
| --- | ---: |
| allaple | 0 |
| tescrypt | 0 |
| ausiv | 1 |
| socks | 1 |
| fuerboos | 3 |
| vb | 6 |
| blocker | 8 |
| cryptinject | 10 |
| vobfus | 10 |
| ditertag | 11 |

Validation không cung cấp mẫu để đánh giá riêng `allaple` và `tescrypt`.
Điểm số riêng của các họ chỉ có vài SHA cần được diễn giải thận trọng.
Cohort vẫn giữ nguyên theo quy tắc đã áp dụng trên train.