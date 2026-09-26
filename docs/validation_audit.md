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