# Kiểm tra bộ dữ liệu SBSMI trước huấn luyện

## Bộ nạp dữ liệu

Mã nguồn: scripts/sbsmi_dataset.py.

- Train: 18.061 mẫu.
- Validation: 8.263 mẫu.
- Validation tháng 02/2020: 3.825 mẫu.
- Validation tháng 03/2020: 4.438 mẫu.
- Bảng nhãn cố định: 51 họ, chỉ số từ 0 đến 50.
- Batch kiểm tra: 4 ảnh train đầu tiên.
- Tensor ảnh: float32, kích thước (4, 1, 64, 64), giá trị trong [0, 1].
- Tensor nhãn: int64.
- Nhãn batch: [3, 21, 17, 17].
- Kết quả kiểm tra bộ nạp: PASS.

Kiểm tra batch chưa phải kiểm tra mọi ảnh bằng bộ nạp.
Không huấn luyện hoặc đánh giá mô hình trong bước này.

## Kiểm tra nội dung trùng lặp — 05/10/2026

Đối chiếu trường disarmed_content_sha256 trong báo cáo preprocessing
của từng mẫu với danh sách train/validation.

- Báo cáo đã đọc: 26.324.
- SHA-256 nội dung disarmed duy nhất: 26.324.
- Nhóm trùng nội dung: 0.
- Nhóm nội dung xuất hiện ở cả train và validation: 0.
- Mẫu validation lặp nội dung train: 0.
- Nhóm cùng nội dung nhưng khác nhãn họ: 0.

Không loại mẫu nào theo kết quả kiểm tra này.

Phạm vi: hash nội dung được ghi ở bước preprocessing, không phải
đọc và hash lại binary trong lần kiểm tra này. Kết quả không thay thế
kiểm tra các biến thể gần giống nhau (near-duplicates).

## Quy tắc sử dụng validation

Theo framework §7:
- Tháng 2 dành cho hiệu chỉnh xác suất và ước lượng reliability.
- Tháng 3 dành cho lựa chọn theo thời gian.
- Chưa sử dụng future test tháng 4–9.

Cách chọn checkpoint và chính sách near-duplicates cần được ghi rõ
trước khi sử dụng kết quả làm đánh giá nghiên cứu chính thức.