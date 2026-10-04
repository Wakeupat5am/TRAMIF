# Trạng thái bộ ảnh SBSMI

## Phạm vi

Bộ ảnh được tạo từ binary BODMAS đã disarm, theo manifest
train/validation và bảng ánh xạ 51 họ đã cố định từ train.

- Train: 18.061 mẫu.
- Validation: 8.263 mẫu.
- Tổng: 26.324 mẫu.
- Chưa tạo bộ ảnh future test trong bước này.

Đầu ra: data/derived/sbsmi_dataset_v1/
Cấu hình: data/derived/sbsmi_dataset_v1/config.json

## Kết quả chạy hoàn tất ngày 04/10/2026

Theo output của scripts/build_sbsmi_dataset.py:

- Created: 1.107.
- Verified existing: 25.217.
- Errors: 0.
- Tổng số mẫu hoàn thành trong lượt chạy: 26.324.

Log:
data/derived/sbsmi_dataset_v1/logs/20261004_144302_905320.csv

Các mẫu CREATED được kiểm tra đầu vào, tạo ảnh và kiểm tra pixel
sau lưu. Các mẫu VERIFIED được kiểm tra lại nội dung đầu vào,
thông tin báo cáo, hash PNG và hash pixel theo cấu hình hiện tại.

Kết quả xác nhận hoàn thành preprocessing theo manifest;
không phải kết quả đánh giá mô hình phân loại.

## Bước tiếp theo

- Xây bộ nạp ảnh SBSMI cho tập đầy đủ.
- Giữ nguyên bảng nhãn trong docs/class_mapping.json.
- Giữ khả năng chọn validation riêng theo tháng để thực hiện
  đúng protocol hiệu chỉnh và lựa chọn mô hình.
- Kiểm tra batch trước khi huấn luyện baseline MalSBSLCNet.