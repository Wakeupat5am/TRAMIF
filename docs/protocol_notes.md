# Ghi chú protocol TRAMIF

Nguồn chính: Temporal_Reliability_Malware_Research_Framework.docx.

## Bài toán — §3.1
Dự đoán họ malware đã có đại diện trong dữ liệu huấn luyện.

## Chia theo ngày quan sát — §11.2
- Train: 01/08/2019–31/01/2020.
- Validation: 01/02/2020–31/03/2020.
- Future test: 01/04/2020–30/09/2020.

Danh sách họ đủ điều kiện phải được chọn từ train. Không dùng nhãn future test để chọn mô hình.

## Trước khi kiểm tra future test — §7
Cố định cách tạo ảnh, mô hình, hiệu chỉnh xác suất và trọng số kết hợp.