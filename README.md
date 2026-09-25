# TRAMIF

TRAMIF là dự án nghiên cứu phân loại họ malware theo thời gian. Câu hỏi nghiên cứu là: thông tin về độ tin cậy của từng cách biểu diễn ảnh, đo từ dữ liệu lịch sử, có giúp kết hợp các mô hình để dự đoán tốt hơn trên các tháng tương lai không? Đây là giả thuyết cần kiểm chứng.

## Đầu vào và đầu ra

Mỗi mẫu cần bytes của file, ngày quan sát, mã SHA-256 và nhãn họ malware để phục vụ huấn luyện hoặc đánh giá. Từ cùng một file, dự án dự kiến tạo ba ảnh: raw-byte, local-entropy và SBSMI. Đầu ra của mô hình là xác suất cho các họ malware đã được chọn từ tập huấn luyện.

## Chia dữ liệu theo thời gian

- Train: tháng 8/2019 đến hết tháng 1/2020.
- Validation: tháng 2/2020 đến hết tháng 3/2020.
- Future test: tháng 4/2020 đến hết tháng 9/2020.

Việc chọn họ malware chỉ dùng dữ liệu train. Mô hình và cách kết hợp phải được cố định trước khi đánh giá trên future test.

## Trạng thái

Đây là workspace thực hiện cá nhân. Chưa có kết quả huấn luyện hay đánh giá mô hình trong workspace này.