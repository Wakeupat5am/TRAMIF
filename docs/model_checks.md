# Kiểm chứng triển khai MalSBSLCNet

## Phạm vi

Các kiểm tra này xác nhận hoạt động kỹ thuật của model và pipeline
nạp dữ liệu. Chưa phải kết quả đánh giá khả năng tổng quát hóa.

## Môi trường đã kiểm tra

- Python trong môi trường .venv của dự án.
- PyTorch: 2.13.0+cu126.
- NumPy: 2.4.6.
- CUDA runtime của PyTorch: 12.6.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU.
- Phép tính CUDA và chuyển đổi NumPy/PyTorch đã chạy thành công.

## Kiến trúc

Triển khai theo Figure 4 và bảng kiến trúc bổ sung của bài báo
A lightweight malware classification method based on short bit
sequence visualization.

- Đầu vào: một kênh, 64 × 64.
- Conv stem: 16 kênh.
- Sáu BaseBlock: 32, 32, 64, 64, 128, 128 kênh.
- Stride lần lượt: 2, 1, 2, 1, 2, 1.
- Adaptive average pooling: 4 × 4.
- Flatten: 2048.
- BatchNorm1d, Dropout 0.4, Linear.
- Đầu ra là logits.
- Cấu hình 9 lớp: 54.409 tham số có thể học.
- Cấu hình TRAMIF 51 lớp: 140.467 tham số có thể học.

BatchNorm và khởi tạo tham số dùng mặc định của PyTorch.
Chưa xác nhận mọi chi tiết triển khai với mã nguồn của tác giả.

## Bảng nhãn

docs/class_mapping.json chứa ánh xạ 51 họ sang chỉ số 0–50.
Thứ tự là tên họ sắp xếp alphabet, lấy từ cohort đã chọn bằng train.
Các nhánh và các tập dữ liệu phải dùng cùng bảng ánh xạ.

## Kiểm tra kiến trúc và batch thật

- Kích thước từng tầng khớp thiết kế.
- Forward/backward trên GPU: PASS.
- Dự đoán một ảnh trong chế độ eval: PASS.
- Đã kiểm tra 10 ảnh SBSMI pilot với ma trận trong báo cáo.
- Batch thật: (4, 1, 64, 64), float32.
- Nhãn: int64; logits: (4, 51).
- Gradient hữu hạn và optimizer cập nhật trọng số: PASS.

## Kiểm tra học thuộc tập nhỏ — 03/10/2026

Dùng đúng 10 mẫu train pilot thuộc 9 họ.
Mạng khởi tạo mới, giữ 51 đầu ra.

- Seed: 42.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Weight decay: 0.000001.
- Batch size: 10.
- Tối đa: 300 bước.
- Kiểm tra mỗi 25 bước bằng model.eval() trên cùng 10 mẫu.
- Tiêu chí kỹ thuật: accuracy 100% và loss dưới 0.1.

| Bước | Loss | Accuracy trên các mẫu train |
| ---: | ---: | ---: |
| 0 | 3,9342 | 10% |
| 25 | 3,5917 | 10% |
| 50 | 2,8590 | 20% |
| 75 | 0,9568 | 100% |
| 100 | 0,0388 | 100% |

Kết quả: PASS tại bước 100; mọi nhãn dự đoán khớp nhãn thật.

Báo cáo:
data/derived/tiny_training/20261003_125805_427887/report.json

Không dùng validation hoặc future test. Không lưu checkpoint.
Đây là kết quả kiểm tra khả năng học thuộc, không phải đánh giá
khả năng phân loại trên mẫu mới.

## Phần còn lại

- Tạo và kiểm tra SBSMI cho toàn bộ manifest train/validation.
- Chuẩn bị bộ nạp dữ liệu và quy trình huấn luyện baseline.
- Áp dụng protocol validation trước khi đánh giá future test.
- Chưa có kết quả baseline trên tập validation hoặc future test.