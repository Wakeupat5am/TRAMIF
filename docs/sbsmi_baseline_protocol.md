# Protocol baseline SBSMI — lượt phát triển đầu tiên

Ngày xác lập: 05/10/2026, trước khi đánh giá validation.

## Dữ liệu và mô hình

- Train: 18.061 mẫu, thuộc 51 họ đã chọn từ train.
- Đầu vào: SBSMI 64 × 64, một kênh, chia pixel cho 255.
- Mô hình: MalSBSLCNet, 51 đầu ra.
- Không dùng augmentation trong lượt này.

## Huấn luyện

- Khởi tạo mới với seed 42.
- Adam, learning rate 0.0001, weight decay 0.000001.
- Cross-entropy không trọng số lớp.
- Batch size 32, shuffle train, không bỏ batch cuối.
- Float32, không dùng mixed precision.
- Learning rate cố định, không có scheduler.
- Huấn luyện 100 epoch, không early stopping.

## Checkpoint

Lưu checkpoint sau mỗi epoch.
Checkpoint được chỉ định cho baseline này là epoch_100.pt.
Không chọn checkpoint theo kết quả trên train hoặc validation.

## Phạm vi sử dụng dữ liệu

Lượt huấn luyện này chỉ dùng train.
Tháng 2 được giữ cho hiệu chỉnh và ước lượng reliability.
Tháng 3 được giữ cho lựa chọn theo thời gian.
Chưa sử dụng future test tháng 4–9.

## Trạng thái kiểm chứng

Lượt kỹ thuật một epoch đã hoàn thành:
- Đủ 18.061 mẫu, 565 batch.
- Thời gian epoch: 45,7 giây.
- Lưu và đọc lại checkpoint: PASS.

Đánh giá checkpoint đó trên train:
- Cross-entropy: 0,8647.
- Accuracy: 80,55%.
- Macro F1: 0,7113.
- Balanced accuracy: 69,81%.

Các chỉ số trên train không đo khả năng tổng quát hóa.
Chính sách near-duplicates và đánh giá nhiều seed còn cần hoàn thiện
trước khi công bố kết quả nghiên cứu chính thức.