# Kết quả baseline SBSMI — seed 42

Ngày ghi nhận: 06/10/2026.

## Phạm vi

- Mô hình: MalSBSLCNet, 51 lớp.
- Train: 18.061 mẫu, tháng 08/2019–01/2020.
- Seed: 42; huấn luyện 100 epoch.
- Checkpoint được chỉ định trước khi đánh giá: epoch_100.pt.
- Chưa sử dụng future test tháng 04–09/2020.

Thư mục lượt huấn luyện:
data/derived/sbsmi_training/20261005_105325_352508/

SHA-256 checkpoint:
0c7d76ec3e077e5f88dd6d455b14a041ee2d701a2dc170d5718928695bcf2c4d

## Validation trước hiệu chỉnh

| Tháng | Mẫu | Họ có mẫu | Accuracy | Macro F1 trên họ có mẫu | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 02/2020 | 3825 | 49 | 81,12% | 0,7144 | 70,22% |
| 03/2020 | 4438 | 47 | 70,39% | 0,7280 | 72,71% |

Tháng 2 vắng allaple và tescrypt.
Tháng 3 vắng allaple, ausiv, socks và tescrypt.
Mô hình vẫn giữ nguyên 51 đầu ra.

## Temperature scaling

- Tối ưu mean NLL không trọng số trên tháng 2.
- Một nhiệt độ chung cho cả 51 lớp.
- Khoảng T: [0.05, 20].
- Nhiệt độ tìm được: 2.519342177667692.
- Nghiệm không chạm biên.
- Giữ nguyên T khi áp dụng tháng 3.
- Không cập nhật trọng số mạng.

| Dữ liệu | NLL trước | NLL sau | Brier trước | Brier sau | ECE trước | ECE sau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tháng 2 — dùng fit T | 1,4709 | 0,8167 | 0,3286 | 0,2810 | 0,1416 | 0,0741 |
| Tháng 3 — forward validation | 1,9284 | 1,0612 | 0,4841 | 0,3856 | 0,2230 | 0,1062 |

Brier được tính bằng tổng sai số bình phương trên các lớp, rồi
trung bình trên mẫu. ECE dùng 15 khoảng confidence bằng nhau.

Kiểm tra nhãn dự đoán trước/sau: PASS, không thay đổi.
Accuracy và F1 giữ nguyên.

## Báo cáo gốc

Các đường dẫn dưới đây nằm trong thư mục lượt huấn luyện:

- validation_2020-02_20261005_200655_311417/report.json
- validation_2020-03_20261005_130355_846029/report.json
- temperature_scaling_20261006_103958_393138/temperature.json
- temperature_scaling_20261006_103958_393138/report.json

Dùng giá trị đầy đủ trong temperature.json khi áp dụng hiệu chỉnh.

## Diễn giải và giới hạn

Cả NLL, Brier và ECE đều giảm trên tháng 3 khi dùng nhiệt độ
ước lượng từ tháng 2. Đây là bằng chứng phát triển cho việc
hiệu chỉnh xác suất của checkpoint SBSMI này.

Chỉ số tháng 2 được đo trên chính dữ liệu dùng fit T,
không phải đánh giá độc lập của phép hiệu chỉnh.

Kết quả hiện chỉ có một seed. Chính sách near-duplicates còn
cần hoàn thiện. Chưa có kết quả future test hoặc multi-view fusion.