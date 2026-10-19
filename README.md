# TRAMIF

TRAMIF là dự án nghiên cứu phân loại họ malware theo thời gian. Câu hỏi nghiên cứu là: thông tin về độ tin cậy của từng cách biểu diễn ảnh, đo từ dữ liệu lịch sử, có giúp kết hợp các mô hình để dự đoán tốt hơn trên các tháng tương lai không? Đây là giả thuyết cần kiểm chứng.

## Đầu vào và đầu ra

Mỗi mẫu cần bytes disarmed, ngày quan sát, mã SHA-256 và nhãn họ malware. Pipeline đã tạo raw-byte, local-entropy và SBSMI cho cohort lịch sử; đầu ra model là xác suất trên các họ được chọn từ train. SHA định danh mẫu khác hash nội dung bytes đã disarm.

## Chia dữ liệu theo thời gian

- Train: tháng 8/2019 đến hết tháng 1/2020.
- Validation: tháng 2/2020 đến hết tháng 3/2020.
- Future test: tháng 4/2020 đến hết tháng 9/2020.

Việc chọn họ malware chỉ dùng dữ liệu train. Mô hình và cách kết hợp phải được cố định trước khi đánh giá trên future test.

## Trạng thái

Checkpoint 18/10/2026 — D062: đã hoàn tất đánh giá Future April–September
trên bảy phương pháp, đủ năm seeds, sau phê duyệt và freeze D060/D061.
Q+D đạt mean-month Macro-F1 64,7573% và worst-month 56,6472%; Q-only tương ứng
64,7494% và56,6301%. Gain mean chỉ+0,0079điểm phần trăm, **không đạt target+2điểm**.
Learned-static đạt64,9366%/57,5183%. Không claim significance hay full Q+D+U validation.
U vẫn diagnostic-only; D descriptive. Retention10/20 và phân tầng file size đã báo cáo.
E2 lịch sử âm được giữ nguyên;không retune beta1. Đây là kết quả fixed-horizon,
không chứng minh khả năng tổng quát hóa trên mọi giai đoạn tương lai.

- [Kết quả Future đầy đủ](docs/FINAL_FUTURE_RESULTS.md).
- [Kết quả lịch sử](docs/historical_research_results.md).
- [Protocol và quyết định](docs/RESEARCH_PROTOCOL_V1.md), [Decision log](DECISION_LOG.md).
- [Reproduction](docs/REPRODUCIBILITY.md), [Methods/Results draft](docs/PAPER_DRAFT.md).
- [Pre-Future readiness](docs/PRE_FUTURE_READINESS_REPORT.md).
- Checkpoint tiếp tục: `automation/continuation_state.json`.

Không dùng tests PASS như bằng chứng giả thuyết nghiên cứu đúng. Không commit dữ
liệu malware hoặc mở Future Test nếu chưa có phê duyệt riêng.
