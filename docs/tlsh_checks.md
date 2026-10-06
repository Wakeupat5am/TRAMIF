# Kiểm chứng TLSH

Ngày kiểm tra: 06/10/2026.

## Mục đích

Kiểm chứng cách tính TLSH trước khi sử dụng để tìm các mẫu
có nội dung gần giống nhau trong BODMAS.

TLSH phục vụ kiểm tra dữ liệu, không phải đặc trưng đầu vào
của mô hình MalSBSLCNet.

## Môi trường và cách tính

- Thư viện: py-tlsh 5.0.0.
- Đầu vào: bytes của binary BODMAS đã disarm.
- Đọc theo khối 65.536 byte.
- Gom đủ 5 byte đầu trước lần gọi update() đầu tiên.
- Không thêm, bỏ hoặc thay đổi bytes đầu vào.

Trên môi trường hiện tại, cập nhật trực tiếp từng byte từ đầu
cho kết quả khác one-shot trên dữ liệu tổng hợp.
Hàm có bộ đệm đầu luồng được sử dụng để xử lý vấn đề này.

## Kiểm tra tổng hợp

- Khoảng cách của cặp hash tham chiếu chính thức: 427 — PASS.
- Tính đối xứng và khoảng cách với chính nó: PASS.
- Hash one-shot của đầu vào tổng hợp khớp giá trị đã ghi nhận.
- 64 phép so sánh streaming với one-shot: PASS.
- Bao gồm khối đọc nhỏ và phần dư cuối luồng.
- Đầu vào quá ngắn hoặc toàn byte 0 trả về TNULL đúng kỳ vọng.

Giá trị 427 chỉ dùng kiểm tra thư viện;
không phải ngưỡng xác định mẫu gần trùng.

## Pilot trên BODMAS

Manifest: data/manifests/preprocessing_pilot.csv.

- 10 mẫu train, thuộc 9 họ.
- Kích thước khoảng 0,003–4,979 MiB.
- Kiểm tra kích thước và CRC: 10/10 PASS.
- Hash streaming khớp chính xác one-shot: 10/10 PASS.
- TLSH hợp lệ: 10.
- TNULL: 0.
- Lỗi: 0.

Báo cáo cục bộ:
data/derived/tlsh_pilot/20261006_044548_937167/report.json

Bảng kết quả:
data/derived/tlsh_pilot/20261006_044548_937167/summary.csv

## Giới hạn và bước tiếp theo

Pilot này kiểm chứng việc tính hash trên các mẫu đã chọn.
Chưa kết luận tỷ lệ TNULL hoặc mức độ gần trùng của toàn bộ dữ liệu.

Chưa chọn ngưỡng tương đồng, chưa nhóm hoặc loại mẫu.
Bước tiếp theo là tính TLSH cho tập train và khảo sát các cặp
tương đồng trong train trước khi quyết định quy tắc xử lý.

Việc xây dựng quy tắc near-duplicate diễn ra sau khi đã xem
kết quả validation của baseline SBSMI. Phải công bố trình tự này;
không mô tả đây là quy tắc đã chốt trước khi xem validation.

Future test chưa được sử dụng trong các bước kiểm tra TLSH này.