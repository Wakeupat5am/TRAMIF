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

## Bảng TLSH toàn bộ train

Ngày hoàn tất: 06/10/2026.

- Tổng số mẫu train: 18.061.
- Lần chạy toàn bộ tạo 18.041 bản ghi và dùng lại 20 bản ghi.
- TLSH hợp lệ: 18.061.
- TNULL: 0.
- Lỗi: 0.
- Complete train inventory: True.

Khi tạo bản ghi, script kiểm tra kích thước và CRC với ZIP index,
đồng thời đối chiếu SHA-256 của bytes disarmed với báo cáo SBSMI.
Bản ghi CACHED được dùng lại; bytes không được băm lại trong lần đó.

Bảng kết quả cục bộ:
data/derived/train_tlsh_v1/train_tlsh.csv

Cấu hình và tổng kết:
data/derived/train_tlsh_v1/config.json
data/derived/train_tlsh_v1/summary.json

Chưa so sánh độ tương đồng giữa các mẫu, chưa chọn ngưỡng,
chưa nhóm hoặc loại mẫu. Hash hợp lệ không chứng minh rằng
dữ liệu không có mẫu gần trùng.

## Đối chiếu bytes nhóm G000007

Nhóm cùng TLSH chứa hai mẫu:
- wacatac, quan sát ngày 15/09/2019.
- autorun, quan sát ngày 20/10/2019.

Cả hai có kích thước 382.368 byte.
SHA-256 nội dung đọc lại khớp báo cáo đã lưu.

So sánh tại cùng vị trí:
- 385 vị trí có byte khác nhau.
- Tỷ lệ byte bằng nhau: 99,899312%.
- 37 vùng khác biệt liên tiếp.

Kết quả hỗ trợ việc xem đây là ứng viên gần trùng về bytes
có nhãn họ không thống nhất. Chưa xác định nhãn nào đúng,
chưa kết luận tương đương hành vi, chưa đổi nhãn hoặc loại mẫu.

Tỷ lệ giống nhau của cặp này là kết quả quan sát;
không được coi là ngưỡng loại mẫu đã được xác lập.

## Đối chiếu bytes nhóm G000038

Ngày kiểm tra: 07/10/2026.

- small: 9.600.947 byte.
- sillyp2p: 10.177.127 byte.
- SHA-256 nội dung của cả hai khớp giá trị đã lưu.
- Toàn bộ file small khớp chính xác phần đầu file sillyp2p.
- File sillyp2p có thêm 576.180 byte ở cuối.

Quan hệ tiền tố được kiểm tra trên toàn bộ bytes của file ngắn,
không suy ra từ việc lấy mẫu các đoạn.

Trong 64 đoạn mẫu dài 4.096 byte, 63 đoạn lặp trong file nguồn;
một đoạn khớp duy nhất ở cùng vị trí. Các đoạn lặp không được
dùng để suy luận vị trí dịch chuyển.

Kết quả xác nhận quan hệ bao chứa bytes giữa hai mẫu khác nhãn.
Chưa kết luận tương đương hành vi, nhãn nào đúng hoặc lịch sử
tạo file. Không đổi nhãn và không loại mẫu.

Báo cáo:
data/derived/mixed_tlsh_pair/20261007_071521_730105/report.json
## Khảo sát hàng xóm TLSH gần nhất trong toàn bộ train

Ngày hoàn tất: 07/10/2026.

Mỗi mẫu được so sánh với 18.060 mẫu train còn lại, loại chính nó.
Dùng phép TLSH diff chuẩn, có thành phần độ dài.
Khi nhiều mẫu đồng hạng gần nhất, thống kê nhãn xét tất cả
các mẫu đồng hạng.

- Số mẫu truy vấn: 18.061.
- Khoảng cách gần nhất nhỏ nhất / trung vị / lớn nhất: 0 / 6 / 264.
- 2.911 mẫu có ít nhất một hàng xóm khác nhãn ở khoảng cách
  nhỏ nhất, chưa giới hạn khoảng cách.
- Complete train audit: True.

| Khoảng cách gần nhất tối đa | Số mẫu truy vấn | Có hàng xóm khác nhãn ở khoảng cách nhỏ nhất |
| ---: | ---: | ---: |
| 0 | 180 | 4 |
| 10 | 10.154 | 1.288 |
| 20 | 11.674 | 1.487 |
| 30 | 12.621 | 1.625 |
| 50 | 13.913 | 1.844 |

Các mức là cộng dồn. Đây là số mẫu truy vấn, không phải số cặp
hoặc số nhóm. Cột cuối không thống kê tất cả các cặp khác nhãn
nằm trong mỗi mức khoảng cách.

180 mẫu ở khoảng cách 0 khớp kết quả kiểm tra nhóm cùng TLSH.
Bốn mẫu có hàng xóm khác nhãn ở khoảng cách 0 thuộc hai nhóm
G000007 và G000038 đã được đối chiếu bytes.

Chưa xác lập ngưỡng near-duplicate hoặc quy tắc gom nhóm.
Không suy ra nhãn sai chỉ từ khoảng cách TLSH.
Không đổi nhãn và không loại mẫu.

Kết quả cục bộ:
data/derived/train_tlsh_neighbors_v1/nearest_neighbors.csv
data/derived/train_tlsh_neighbors_v1/summary.json
data/derived/train_tlsh_neighbors_v1/config.json