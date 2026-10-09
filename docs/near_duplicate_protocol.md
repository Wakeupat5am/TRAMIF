# Chính sách mẫu trùng và gần trùng — TRAMIF v1

Ngày đề xuất: 09/10/2026.

Trạng thái: đặc tả trước khi áp dụng lên validation.
Chưa hoàn tất kiểm tra giữa các giai đoạn.

## 1. Mục đích và căn cứ

Framework §11.2 yêu cầu loại bản sao chính xác, có chính sách
nhóm mẫu gần trùng dựa trên dữ liệu lịch sử, xử lý bản sao đến sau
trong đánh giá và báo cáo độ nhạy theo mức độ lọc.

TLSH phục vụ kiểm soát dữ liệu và tạo nhóm.
TLSH không phải đầu vào của các mô hình phân loại.

## 2. Trình tự thực tế

Đã tính TLSH cho toàn bộ 18.061 mẫu train và khảo sát các nhóm
cùng hash, khoảng cách gần nhất và hai cặp khác nhãn.

Đã xem kết quả validation của baseline SBSMI trước khi xây dựng
chính sách này. Không mô tả chính sách là đã được chốt trước lần
đánh giá validation đó.

Chưa dùng future test để xây dựng hoặc điều chỉnh chính sách.

Các báo cáo và checkpoint trước đây được giữ nguyên.
Kết quả validation trước lọc là kết quả phát triển ban đầu.

## 3. Phạm vi dữ liệu

Áp dụng trên cohort 51 họ đã chọn từ train.

Train: 01/08/2019 đến trước 01/02/2020.
Validation: 01/02/2020 đến trước 01/04/2020.
Future test: tháng 4 đến hết tháng 9/2020.

Cả ba biểu diễn của một mẫu dùng chung quyết định giữ hoặc loại.
Không di chuyển mẫu tương lai vào train.

## 4. Quan hệ trùng và gần trùng

Bản sao chính xác:
SHA-256 của bytes disarmed giống nhau.

Quan hệ gần trùng chính:
Hai TLSH hợp lệ có khoảng cách chuẩn tlsh.diff bằng 0.

Hai mức kiểm tra độ nhạy:
- tlsh.diff <= 10.
- tlsh.diff <= 20.

Dùng khoảng cách có thành phần độ dài.
Không dùng nhãn họ để quyết định hai mẫu có quan hệ gần trùng.

Các ngưỡng là lựa chọn triển khai của dự án, không phải ngưỡng
bắt buộc từ framework hoặc bằng chứng về tương đương hành vi.

Khoảng cách lớn hơn ngưỡng không chứng minh hai mẫu độc lập.
Chính sách chính có thể bỏ sót mẫu gần trùng ở khoảng cách khác 0.

## 5. Xử lý train

Bản sao chính xác, nếu phát hiện, giữ mẫu sớm nhất.
Nếu cùng timestamp, dùng thứ tự SHA để quyết định nhất quán.

Giữ các mẫu train khác bytes dù chúng có quan hệ gần trùng.
Không đổi nhãn dựa trên TLSH.

Tạo nhóm train cho từng mức chính sách:
nối các mẫu có quan hệ trùng hoặc gần trùng, rồi lấy các thành phần
liên thông làm nhóm.

Thành phần liên thông có thể chứa hai mẫu không trực tiếp gần nhau,
nhưng được nối qua các mẫu trung gian. Phải báo cáo kích thước nhóm
lớn nhất và số nhóm để thể hiện ảnh hưởng này.

Các nhóm phục vụ lấy mẫu lại khi ước lượng bất định.
Nhóm có nhiều nhãn được ghi nhận; không tự sửa nhãn.

## 6. Xử lý validation theo thời gian

Sắp xếp mẫu theo timestamp, sau đó theo SHA.

Loại khỏi đánh giá một mẫu nếu nó trùng chính xác hoặc có quan hệ
gần trùng với bất kỳ mẫu nào xuất hiện trước trong phạm vi kiểm tra.

Tập mẫu trước đó gồm train và các mẫu validation đã xét,
kể cả mẫu đã bị loại. Vì vậy, một chuỗi mẫu tương tự cũng có thể
dẫn tới loại các bản đến sau.

Quyết định chỉ dựa trên nội dung và thứ tự thời gian.
Không dùng nhãn, dự đoán hoặc accuracy để quyết định loại.

Mỗi mẫu bị loại phải ghi được mẫu trước đó làm căn cứ,
loại quan hệ và khoảng cách TLSH nếu có.

Chỉ loại khỏi manifest đánh giá; không xóa binary hoặc ảnh.

## 7. TLSH không hợp lệ

Nếu TLSH là TNULL, vẫn kiểm tra trùng chính xác bằng SHA-256.

Không gán khoảng cách giả và không coi TNULL là bằng chứng
mẫu không gần trùng. Ghi rõ số mẫu không kiểm tra được bằng TLSH.

## 8. Đầu ra bắt buộc của bước lịch sử

- Bảng TLSH validation có kiểm tra nguồn bytes.
- Bảng nhóm train cho ba mức chính sách.
- Manifest quyết định giữ/loại validation cho từng mức.
- Báo cáo số giữ/loại theo tháng và họ.
- Số nhóm nhiều nhãn, nhóm lớn nhất và số TNULL.
- Mã kiểm tra cấu hình và các bảng đầu vào.

## 9. Điều kiện kết thúc bước lịch sử

- Có kết quả cho đủ 18.061 mẫu train và 8.263 mẫu validation.
- Mỗi mẫu validation có đúng một quyết định tại mỗi mức chính sách.
- Tổng số giữ và loại bằng số đầu vào.
- Mỗi quyết định loại có căn cứ hợp lệ từ thời điểm trước.
- Không còn quan hệ bị cấm giữa mẫu validation được giữ và các
  mẫu trước nó, theo đúng mức chính sách.
- Các nhánh dùng chung manifest giữ/loại.

Không mở rộng khảo sát ngưỡng hoặc soi thêm từng cặp sau bước này,
trừ khi phát hiện lỗi triển khai ảnh hưởng đến quyết định dữ liệu.

## 10. Quan hệ với huấn luyện và fusion

Nếu train không thay đổi, không bắt buộc huấn luyện lại chỉ vì
manifest validation thay đổi.

Hiệu chỉnh xác suất trên tháng 2 và lựa chọn trên tháng 3 phải dùng
manifest tương ứng với chính sách chính. Không tái sử dụng trực tiếp
temperature đã fit trên validation chưa lọc.

Các mức độ nhạy được báo cáo riêng, không dùng để chọn ngưỡng
có kết quả phân loại cao nhất.

## 11. Future test

Sau khi cố định mô hình, hiệu chỉnh và quy tắc fusion, áp dụng
cùng chính sách theo thứ tự thời gian lên future test.

Không đổi ngưỡng theo kết quả future test.
Lọc dữ liệu không được cập nhật trọng số hoặc quy tắc fusion.

Chỉ coi kiểm tra xuyên giai đoạn hoàn tất khi bước future test
cũng đã được thực hiện và báo cáo.