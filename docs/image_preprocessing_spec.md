# Đặc tả tạo ảnh từ binary

Trạng thái: bản nháp. Cần chốt các tham số trước khi dùng validation
để chọn mô hình; không điều chỉnh chúng theo kết quả future test.

## Đầu vào chung

Mỗi mẫu được nhận diện bằng SHA trong manifest. Ba ảnh của một mẫu
phải được tạo từ cùng một dãy bytes của binary BODMAS đã disarm.
Không dùng `bodmas_feature_vectors.npz.zip` để dựng lại các ảnh này.
Kết quả nghiên cứu phải ghi rõ đầu vào là binary đã disarm.

## Ba cách biểu diễn

1. **Raw-byte:** biểu diễn giá trị từng byte (0–255) thành mức xám.
2. **Entropy:** biểu diễn entropy tính trên các đoạn bytes cục bộ.
3. **SBSMI:** biểu diễn chuỗi bit ngắn theo phương pháp của bài báo SBSMI.

## Tham số cần xác nhận từ tài liệu

- Số bytes đầu vào tối đa cho mỗi mẫu.
- Kích thước và cách sắp xếp pixel của từng ảnh.
- Cách xử lý khi file ngắn hoặc dài hơn kích thước cần thiết.
- Kích thước cửa sổ, bước trượt và cách đưa entropy về mức xám.
- Quy tắc tạo giá trị pixel SBSMI, đặc biệt với nhóm bit cuối chưa đủ độ dài.

## SBSMI — quy tắc đã xác nhận

Nguồn: bài báo SBSMI, Section 4.1, trang 5–6, công thức (1)–(2).

Dãy bit được chia thành các khối không chồng lấp, dài l bit,
bắt đầu từ bit đầu tiên.

Nếu khối cuối chưa đủ l bit, vẫn chuyển các bit còn lại thành
số nguyên; các bit thiếu ở phía trọng số cao được coi là 0.
Ví dụ, với l = 6, khối cuối `11` có giá trị 3, tương đương
`000011`; không chuyển thành `110000`.

Với hai trạng thái liên tiếp m và n, f(m,n) là số lần quan sát
chuyển tiếp m → n. Xác suất được chuẩn hóa theo từng hàng:

P(m,n) = f(m,n) / sum_k f(m,k)

Ma trận có kích thước 2^l × 2^l.
Mỗi ô được chuyển thành pixel bằng floor(P(m,n) × 255).
Hàng có tổng số đếm bằng 0 được giữ bằng 0; framework §4.3
cũng quy định rõ hàng không có lượt chuyển tiếp là hàng 0.

Bài báo minh họa l = 6, tương ứng ảnh 64 × 64.
Framework TRAMIF §4.3 chọn trạng thái 6 bit cho thiết kế ban đầu.

### Chi tiết triển khai hiện tại

- Đọc bit theo thứ tự MSB-first trong mỗi byte. Thứ tự này đã
  được kiểm tra bằng ví dụ tính tay; phần văn bản đã đối chiếu
  chưa nêu rõ thứ tự bit trong từng byte.
- Giữ phần bit dư giữa các byte để ghép với byte tiếp theo.
- Giữ khối cuối chưa đủ bit bằng giá trị số nguyên của nó.
- Hàm mặc định dùng l = 6 và hỗ trợ l từ 1 đến 8.
  Giới hạn này là lựa chọn triển khai.
- Chuỗi có ít hơn hai trạng thái bị từ chối vì không có chuyển tiếp.
  Đây là quy tắc đầu vào của code hiện tại.
- `scripts/sbsmi_core.py` giữ phiên bản đọc toàn bộ trạng thái
  để làm đối chứng trên dữ liệu nhỏ.
- `scripts/sbsmi_stream.py` đọc file theo từng khối, mặc định
  65.536 byte, và đếm chuyển tiếp ngay khi đọc.
- Bit dư và trạng thái trước được giữ xuyên qua ranh giới khối.
  Nhóm cuối thiếu bit chỉ được xử lý khi hết toàn bộ file.

### Kiểm chứng đã hoàn thành

Bộ kiểm tra hiện có 11 test, tất cả đã vượt qua:

- Sáu test kiểm tra phần lõi: thứ tự bit, đọc xuyên ranh giới byte,
  đếm chuyển tiếp, chuẩn hóa, làm tròn và giữ nhóm bit cuối.
- Năm test kiểm tra bộ đọc từng khối: đối chiếu với bản tham chiếu
  ở nhiều kích thước khối và độ dài trạng thái, đối chiếu phép tính
  tay, từ chối đầu vào không có chuyển tiếp và kích thước khối bằng 0.

Script `scripts/export_sbsmi_demo.py` đã đọc file mẫu hai byte
FF FF bằng bộ đọc từng khối và xuất ảnh PNG 64 × 64, chế độ L
(grayscale 8-bit). Đọc lại PNG cho kết quả khớp toàn bộ 4.096 pixel
với ma trận trước khi lưu.

Các kiểm tra dùng dữ liệu mô phỏng.
Chưa kiểm chứng trên binary BODMAS thật hoặc đo thời gian và
bộ nhớ trên file lớn.