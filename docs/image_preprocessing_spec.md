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

Nguồn: bài báo SBSMI, Section 4, công thức (1)–(2), trang 5–6.

Dãy bit được chia thành các khối dài `l` bit. Với hai khối liên tiếp
`m` và `n`, `f(m,n)` là số lần quan sát chuyển tiếp `m → n`.
Xác suất chuyển tiếp là `P(m,n) = f(m,n) / sum_k f(m,k)`.
Các xác suất tạo thành ma trận `2^l × 2^l`.
Mỗi ô được chuyển thành pixel 8-bit bằng `floor(P(m,n) × 255)`.

Bài báo minh họa `l = 6`, tương ứng ảnh 64 × 64.

Algorithm 1 mô tả việc đọc liên tiếp các khối bit, đếm chuyển tiếp,
chuẩn hóa theo hàng và chuyển xác suất thành mức xám. Phần thuật toán
đã đối chiếu chưa nêu rõ cách ReadBits xử lý khối cuối thiếu bit
hoặc cách chuẩn hóa một hàng có tổng số đếm bằng 0.

### Quy ước trong bản triển khai hiện tại

Các mục dưới đây mô tả code TRAMIF, chưa được coi là những chi tiết
đã xác nhận đầy đủ từ bài báo:

- Đọc bit từ bit có trọng số cao đến bit có trọng số thấp trong mỗi
  byte (MSB-first).
- Chia dãy bit liên tục thành các khối không chồng lấp. Phần bit
  còn lại giữa hai byte được giữ để ghép với byte tiếp theo.
- Hàm mặc định dùng l = 6; tham số thí nghiệm vẫn cần được chốt.
- Hàng không có lượt chuyển tiếp được biểu diễn bằng các pixel 0.
- Nếu cuối toàn bộ dữ liệu còn thiếu bit để tạo một khối, hàm hiện
  báo ValueError. Đây là hành vi tạm thời, chưa dùng để xử lý toàn
  bộ binary thật.
- Chuỗi có ít hơn hai trạng thái bị từ chối vì không có chuyển tiếp.

### Kiểm chứng đã hoàn thành

Năm test tự động đã vượt qua, kiểm tra:

1. Đọc trạng thái xuyên qua ranh giới byte.
2. Thứ tự đọc MSB-first.
3. Chiều chuyển tiếp và chuẩn hóa theo hàng.
4. Làm tròn xuống khi chuyển xác suất thành mức xám.
5. Báo lỗi khi khối cuối chưa đủ bit.

Các test xác nhận kết quả trên dữ liệu nhỏ đã tính tay.
Chưa kiểm chứng toàn bộ quy trình đọc binary thật và xuất ảnh,
hoặc mức sử dụng bộ nhớ khi xử lý file lớn.