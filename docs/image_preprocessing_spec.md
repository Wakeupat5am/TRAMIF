# Đặc tả tạo ảnh từ binary

## Trạng thái và phạm vi

Đã triển khai và kiểm thử các thành phần tạo ảnh SBSMI, raw-byte
và local entropy. Đã chạy pilot ba biểu diễn trên một mẫu BODMAS
thuộc train.

Đây là cấu hình triển khai hiện tại, chưa phải xác nhận toàn bộ
pipeline đã sẵn sàng xử lý dataset. Chưa có kết quả huấn luyện
hoặc đánh giá mô hình.

Mọi thay đổi tham số phải được ghi lại. Không điều chỉnh biểu diễn
theo kết quả future test.

## Nguồn thiết kế

- Temporal_Reliability_Malware_Research_Framework.docx:
  §4.1 raw-byte, §4.2 entropy, §4.3 SBSMI, §4.4 kiểm soát triển khai.
- A lightweight malware classification method based on short bit
  sequence visualization: Section 4.1, trang 5–6, công thức (1)–(2)
  và Algorithm 1 cho SBSMI.
- README BODMAS: đặc điểm binary disarmed và ý nghĩa metadata.

## Đầu vào chung

Ba biểu diễn của một mẫu phải được tạo từ cùng bytes của binary
BODMAS đã disarm. Hiện pilot đọc file trong data/raw/altered.

Không dùng bodmas.npz để dựng lại các ảnh này. Pipeline chỉ đọc
binary như dữ liệu, không thực thi hoặc khôi phục khả năng thực thi.

Phân biệt hai mã nhận diện:

- sample_identifier: SHA được ghi trong metadata và tên file.
- disarmed_content_sha256: SHA-256 tính trên bytes disarm thực tế.

Hai mã này không bắt buộc bằng nhau. Các báo cáo của cùng mẫu
phải có cùng disarmed_content_sha256.

## Raw-byte

Nguồn thiết kế: framework §4.1.

- Xếp bytes theo thứ tự offset vào các hàng rộng 256.
- Giá trị byte nằm trong khoảng 0–255.
- Hàng cuối thiếu ô được đệm bằng 0.
- Mask bằng 1 tại byte thật và bằng 0 tại ô đệm.
- Byte thật có giá trị 0 vẫn có mask bằng 1.
- Resize theo diện tích có xét mask về 64 × 64.
- Giữ kết quả resize dạng số thực; chia cho 255 để về khoảng 0–1.

Triển khai:
- scripts/raw_byte_core.py: bytes_to_layout().
- scripts/masked_area.py: masked_area_resize().
- scripts/run_bodmas_raw_byte_pilot.py: pilot một mẫu.

## Local entropy

Nguồn thiết kế: framework §4.2.

- Tính Shannon entropy của phân phối giá trị byte.
- Cửa sổ dài 256 byte, bước trượt 128 byte.
- Chia entropy tính bằng bit cho 8 để về khoảng 0–1.
- Mỗi offset nhận trung bình entropy chuẩn hóa của các cửa sổ
  phủ lên offset đó.
- Xếp các giá trị lên cùng layout rộng 256 và dùng cùng mask
  như raw-byte.
- Resize theo diện tích có xét mask về 64 × 64.
- Không chia thêm cho 255 sau resize.

Quy tắc biên được chọn cho triển khai hiện tại:

- File ngắn hơn một cửa sổ: dùng toàn bộ bytes có thật.
- Cửa sổ cuối thiếu bytes: chỉ dùng phần còn lại, không thêm byte 0.
- Dừng tại cửa sổ đầu tiên chạm cuối file.
- Đầu vào rỗng bị từ chối.

Framework yêu cầu ghi rõ cách xử lý biên; các quy tắc trên là
lựa chọn triển khai của dự án.

Triển khai:
- scripts/entropy_core.py: shannon_entropy(), local_entropy_values().
- scripts/run_bodmas_entropy_pilot.py: pilot một mẫu.

## Phép resize dùng chung

Với mỗi pixel đầu ra:

pixel = tổng(value × mask × diện tích giao)
        / tổng(mask × diện tích giao)

- Ô đệm không tham gia tử số hoặc mẫu số.
- Vùng không có dữ liệu hợp lệ trả về 0.
- Tính diện tích giao bằng hệ tọa độ nguyên đã nhân tỷ lệ.
- Kết quả trung bình giữ dạng số thực.

Quy tắc trả về 0 cho vùng không có dữ liệu là lựa chọn triển khai.

## SBSMI

Nguồn: bài báo SBSMI Section 4.1 và framework §4.3.

- Chia bitstream thành các khối không chồng lấp dài l bit.
- Cấu hình hiện tại dùng l = 6, tạo ma trận 64 × 64.
- Đếm chuyển tiếp giữa hai trạng thái liên tiếp.
- Chuẩn hóa theo tổng số chuyển tiếp đi ra của từng hàng.
- Pixel bằng floor(255 × xác suất chuyển tiếp).
- Hàng không có chuyển tiếp được giữ bằng 0.
- Giữ khối cuối thiếu bit bằng giá trị số nguyên của các bit còn lại;
  các bit thiếu phía trọng số cao được coi là 0.
- Ví dụ FF FF với l = 6 tạo các trạng thái [63, 63, 15].

Chi tiết triển khai:

- Đọc MSB-first trong mỗi byte. Đã kiểm tra bằng ví dụ tính tay;
  phần văn bản đã đối chiếu chưa nêu rõ thứ tự bit trong từng byte.
- Giữ bit dư và trạng thái trước xuyên qua ranh giới byte/khối đọc.
- Bộ đọc từng khối mặc định dùng 65.536 byte.
- Code hỗ trợ l từ 1 đến 8; đây là giới hạn triển khai.
- Ít hơn hai trạng thái bị từ chối vì không có chuyển tiếp.

Triển khai:
- scripts/sbsmi_core.py: bản tham chiếu cho dữ liệu nhỏ.
- scripts/sbsmi_stream.py: file_to_sbsmi().
- scripts/run_bodmas_sbsmi_pilot.py: pilot một mẫu.

## Định dạng đầu ra và chuẩn hóa

Raw-byte và entropy:
- JSON lưu normalized_matrix dạng số thực trong khoảng 0–1.
- PNG chỉ dùng để xem thử, pixel = floor(255 × normalized_value).
- Không dùng PNG thay ma trận số thực trong pipeline huấn luyện
  mà không ghi nhận thay đổi về lượng tử hóa.

SBSMI:
- PNG giữ ma trận số nguyên 0–255 theo quy tắc của biểu diễn.
- Khi đưa vào mô hình, dự kiến chia các pixel này cho 255.
- Bước nạp dữ liệu huấn luyện chưa được triển khai.

Mỗi pilot lưu báo cáo về mẫu, hash nội dung và cấu hình.
Kiểm tra pixel round-trip so sánh toàn bộ pixel trước khi lưu
với pixel đọc lại từ PNG.

## Bằng chứng kiểm chứng hiện có

Output kiểm thử do người thực hiện chạy ghi nhận 36 test pass:
- 6 test SBSMI core.
- 5 test SBSMI stream.
- 5 test raw-byte layout.
- 7 test masked area resize.
- 6 test Shannon entropy.
- 7 test local entropy.

Pilot BODMAS:
- Split: train.
- Family: autoit.
- Sample identifier:
  7f2d1e70b3dcb0c8e85a8e7c88ef2594c2f371055ba49e0941764c9df9cc798e
- Disarmed content SHA-256:
  84222d1561877afa0e7f39531afa1783bf6e9a4b5279302adf2cf271e8ee8692
- Kích thước: 337.866 byte.
- Layout raw-byte/entropy: 1.320 hàng × 256 cột, có 54 ô đệm.
- Cả ba ảnh: 64 × 64, PNG grayscale mode L.
- Ba pilot báo CRC khớp ZIP index và pixel round-trip PASS.
- Đã đối chiếu ba JSON: cùng sample identifier và hash nội dung.

Các kết quả trên được ghi lại từ output chạy trước đó;
không phải bộ test được tự động chạy khi đọc tài liệu này.

## Phần chưa được xác nhận

- Tốc độ và bộ nhớ trên các file lớn.
- Tính toàn vẹn nội dung của toàn bộ dataset đã giải nén.
- Xử lý hàng loạt, ghi nhận mẫu lỗi và tiếp tục khi gián đoạn.
- Pipeline nạp ba biểu diễn cho huấn luyện.
- Hiệu quả phân loại và khả năng tổng quát hóa theo thời gian.

Test và pilot một mẫu không chứng minh các mục trên đã hoàn thành.