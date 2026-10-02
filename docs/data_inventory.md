# Kiểm kê dữ liệu BODMAS

Nguồn: thư mục BODMAS trên Google Drive.

| File | Dung lượng Drive hiển thị | Vai trò | Trạng thái tại máy |
| --- | ---: | --- | --- |
| bodmas_metadata.csv | 12,1 MB | SHA-256, thời điểm quan sát và nhãn họ | Đã tải vào data/raw/bodmas_metadata.csv |
| bodmas_feature_vectors.npz.zip | 568,5 MB | Đặc trưng có sẵn; không tạo được ba ảnh từ các vector này | Chưa tải |
| BODMAS_disarmed_malware_binaries.zip | 67,92 GB | Chứa bytes cần cho ba ảnh raw-byte, entropy và SBSMI | Chưa tải |

ZIP binaries lớn, nên cần chọn cách xử lý dung lượng trước khi tải và giải nén. README của bộ dữ liệu mô tả các binaries là “disarmed”; ảnh tạo từ chúng cần ghi rõ đặc điểm này trong thí nghiệm.

## Dung lượng và quyết định hiện tại

Ngày 25/09/2026, ổ đĩa dự kiến dùng cho dữ liệu còn khoảng 300 GB trống (theo kiểm tra trên máy cá nhân). README BODMAS ước tính cần khoảng 370 GB nếu giữ đồng thời ZIP, bản disarmed đã giải nén và bản binaries được khôi phục. Vì vậy, hiện chưa tải ZIP lớn hoặc giải nén toàn bộ; cách xử lý sẽ được quyết định sau khi kiểm tra nhu cầu bytes của thí nghiệm.

## Dấu nhận diện bản metadata đã tải

- File tại máy: `data/raw/bodmas_metadata.csv`
- SHA-256 của toàn bộ CSV: `5F1DC793CAC4A540EF9E0C40FD7897142095621790D3C6F2D6637971806F4C50`
- Ngày kiểm tra: 25/09/2026

Mã này nhận diện bản CSV dùng trong dự án; nó khác với cột `sha`, vốn nhận diện từng mẫu trong bảng.

## Diễn giải nhãn và binary

Theo README BODMAS, trường `family` trống biểu thị mẫu benign.
Trong giai đoạn train có 26.523 dòng benign và 20.333 dòng có nhãn họ malware;
validation có 7.280 dòng benign và 9.259 dòng có nhãn họ.

Thí nghiệm tạo ảnh dự kiến dùng bytes của các binary đã disarm và phải ghi rõ
đặc điểm này khi báo cáo kết quả. Chưa khôi phục hoặc lưu toàn bộ binary gốc.
Trước khi tải ZIP 67,92 GB, cần lập danh sách SHA cần dùng và kiểm tra cách
các file được đặt tên trong ZIP để chỉ giải nén phần cần thiết.
---------------------------------------------------------------------------------
## Manifest train và validation

Chạy `python scripts/build_train_val_manifest.py` từ thư mục gốc của dự án
để tạo `data/manifests/train_validation_manifest.csv`.

Manifest lấy các SHA thuộc 51 họ đã cố định từ train:
18.061 SHA train và 8.263 SHA validation, tổng cộng 26.324 SHA duy nhất.
Mỗi dòng ghi SHA, split, timestamp và family. Script báo lỗi nếu một SHA
xuất hiện hai lần trong manifest.

Manifest nằm trong `data/` và không được đưa lên GitHub; script tạo manifest
được lưu trong Git. Bước này chưa dùng nhãn future test và chưa tải binary.

## Trạng thái sau giải nén và pilot

- Binary disarmed hiện nằm trong `data/raw/altered`.
- `bodmas.npz` được giữ lại; ZIP đặc trưng bên ngoài đã xóa.
- ZIP binary gốc đã xóa sau khi người thực hiện quyết định.
- Đã kiểm tra đủ 26.324 mẫu train/validation trong manifest:
  18.061 train và 8.263 validation; không thiếu file, không sai
  kích thước so với ZIP index đã lưu.
- Kiểm tra tên và kích thước không xác nhận toàn vẹn nội dung.
  Chưa kiểm tra CRC toàn bộ dataset, kể cả phần future test.
- Một mẫu train thuộc họ autoit đã vượt qua kiểm tra CRC và
  tạo thành công ba ảnh SBSMI, raw-byte, local entropy.
- Ba báo cáo pilot ghi cùng SHA-256 nội dung disarm và đều
  có kết quả kiểm tra pixel sau lưu là PASS.
- Chi tiết cấu hình và bằng chứng nằm trong
  `docs/image_preprocessing_spec.md`.

Hai script `inspect_bodmas_zip.py` và `build_zip_index.py`
cần ZIP binary gốc nếu chạy lại. Index đã tạo vẫn được giữ tại
`data/manifests/train_validation_zip_index.csv`.

Các pilot hiện đọc trực tiếp từ `data/raw/altered`.