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
