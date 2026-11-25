# Phân tích dữ liệu tiêu dùng bằng Python

Khảo sát hành vi đánh giá sản phẩm, dự báo xu hướng và ước lượng nguy cơ đánh giá tiêu cực trên dữ liệu thương mại điện tử Việt Nam (sàn Shopee, bộ dữ liệu **ViEcomRec**).

Đây là mã nguồn của bài tập lớn, viết bằng Python với NumPy, pandas, matplotlib, seaborn và scikit-learn.

## Mục lục

- [Nội dung](#nội-dung)
- [Dữ liệu](#dữ-liệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Cách chạy](#cách-chạy)
- [Dữ liệu giả để chạy thử](#dữ-liệu-giả-để-chạy-thử)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [Lưu ý khi diễn giải kết quả](#lưu-ý-khi-diễn-giải-kết-quả)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)
- [Nhóm thực hiện](#nhóm-thực-hiện)
- [Trích dẫn](#trích-dẫn)

## Nội dung

Dự án gồm bốn phần phân tích, mỗi phần là một file chạy riêng:

| Bước | File | Nội dung |
|---|---|---|
| 1 | `01_lam_sach.py` | Khảo sát và làm sạch dữ liệu (thiếu, trùng, điểm và thời gian không hợp lệ), lưu dữ liệu sạch |
| 2 | `02_truc_quan.py` | Trực quan hóa: phân phối điểm, xu hướng theo tháng, thương hiệu, xuất xứ, hành vi người dùng |
| 3 | `03_phan_khuc.py` | Mức độ tập trung của đánh giá và phân khúc sản phẩm theo **phổ biến x chất lượng cảm nhận** (4 nhóm) |
| 4 | `04_du_bao.py` | Dự báo số lượt đánh giá theo tháng, so sánh 4 mô hình (Naive, Naive theo mùa, trung bình trượt, hồi quy tuyến tính) |
| 5 | `05_phan_loai.py` | Ước lượng nguy cơ một đánh giá là tiêu cực (hồi quy logistic, rừng ngẫu nhiên) và lập danh sách sản phẩm nguy cơ cao |

Các mô hình dự báo và phân loại đều được so với **mô hình cơ sở (baseline)** và đánh giá trên tập kiểm tra tách theo thời gian.

## Dữ liệu

- **Bộ dữ liệu:** ViEcomRec, gồm đánh giá của người dùng về các sản phẩm sữa rửa mặt trên Shopee, kèm thuộc tính sản phẩm (thương hiệu, xuất xứ, loại da, dung tích...).
- **Nguồn:** https://github.com/linh222/face_cleanser_recommendation_dataset
- **Giấy phép:** xem README của kho mã nguồn trên và tuân thủ điều kiện sử dụng. Chỉ dùng cho mục đích học tập, không công bố thông tin định danh người dùng.
- Dữ liệu gốc **không được đưa vào kho mã nguồn này** (trừ khi giấy phép cho phép). Hãy tải về và đặt vào thư mục `data/`.

Bộ dữ liệu không có giá bán hay mã hóa đơn nên dự án không phân tích doanh thu hay chi tiêu.

## Cấu trúc thư mục

```
cd3-ptdltd/
├── config.py          # cấu hình: đường dẫn dữ liệu, tên cột, tham số phân tích (file duy nhất cần chỉnh)
├── utils.py           # hàm dùng chung: đọc file, chuẩn hóa, lưu hình, ghi số liệu tổng hợp
├── 01_lam_sach.py     # bước 1: khảo sát và làm sạch
├── 02_truc_quan.py    # bước 2: trực quan hóa
├── 03_phan_khuc.py    # bước 3: phân khúc sản phẩm
├── 04_du_bao.py       # bước 4: dự báo theo tháng
├── 05_phan_loai.py    # bước 5: nguy cơ đánh giá tiêu cực
├── run_all.py         # chạy lần lượt bước 1 đến 5
├── main.ipynb         # notebook gọi lần lượt các bước (Jupyter / Google Colab)
├── gen_fake_data.py   # tạo dữ liệu GIẢ để thử chương trình (xem mục "Dữ liệu giả")
├── data/              # dữ liệu đầu vào
├── outputs/           # bảng kết quả (CSV) và outputs/tong_hop_so_lieu.json
└── figures/           # biểu đồ (PNG)
```

Các bước nối với nhau qua dữ liệu đã làm sạch do bước 1 tạo ra, nên có thể chạy riêng từng bước sau khi bước 1 đã chạy xong.

## Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
git clone https://github.com/duongx2004/cd3-ptdltd.git
cd cd3-ptdltd
pip install numpy "pandas>=2.0" matplotlib seaborn "scikit-learn>=1.2" openpyxl
```

Trên Google Colab các thư viện đã có sẵn, chỉ cần tải thư mục dự án lên.

## Cấu hình

Mở `config.py` và điền:

| Tham số | Ý nghĩa |
|---|---|
| `INTERACTION_FILES` | Danh sách file đánh giá trong `data/`. Nếu dữ liệu chia train/val/test thì liệt kê hết, code tự gộp |
| `ITEMS_FILE` | File thuộc tính sản phẩm; để `None` nếu thuộc tính đã nằm trong file đánh giá |
| `COLS` | Tên cột người dùng, sản phẩm, điểm, thời gian, nội dung. Để `None` thì code tự đoán theo từ khóa và in ra để kiểm tra |
| `BRAND_COL`, `ORIGIN_COL`, `ITEM_CAT_COLS` | Cột thương hiệu, xuất xứ và các thuộc tính đưa vào mô hình |
| `NEG_THRESHOLD` | Điểm nhỏ hơn hoặc bằng giá trị này là đánh giá tiêu cực (chọn sau khi xem phân phối điểm) |
| `N_MIN`, `N_MIN_RISK` | Số đánh giá tối thiểu để so sánh điểm, phân khúc và vào danh sách nguy cơ |
| `TEST_MONTHS`, `HORIZON` | Số tháng kiểm tra và số tháng dự báo tương lai |
| `TRAIN_FRAC` | Tỷ lệ huấn luyện khi chia theo thời gian cho bài toán phân loại |

Lần đầu, để `INTERACTION_FILES` trống rồi chạy bước 1: chương trình liệt kê các file và tên cột trong `data/` để bạn điền.

## Cách chạy

**Chạy toàn bộ:**
```bash
python run_all.py
```

**Chạy từng bước:**
```bash
python 01_lam_sach.py
python 02_truc_quan.py
python 03_phan_khuc.py
python 04_du_bao.py
python 05_phan_loai.py
```

**Notebook:** mở `main.ipynb` và chạy lần lượt các cell, biểu đồ hiển thị ngay trong notebook.

Nên chạy bước 1 trước và đọc kỹ phần khảo sát (số dòng, phân phối điểm, khoảng thời gian) để chọn `NEG_THRESHOLD`, `N_MIN`, `TEST_MONTHS` cho phù hợp, rồi mới chạy các bước còn lại.

## Dữ liệu giả để chạy thử

`gen_fake_data.py` tạo bộ dữ liệu **giả** để kiểm tra chương trình chạy được từ đầu đến cuối khi chưa có dữ liệu thật.

```bash
python gen_fake_data.py
```

> **Quan trọng:** kết quả chạy trên dữ liệu giả (các bảng, biểu đồ, độ chính xác của mô hình) **không phản ánh thực tế và không được dùng làm kết luận** trong báo cáo. Các cột do script tự sinh (ví dụ cột phân hạng chất lượng) có thể được tạo ra có chủ đích để liên hệ với điểm đánh giá, nên sẽ làm mô hình trông tốt hơn thực tế. Trước khi viết báo cáo, hãy xóa kết quả cũ, đặt dữ liệu ViEcomRec thật vào `data/`, chỉnh `config.py` và chạy lại toàn bộ.

## Kết quả đầu ra

| Kết quả | File |
|---|---|
| Nhật ký số dòng sau mỗi bước làm sạch | `outputs/bang_3_3_nhat_ky_lam_sach.csv` |
| Phân khúc sản phẩm | `outputs/bang_3_2_phan_khuc_san_pham.csv` |
| So sánh mô hình dự báo, dự báo các tháng tới | `outputs/bang_3_3_so_sanh_du_bao.csv`, `outputs/du_bao_cac_thang_toi.csv` |
| So sánh mô hình phân loại | `outputs/bang_3_4_so_sanh_phan_loai.csv` |
| Top sản phẩm nguy cơ đánh giá tiêu cực cao | `outputs/bang_3_5_san_pham_nguy_co_cao.csv` |
| Độ quan trọng đặc trưng | `outputs/do_quan_trong_dac_trung.csv` |
| Toàn bộ số liệu quan trọng gom lại | `outputs/tong_hop_so_lieu.json` |
| Biểu đồ | `figures/*.png` |

Các file CSV lưu mã hóa UTF-8 có BOM nên mở bằng Excel vẫn đúng dấu tiếng Việt.

## Lưu ý khi diễn giải kết quả

- **Chia theo thời gian:** tập kiểm tra luôn nằm sau tập huấn luyện, không chia ngẫu nhiên.
- **Tránh rò rỉ dữ liệu:** đặc trưng lịch sử của sản phẩm (số đánh giá trước đó, tỷ lệ tiêu cực trước đó) chỉ tính từ các đánh giá đứng trước đánh giá đang xét. Khi để `ITEM_CAT_COLS = None`, chương trình tự chọn các cột thuộc tính; **hãy kiểm tra danh sách cột được chọn** và loại bỏ mọi cột được tính từ điểm đánh giá, vì cột như vậy làm kết quả đẹp giả tạo.
- **Baseline:** mô hình chỉ có ý nghĩa nếu thắng được baseline. Nếu không thắng, kết luận trung thực là mô hình đơn giản đã đủ.
- **Dự báo** là dự báo số lượt đánh giá, không phải doanh số. Chuỗi dưới 24 tháng thì hồi quy chỉ có xu hướng, chưa có mùa vụ.
- **"Sản phẩm nguy cơ cao"** là sản phẩm có xác suất bị đánh giá thấp trung bình cao so với các sản phẩm khác, không có nghĩa là sản phẩm kém chất lượng hay nguy hiểm.
- Các ngưỡng (`NEG_THRESHOLD`, `N_MIN`, ngưỡng phân khúc) là quy ước của nhóm và cần được nêu rõ trong báo cáo.

## Xử lý lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `Hãy điền INTERACTION_FILES ...` | Điền `config.py` theo danh sách file in ra, chạy lại bước 1 |
| `Không đoán được cột: [...]` | Điền tay tên cột vào `COLS` trong `config.py` |
| `Bước '...' làm mất toàn bộ dữ liệu` | Kiểm tra tên cột điểm, thang điểm, định dạng cột thời gian |
| `Chưa có ... clean` | Chạy bước 1 trước |
| `Không có sản phẩm nào đủ N_MIN ...` | Giảm `N_MIN` |
| `Chuỗi quá ngắn để dự báo ...` | Giảm `TEST_MONTHS`, hoặc bỏ bước 4 và nêu lý do trong báo cáo |
| Cảnh báo lớp tiêu cực quá nhỏ | Tăng `NEG_THRESHOLD` và giải thích lựa chọn |

## Nhóm thực hiện

| Họ tên | MSSV | Phần phụ trách |
|---|---|---|
| [Họ tên] | [MSSV] | [Bước 1: làm sạch] |
| [Họ tên] | [MSSV] | [Bước 2: trực quan hóa] |
| [Họ tên] | [MSSV] | [Bước 3: phân khúc] |
| [Họ tên] | [MSSV] | [Bước 4: dự báo] |
| [Họ tên] | [MSSV] | [Bước 5: phân loại] |

**Giảng viên hướng dẫn:** [Họ tên]

## Trích dẫn

Tran, Q.-L., Nguyen, B. T., Jones, G. J. F., & Gurrin, C. (2024). ViEcomRec: A dataset for recommendation in Vietnamese e-commerce. In *Computational Data and Social Networks (CSoNet 2023)*, LNCS 14479 (pp. 74–82). Springer. https://doi.org/10.1007/978-981-97-0669-3_7