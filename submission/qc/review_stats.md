# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 520; đối tượng sau duyệt: 476; ngưỡng accepted: IoU >= 0.9
- Đã **loại 31 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 362 | 362/520 = 69.6% | 520 box máy |
| edited (sửa) | 48 | 48/520 = 9.2% | 520 box máy |
| deleted (xoá) | 110 | 110/520 = 21.2% | 520 box máy |
| added (vẽ thêm) | 66 | 66/476 = 13.9% | 476 đối tượng sau duyệt |

Trong số deleted + added, **14 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 52.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 216 | 20 | 50 | 17 |
| bicycle | 6 | 0 | 10 | 4 |
| motorbike | 103 | 21 | 20 | 16 |
| car | 27 | 0 | 3 | 20 |
| bus | 1 | 0 | 1 | 3 |
| truck | 1 | 0 | 1 | 2 |
| three_wheeler | 8 | 7 | 25 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 53 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 40 | 22 | 13 | 64.5% (KTC 95%: 52–75%) | 75.5% (KTC 95%: 62–85%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 9/12 = 75.0% (KTC 95%: 47–91%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 2/3
  - loose: 2/3
  - missing: 3/3
  - extra: 2/3
  - lọt: img_031.jpg (wrong_class), img_003.jpg (extra), img_010.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
