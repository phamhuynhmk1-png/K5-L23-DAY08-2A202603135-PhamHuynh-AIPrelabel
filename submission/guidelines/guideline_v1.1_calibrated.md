# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603135 - Phạm Huỳnh

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Gán **từng xe** nếu nhìn thấy ≥ 20% và box cao ≥ 20 px, dù đông. Chỉ phần ở xa, box < 20 px (cụm ≥ 5 vật) mới dùng `ignore_region`. Ngưỡng là **kích thước và mức che**, không phải mật độ. | img_018 | CAL-04 |
| 4 | Người ngồi trong xe ba bánh có mui che: ngưỡng "còn thấy ≥ 20%" có hợp lý không? | **Giữ ngưỡng 20%** (mục 4.1.2). Ca ước lượng sát 20%: vẫn gán `person` (box quanh phần nhìn thấy) và **tick `can_review`**. | img_020 | CAL-02 |
| 1, 3, 5, 6, 7 | Xe đạp điện; hàng nhô ra; kính xe; box chồng ảnh đêm; conf tối thiểu | **Chưa chốt**: trong 28 ảnh đã gán / duyệt chưa gặp ca đủ rõ để làm ảnh ví dụ. Giữ nguyên v1.0. | — | — |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_013 | Mask tuk-tuk có người lái và 3 người ngồi phía sau | SAM 2.1 tiny, 3 click (2 dương, 1 âm): mask xe dính 3 người phía sau | Mask `three_wheeler` **không gồm người**; mỗi người một mask `person` riêng | **Mục 4.2, thêm điều 5:** mask xe chỉ gồm xe + mui + thùng gắn liền; người lái / người ngồi là mask `person` riêng. Dùng SAM thì thêm một click âm lên từng người (img_013: 4 click âm + box → mask đúng). (CAL-01) |
| img_002, img_011, img_032 | Xe ba bánh (tuk-tuk, xích lô) trong ảnh có pre-label | YOLO26: gọi `truck` (img_002 conf 0.37, img_011 conf 0.28), sót xích lô hoặc gọi `bicycle` (img_032); 14 ca đổi lớp trong batch | Máy sai, định nghĩa lớp v1.0 đã đúng | **Mục 4.3, thêm:** khi duyệt pre-label, mọi box `car` / `truck` / `bicycle` của detector COCO phải **kiểm số bánh** trước khi Accept; xe ba bánh đổi sang `three_wheeler`. Lớp `three_wheeler` lấy từ YOLOE. (CAL-03) |

Không đưa vào guideline (lỗi người duyệt, guideline không mơ hồ): img_010, box `bicycle` lệch bị xoá thay vì kéo lại (CAL-05).

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu hỏi mở 2 (xe máy dày gần camera: từng xe nếu ≥ 20 px) và 4 (giữ ngưỡng 20% + `can_review`); thêm mục 4.2 điều 5 (mask xe ba bánh không gồm người); thêm quy tắc review ở 4.3 (kiểm số bánh với box car/truck/bicycle của máy) | CAL-04, CAL-02, CAL-01, CAL-03 |
