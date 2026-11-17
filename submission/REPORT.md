# Báo cáo Day 08 — 2A202603135 - Phạm Huỳnh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26n (`yolo26n.pt`, CPU) | person 0.15 · motorbike 0.15 · bicycle 0.1 · car 0.45 · bus / truck 0.3 (mặc định, 0 vật trong gold) | r01/r02: lên conf 0.5 chung mất 67/148 box, phần lớn là vật thật (img_007 motorbike 9→3, img_032 person 17→5) → cần ngưỡng theo lớp. Sweep: car 0.45 vẫn recall 7/7. person / motorbike không ngưỡng nào đạt recall 90% nên tôi hạ xuống 0.15 (ưu tiên recall: xoá box thừa nhanh hơn vẽ vật bị sót); bicycle 0.1 bắt 2/6 mà không thêm box thừa |
| three_wheeler | YOLOE-26s-seg, prompt mặc định từ `taxonomy.json` (`--classes three_wheeler`) | 0.15 | YOLO26 không có lớp xe ba bánh (r01: img_002, img_011 gọi `truck`; img_032 sót 2 xích lô). r03: YOLOE tìm đúng xe ba bánh img_002 (0.93), img_011 (0.28), 3 xích lô img_032 (0.92 / 0.48 / 0.27). Sweep: 0.15 cho recall 6/6, 0.3 chỉ 4/6 |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "car=0.45,three_wheeler=0.15,motorbike=0.15,person=0.15,bicycle=0.1" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` → 599 box (28 ảnh)
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 tiny (`sam2.1_t.pt`), vì bám biên tốt hơn MobileSAM (SAM 1) trên cùng 3 click ở img_013 (MobileSAM còn tách mask thành 2 mảnh). Tốc độ: 5.16 s/ảnh trên CPU kèm YOLO26 (r04).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Xe ba bánh bị YOLO26 gọi `truck` (img_002 conf 0.37, img_011 conf 0.28): đây là lỗi lớp, tăng ngưỡng chỉ làm box biến mất (r02: truck 1→0) chứ không thành `three_wheeler`. Sửa bằng YOLOE (prompt three_wheeler) đặt ưu tiên cao hơn khi merge, và khi review đổi lớp bằng tay (14 ca đổi lớp trong `review_stats`); thêm quy tắc kiểm số bánh (CAL-03).
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): điểm thấp mà đúng: xe ba bánh bị cắt mép trái img_011 (YOLOE 0.28), xích lô img_032 (0.27). Điểm cao hơn mà sai: 3 box rác của YOLOE ở img_002 có conf 0.37–0.40, cao hơn hai box đúng trên. YOLOE còn gán `three_wheeler` 0.12 cho một ô tô thật ở img_013. Vì vậy conf của box đúng và box rác chồng nhau, điểm không phải xác suất đúng.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng được chọn và chấm trên cùng 53 đối tượng gold (in-sample). Mỗi lớp rất ít vật (car 7, three_wheeler 6, bicycle 6, bus / truck 0), nên chênh 1 vật là recall đổi 14–17%; bus / truck không chọn được ngưỡng riêng.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 362 | 69.6% | 520 box máy |
| edited | 48 | 9.2% | 520 box máy |
| deleted | 110 | 21.2% | 520 box máy |
| added | 66 | 13.9% | 476 đối tượng sau duyệt |

Nguồn: `qc/review_stats.md` (24 ảnh batch; đã loại 31 đối tượng quanh bẫy). Trong deleted + added có 14 ca đổi lớp → máy sót thật ≈ 66 − 14 = 52 vật.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? `bicycle`: trong batch chỉ 6 box máy được giữ, 10 bị xoá, phải vẽ thêm 4 (tức xoá 10/16 box máy). Trên gold, ở ngưỡng mặc định 0.3 recall là 0/6, hạ xuống 0.1 cũng chỉ 2/6. Xe đạp còn là nơi 2 bẫy lọt (img_010 loose, img_031 wrong_class) và lỗi QC img_031. YOLO26n (cỡ nhỏ) gần như không nhận ra xe đạp trong cảnh phố đông; ở img_032 nó còn gán `bicycle` cho một phần xích lô.
- So với gold: precision / recall của **máy** 64.5% (40/62 box máy, KTC 95% 52–75%) / 75.5% (40/53, KTC 95% 62–85%) (mẫu số recall: 53 đối tượng gold, 4 ảnh)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 9/12 = 75% (KTC 95% 47–91%, dưới ngưỡng 80%); theo loại: missing 3/3, wrong_class 2/3, loose 2/3, extra 2/3.
- Bẫy lọt và vì sao (đối chiếu `traps_key.json` với `batch_reviewed.json` sau khi export):
  - `img_003` (extra, box `truck` ma): box giữ nguyên y hệt (IoU 1.0). Bẫy ở ảnh đầu tiên của lô; có thể lúc đó tôi chưa quen nhịp, chưa làm bước 0 (quét ảnh trước), hoặc tin box truck. Bước bị bỏ: bước 2 (thừa).
  - `img_010` (loose, box `bicycle` lệch): tôi nhầm giữa "box lệch" và "box thừa" nên xoá box thay vì kéo lại; bước 4 (lỏng) bị thay bằng xoá, xe đạp này hiện không có nhãn.
  - `img_031` (wrong_class, xe đạp bị đổi thành `motorbike`): bẫy lọt thật. Nhãn cuối có box `bicycle` (id 418 trong ảnh QC) nhưng lệch xa box gốc (IoU 0.33), và QC (Part 6) xác nhận box này là box thừa.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Accepted 69.6% (362/520) nhưng bắt bẫy chỉ 9/12. Cả 3 bẫy missing tôi đều bắt (tôi có quét tìm vật sót), nhưng cả 3 bẫy lọt đều nằm trên **box máy có sẵn**: giữ nguyên box ma (img_003), xử lý sai box lệch (img_010), để lại box sai (img_031). Nghĩa là tôi kiểm vật thiếu kỹ hơn kiểm box máy: khi box trông hợp lý tôi có xu hướng Accept (lỗi "làm theo" của thiên kiến tự động hoá). Vì vậy accepted cao không chứng minh máy tốt; một phần accepted có thể là box tôi chưa thật sự kiểm.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 5.0 phút/ảnh (20 phút / 4 ảnh) vs duyệt 3.5 phút/ảnh (85 phút / 24 ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): tôi gán gold trước nên khi review đã quen quy tắc và loại cảnh, làm thời gian review thấp hơn thực tế của người mới. Gold chỉ 4 ảnh (mẫu nhỏ), còn batch có 6 ảnh rất đông; thời gian là số phút làm tròn, không tính giờ nghỉ. So sánh 5.0 vs 3.5 phút/ảnh chỉ là ước lượng thô.
- Kết luận: **có, cho mọi lớp**, dựa trên thời gian duyệt 3.5 phút/ảnh so với gán tay 5.0 phút/ảnh (nhanh hơn khoảng 30%) và 69.6% box máy (362/520) được giữ nguyên. Điều kiện: máy chỉ là gợi ý. Recall của máy trên gold là 75.5% (40/53) và máy sót khoảng 52 vật trong batch, nên bước tìm vật sót là bắt buộc. Riêng `bicycle` (xoá 10/16) và `three_wheeler` (xoá 25/40) phải kiểm từng box.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Ghi số vật ở bước 0:** mỗi ảnh ẩn nhãn máy, nhẩm đếm và ghi số vật theo lớp trước khi bật nhãn; sau khi duyệt so với số box cuối, lệch > 2 thì quét lại. Kiểm tra: có số đếm cho đủ 24/24 ảnh (chống lọt kiểu img_003).
  2. **Kiểm từng box car / truck / bicycle của máy:** click từng box, kiểm số bánh và vật thật trước khi Accept. Kiểm tra: số box đã mở / tổng box car + truck + bicycle = 100% (chống lọt kiểu img_003 truck ma, img_031 bicycle sai).
  3. **Box lệch thì kéo, không xoá:** chỉ xoá khi chắc chắn không có vật; box có vật nhưng lệch thì kéo lại. Kiểm tra: trong QC, số box bị xoá mà vật vẫn còn trong ảnh = 0 (chống lọt kiểu img_010).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (tuk-tuk) | SAM 2.1 tiny (`sam2.1_t.pt`) | 3 (2 dương, 1 âm); mask 39.4% ảnh | có: dính 3 người ngồi phía sau | có lỗi biên; click âm chỉ loại được người lái, không loại 3 người phía sau | `tuktuk_img013.jpg` |
| img_013 (tuk-tuk, so phiên bản) | SAM 1 MobileSAM (`mobile_sam.pt`) | 3 (cùng click); mask 35.2% ảnh, 2 polygon | dính người một chút, không đáng kể | có lỗi biên; biên kém SAM 2.1, mask bị tách 2 mảnh | `tuktuk_img013_mobilesam.jpg` |
| img_008 (xe ba bánh) | SAM 2.1 tiny | 1 dương + box gợi ý; mask 14.0% ảnh. Lần đầu chỉ 1 click dương: SAM không trả mask | không gồm người | có lỗi biên | `three_wheeler_img008.jpg` |
| img_032 (xích lô) | SAM 2.1 tiny | 1 dương + box gợi ý; mask 8.7% ảnh. Lần đầu chỉ 1 click dương: mask 0.3% ảnh (chỉ một chi tiết nhỏ) | không dính người | có lỗi biên | `xichlo_img032.jpg` |
| img_013 (tuk-tuk, sửa) | SAM 2.1 tiny | 7 (2 dương, 4 âm lên người lái + 3 người phía sau) + box tuk-tuk; mask 30.4% ảnh (giảm từ 39.4%) | không: đã loại cả người lái và 3 người phía sau | còn lỗi biên nhẹ | `tuktuk_img013_neg.jpg` |

- So phiên bản: SAM 2.1 tiny bám biên tốt hơn MobileSAM trên cùng 3 click ở img_013.
- Sửa mask tuk-tuk img_013: một click âm chỉ đẩy vùng quanh điểm đó; lần đầu chỉ có click âm (720,300) trúng người lái nên 3 người phía sau vẫn bị gộp vào xe. Sửa bằng cách thêm một click âm lên mỗi người cần loại (người lái (720,300) và 3 người phía sau (469,236), (540,202), (387,256), lấy tâm box `person` trong nhãn đã duyệt) và dùng box tuk-tuk (98,19)–(910,682) làm khung giới hạn → mask giảm từ 39.4% xuống 30.4% ảnh (`tuktuk_img013_neg.jpg`). Kết quả: đã loại được 3 người phía sau, còn lỗi biên nhẹ. Cần 7 click + box (so với 3 click ban đầu) để mask đúng quy tắc "người là vật riêng".
- Lỗi mask rút ra: chỉ 1 click dương có thể không ra mask (img_008) hoặc chỉ ra một chi tiết nhỏ (img_032, 0.3% ảnh); thêm box gợi ý thì ra mask cả xe.
- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? Giúp: 4.2.4 (hai vật chạm nhau, mỗi vật một polygon, không chồng) và mục 2 (người là vật riêng) cho biết người trên tuk-tuk phải tách khỏi mask xe. Thiếu: 4.2 không nói rõ mask xe ba bánh có gồm người ngồi trong xe không (img_013) → chốt ở CAL-01 (không gồm). Theo 4.2.1 (lệch biên ≤ ~3 px), cả 4 mask đều còn lỗi biên nên chưa đạt nếu dùng làm nhãn cuối.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: không thử (chỉ làm offline bằng `sam2_points.py`).

## 6. Calibration (Part 6)

- Số quyết định: 5 (`calibration_decisions.csv`: 3 `cau_hoi_mo`, 1 `loi_may`, 1 `bay_lot`); quan trọng nhất (dẫn `CAL-xx`): CAL-03, vì lỗi gọi sai lớp xe ba bánh là lỗi lặp nhiều nhất của máy (14 ca đổi lớp; three_wheeler xoá 25/40) và không sửa được bằng ngưỡng; quy tắc kiểm số bánh khi review chặn được lỗi này.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu hỏi mở 2 (CAL-04: xe máy dày gần camera gán từng xe nếu box ≥ 20 px) và 4 (CAL-02: giữ ngưỡng 20% + `can_review`); thêm 4.2 điều 5 (CAL-01: mask xe ba bánh không gồm người); thêm quy tắc review ở 4.3 (CAL-03: kiểm số bánh với box car/truck/bicycle của máy). CAL-05 (img_010) là lỗi người duyệt, không vào guideline. Nhãn đã sửa theo CAL-04 trong `qc_fixed.json` (img_018, img_027, img_034).
- Nguyên nhân gốc: chủ yếu là **máy** (CAL-03: định nghĩa lớp v1.0 đã đúng, máy gọi sai). Ngoài ra: guideline mơ hồ ở CAL-01 / 02 / 04 (đã chốt trong v1.1), và lỗi của tôi khi duyệt ở CAL-05 (img_010), không cần đổi guideline.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` vì batch không có box nào tick + 5 ngẫu nhiên, seed 42: img_003, img_006, img_014, img_031, img_035), 90 đối tượng; người kiểm: tôi (**hạn chế**: tôi vừa là người duyệt).
- Lỗi theo **đối tượng**: 2/90 = 2.2% · theo **ảnh**: 2/5 = 40.0% (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: chỉ có `thua` (2 lỗi): img_003 box `truck` id 2 (chính là bẫy extra bị lọt), img_031 box `bicycle` id 418. Không thấy lỗi loại khác trong mẫu.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 2 ảnh lỗi / 5 ảnh → cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là 77%. Với 5 ảnh, kể cả 0 lỗi thì cận trên vẫn khoảng 3/5 = 60% (quy tắc 3), nên mẫu này không thể chứng minh ngưỡng ≤ 2%; con số 40% cũng chỉ là ước lượng rất thô.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): script kiểm được tên ảnh, lớp có trong taxonomy, box trong ảnh, box < 20 px, accepted / edited / deleted / added, bẫy và tỉ lệ lỗi theo phiếu QC. Không kiểm được: box có ôm sát vật thật không, lớp có đúng thực tế không, vật mà cả tôi và máy cùng bỏ sót (ngoài gold), và tôi có làm đủ checklist không. QC do chính tôi kiểm nên có thể bỏ sót đúng những lỗi tôi đã bỏ sót khi review.

**Quyết định: Rework**, ngày 2026-11-17. Lý do (đi theo Hold → Rework → Release): không Hold, vì 2 lỗi QC đều là box thừa (lỗi duyệt), không do guideline mơ hồ, và các câu hỏi mở gặp phải đã chốt trong v1.1. Rework, vì ảnh lỗi 2/5 = 40% vượt xa ngưỡng 2%, và bắt bẫy 9/12 = 75% dưới ngưỡng 80% nên lô duyệt chưa đáng tin. Không Release vì cả hai điều kiện đều chưa đạt.
Nếu Rework / Hold, việc cần làm tiếp: đã làm vòng sửa đầu: `qc_fixed.json` sửa 5 ảnh (img_003 xoá truck thừa, img_031 xoá bicycle thừa, img_018 / 027 / 034 bỏ 6 box < 20 px theo CAL-04), `build_release.py` còn 484 đối tượng. Còn phải: duyệt lại lô theo 3 quy tắc ở mục 4 (ưu tiên box bicycle / car / truck và 6 ảnh đông), rồi QC trên mẫu mới (`qc_sample.py --force` với seed khác, vì 5 ảnh cũ tôi đã biết lỗi) trước khi xét Release.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Claude Code (trong VS Code) để: kiểm tra môi trường, chạy các script (`prepare_workdirs`, `merge_prelabels`, `inject_traps`, `review_stats`, `sam2_points`, `build_release`, `qc_sample`, `qc_report`), giải thích GUIDE / guideline / số đo, đếm box và tính IoU giữa các lần chạy (không xem ảnh), ghi lại nhận xét và quyết định của tôi vào file, commit git. AI **không** xem ảnh, không gán gold, không review batch, không kiểm QC. Nhận xét model, quyết định ngưỡng / calibration và kết luận là của tôi; tôi kiểm bằng cách mở `viz/`, ảnh mask, ảnh QC và đối chiếu với số trong `qc/*`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không. Tôi gán và sửa gold (xoá box < 20 px, tick `can_review` img_020; export 3 lần) xong trước khi chạy bất kỳ model nào; AI không xem ảnh gold, chỉ giải thích quy tắc trong guideline.
- Ca khó khi gán gold (box tick `can_review`):
  - `img_020`: người ngồi trong tuk-tuk, chỉ thấy khoảng 20% (sát ngưỡng che 4.1.2) → vẫn gán `person`, tick `can_review`. Liên quan câu hỏi mở số 4 của v1.0.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11, GPU NVIDIA RTX 3050 Laptop 4 GB nhưng venv cài torch bản CPU (2.15.0+cpu) nên mọi model chạy trên CPU (ultralytics 8.4.215, venv `C:\Users\admin\day08\.venv`); model bộ 8 GB (`yolo26n`, `yoloe-26s-seg`, `sam2.1_t`, `mobile_sam`) thay cho `yolo26m` / `sam2.1_b` trong GUIDE; CVAT v2.82.0 trên server. Vướng: PowerShell chặn `Activate.ps1` → chạy bằng Git Bash (`source .../activate`).
- Dùng cache `data/cache/` thay vì tự chạy? không, tự chạy toàn bộ trên CPU (`work/runs.csv` r01–r06).
- Lỗi / bug script đã gặp (nếu có): không sửa script. `sam2_points.py` báo "SAM không trả mask nào" ở img_008 khi chỉ có 1 click dương (xử lý bằng thêm `--box`).
- Một điều bạn làm khác lần sau: kiểm kỹ ảnh đầu lô: bẫy lọt ở ngay ảnh đầu tiên (img_003), lúc chưa quen nhịp; lần sau làm thử 1–2 ảnh trước khi bấm giờ và duyệt lại ảnh đầu sau khi xong lô.
