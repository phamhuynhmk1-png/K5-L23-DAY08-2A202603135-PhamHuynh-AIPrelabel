# Thông tin học viên

- Họ và tên: Phạm Huỳnh
- MSHV: 2A202603135
- Lớp (L2 / L3 / khác): L2A
- Tài khoản GitHub: phamhuynhmk1-png
- Repo bài nộp (URL): https://github.com/phamhuynhmk1-png/K5-L23-DAY08-2A202603135-PhamHuynh-AIPrelabel
- Commit chốt bài (hash): 893ed2a
- Máy: Windows 11, GPU NVIDIA RTX 3050 Laptop 4 GB, nhưng venv cài torch bản CPU (2.15.0+cpu) nên mọi model chạy trên CPU
- CVAT (Docker): v2.82.0 (chạy trên server)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 362 / 48 / 110 = 69.6% / 9.2% / 21.2% (mẫu số: 520 box máy); added: 66 = 13.9% (mẫu số: 476 đối tượng)
- So với gold: precision / recall của **máy** 64.5% (40/62) / 75.5% (40/53)
- Bắt bẫy: 9/12 = 75%
- Thời gian: gán tay 5.0 phút/ảnh (20/4); duyệt có pre-label 3.5 phút/ảnh (85/24)
- Lỗi QC: 40% theo ảnh (2/5), 2.2% theo đối tượng (2/90)
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi QC 2/5 = 40% > 2% và bắt bẫy 9/12 = 75% < 80%; đã sửa 5 ảnh (`qc_fixed.json`), cần QC mẫu mới trước khi Release.
