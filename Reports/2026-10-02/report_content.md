# Nội dung báo cáo — Buổi meeting với cô, 02/10/2026

> Đề tài: Phân đoạn thân tằm + chẩn đoán bệnh (Healthy / Grasserie) bằng mô hình multi-task.
> Tài liệu này là dàn ý để nói, không phải bản báo cáo đầy đủ. Mục nào còn là giả thuyết thì đã ghi rõ.

---

## 0. Tóm tắt 30 giây (nói đầu buổi)

- Em huấn luyện 2 mô hình multi-task (VM-UNet, Swin-UNet) để vừa phân đoạn thân tằm vừa phân loại bệnh từ một ảnh.
- Server huấn luyện đang lỗi nên em **dùng lại checkpoint của tuần trước (21/09)** và đánh giá lại, chưa train thêm được.
- **Trên ảnh tằm đơn (SAM3, 100 ảnh):** Dice 86,7% (VM-UNet) và 91,7% (Swin-UNet), phân loại đúng 100/100.
- **Trên ảnh nhiều con (Silkynet):** cả hai mô hình gần như thất bại (Dice ước lượng khoảng 9% và 1,5%).
- Em phát hiện bảng kết quả cũ bị **lệch nhãn**, nên em đánh giá lại và tách riêng hai miền dữ liệu.

---

## 1. Bài toán và kiến trúc (1–2 phút)

- Đầu vào: 1 ảnh tằm. Đầu ra: mask thân tằm `[B,1,H,W]` và nhãn bệnh `[B,2]`.
- Một encoder dùng chung, hai nhánh: decoder phân đoạn và nhánh phân loại (GAP + FC trên đặc trưng sâu nhất).
- Loss: `L = λ_seg · (BCE + Dice) + λ_cls · CrossEntropy`.
- Hai backbone so sánh: **VM-UNet** (Visual Mamba) và **Swin-UNet** (Transformer).

## 2. Dữ liệu (2 phút)

| Bộ | Nội dung |
|---|---|
| SAM3 (YOLO + mask pseudo) | Ảnh 1 con, có nhãn Healthy/Grasserie. Mask do SAM3 sinh ra từ bounding box, **chưa được người kiểm tra** |
| Silkynet | Ảnh nhiều con, **không có nhãn bệnh** |
| Mixed 10k | Train gồm 9.940 ảnh sinh bằng augmentation (copy-paste tằm đè nhau, dán lá dâu che, xoay/lật). Val 110, test 169 (100 SAM3 + 69 Silkynet) |

Điểm cần nói: 9.940 ảnh train sinh ra từ chỉ vài trăm ảnh gốc, nên **nhiều về số lượng nhưng ít đa dạng**.

## 3. Thiết lập thực nghiệm (1 phút)

- Checkpoint đánh giá: `multitask_vmunet_v1/2026-09-21_2253` (đầu vào 256 px) và `multitask_swinunet/2026-09-21_2258` (đầu vào 224 px).
- Ngưỡng mask 0,5. Chỉ số: Dice, IoU, Accuracy, Precision, Recall, F1.
- Lần đánh giá mới chạy bằng CPU (GPU máy đang bận), nên có thể lệch rất nhỏ so với lần chạy GPU cũ.

## 4. Kết quả chính (3 phút)

### 4.1. Chỉ trên ảnh SAM3 (100 ảnh: 58 Healthy, 42 Grasserie)

| Mô hình | Dice | IoU | Accuracy | F1 |
|---|---|---|---|---|
| VM-UNet | 86,67% | 78,19% | 100% | 100% |
| Swin-UNet | 91,71% | 85,49% | 100% | 100% |

- Lớp dương cho Precision/Recall/F1 là **Grasserie**.
- Swin-UNet ổn định hơn: median Dice 0,937 so với 0,908; số ảnh Dice < 0,8 là 6 so với 19.
- Chi tiết: `runs/evaluation_2026-09-22/sam3_only/`.

### 4.2. Toàn bộ test 169 ảnh (kết quả cũ — dùng có điều kiện)

| Mô hình | Dice | Accuracy |
|---|---|---|
| VM-UNet | 54,90% | 73,96% |
| Swin-UNet | 54,87% | 100% |
| Mask2Former | 56,79% | 59,17% |

**Không nên diễn giải bảng này như năng lực chẩn đoán**, vì lý do ở mục 5.1.

## 5. Phân tích và phát hiện (5 phút — phần quan trọng nhất)

### 5.1. Bảng 169 ảnh bị lệch nhãn (đã kiểm chứng một phần)
- 69 ảnh Silkynet **không có file nhãn nào**. Code dataset hiện tại gán `-1` (bỏ qua), nhưng kết quả cũ có 111 TN = 42 Grasserie + 69 Silkynet, tức là lúc đó chúng bị tính là Grasserie.
- Hệ quả: Accuracy 100% của Swin-UNet và các FP của VM-UNet/Mask2Former **không phản ánh khả năng chẩn đoán**. Đây là suy luận từ confusion matrix, chưa chạy lại bằng code cũ để chứng minh.

### 5.2. Mô hình thất bại trên ảnh Silkynet (ước lượng)
- Từ Dice tổng và Dice SAM3, Dice trên 69 ảnh Silkynet khoảng **9% (VM-UNet)** và **1,5% (Swin-UNet)**.
- Mask Silkynet đã kiểm tra: hợp lệ (0/255, không rỗng). Vậy đây là thất bại thật của mô hình, không phải lỗi dữ liệu.
- Nguyên nhân khả dĩ:
  1. **Val không đại diện test** (val chỉ có khoảng 10 ảnh nhiều con, test có 69), nên checkpoint "best" được chọn mà không nhìn miền Silkynet. *(từ cấu trúc dữ liệu)*
  2. Copy-paste chỉ lấy cutout từ SAM3, nên mô hình chưa thấy cảnh nhiều con thật. *(đã thấy trong code)*
  3. Ảnh bị resize 640 → 224/256 có thể làm mất tằm nhỏ. *(giả thuyết)*
  4. Định nghĩa mask khác nhau giữa hai nguồn. *(giả thuyết)*

### 5.3. Ngay cả trên ảnh SAM3, chất lượng biên chưa tốt
- VM-UNet có đốm nhiễu ở nền; biên Swin-UNet có dạng bậc thang (đặc trưng patch của Swin).
- **Mask GT cũng nhiễu:** ví dụ `Grasserie-1228` có Dice 0,38 nhưng mask dự đoán lại ôm đúng thân, còn mask SAM3 vỡ và lệch. Chỉ 1/100 ảnh bị vỡ nặng; độ gồ ghề của biên GT tương quan âm với Dice (−0,67 với Swin, −0,40 với VM-UNet).
- Nguyên nhân khả dĩ: ngưỡng cố định không hậu xử lý, giảm độ phân giải, loss chỉ tối ưu vùng chồng phủ chứ không phạt riêng biên.

## 6. Hạn chế (nói thẳng để cô không phải hỏi)

- Chưa train lại được vì server lỗi; mọi kết quả đến từ **một lần train duy nhất**, chưa có nhiều seed nên chưa đo được độ biến thiên.
- Accuracy 100% trên 100 ảnh chưa chắc là tổng quát hoá (test nhỏ, Healthy/Grasserie có thể khác nhau rõ về điều kiện chụp).
- Mask SAM3 là pseudo-label chưa kiểm tra, nên điểm test bị chặn trên bởi chất lượng nhãn.
- Mask2Former có số liệu cũ nhưng **không tái lập được** (code và checkpoint không còn trong repo), nên chỉ coi là tham khảo.
- Cần xác nhận checkpoint VM-UNet của bảng "clean" cũ: Dice 81,45% ≠ 86,67% của lần đánh giá này, trong khi Swin-UNet khớp (91,70% so với 91,71%).
- Combined Score (0,7·Dice + 0,3·Acc) là công thức tự đặt, nên không dùng làm tiêu chí chính.

## 7. Kế hoạch tiếp theo

**Không cần server (có thể làm trước buổi sau):**
1. Hậu xử lý (giữ thành phần liên thông lớn nhất, opening) và đo lại.
2. Thêm Boundary IoU / HD95 để đo chất lượng biên bằng số.
3. Chạy riêng 69 ảnh Silkynet để có Dice thật thay cho ước lượng.
4. Liệt kê và xử lý các ảnh GT bất thường; tự sửa tay khoảng 20 mask để có tập test tham chiếu sạch.

**Khi server hoạt động lại:**
5. Val cân bằng giữa hai miền; chọn checkpoint theo cả hai.
6. Thêm boundary loss (thư mục `boundaries/` của SAM3 đã có sẵn); thử đầu vào độ phân giải cao hơn.
7. Ablation `λ_seg` / `λ_cls`; so sánh joint training với segfirst 2 giai đoạn.
8. Chạy nhiều seed để báo cáo trung bình ± độ lệch chuẩn.

## 8. Câu hỏi muốn hỏi cô

1. Với ảnh Silkynet (không có nhãn bệnh), nên tách thành bài toán chỉ phân đoạn, hay bổ sung nhãn?
2. Có chấp nhận dùng mask pseudo-label của SAM3 làm GT test không, hay cần một tập kiểm tra bằng tay?
3. Cô muốn báo cáo cuối tập trung vào so sánh kiến trúc (Mamba vs Transformer) hay vào pipeline dữ liệu?
4. Nên báo cáo hai miền dữ liệu như hai bảng riêng, hay cố gắng cải thiện model để gộp thành một?

---

## Phụ lục: file tham chiếu

- Kết quả SAM3-only: `runs/evaluation_2026-09-22/sam3_only/` (có `NOTE.md`, `per_image.csv`, ảnh trực quan)
- Kết quả cũ 169 ảnh: `runs/evaluation_2026-09-22/summary_report.md`
- Script augment: `utils/augment_mixed_10k.py`
- Dataset đa nhiệm: `src/dataset_multitask.py`
