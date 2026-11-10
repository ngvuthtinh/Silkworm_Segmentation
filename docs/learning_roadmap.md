# 🗺️ Lộ trình học ML/DL/CV từ gốc — chỉ học thứ đồ án của bạn dùng

> Dành cho: người có nền **Software Development**, chưa học ML/DL/CV, cần **hiểu và bảo vệ** đồ án
> *VM-UNet + spatial embedding để tách và chẩn đoán từng con tằm*.
>
> Nguyên tắc: **mỗi bài học kết thúc bằng một đoạn code có thật trong repo này**. Nếu một kiến thức không nối được
> vào đồ án thì nó nằm ở mục "Bỏ qua" (cuối file), đừng học.

---

## 0. Bản đồ: đồ án của bạn là một chuỗi 6 mắt xích

```
Ảnh (3×256×256)
   │  [A] Dữ liệu            src/dataset_instance.py      ← ảnh + mask từng con + nhãn bệnh
   ▼
VM-UNet (encoder–decoder)    models/vmunet/.../vmamba.py   ← [B] Kiến trúc: Mamba + U-Net
   │  [C] Đầu ra 5 kênh/pixel experiments/multitask_vmunet_spatialonly/model.py
   │      thân | bệnh | offset(x,y) | σ | seed
   ▼
Loss                         src/losses.py                 ← [D] BCE+Dice (thân, bệnh) + SpatialEmbeddingLoss
   │  (huấn luyện: AdamW, lr 1e-4, 40 epoch, AMP)          ← [E] Huấn luyện: experiments/.../train.py
   ▼
Hậu xử lý: gom pixel quanh seed   src/metrics.py::cluster_spatial_embeddings
   ▼
Từng con + chẩn đoán           ← [F] Đánh giá: inst_f1, e2e_f1, grasserie_recall (src/metrics.py)
```

Khi bảo vệ, hội đồng sẽ hỏi theo đúng thứ tự này. Mỗi giai đoạn dưới đây gắn với một mắt xích.

| Giai đoạn | Học gì | Mắt xích | Thời gian gợi ý |
|---|---|---|---|
| 1 | Mạng nơ-ron & huấn luyện | E | 3–4 ngày |
| 2 | CNN & ảnh | B | 2–3 ngày |
| 3 | Segmentation & U-Net | B, C, F | 3 ngày |
| 4 | Attention & Transformer | B | 2 ngày |
| 5 | State Space Model → Mamba → VMamba → VM-UNet | **B (cốt lõi)** | 4–5 ngày |
| 6 | Instance segmentation & spatial embedding | **C, D (cốt lõi)** | 4–5 ngày |
| 7 | Metric & đọc log | F | 1–2 ngày |
| 8 | Nối vào việc cải tiến kiến trúc | B | liên tục |

**Lộ trình rút gọn nếu chỉ còn 2 tuần (tới họp 13/11):** làm giai đoạn 1 → 3 ở mức "hiểu ý tưởng"
(mỗi giai đoạn 1–2 ngày), rồi dồn thời gian cho **5 và 6**. Hai giai đoạn này là phần khó bảo vệ nhất vì là đóng góp của bạn.

---

## Giai đoạn 1 — Mạng nơ-ron & huấn luyện (nền của mọi thứ)

**Vì sao cần:** mọi dòng trong `train.py` (optimizer, loss, epoch, AMP, seed) đều thuộc phần này.

**Ánh xạ sang tư duy lập trình:** một model là một **hàm có tham số** `y = f(x; θ)`. Huấn luyện là vòng lặp
"gọi hàm → đo sai số → chỉnh θ một chút theo hướng làm sai số giảm". Không có gì ma thuật hơn thế.

### Khái niệm phải nắm
- Tensor và **shape** (`[B, C, H, W]` = batch, kênh, cao, rộng). Đọc shape là kỹ năng debug số 1.
- Neuron / lớp tuyến tính, hàm kích hoạt (ReLU, sigmoid, tanh, GELU/SiLU).
- **Loss function** — đo "sai bao nhiêu". **Gradient** — hướng chỉnh θ. **Backpropagation** — cách tính gradient tự động.
- **Gradient descent / SGD / Adam / AdamW**; learning rate; weight decay.
- Batch, epoch, iteration. Train / valid / test và **vì sao phải tách** (đồ án bạn tách theo *scene*, không theo ảnh — hiểu vì sao: tránh rò rỉ).
- Overfitting, regularization: dropout, **drop path** (`drop_path_rate=0.2` trong config), weight decay.
- Mixed precision (AMP), seed và tính tái lập.
- **Transfer learning / pretrained**: bạn nạp `vmamba_small_e238_ema.pth` (VMamba huấn luyện sẵn trên ImageNet) cho encoder.

### Học từ đâu (chọn 1 video + 1 bài tập)
- 3Blue1Brown, playlist *Neural Networks* (4 video, ~1 giờ): trực giác về gradient descent và backprop.
- Andrej Karpathy, *Neural Networks: Zero to Hero*, **chỉ video 1 (micrograd)**: tự viết backprop ~100 dòng — sau đó bạn sẽ không còn sợ chữ "backprop".
- PyTorch Tutorials, phần *Learn the Basics* (Tensors → Datasets → Build Model → Autograd → Optimization).

### Làm trong repo
1. Mở [`experiments/multitask_vmunet_spatialonly/train.py`](../experiments/multitask_vmunet_spatialonly/train.py), tìm theo thứ tự:
   `seed` → tạo dataset/dataloader → tạo model → optimizer → vòng `for epoch` → `loss.backward()` → `optimizer.step()` → validate → lưu `best.pth`.
   Ghi chú từng bước một dòng bằng tiếng Việt.
2. Trả lời: vì sao có `model.train()` và `model.eval()`? vì sao validate dùng `torch.no_grad()`? `best.pth` chọn theo metric nào (`e2e_f1`) và vì sao không chọn theo loss?
3. Chạy `--smoke-test` (5 batch train + 3 batch val) và đọc từng dòng log in ra.

### Tự kiểm tra (nếu trả lời được là qua)
- Giải thích bằng 3 câu: "train một model là gì".
- Loss giảm mà metric không tăng thì có thể do đâu?
- `lr=1e-4` quá lớn / quá nhỏ thì thấy gì trong log?

---

## Giai đoạn 2 — CNN & ảnh: "model nhìn ảnh thế nào"

**Vì sao cần:** hướng chính của bạn là *"model phải thấy cả con tằm, không chỉ từng mảnh"*. Đó chính là bài toán
**receptive field** (vùng ảnh mà một neuron nhìn thấy). Không hiểu nó thì không hiểu vì sao đổi kiến trúc lại có ý nghĩa.

### Khái niệm phải nắm
- Ảnh là tensor; kênh màu; chuẩn hoá (normalize).
- **Convolution**: kernel trượt trên ảnh; stride; padding; số tham số của một lớp conv.
- Feature map; pooling / downsampling; upsampling.
- **Receptive field**: chồng nhiều conv/pooling thì mỗi neuron nhìn được vùng lớn dần. Vùng nhìn nhỏ ⇒ chỉ thấy "mảnh", không thấy "cả con".
- Batch Norm / Layer Norm: ổn định huấn luyện (VM-UNet dùng **LayerNorm**).
- Residual connection (`x + f(x)`): xem `VSSBlock.forward` trong `vmamba.py:532` — đúng một dòng `input + drop_path(...)`.
- Data augmentation: lật, xoay, đổi sáng; trong dự án còn có **ảnh ghép** (composed) — hiểu vì sao đồ án dùng chúng.

### Học từ đâu
- Stanford **CS231n**, ghi chú *Convolutional Networks* (cs231n.github.io) — đọc phần Conv layer, Pooling, ConvNet Architectures.
- d2l.ai (Dive into Deep Learning), chương *Convolutional Neural Networks* — đọc 6.2–6.5 và 7.x.
- Bài viết "Understanding the Effective Receptive Field in Deep CNNs" (Luo et al., 2016) — chỉ cần đọc phần mở đầu và hình minh hoạ.

### Làm trong repo
- Tự tay vẽ: ảnh 256×256 → PatchEmbed (patch_size=4) → 64×64 → PatchMerging → 32×32 → 16×16 → 8×8. Ghi số kênh mỗi tầng (`dims=[96,192,384,768]`, trong `VSSM.__init__`).
  Ở tầng cuối (8×8), một "token" đại diện cho vùng 32×32 pixel gốc. Con tằm trong ảnh 256² to cỡ nào so với con số đó?

### Tự kiểm tra
- Receptive field là gì, tăng nó bằng những cách nào?
- Vì sao con tằm bị lá che thành 3 mảnh lại là bài toán *receptive field / ngữ cảnh toàn cục*?

---

## Giai đoạn 3 — Segmentation & U-Net

**Vì sao cần:** VM-UNet = U-Net với khối Mamba thay cho khối conv. Phải hiểu U-Net trước.

### Khái niệm phải nắm
- Phân loại ảnh vs **semantic segmentation** (mỗi pixel một nhãn) vs **instance segmentation** (mỗi *vật thể* một id) — đồ án thuộc loại cuối.
- **Encoder–decoder**: encoder nén ảnh lấy ngữ cảnh, decoder phóng to lại để ra bản đồ pixel.
- **Skip connection** của U-Net: nối đặc trưng chi tiết từ encoder sang decoder (xem `forward_features_up` trong `vmamba.py:790`, dòng `x + skip_list[...]`).
- Đầu ra theo pixel: `final_conv` 1×1 biến đặc trưng thành `num_classes` kênh (`vmamba.py:745`). Đồ án bạn: 5 kênh.
- Logit vs xác suất (sigmoid); ngưỡng (`body_thr=0.5`).
- **Loss cho segmentation**: BCE (đúng/sai từng pixel) và **Dice** (độ trùng khớp vùng). Vì sao cộng cả hai: BCE bị chi phối bởi nền khi vật thể nhỏ; Dice bù lại.
- **Mask ignore / valid**: loss bỏ qua vùng không có nhãn (`_masked_bce_dice` trong `src/losses.py:157`).

### Học từ đâu
- Paper **U-Net** (Ronneberger et al., 2015) — ngắn, đọc hết.
- CS231n, bài *Detection and Segmentation*; hoặc d2l.ai chương *Computer Vision* (semantic segmentation).
- Bài blog bất kỳ giải thích "Dice loss vs BCE" — chỉ cần hiểu công thức Dice.

### Làm trong repo
1. Đọc `_masked_bce_dice` ([`src/losses.py:157`](../src/losses.py)) và `BodyDiseaseLoss` (`:210`). Vẽ một ví dụ số nhỏ 4×4 để tự tính tay BCE và Dice.
2. Vì sao **loss bệnh chỉ tính trong vùng thân** (`BCE inside body`)? (Gợi ý: nền không có bệnh, tính vào chỉ làm nhiễu.)
3. Vì sao ảnh dày `dense_seg_v2` đặt `disease = -1` và loss bệnh bỏ qua chúng?

### Tự kiểm tra
- Vẽ lại sơ đồ U-Net và chỉ ra skip connection giúp gì.
- Semantic vs instance segmentation: hai con tằm chạm nhau thì semantic cho ra gì, instance cho ra gì?

---

## Giai đoạn 4 — Attention & Transformer (để hiểu *vì sao* có Mamba)

**Vì sao cần:** Mamba được sinh ra để thay Transformer ở chỗ Transformer yếu. Không hiểu Transformer thì phần
"Mamba tốt hơn ở đâu" không trả lời được. **Không cần học sâu**, chỉ cần nắm ý tưởng và điểm yếu.

### Khái niệm phải nắm
- **Self-attention**: mỗi vị trí nhìn *tất cả* vị trí khác và quyết định chú ý vào đâu (Query, Key, Value).
- Ưu: receptive field **toàn cục ngay từ lớp đầu**. Nhược: chi phí **O(N²)** theo số token — ảnh lớn thì rất đắt.
- Vision Transformer (ViT): cắt ảnh thành patch, mỗi patch là một token. **Swin Transformer**: attention trong cửa sổ cục bộ để giảm chi phí (dự án có `swin_unet` làm tham chiếu cũ).
- Positional encoding: vì attention không biết thứ tự/vị trí.

### Học từ đâu
- Jay Alammar, *The Illustrated Transformer* (bài blog, có hình).
- 3Blue1Brown, *Attention in transformers, visually explained*.
- Paper **ViT** ("An Image is Worth 16x16 Words") — đọc Abstract + hình 1.

### Tự kiểm tra
- Vì sao attention là O(N²)? Với ảnh 256×256 patch 4 có bao nhiêu token, và bình phương lên là bao nhiêu?
- Attention và convolution khác nhau chỗ nào về "vùng nhìn thấy"?

---

## Giai đoạn 5 — SSM → Mamba → VMamba → VM-UNet ⭐ (phần cốt lõi của bảo vệ)

**Vì sao cần:** đây là backbone bạn đang dùng và đang muốn sửa. Bạn phải giải thích được *nó chạy ra sao* và *train thế nào*.

### Thứ tự học (đừng nhảy cóc)

**5.1 State Space Model (SSM) cơ bản**
- Ý tưởng: duyệt chuỗi từng phần tử, giữ một **trạng thái ẩn** `h` (bộ nhớ nén). Mỗi bước: `h_t = A·h_{t-1} + B·x_t`, `y_t = C·h_t`.
- Giống RNN nhưng có cấu trúc toán học để huấn luyện song song được.
- Chi phí **O(N)** theo độ dài chuỗi (so với O(N²) của attention).

**5.2 Mamba (Gu & Dao, 2023)** — điểm mới là **selective** SSM
- Tham số B, C, Δ **phụ thuộc vào nội dung đầu vào** ⇒ model tự quyết định nhớ gì / quên gì.
- Cần **selective scan** để chạy nhanh: đó là kernel CUDA từ `mamba_ssm` mà CLAUDE.md nhắc
  (không có thì ~18 s/step thay vì ~0.1 s). Trong `vmamba.py:12` có bản tham chiếu thuần PyTorch `selective_scan_ref` — đọc nó để hiểu vòng lặp thật sự.
- Điều phải nói được: *Mamba xử lý chuỗi 1 chiều, ảnh là 2 chiều ⇒ cần cách biến ảnh thành chuỗi.*

**5.3 VMamba (Liu et al., 2024)** — đưa Mamba vào ảnh
- **SS2D (2D Selective Scan)**: duyệt ảnh theo **4 hướng** (trái→phải, phải→trái, trên→dưới, dưới→trên), chạy SSM mỗi hướng rồi cộng lại. Xem class `SS2D` (`vmamba.py:306`).
- **VSSBlock** (`vmamba.py:532`) = `LayerNorm → SS2D → cộng residual`. **VSSLayer** = nhiều VSSBlock + PatchMerging. **VSSM** (`:684`) = cả mạng.
- Chú ý: `d_state=16` là kích thước "bộ nhớ" của SSM.

**5.4 VM-UNet (Ruan & Xiang, 2024)** — ghép VMamba vào khung U-Net
- Encoder: VSSLayer + PatchMerging (giảm kích thước). Decoder: VSSLayer_up + PatchExpand (phóng to). Skip connection giữa hai bên.
- `depths=[2,2,2,2]`, `depths_decoder=[2,2,2,1]` = số VSSBlock mỗi tầng; `dims=[96,192,384,768]`.
- Bạn không viết lại backbone: `MultiTaskVMUNet` ([`experiments/multitask_vmunet/model.py`](../experiments/multitask_vmunet/model.py)) chỉ bọc `VSSM`, đổi số kênh đầu ra,
  và nạp pretrained cho encoder rồi **soi gương sang decoder** (`load_pretrained`).

### Học từ đâu
- Maarten Grootendorst, *A Visual Guide to Mamba and State Space Models* (tìm theo tên; có hình rất dễ hiểu) — **đọc đầu tiên**.
- Paper **Mamba**: *Linear-Time Sequence Modeling with Selective State Spaces* — đọc Intro + mục Selective SSM, bỏ qua chứng minh.
- Paper **VMamba**: *Visual State Space Model* — đọc mục SS2D và hình về "Cross-Scan".
- Paper **VM-UNet**: *Vision Mamba UNet for Medical Image Segmentation* — ngắn, khớp trực tiếp với `models/vmunet/`.
- (Tuỳ chọn) S4 / HiPPO — chỉ khi hội đồng hay hỏi sâu về toán.

### Làm trong repo (đây là phần "đào code")
1. In shape qua từng tầng: viết 15 dòng script nạp `SpatialOnlyVMUNet`, đưa vào tensor `[1,3,256,256]`, in shape sau `patch_embed`, mỗi `layers[i]`, mỗi `layers_up[i]`, và `final_conv`. Đối chiếu với hình bạn vẽ ở Giai đoạn 2.
2. Đọc `SS2D.forward` và tìm: 4 hướng quét nằm ở đâu? `selective_scan_fn` được gọi thế nào?
3. Đếm tham số: `sum(p.numel() for p in model.parameters())` — đối chiếu "27.4 M" trong CLAUDE.md.
4. Đọc `SpatialOnlyVMUNet.__init__` ([`model.py:26`](../experiments/multitask_vmunet_spatialonly/model.py)): vì sao khởi tạo offset = 0 và σ = 1 (`fill_(0)`, `fill_(1)`)? (Gợi ý: để lúc đầu mọi pixel chưa "kéo" đi đâu, huấn luyện ổn định.)

### Tự kiểm tra
- Mamba khác Transformer ở đâu (độ phức tạp, cách nhớ thông tin)? Khác CNN ở đâu?
- SS2D quét 4 hướng để làm gì? Nếu chỉ quét 1 hướng thì pixel ở góc nào được "thấy" ít ngữ cảnh nhất?
- Vì sao con tằm bị chia nhiều mảnh có thể khó với cách quét này? *(Đây là giả thuyết để bạn tự kiểm chứng, đừng khẳng định khi chưa có thí nghiệm.)*

---

## Giai đoạn 6 — Instance segmentation & Spatial Embedding ⭐ (đóng góp của bạn)

**Vì sao cần:** đây là lý do đồ án tách được từng con và là hướng cô đã chốt. Nó cũng là chỗ bạn phải bảo vệ *"vì sao bỏ watershed"*.

### Thứ tự học

**6.1 Vì sao semantic segmentation chưa đủ.** Hai con chạm nhau ⇒ mask thân là một mảng liền, không biết có mấy con.

**6.2 Các cách tách con (để so sánh khi bảo vệ)**
| Cách | Ý tưởng | Trong dự án |
|---|---|---|
| Watershed trên biên | Dự đoán biên → làm "hạt giống" → lan ra | B1, E1, E2 (`split_instances`) — **đã bỏ** |
| Proposal-based (Mask R-CNN) | Đề xuất hộp rồi cắt mask | không dùng |
| Dynamic / SOLO(v2) | Dự đoán mask theo vị trí lưới | `experiments/solov2` — chỉ tham chiếu |
| **Embedding (Discriminative loss)** | Mỗi pixel ra một vector; pixel cùng con gần nhau | `multitask_vmunet_disc` — tham chiếu |
| **Spatial embedding (Neven 2019)** | Mỗi pixel *trỏ về tâm* con của nó | **E2 / E3 — hướng chính** |

**6.3 Spatial embedding (Neven et al., CVPR 2019) — đọc kỹ nhất**
- Mỗi pixel `i` dự đoán **offset** trỏ về tâm con. Embedding: `e_i = tanh(offset_i) + toạ_độ_i`. Pixel cùng con ⇒ `e_i` cùng dồn về một điểm.
- Mỗi con có một **σ** (độ rộng vùng); xác suất pixel thuộc con k: `φ_k(e_i) = exp(−‖e_i − C_k‖² · exp(10σ̄_k))` (gaussian).
- **Seed map**: mỗi pixel dự đoán "mình gần tâm cỡ nào"; chỗ seed cao nhất sẽ làm tâm lúc suy luận.
- **Loss** (xem docstring [`SpatialEmbeddingLoss`](../src/losses.py), `src/losses.py:349`): `w_inst·Lovász + w_var·L_var + w_seed·L_seed` = `1, 10, 1`.
  - *Lovász-hinge*: tối ưu trực tiếp **IoU** của mask từng con (thay vì BCE).
  - *L_var*: σ của các pixel trong một con giống nhau.
  - *L_seed*: seed bắt chước φ.
- **Suy luận (không watershed):** lặp "chọn pixel chưa gán có seed cao nhất → làm tâm → lấy mọi pixel có φ > 0.5 → đánh dấu đã gán" — xem `cluster_spatial_embeddings` ([`src/metrics.py:207`](../src/metrics.py)).
- **Điểm hay cho đồ án:** mọi mảnh của một con *cùng trỏ về một tâm* ⇒ một id cho cả con dù bị lá che. Đây chính là lý do bạn chọn nó thay watershed (xem `meetings/2026-10-30.md`, mục 4).

### Học từ đâu
- Paper **Neven et al. 2019**: *Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth*. Đọc kỹ mục 3 (Method), xem hình 1–2.
- Paper **De Brabandere et al. 2017**: *Semantic Instance Segmentation with a Discriminative Loss Function* (tiền thân, ngắn).
- Paper **Lovász-Softmax** (Berman et al., 2018) — chỉ cần hiểu *"surrogate cho IoU"*, bỏ phần chứng minh.
- Đọc thêm file scoping review của bạn: `Reports/Meeting_2026-10-30/literature/scoping_review_occlusion_fragmentation.md`.

### Làm trong repo
1. Chạy `python -m experiments.multitask_vmunet_spatialonly.test_spatialonly` (nếu test chạy được) và đọc để biết loss được kiểm chứng ra sao.
2. Lấy 1 ảnh có 2–3 con chạm nhau, chạy inference, rồi **vẽ**: (a) kênh thân, (b) offset thành mũi tên, (c) kênh seed, (d) kết quả gom. Nhìn bằng mắt là cách hiểu nhanh nhất.
3. Thử đổi `seed_thr` (0.5 → 0.3 / 0.7) trên vài ảnh `valid`, quan sát số con thay đổi thế nào. Vì sao tăng/giảm lại gộp/tách nhầm?
4. Giải thích bằng lời, không công thức: *"vì sao hai mảnh rời của một con cùng một id mà watershed thì không làm được?"*

### Tự kiểm tra
- Vẽ lên giấy: 2 con tằm chạm nhau ⇒ các mũi tên offset trông thế nào?
- σ để làm gì? Seed để làm gì? Thiếu một trong hai thì suy luận hỏng ở đâu?
- Vì sao *không thêm loss mới* được cô chấp nhận còn *đổi kiến trúc* thì được khuyến khích?

---

## Giai đoạn 7 — Metric & đọc `train_log.csv`

**Vì sao cần:** mọi kết luận trong luận văn là con số. Hiểu sai metric là sai cả bài.

| Metric | Ý nghĩa | Vì sao có |
|---|---|---|
| `body_dice` | Dice pixel của kênh thân | chỉ đo "thấy tằm hay không", chưa đo tách con |
| `inst_f1` | tách đúng con (IoU ≥ 0.5) | đo bài toán instance |
| `cls_acc_matched` | trong các con tách đúng, đoán đúng bệnh | tách biệt lỗi chẩn đoán với lỗi tách |
| `e2e_f1` | tách đúng **và** chẩn đoán đúng | **điểm chính**, chọn `best.pth` |
| `grasserie_recall` | tỉ lệ con bệnh bị phát hiện | bỏ sót con bệnh là lỗi đắt nhất |

### Khái niệm phải nắm
- Precision, Recall, F1; TP/FP/FN. IoU. **Matching** (ghép dự đoán với đáp án): `match_instances` ([`src/metrics.py:269`](../src/metrics.py)).
- Vì sao quy ước "một con = một id" làm điểm watershed bị trần (~0.79) — xem CLAUDE.md §3.4.
- Một seed, một lần chạy ⇒ chênh nhỏ (≤ 0.015) là **nhiễu**, không phải cải tiến. Hiểu "variance giữa các lần train".
- Ablation study: đổi **một** thứ mỗi lần (đúng quy tắc cô đặt).

### Làm trong repo
- Mở một `train_log.csv` bất kỳ trong `Reports/Meeting_*/checkpoints/`. Vẽ `loss` và `e2e_f1` theo epoch. Tìm epoch tốt nhất. Có dấu hiệu overfit không?
- Tự tính tay `inst_f1` cho một ảnh 3 con (ví dụ 2 TP, 1 FP, 1 FN).

---

## Giai đoạn 8 — Nối vào việc cải tiến kiến trúc

Đến đây bạn đủ nền để **tham gia quyết định** thay vì chỉ chạy lệnh. Mục tiêu cô đặt:
*"model hiểu high level của con tằm — thấy cả con, không chỉ từng mảnh"*, giữ nguyên loss spatial embedding.

Các câu hỏi bạn nên tự trả lời được trước họp 13/11 (đây là **câu hỏi nghiên cứu**, chưa phải kết luận):
1. Lỗi của baseline E3 nằm ở đâu: pixel của mảnh nhỏ không biết mình thuộc con nào (thiếu ngữ cảnh toàn cục), hay seed bị chia đôi, hay σ sai? → nhìn ảnh lỗi trước khi thiết kế.
2. Receptive field hiệu dụng của VM-UNet 256² là bao nhiêu? Tầng 8×8 (mỗi token ~32 px) có đủ chứa một con và các mảnh bị che của nó không?
3. Thứ tự quét 4 hướng trong SS2D có thiên vị gì với con nằm chéo / cong không?
4. Đầu ra có nên đọc từ đặc trưng đa tầng (decoder đa tỉ lệ) thay vì chỉ tầng cuối?
5. Số `depths`, kích thước ảnh vào (256² vs lớn hơn), `d_state` có ảnh hưởng gì tới khả năng thấy "cả con"?

Mỗi ý tưởng muốn thử: (a) viết giả thuyết 1 câu, (b) đổi **một** thứ trong `experiments/<tên>/model.py` (không sửa `models/`), (c) train, (d) ghi vào `Reports/experiment_results.xlsx` đúng quy trình trong CLAUDE.md.
Bạn có thể nhờ Claude giúp phần (b) và smoke test; bạn tự chạy train đầy đủ trong tmux.

---

## 🎤 Bảo vệ: câu hỏi hội đồng chắc chắn hỏi (và nơi tìm câu trả lời)

| Câu hỏi | Giai đoạn | Chỗ tham chiếu |
|---|---|---|
| Vì sao chọn Mamba / VM-UNet chứ không phải U-Net hay Transformer? | 4, 5 | paper VMamba, VM-UNet; so sánh O(N) vs O(N²) |
| Mamba hoạt động thế nào? Selective nghĩa là gì? | 5 | `selective_scan_ref` ở `vmamba.py:12` |
| SS2D quét ảnh thế nào? | 5 | `vmamba.py:306` |
| Vì sao dùng pretrained VMamba? Decoder có pretrained không? | 1, 5 | `load_pretrained` trong `multitask_vmunet/model.py` |
| Model có mấy đầu ra, mỗi kênh để làm gì? | 3, 6 | `spatialonly/model.py` |
| Vì sao loss bệnh chỉ trong vùng thân? Vì sao có vùng ignore? | 3 | `src/losses.py` |
| Spatial embedding là gì? σ, seed, Lovász để làm gì? | 6 | `SpatialEmbeddingLoss` |
| Vì sao bỏ watershed? | 6 | `meetings/2026-10-30.md` mục 4 |
| Vì sao baseline là E3 chứ không phải E2? | 6, 7 | `meetings/2026-10-30.md` |
| Tách một con khi bị che thành nhiều mảnh thế nào? | 6 | quy ước "một id = một con" |
| Train thế nào: optimizer, lr, epoch, batch, AMP, seed? | 1 | `config.py`, `train.py` |
| Nhãn là mask giả từ SAM 3 — có đáng tin không? | dữ liệu | `DATASET_STRUCTURE.md`; chưa chuyên gia kiểm tra |
| Vì sao split theo scene? | 1 | `DATASET_STRUCTURE.md` |
| `e2e_f1` đo gì? Vì sao chọn nó? | 7 | `src/metrics.py` |
| Sai số một seed là bao nhiêu? Cải tiến này có thật không? | 7 | ablation, nhiều seed |
| Ảnh dày ngoài miền huấn luyện thì sao? | 6, 7 | `data/test`, Limitation 2 |
| Hạn chế của đồ án? | tất cả | `Reports/Meeting_2026-10-17/results/limitations_summary.md` |

**Cách luyện:** mỗi tối chọn 2 câu, trả lời **thành tiếng** 2 phút không nhìn tài liệu. Câu nào ấp úng thì quay lại giai đoạn tương ứng.

---

## 📅 Gợi ý lịch (14 ngày tới họp 13/11, ~3 giờ/ngày)

| Ngày | Việc |
|---|---|
| 1–2 | GĐ1: video 3Blue1Brown + micrograd + đọc `train.py`, chạy smoke test |
| 3–4 | GĐ2: CNN, receptive field; vẽ sơ đồ shape qua VM-UNet (đã in shape bằng script) |
| 5–6 | GĐ3: U-Net, Dice/BCE; tính tay loss; đọc `src/losses.py` phần thân/bệnh |
| 7 | GĐ4: Transformer ở mức ý tưởng + vì sao O(N²) |
| 8–10 | GĐ5: SSM → Mamba → VMamba → VM-UNet; đào `vmamba.py`; trả lời bộ câu hỏi 5 |
| 11–13 | GĐ6: Neven 2019, `SpatialEmbeddingLoss`, vẽ offset/seed trên ảnh thật |
| 14 | GĐ7 + luyện 6 câu hỏi bảo vệ; viết 5 câu hỏi nghiên cứu cho cô (GĐ8) |

Sau 13/11: học GĐ8 song song với việc thử kiến trúc; học thêm theo nhu cầu.

---

## 📖 Từ điển nhanh (EN → VI)

| English | Tiếng Việt / giải thích ngắn |
|---|---|
| tensor | mảng nhiều chiều |
| logit | giá trị thô trước sigmoid |
| backbone | thân mạng chính (ở đây là VSSM) |
| encoder / decoder | phần nén / phần phóng to lại |
| skip connection | nối tắt từ encoder sang decoder |
| receptive field | vùng ảnh mà một neuron nhìn thấy |
| token / patch | một mảnh ảnh được coi như một "từ" |
| state space model | mô hình không gian trạng thái, bộ nhớ nén `h` |
| selective scan | quét chọn lọc, lõi của Mamba |
| embedding | vector đặc trưng của một pixel |
| offset | độ lệch trỏ về tâm con |
| seed | điểm "gần tâm", dùng chọn tâm khi suy luận |
| bandwidth (σ) | độ rộng vùng của một con |
| IoU | giao / hợp của hai mask |
| ablation | thí nghiệm bỏ/đổi từng thành phần để xem đóng góp |
| baseline | phương pháp có sẵn làm mốc so sánh |
| overfitting | học thuộc train, kém trên dữ liệu mới |
| out-of-domain | dữ liệu khác miền huấn luyện (ảnh dày trong `data/test`) |

---

## 🚫 Bỏ qua (chưa cần, đừng tốn thời gian)

- Toán sâu: chứng minh hội tụ của SGD, chứng minh Lovász extension, HiPPO/S4 chi tiết.
- GAN, diffusion, LLM, reinforcement learning.
- Tự cài đặt lại Mamba bằng CUDA/Triton.
- Object detection thuần (YOLO) — chỉ cần biết nó cho box để SAM 3 tạo mask.
- Các model khác (SOLOv2, Swin-UNet): chỉ đọc ghi chú tham chiếu, không cần học sâu (cô đã chốt dừng thử model khác).

---

## ✅ Cách dùng file này hiệu quả

1. Mỗi giai đoạn: **xem/đọc ý tưởng trước → mở code tương ứng → tự giải thích lại bằng lời**. Đừng đọc paper khi chưa xem video/blog dẫn nhập.
2. Ghi vào một file ghi chú riêng "câu tôi chưa hiểu" — mang tới hỏi cô hoặc hỏi Claude.
3. Khi hiểu một khái niệm, thêm 1–2 dòng vào `meetings/<ngày>.md` hoặc bản nháp luận văn bằng lời của chính bạn: đó chính là Section 3–4 sau này.
4. Nếu chạm chỗ nào trong file này mà bạn thấy "mơ hồ" hoặc muốn tôi đào code cùng, bảo tôi: tôi có thể viết script in shape, vẽ offset/seed, hay giải thích từng hàm một.
