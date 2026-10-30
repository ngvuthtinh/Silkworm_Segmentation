"""
src/losses.py — Shared Loss Functions for Silkworm Segmentation experiments.

Implements:
    - DiceLoss (Soft Dice Loss)
    - SegLoss (BCE + Dice)
    - ClsLoss (CrossEntropyLoss)
    - SegFirstMultiTaskLoss (handles Dataset A, Dataset B, or Joint loss)
    - InstanceMultiTaskLoss (body + boundary + disease per pixel)
    - DiscriminativeLoss (per-pixel embedding loss; groups pieces of one larva regardless of connectivity)
    - SpatialEmbeddingLoss (offset-to-centre + learnable margin + seed map; Neven et al. 2019; Lovász-hinge on each larva)

Usage:
    from src.losses import DiceLoss, SegLoss, ClsLoss, SegFirstMultiTaskLoss

Override per-experiment:
    Create losses.py inside your experiment folder to override these defaults.
    The experiment's train.py will prefer the local version via try/except import.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset_multitask import IGNORE_LABEL


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : [B, 1, H, W] (probabilities or logits after sigmoid)
        target : [B, 1, H, W] (binary ground truth)
        """
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class SegLoss(nn.Module):
    """Combined BCEWithLogitsLoss + Dice Loss for segmentation."""

    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, mask_logits: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(mask_logits, mask_gt)
        mask_pred = torch.sigmoid(mask_logits)
        loss_dice = self.dice(mask_pred, mask_gt)
        return self.w_bce * loss_bce + self.w_dice * loss_dice


def masked_cross_entropy(
    class_logits: torch.Tensor,
    class_gt: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = IGNORE_LABEL,
) -> torch.Tensor:
    """
    CrossEntropy chỉ tính trên các mẫu CÓ nhãn bệnh (class_gt != ignore_index).

    Ảnh Silkynet không có nhãn bệnh (-1) sẽ không đóng góp gradient cho head phân loại,
    nhưng vẫn huấn luyện head phân đoạn bình thường. Nếu cả batch không có nhãn, trả về 0
    (có gradient graph) thay vì NaN — vì CE mean-reduction trên tập rỗng cho ra NaN.
    """
    valid = class_gt != ignore_index
    if not valid.any():
        return class_logits.sum() * 0.0
    return F.cross_entropy(
        class_logits[valid], class_gt[valid], label_smoothing=label_smoothing
    )


class ClsLoss(nn.Module):
    """CrossEntropyLoss for binary classification (bỏ qua mẫu không có nhãn = -1)."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, class_logits: torch.Tensor, class_gt: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(class_logits, class_gt, self.label_smoothing)


class SegFirstMultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with Segmentation-First priority.

    Weighting:
        L = lambda_seg * L_seg + lambda_cls * L_cls

    Default lambda_seg = 1.0, lambda_cls = 0.1 (low impact to protect backbone).
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_cls: float = 0.1,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.seg_loss_fn = SegLoss(w_bce=w_bce, w_dice=w_dice)
        self.cls_loss_fn = ClsLoss()

    def forward(
        self,
        mask_pred: torch.Tensor | None = None,
        mask_gt: torch.Tensor | None = None,
        class_logits: torch.Tensor | None = None,
        class_gt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes total loss depending on available targets.

        Returns
        -------
        (total_loss, seg_loss_val, cls_loss_val)
        """
        device = (
            mask_pred.device
            if mask_pred is not None
            else (class_logits.device if class_logits is not None else torch.device("cpu"))
        )

        l_seg = torch.tensor(0.0, device=device)
        l_cls = torch.tensor(0.0, device=device)

        if mask_pred is not None and mask_gt is not None:
            l_seg = self.seg_loss_fn(mask_pred, mask_gt)

        if class_logits is not None and class_gt is not None:
            l_cls = self.cls_loss_fn(class_logits, class_gt)

        total_loss = self.lambda_seg * l_seg + self.lambda_cls * l_cls
        return total_loss, l_seg, l_cls


def _masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """BCE + soft Dice chỉ trên pixel có weight = 1 (pixel weight = 0 không đóng góp gradient)."""
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / weight.sum().clamp_min(1.0)
    prob = torch.sigmoid(logits) * weight
    tgt = target * weight
    dice = 1.0 - (2.0 * (prob * tgt).sum() + eps) / (prob.sum() + tgt.sum() + eps)
    return bce + dice


class InstanceMultiTaskLoss(nn.Module):
    """
    Loss cho mô hình phân đoạn TỪNG CON + chẩn đoán TỪNG CON (3 kênh đầu ra theo pixel).

        L = λ_body·(BCE+Dice)(thân) + λ_bnd·(BCE+Dice)(biên) + λ_dis·BCE(bệnh | trong thân)

    - Biên chiếm rất ít pixel nên cần Dice để không bị "học" thành toàn 0.
    - Loss bệnh chỉ tính trên pixel thuộc thân tằm (GT): nền không có khái niệm khỏe/bệnh.
    - `valid` (tuỳ chọn): 0 = vùng bỏ qua (con tằm không có nhãn) → không tính vào bất kỳ loss nào.
    """

    def __init__(self, lambda_body: float = 1.0, lambda_boundary: float = 1.0, lambda_disease: float = 1.0):
        super().__init__()
        self.lambda_body = lambda_body
        self.lambda_boundary = lambda_boundary
        self.lambda_disease = lambda_disease

    def forward(
        self,
        logits: torch.Tensor,
        body_gt: torch.Tensor,
        boundary_gt: torch.Tensor,
        disease_gt: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        logits: [B, 3, H, W] — kênh 0 = thân, 1 = biên, 2 = bệnh (logit thô)
        Trả về (total, l_body, l_boundary, l_disease).
        """
        logits = logits.float()
        w = torch.ones_like(body_gt) if valid is None else valid.float()
        l_body = _masked_bce_dice(logits[:, 0:1], body_gt, w)
        l_bnd = _masked_bce_dice(logits[:, 1:2], boundary_gt, w)

        inside = (body_gt > 0.5) & (w > 0.5) & (disease_gt >= 0)      # disease_gt = -1: con chưa biết bệnh → không tính loss bệnh
        if inside.any():
            l_dis = F.binary_cross_entropy_with_logits(logits[:, 2:3][inside], disease_gt[inside])
        else:
            l_dis = logits[:, 2:3].sum() * 0.0

        total = self.lambda_body * l_body + self.lambda_boundary * l_bnd + self.lambda_disease * l_dis
        return total, l_body, l_bnd, l_dis


class BodyDiseaseLoss(nn.Module):
    """
    Phần "không có biên" của InstanceMultiTaskLoss: model tách con bằng spatial embeddings nên không còn kênh biên.

        L = λ_body·(BCE+Dice)(thân) + λ_dis·BCE(bệnh | trong thân)

    Hai số hạng giữ đúng công thức và đúng λ của InstanceMultiTaskLoss, nên so sánh công bằng với run có kênh biên.
    """

    def __init__(self, lambda_body: float = 1.0, lambda_disease: float = 1.0):
        super().__init__()
        self.lambda_body = lambda_body
        self.lambda_disease = lambda_disease

    def forward(
        self,
        logits: torch.Tensor,
        body_gt: torch.Tensor,
        disease_gt: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """logits: [B, 2, H, W] — kênh 0 = thân, 1 = bệnh. Trả về (total, l_body, l_disease)."""
        logits = logits.float()
        w = torch.ones_like(body_gt) if valid is None else valid.float()
        l_body = _masked_bce_dice(logits[:, 0:1], body_gt, w)
        inside = (body_gt > 0.5) & (w > 0.5) & (disease_gt >= 0)      # disease_gt = -1: con chưa biết bệnh → không tính loss bệnh
        if inside.any():
            l_dis = F.binary_cross_entropy_with_logits(logits[:, 1:2][inside], disease_gt[inside])
        else:
            l_dis = logits[:, 1:2].sum() * 0.0
        return self.lambda_body * l_body + self.lambda_disease * l_dis, l_body, l_dis


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss (De Brabandere et al., 2017, arXiv:1708.02551) cho embedding theo pixel.

    Mỗi pixel thuộc một con được ánh xạ thành một vector; loss kéo các pixel CÙNG một con về gần nhau và đẩy
    các con KHÁC nhau ra xa, KHÔNG phụ thuộc hai pixel có liền kề hay không. Nhờ vậy hai mảnh của một con bị lá
    che (rời nhau) vẫn được kéo về cùng một cụm — điều mà việc tách theo thành phần liền kề không làm được.

        L = α·L_var + β·L_dist + γ·L_reg                           (công thức 4 của paper)
        L_var  = mean_c  mean_{i∈c}  [ ‖μ_c − x_i‖ − δ_v ]₊²       (kéo pixel về tâm cụm, hinge δ_v)
        L_dist = mean_{c_A≠c_B}      [ 2δ_d − ‖μ_cA − μ_cB‖ ]₊²    (đẩy các tâm cụm, hinge 2δ_d)
        L_reg  = mean_c ‖μ_c‖                                       (giữ embedding không phình ra)

    - Trung bình L_var theo TỪNG CỤM rồi theo các cụm (đúng công thức 1), nên con nhỏ/mảnh nhỏ nặng bằng con lớn.
    - Chỉ tính pixel có `inst > 0` (nền không phải một cụm) và `valid = 1` (vùng bỏ qua không đóng góp).
    - Vector hoá bằng index_add (không dựng tensor one-hot [pixel × con × chiều]), nên nhẹ bộ nhớ ở 256².
    - Cần δ_d > 2δ_v để mỗi pixel gần tâm cụm của chính nó hơn mọi tâm khác (mục 3.2 của paper).
    """

    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        eps: float = 1e-8,
    ):
        super().__init__()
        if delta_dist <= 2 * delta_var:
            raise ValueError(f"Cần delta_dist > 2*delta_var (nhận {delta_dist} và {delta_var})")
        self.delta_var, self.delta_dist = delta_var, delta_dist
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.eps = eps

    def forward(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        emb:   [B, D, H, W] — embedding thô của mô hình
        inst:  [B, H, W] int64 — id từng con (0 = nền)
        valid: [B, 1, H, W] (tuỳ chọn) — 0 = vùng bỏ qua
        Trả về (total, l_var, l_dist, l_reg), trung bình trên các ảnh có ít nhất một con.
        """
        emb = emb.float()
        b_size, dim = emb.shape[:2]
        zero = emb.sum() * 0.0                                   # giữ đồ thị gradient khi ảnh không có con nào
        acc_var = acc_dist = acc_reg = zero
        n_used = 0
        for b in range(b_size):
            ids = inst[b].reshape(-1)
            keep = ids > 0
            if valid is not None:
                keep = keep & (valid[b].reshape(-1) > 0.5)
            if not bool(keep.any()):
                continue
            x = emb[b].reshape(dim, -1)[:, keep].t()             # [M, D] pixel thuộc một con
            uniq, inv = torch.unique(ids[keep], return_inverse=True)
            n_c = uniq.numel()
            counts = torch.bincount(inv, minlength=n_c).to(x.dtype)
            means = torch.zeros(n_c, dim, device=x.device, dtype=x.dtype).index_add(0, inv, x) / counts[:, None]

            d = ((x - means[inv]).pow(2).sum(1) + self.eps).sqrt()          # ‖μ_c − x_i‖ (eps tránh NaN ở 0)
            hinge = (d - self.delta_var).clamp_min(0.0).pow(2)
            l_var = (torch.zeros(n_c, device=x.device, dtype=x.dtype).index_add(0, inv, hinge) / counts).mean()

            if n_c > 1:
                dm = ((means[:, None] - means[None]).pow(2).sum(-1) + self.eps).sqrt()
                off_diag = ~torch.eye(n_c, dtype=torch.bool, device=x.device)
                l_dist = ((2 * self.delta_dist - dm).clamp_min(0.0).pow(2) * off_diag).sum() / (n_c * (n_c - 1))
            else:
                l_dist = zero
            l_reg = (means.pow(2).sum(1) + self.eps).sqrt().mean()

            acc_var, acc_dist, acc_reg = acc_var + l_var, acc_dist + l_dist, acc_reg + l_reg
            n_used += 1

        if n_used == 0:
            return zero, zero, zero, zero
        l_var, l_dist, l_reg = acc_var / n_used, acc_dist / n_used, acc_reg / n_used
        return self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg, l_var, l_dist, l_reg


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient của mở rộng Lovász của Jaccard loss theo thứ tự lỗi đã sắp xếp (Berman & Blaschko, 2018)."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovász-hinge (surrogate lồi của 1 − IoU) trên các pixel đã trải phẳng; logits [N], labels [N] ∈ {0, 1}."""
    if labels.numel() == 0:
        return logits.sum() * 0.0
    errors = 1.0 - logits * (2.0 * labels - 1.0)
    errors_sorted, perm = torch.sort(errors, descending=True)
    return torch.dot(F.relu(errors_sorted), _lovasz_grad(labels[perm]))


class SpatialEmbeddingLoss(nn.Module):
    """
    Loss "spatial embeddings" (Neven et al., CVPR 2019, arXiv:1906.11109) cho phân đoạn TỪNG CON.

    Mỗi pixel dự đoán một vector dịch (offset) trỏ về tâm con của nó, một bề rộng σ và một điểm "hạt giống" (seed):
        e_i = tanh(offset_i) + x_i                 embedding không gian (x_i = toạ độ chuẩn hoá, ảnh vuông → [0, 1]²)
        φ_k(e_i) = exp(−Σ (e_i − C_k)² · exp(10·σ̄_k))   xác suất pixel i thuộc con k, σ̄_k = σ trung bình trên con k
    Mỗi con k có ba số hạng:
        L_inst = Lovász-hinge(φ_k·2 − 1, mask_k)   tối ưu trực tiếp IoU của mask con k (không phụ thuộc mask liền kề hay không)
        L_var  = mean (σ_i − σ̄_k)²                 các pixel của cùng con có σ giống nhau (để lúc suy luận lấy σ ở một điểm là đủ)
        L_seed = (seed_i − φ_k(e_i))²              seed ≈ độ gần tâm; nền hồi quy về 0
    Tổng: w_inst·L_inst + w_var·L_var + w_seed·L_seed (mặc định 1, 10, 1 như repo gốc), trung bình theo con rồi theo ảnh.

    **Một id = một con, mọi mảnh dùng chung id**: mask_k gồm TẤT CẢ mảnh của con k, nên các pixel của mọi mảnh cùng trỏ về một
    tâm (tâm là trung bình toạ độ của cả con, có thể nằm trong khoảng trống bị lá che) và cùng nằm trong một vùng gaussian.
    - to_center=True: tâm = trung bình toạ độ của con; False: tâm = trung bình embedding (tâm học được).
    - n_sigma = 1 (vùng tròn) hoặc 2 (elip theo trục x, y — hợp con dài).
    - `valid` (tuỳ chọn): 0 = vùng bỏ qua (con không nhãn), không đóng góp vào bất kỳ số hạng nào.
    - Vector hoá theo [ảnh × con × pixel] (không lặp Python theo từng con): nhanh hơn ~10 lần, cùng kết quả với bản lặp (xem test).

    Viết lại từ paper và công thức trong repo gốc (CC BY-NC 4.0: dùng cho nghiên cứu, không thương mại — xem notes.md).
    """

    def __init__(
        self,
        to_center: bool = True,
        n_sigma: int = 2,
        foreground_weight: float = 10.0,
        w_inst: float = 1.0,
        w_var: float = 10.0,
        w_seed: float = 1.0,
    ):
        super().__init__()
        if n_sigma not in (1, 2):
            raise ValueError("n_sigma phải là 1 (tròn) hoặc 2 (elip)")
        self.to_center, self.n_sigma = to_center, n_sigma
        self.foreground_weight = foreground_weight
        self.w_inst, self.w_var, self.w_seed = w_inst, w_var, w_seed

    @staticmethod
    def coordinate_map(height: int, width: int, device: torch.device) -> torch.Tensor:
        """[2, H, W]: x ∈ [0, 1] theo cột, y ∈ [0, 1] theo hàng (chuẩn hoá theo từng cạnh ảnh)."""
        xm = torch.linspace(0, 1, width, device=device).view(1, 1, -1).expand(1, height, width)
        ym = torch.linspace(0, 1, height, device=device).view(1, -1, 1).expand(1, height, width)
        return torch.cat((xm, ym), 0)

    def forward(
        self,
        pred: torch.Tensor,
        inst: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pred:  [B, 2 + n_sigma + 1, H, W] thô — kênh 0-1 offset, kênh 2.. σ, kênh cuối seed
        inst:  [B, H, W] int64 — id từng con (0 = nền)
        valid: [B, 1, H, W] (tuỳ chọn) — 0 = vùng bỏ qua
        Trả về (total, L_inst, L_var, L_seed), trung bình theo ảnh.
        """
        pred = pred.float()
        b_size, _, height, width = pred.shape
        n, n_pix, dev = self.n_sigma, height * width, pred.device
        xym = self.coordinate_map(height, width, dev).reshape(2, n_pix)
        emb = torch.tanh(pred[:, 0:2]).reshape(b_size, 2, n_pix) + xym                    # [B, 2, P]
        sigma = pred[:, 2:2 + n].reshape(b_size, n, n_pix)                                # [B, n, P]
        seed = torch.sigmoid(pred[:, 2 + n]).reshape(b_size, n_pix)                       # [B, P]
        ids = inst.reshape(b_size, n_pix)
        vmask = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid.reshape(b_size, n_pix) > 0.5

        # đánh lại id mỗi ảnh thành 1..K (0 = nền hoặc vùng bỏ qua) bằng MỘT lệnh unique cho cả batch; chỉ pixel hợp lệ mới có id > 0
        sel = vmask & (ids > 0)
        idx = torch.zeros_like(ids)
        k_max = 0
        if bool(sel.any()):
            img_no = torch.arange(b_size, device=dev)[:, None].expand_as(ids)
            uniq, inv = torch.unique((img_no[sel] << 20) + ids[sel], return_inverse=True)   # khoá = (ảnh, id); id < 2^20
            img_of = uniq >> 20
            per_img = torch.bincount(img_of, minlength=b_size)                              # số con mỗi ảnh
            idx[sel] = inv - (torch.cumsum(per_img, 0) - per_img)[img_of[inv]] + 1
            k_max = int(per_img.max())

        seed_bg = ((seed ** 2) * ((idx == 0) & vmask)).sum(1)                             # [B] nền hồi quy về 0
        if k_max == 0:
            seed_loss = (seed_bg / n_pix).mean()
            zero = pred.sum() * 0.0
            return self.w_seed * seed_loss + zero, zero, zero, seed_loss

        # Tính đồng thời cho mọi con của mọi ảnh: tensor [B, K, P] thay cho vòng lặp Python theo từng con
        in_mask = idx[:, None, :] == torch.arange(1, k_max + 1, device=dev)[None, :, None]    # [B, K, P]; mọi mảnh của con k
        cnt = in_mask.sum(-1)                                                             # [B, K]
        exist = cnt > 0
        cntf = cnt.clamp_min(1).to(pred.dtype)
        inf = in_mask.to(pred.dtype)
        if self.to_center:
            center = torch.einsum("bkp,cp->bkc", inf, xym) / cntf[..., None]              # tâm = trung bình toạ độ của con
        else:
            center = torch.einsum("bkp,bcp->bkc", inf, emb) / cntf[..., None]             # tâm học được = trung bình embedding
        s = torch.einsum("bkp,bnp->bkn", inf, sigma) / cntf[..., None]                    # σ trung bình mỗi con [B, K, n]
        var_k = (inf[:, :, None, :] * (sigma[:, None] - s.detach()[..., None]) ** 2).sum((-1, -2)) / (cntf * n)
        s_exp = torch.exp(s * 10.0)
        d2 = (emb[:, None] - center[..., None]) ** 2                                      # [B, K, 2, P]
        dist = torch.exp(-(d2 * s_exp[..., None]).sum(2))                                 # φ_k(e_i) [B, K, P]

        # Lovász-hinge của mọi (ảnh, con) cùng lúc; pixel vùng bỏ qua được đẩy xuống cuối bảng sắp xếp (đóng góp 0)
        errors = 1.0 - (dist * 2 - 1) * (2.0 * inf - 1.0)
        errors = torch.where(vmask[:, None, :], errors, torch.full_like(errors, -1e4))
        err_sorted, perm = torch.sort(errors, dim=-1, descending=True)
        gt_sorted = torch.gather(inf, -1, perm)
        gts = inf.sum(-1, keepdim=True)
        jaccard = 1.0 - (gts - gt_sorted.cumsum(-1)) / (gts + (1.0 - gt_sorted).cumsum(-1))
        grad = torch.cat([jaccard[..., :1], jaccard[..., 1:] - jaccard[..., :-1]], dim=-1)
        inst_k = (F.relu(err_sorted) * grad).sum(-1)                                      # [B, K]

        n_obj = exist.sum(1).clamp_min(1).to(pred.dtype)                                  # số con mỗi ảnh
        inst_b = (inst_k * exist).sum(1) / n_obj
        var_b = (var_k * exist).sum(1) / n_obj
        seed_fg = self.foreground_weight * (inf * (seed[:, None, :] - dist.detach()) ** 2).sum((1, 2))
        seed_b = (seed_bg + seed_fg) / n_pix
        tot_inst, tot_var, tot_seed = inst_b.mean(), var_b.mean(), seed_b.mean()
        total = self.w_inst * tot_inst + self.w_var * tot_var + self.w_seed * tot_seed
        return total, tot_inst, tot_var, tot_seed
