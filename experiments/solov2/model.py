"""
model.py — SilkSOLOv2: SOLOv2 của AdelaiDet + hai thay đổi cho data tằm (chỉ ở bước dựng nhãn và loss phân loại, kiến trúc giữ nguyên).

Data tằm có hai loại nhãn mà SOLOv2 gốc không có:
  - con CHƯA BIẾT BỆNH (lớp 3 trong masks_cls, vd data/dense_seg_v2): biết là tằm nhưng không biết Khỏe/Bệnh.
    → category_id = UNKNOWN (-1). Mask của con vẫn được học (dice loss). Ở nhánh phân loại, ô lưới của con này được dạy
      "có ít nhất một lớp đúng": p_có_tằm = 1 - Π_c (1 - σ(x_c)) → 1 (focal loss trên p_có_tằm), không ép lớp nào.
  - VÙNG BỎ QUA (masks_ignore: con tằm không có nhãn, mảnh vụn): → category_id = IGNORE (-2). Không có mask, các ô lưới của nó
    không tính loss phân loại (không bị dạy là nền).
Con có lớp bình thường (0 = Grasserie, 1 = Healthy) học y như SOLOv2 gốc.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from fvcore.nn import sigmoid_focal_loss_jit

from experiments.solov2 import adet_shim  # noqa: F401
from adet.modeling.solov2.loss import dice_loss
from adet.modeling.solov2.solov2 import SOLOv2
from adet.modeling.solov2.utils import center_of_mass, imrescale

UNKNOWN, IGNORE = -1, -2


def silk_cate_loss(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float, gamma: float) -> torch.Tensor:
    """
    preds [N, C] logit; labels [N]: 0..C-1 = có lớp, C = nền, UNKNOWN = tằm chưa biết lớp, IGNORE = bỏ qua. Trả về tổng loss (chưa chia).
    """
    preds = preds.float()
    known = (labels >= 0) & (labels < num_classes)
    use = known | (labels == num_classes)
    onehot = torch.zeros_like(preds)
    onehot[known, labels[known]] = 1.0
    loss = sigmoid_focal_loss_jit(preds[use], onehot[use], gamma=gamma, alpha=alpha, reduction="sum")
    unk = labels == UNKNOWN
    if unk.any():
        s = F.softplus(preds[unk]).sum(1)                       # = -log Π_c (1 - σ(x_c)) ≥ 0
        p_any = -torch.expm1(-s)                                # 1 - Π_c (1 - σ(x_c))
        loss = loss + (alpha * torch.exp(-gamma * s) * -torch.log(p_any.clamp_min(1e-12))).sum()
    return loss


@META_ARCH_REGISTRY.register()
class SilkSOLOv2(SOLOv2):
    """SOLOv2 hiểu UNKNOWN / IGNORE (xem đầu file). Phần còn lại kế thừa nguyên bản."""

    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_labels_raw = gt_instances[img_idx].gt_classes
        device = gt_bboxes_raw.device                              # bản gốc lấy gt_labels_raw[0]: lỗi khi ảnh không có con nào
        n = len(gt_labels_raw)
        gt_masks_raw = gt_instances[img_idx].gt_masks.tensor if n else torch.zeros((0, 1, 1), dtype=torch.bool, device=device)
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        for (lower_bound, upper_bound), stride, num_grid in zip(self.scale_ranges, self.strides, self.num_grids):
            cate_label = torch.full([num_grid, num_grid], self.num_classes, dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
            ins_label, grid_order = [], []
            hit = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten() if n else gt_areas.new_zeros(0).long()
            if len(hit):
                labels = gt_labels_raw[hit]
                hit = torch.cat([hit[labels != IGNORE], hit[labels == IGNORE]])     # vùng bỏ qua xử lý sau cùng: không đè lên con thật
                gt_bboxes, gt_labels, gt_masks = gt_bboxes_raw[hit], gt_labels_raw[hit], gt_masks_raw[hit]
                half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
                half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma
                center_ws, center_hs = center_of_mass(gt_masks)
                valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0
                masks_np = imrescale(gt_masks.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy(), scale=1.0 / 4)
                if masks_np.ndim == 2:
                    masks_np = masks_np[..., None]
                seg_masks = torch.from_numpy(masks_np).to(dtype=torch.uint8, device=device).permute(2, 0, 1)
                up_h, up_w = mask_feat_size[0] * 4, mask_feat_size[1] * 4
                for seg_mask, gt_label, half_h, half_w, center_h, center_w, ok in zip(
                        seg_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                    if not ok:
                        continue
                    coord_w = int((center_w / up_w) // (1.0 / num_grid))
                    coord_h = int((center_h / up_h) // (1.0 / num_grid))
                    top_box = max(0, int(((center_h - half_h) / up_h) // (1.0 / num_grid)))
                    down_box = min(num_grid - 1, int(((center_h + half_h) / up_h) // (1.0 / num_grid)))
                    left_box = max(0, int(((center_w - half_w) / up_w) // (1.0 / num_grid)))
                    right_box = min(num_grid - 1, int(((center_w + half_w) / up_w) // (1.0 / num_grid)))
                    top, down = max(top_box, coord_h - 1), min(down_box, coord_h + 1)
                    left, right = max(coord_w - 1, left_box), min(right_box, coord_w + 1)
                    if int(gt_label) == IGNORE:
                        region = cate_label[top:down + 1, left:right + 1]
                        region[region == self.num_classes] = IGNORE          # chỉ các ô đang là nền
                        continue
                    cate_label[top:down + 1, left:right + 1] = gt_label
                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            cur = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                            cur[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                            ins_label.append(cur)
                            ins_ind_label[i * num_grid + j] = True
                            grid_order.append(i * num_grid + j)
            ins_label_list.append(torch.stack(ins_label, 0) if ins_label
                                  else torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device))
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def loss(self, cate_preds, kernel_preds, ins_pred, targets):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = targets
        ins_labels = [torch.cat(list(level), 0) for level in zip(*ins_label_list)]
        kernel_preds = [[kp.view(kp.shape[0], -1)[:, go] for kp, go in zip(kp_level, go_level)]
                        for kp_level, go_level in zip(kernel_preds, zip(*grid_order_list))]
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.size()[-1] == 0:
                    continue
                cur = ins_pred[idx, ...]
                H, W = cur.shape[-2:]
                N, I = kernel_pred.shape
                b_mask_pred.append(F.conv2d(cur.unsqueeze(0), kernel_pred.permute(1, 0).view(I, -1, 1, 1), stride=1).view(-1, H, W))
            ins_pred_list.append(torch.cat(b_mask_pred, 0) if b_mask_pred else None)

        num_ins = torch.cat([torch.cat([x.flatten() for x in level]) for level in zip(*ins_ind_label_list)]).sum()
        dice = [dice_loss(torch.sigmoid(inp), tgt) for inp, tgt in zip(ins_pred_list, ins_labels) if inp is not None]
        loss_ins = (torch.cat(dice).mean() if dice else ins_pred.sum() * 0.0) * self.ins_loss_weight

        flat_labels = torch.cat([torch.cat([x.flatten() for x in level]) for level in zip(*cate_label_list)])
        flat_preds = torch.cat([cp.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for cp in cate_preds])
        loss_cate = self.focal_loss_weight * silk_cate_loss(flat_preds, flat_labels, self.num_classes,
                                                            self.focal_loss_alpha, self.focal_loss_gamma) / (num_ins + 1)
        return {"loss_ins": loss_ins, "loss_cate": loss_cate}
