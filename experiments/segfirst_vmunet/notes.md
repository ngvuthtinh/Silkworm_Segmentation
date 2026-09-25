# Experiment Notes — segfirst_vmunet

## Mô tả
Segmentation-First Multi-Task VM-UNet: train 2 phase, backbone là VMamba (VSSM).  
Phase 1 pre-train phân đoạn, Phase 2 fine-tune đa nhiệm với differential LR.

## Kết quả các lần chạy

| Run | Phase 1 Best Dice | Phase 2 Best Dice | Cls Acc | Ghi chú |
|-----|-------------------|-------------------|---------|---------|
| (chưa có) | - | - | - | - |

## Config đáng chú ý
- `lambda_cls = 0.1` → giữ ưu tiên segmentation trong Phase 2
- `lr_backbone = 1e-5` → bảo vệ backbone khỏi bị overwrite bởi cls loss
- Augmentation: horizontal flip, vertical flip, rotation (90/180/270°)

## TODO / Thử nghiệm tiếp theo
- [ ] Thử tăng `lambda_cls` lên 0.2 xem Cls Acc có tăng mà không mất Dice không
- [ ] Thử thêm ColorJitter augmentation vào Dataset A
- [ ] So sánh với segfirst_swinunet trên cùng tập test
