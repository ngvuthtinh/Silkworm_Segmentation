#!/bin/bash
# ==============================================================================
# Script huấn luyện SEGMENTATION-FIRST MULTI-TASK SWIN-UNET (2 Pha)
# Đồng bộ 100% phương pháp luận với SegFirst Multi-Task VM-UNet:
#   - Phase 1: Pre-training Segmentation (25 epochs)
#   - Phase 2: Multi-Task Fine-Tuning Differential LR (25 epochs)
# ==============================================================================

DATA_DIR="${data_dir:-../data/silkworm_mixed_dataset}"
OUT_DIR="${out_dir:-./segfirst_swin_out}"
CFG="${cfg:-configs/swin_tiny_patch4_window7_224_lite.yaml}"
BATCH_SIZE="${batch_size:-6}"      # Mặc định 6 (~1.4GB VRAM) hoặc 12 khi GPU trống
PHASE1_EPOCHS="${p1:-25}"
PHASE2_EPOCHS="${p2:-25}"
GPU="${gpu:-0}"                    # Chọn GPU 0 hoặc 1

echo "=============================================================================="
echo " 🌟 KHỞI ĐỘNG SEGFIRST MULTI-TASK SWIN-UNET (QUY TRÌNH 2 PHA ĐỒNG BỘ VM-UNET)"
echo "=============================================================================="
echo " Dữ liệu          : $DATA_DIR"
echo " Thư mục Checkpoint: $OUT_DIR"
echo " Batch Size       : $BATCH_SIZE"
echo " Phase 1 Epochs   : $PHASE1_EPOCHS"
echo " Phase 2 Epochs   : $PHASE2_EPOCHS"
echo " GPU              : $GPU"
echo "=============================================================================="

CUDA_VISIBLE_DEVICES=$GPU /home/subnh5/miniconda3/envs/swin_unet/bin/python train_segfirst_swin_unet.py \
    --data_dir "$DATA_DIR" \
    --cfg "$CFG" \
    --output_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --phase1_epochs "$PHASE1_EPOCHS" \
    --phase2_epochs "$PHASE2_EPOCHS" \
    --lr_backbone 1e-5 \
    --lr_seg 1e-4 \
    --lr_cls 1e-3 \
    --lambda_seg 1.0 \
    --lambda_cls 0.1 \
    --gpu "$GPU"
