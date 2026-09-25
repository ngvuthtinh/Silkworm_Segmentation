#!/bin/bash
# ==============================================================================
# Script huấn luyện MULTI-TASK Swin-Unet trên Silkworm Mixed Dataset
# Nhiệm vụ:
#   1. Phân đoạn hình dáng con tằm (Segmentation)
#   2. Chẩn đoán bệnh tật (Classification: Healthy vs Grasserie)
# ==============================================================================

# Tham số mặc định
DATA_DIR="${data_dir:-../data/silkworm_mixed_dataset}"
OUT_DIR="${out_dir:-./multitask_silkworm_out}"
CFG="${cfg:-configs/swin_tiny_patch4_window7_224_lite.yaml}"
BATCH_SIZE="${batch_size:-6}"   # Mặc định 6 (~1.4GB VRAM GPU). Có thể tăng lên 12/24 khi GPU trống
EPOCHS="${epochs:-100}"
LR="${lr:-0.01}"
LAMBDA_SEG="${lambda_seg:-1.0}"
LAMBDA_CLS="${lambda_cls:-0.1}" # Trọng số phân loại tương tự SegFirst VM-UNet
GPU="${gpu:-1}"                 # Sử dụng GPU 1 (đang còn ~3.2GB trống)

echo "=============================================================================="
echo " 🐛 KHỞI ĐỘNG HUẤN LUYỆN MULTI-TASK SWIN-UNET (SEGMENTATION + CLASSIFICATION)"
echo "=============================================================================="
echo " Dữ liệu          : $DATA_DIR"
echo " Thư mục kết quả  : $OUT_DIR"
echo " Batch Size       : $BATCH_SIZE"
echo " Số Epochs        : $EPOCHS"
echo " Learning Rate    : $LR"
echo " Trọng số Loss    : Seg=$LAMBDA_SEG, Cls=$LAMBDA_CLS"
echo " GPU Chỉ định     : $GPU"
echo "=============================================================================="

CUDA_VISIBLE_DEVICES=$GPU /home/subnh5/miniconda3/envs/swin_unet/bin/python train_multitask_silkworm.py \
    --data_dir "$DATA_DIR" \
    --cfg "$CFG" \
    --output_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --img_size 224 \
    --max_epochs "$EPOCHS" \
    --base_lr "$LR" \
    --lambda_seg "$LAMBDA_SEG" \
    --lambda_cls "$LAMBDA_CLS" \
    --gpu "$GPU"
