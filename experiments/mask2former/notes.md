# Mask2Former Silkworm Segmentation Experiment Notes

## Overview
- **Model**: Mask2Former (ResNet-50 + MSDeformAttn Pixel Decoder + Masked-attention Transformer Decoder)
- **Dataset**: `data/silkworm_mixed_dataset` (Train: 570 valid pairs, Val: 110 pairs, Test: 169 pairs)
- **Hyperparameters (Synchronized with Swin-Unet)**:
  - Input resolution: 224 x 224
  - Batch size: 6
  - Optimizer: AdamW
  - Base Learning Rate: 1e-4
  - Weight Decay: 0.01
  - Max Iterations: 5000 (~50 epochs)
  - AMP: Enabled
  - Loss: Cross-Entropy + Binary Dice / Mask Loss
  - Device: 2x NVIDIA A100-SXM4-80GB

## Running Training
```bash
conda activate mask2former
python experiments/mask2former/train.py
```

## Running Evaluation
```bash
conda activate mask2former
python experiments/mask2former/evaluate.py runs/mask2former/model_final.pth
```
