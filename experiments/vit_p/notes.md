# ViT-P Silkworm Segmentation Experiment Notes

## Configuration & Hyperparameters (Synchronized with `segfirst_swinunet`)
- **Model**: ViT-P with DINOv2-Small backbone (`dinov2_vits14`)
- **Input Size**: `224x224`
- **Batch Size**: `6`
- **Epochs**: `50`
- **Optimizer**: `AdamW` (LR: `1e-4`, Weight Decay: `0.01`, Betas: `(0.9, 0.999)`)
- **LR Scheduler**: `CosineAnnealingLR`
- **Loss Function**: `CrossEntropyLoss`
- **Mixed Precision**: `AMP Enabled (fp16)`
- **Device**: `cuda:1` (dedicated to avoid GPU 0 collision with Mask2Former)
- **Dataset**: `data/silkworm_mixed_dataset` (Train: 570, Valid: 110, Test: 169)
- **Points per sample**: 32 points (balanced foreground / background)
