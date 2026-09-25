# 📝 SegFirst Multi-Task Swin-Unet Experiment Notes

## Architecture Overview
- **Backbone**: SwinTransformerSys (Shifted Windows Vision Transformer)
- **Input Size**: 224x224 (native patch size 4, window size 7)
- **Segmentation Head**: Patch Expanding Decoder
- **Classification Head**: Bottleneck feature (768-d) -> Global Average Pooling -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(2)

## Training Strategy
- **Phase 1**: Pure segmentation pre-training on `data/silkworm_mixed_dataset`.
- **Phase 2**: Multi-task fine-tuning with differential learning rates:
  - Backbone: `1e-5`
  - Seg Decoder: `1e-4`
  - Cls Head: `1e-3`
  - Loss balancing: `lambda_seg = 1.0`, `lambda_cls = 0.1`

## Run Command
```bash
python -m experiments.segfirst_swinunet.train
```

Results are saved to `runs/segfirst_swinunet/<timestamp>/`.
