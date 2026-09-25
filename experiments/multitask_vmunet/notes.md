# 📝 Joint Multi-Task VM-UNet Experiment Notes

## Architecture Overview
- **Backbone**: VM-UNet (Visual Mamba State Space)
- **Segmentation Head**: 1x1 Conv from decoder output
- **Classification Head**: Bottleneck feature extracted via forward hook on `backbone.layers[3]` -> GAP -> FC(256) -> FC(2)

## Training Strategy
- Joint training on Dataset A (Segmentation) and Dataset B (YOLO Classification) simultaneously.
- Loss: `lambda_seg * (BCE + Dice) + lambda_cls * CrossEntropy`

## Run Command
```bash
python -m experiments.multitask_vmunet.train
```

Results are saved to `runs/multitask_vmunet/<timestamp>/`.
