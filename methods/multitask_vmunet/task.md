# Multi-Task VM-UNet — Task List

- [x] Khám phá dataset thực tế (Silkynet 178 pairs + YOLO 7978 train)
- [x] `model.py` — MultiTaskVMUNet (VSSM backbone + seg head + cls head via hook)
- [x] `losses.py` — SegLoss (BCE+Dice), ClsLoss (CE), MultiTaskLoss (conditional)
- [x] `dataset.py` — SegDataset (Silkynet layout) + ClsDataset (YOLO binary mapping)
- [x] `config.py` — MultiTaskConfig dataclass với all hyperparams
- [x] `train.py` — Interleaved multi-task loop + AMP + TensorBoard + checkpoint
- [x] `inference.py` — Single/batch inference + visualization (3-4 panels)
- [x] `__init__.py` — Package exports
- [x] Dataset loading test PASSED (train=99 seg, 7978 cls | val=10 seg, 499 cls)
- [x] Model static test PASSED (27.63M params | backbone=27.43M | cls_head=197K)
- [x] Loss conditional logic PASSED (seg-only cls=0 ✓ | cls-only seg=0 ✓ | both ✓)
- [!] Forward pass: cần GPU (CPU timeout expected với VSSM/Mamba)

