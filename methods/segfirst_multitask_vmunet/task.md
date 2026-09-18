# Segmentation-First Multi-Task VM-UNet — Task & Deliverables List

- [x] `config.py` — `SegFirstConfig` dataclass supporting Phase 1 & Phase 2 params, differential LRs, and loss weights ($\lambda_{\text{seg}}=1.0, \lambda_{\text{cls}}=0.1$).
- [x] `model.py` — `SegFirstVMUNet` architecture (VSSM backbone, Seg Head, Classification Head on Bottleneck via forward hook, differential parameter group builder).
- [x] `dataset.py` — `SilkynetSegDataset` (Dataset A: image, mask) + `YOLOClsDataset` (Dataset B: image, class_label).
- [x] `losses.py` — `SegFirstMultiTaskLoss` (DiceLoss + BCE + CrossEntropy) handling Dataset A, Dataset B, and joint loss calculation.
- [x] `train.py` — Two-Phase training pipeline:
  - Phase 1: Segmentation Pre-training on Dataset A.
  - Phase 2: Multi-Task Fine-Tuning with differential learning rates (`lr_backbone=1e-5`, `lr_seg=1e-4`, `lr_cls=1e-3`) and low $\lambda_{\text{cls}}=0.1$.
- [x] `inference.py` — Single/batch image inference & 4-panel visualizer (Input, Mask Pred, Overlay Segment, Classification Badge).
- [x] `__init__.py` — Package exports.
