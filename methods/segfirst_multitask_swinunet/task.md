# Segmentation-First Multi-Task Swin-Unet — Task & Deliverables List

- [x] `config.py` — `SegFirstSwinConfig` dataclass supporting Phase 1 & Phase 2 params, differential LRs (`lr_backbone=1e-5`, `lr_seg=1e-4`, `lr_cls=1e-3`), and loss weights ($\lambda_{\text{seg}}=1.0, \lambda_{\text{cls}}=0.1$).
- [x] `model.py` — `SegFirstSwinUnet` architecture (Swin-T backbone & UNet decoder, Classification Head on Bottleneck, differential parameter group builder).
- [x] `dataset.py` — `SilkynetSegDataset` (Dataset A: image, mask) + `YOLOClsDataset` (Dataset B: image, class_label).
- [x] `losses.py` — `SegFirstMultiTaskLoss` (DiceLoss + BCE + CrossEntropy) handling Dataset A, Dataset B, and joint loss calculation.
- [x] `train.py` — Two-Phase training pipeline:
  - Phase 1: Segmentation Pre-training on Dataset A.
  - Phase 2: Multi-Task Fine-Tuning with differential learning rates and low $\lambda_{\text{cls}}=0.1$.
- [x] `inference.py` — Single/batch image inference & 4-panel visualizer (Input, Mask Pred, Overlay Segment, Classification Badge).
- [x] `instance_inference.py` — Per-silkworm instance segmentation, contour extraction, cropping, and individual health classification.
- [x] `__init__.py` — Package exports.
