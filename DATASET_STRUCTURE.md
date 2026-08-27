# Silkworm Segmentation Project & Dataset Structure

This project provides instance segmentation and lesion detection methods for silkworms, organized by method.

```text
Silkworm_Segmentation/
├── methods/
│   ├── deep_watershed/
│   │   ├── model.py                  # Attention U-Net model & DiceLoss
│   │   ├── boundary_pipeline.py      # Label prep, training, and single-image inference CLI
│   │   ├── instance_pipeline.py      # Modular 4-step instance segmentation engine
│   │   └── interactive_viewer.py     # Interactive GUI / terminal inference viewer
│   ├── box_guided_watershed/         # Box-Guided Watershed method
│   ├── vmunet/                       # VM-UNet model method
│   └── silkynet/                     # Silkynet U-Net method
├── utils/
│   └── yolo_bbox_to_masks.py         # Converter for YOLO bbox annotations to masks & boundaries
├── data/
│   ├── images/                       # RGB training images
│   ├── masks/                        # Binary object masks
│   ├── boundaries/                   # Generated boundary ground-truth maps
│   ├── checkpoints/                  # Saved model checkpoints
│   └── outputs/                      # Output visualization images
└── logs/                             # Training log files
```

## Deep Watershed Pipeline Flow

1. **Convert YOLO bboxes (if using YOLO raw annotations):**
```bash
python -m utils.yolo_bbox_to_masks --input-root "data/Silkworm Diseases.v1i.yolo26" --output-root "data/converted"
```

2. **Generate boundary labels from masks:**
```bash
python -m methods.deep_watershed.boundary_pipeline prepare-labels --mask-dir data/masks --output-dir data/boundaries
```

3. **Train the model:**
```bash
python -m methods.deep_watershed.boundary_pipeline train --image-dir data/images --boundary-dir data/boundaries --save-path data/checkpoints/boundary_model.pt
```

4. **Run single-image inference:**
```bash
python -m methods.deep_watershed.boundary_pipeline infer --image-path path/to/test_image.png --checkpoint data/checkpoints/boundary_model.pt
```

5. **Run interactive viewer (GUI / Terminal):**
```bash
python -m methods.deep_watershed.interactive_viewer --image-dir "data/Silkworm Diseases.v1i.yolo26/test/images" --checkpoint data/checkpoints/boundary_model.pt
```

## Notes

- Image and boundary filenames should match.
- Boundary images are generated as 1-channel binary contour maps (2px thickness).
- Inference results are saved to `data/outputs/`.

## Repo Cleanup Guidance

Keep these as source inputs or reusable assets:
- `data/raw/`
- `data/Silkworm Diseases.v1i.yolo26/`
- `data/ae/`
- `data/isolated/`

Do not commit these generated or temporary artifacts:
- `data/converted/`
- `data/checkpoints/`
- `data/outputs/`
- `__pycache__/`
- `*.pyc`
