import os
import glob
import numpy as np
from PIL import Image
import random

def prepare_silkworm_dataset():
    src_img_dir = "methods/silkynet/data/output20221127/JPEGImages"
    src_mask_dir = "methods/silkynet/data/output20221127/SegmentationClassPNG"

    output_dir = "data/silkworm"
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_mask_dir = os.path.join(output_dir, "train", "masks")
    val_img_dir = os.path.join(output_dir, "val", "images")
    val_mask_dir = os.path.join(output_dir, "val", "masks")

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(src_mask_dir, "*.png")))
    print(f"Found {len(mask_files)} mask files.")

    paired_files = []
    for mask_path in mask_files:
        basename = os.path.basename(mask_path).replace(".png", ".jpg")
        img_path = os.path.join(src_img_dir, basename)
        if os.path.exists(img_path):
            paired_files.append((img_path, mask_path))

    print(f"Paired {len(paired_files)} images and masks.")

    random.seed(42)
    random.shuffle(paired_files)

    num_val = max(1, int(len(paired_files) * 0.2))
    val_pairs = paired_files[:num_val]
    train_pairs = paired_files[num_val:]

    def process_and_save(pairs, img_dest, mask_dest):
        for idx, (img_path, mask_path) in enumerate(pairs):
            name = f"{idx:04d}.png"
            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(img_dest, name))

            mask_arr = np.array(Image.open(mask_path).convert("L"))
            # Ensure mask values are 0 and 255 for standard binary mask format
            mask_arr = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
            mask_img = Image.fromarray(mask_arr)
            mask_img.save(os.path.join(mask_dest, name))

    process_and_save(train_pairs, train_img_dir, train_mask_dir)
    process_and_save(val_pairs, val_img_dir, val_mask_dir)

    print(f"Saved {len(train_pairs)} training samples to {train_img_dir}")
    print(f"Saved {len(val_pairs)} validation samples to {val_img_dir}")

if __name__ == "__main__":
    prepare_silkworm_dataset()
