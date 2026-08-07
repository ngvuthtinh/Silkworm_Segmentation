import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_image(img_name, img_dir, label_dir, out_dir, img_size=(256, 256)):
    base_name = os.path.splitext(img_name)[0]
    out_file = os.path.join(out_dir, base_name + '.npz')
    
    if os.path.exists(out_file):
        return True
        
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, base_name + '.txt')
    
    image = cv2.imread(img_path)
    if image is None:
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    
    H, W = img_size[1], img_size[0]
    pseudo_label = np.zeros((H, W), dtype=np.uint8)
    gate_mask = np.zeros((H, W), dtype=np.uint8)
    
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bboxes.append([float(p) for p in parts[:5]])
                    
    for bbox in bboxes:
        _, x_center, y_center, width, height = bbox
        x1 = max(0, int((x_center - width / 2) * W))
        y1 = max(0, int((y_center - height / 2) * H))
        x2 = min(W, int((x_center + width / 2) * W))
        y2 = min(H, int((y_center + height / 2) * H))
        
        if y2 > y1 and x2 > x1:
            gate_mask[y1:y2, x1:x2] = 1
            crop_rgb = image[y1:y2, x1:x2]
            crop_h, crop_w = crop_rgb.shape[:2]
            
            if crop_h > 4 and crop_w > 4:
                rect = (1, 1, crop_w - 2, crop_h - 2)
                mask = np.zeros((crop_h, crop_w), np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                try:
                    cv2.grabCut(crop_rgb, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
                    organic_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
                except Exception:
                    organic_mask = np.ones((crop_h, crop_w), dtype=np.uint8)
            else:
                organic_mask = np.ones((crop_h, crop_w), dtype=np.uint8)
                
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed_mask = cv2.morphologyEx(organic_mask, cv2.MORPH_CLOSE, kernel_close)
            
            solid_mask = np.zeros_like(closed_mask)
            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(solid_mask, contours, -1, 1, thickness=-1)
            else:
                solid_mask = closed_mask
                
            k_size = max(3, int(min(x2-x1, y2-y1) * 0.1))
            k_size = k_size if k_size % 2 == 1 else k_size + 1
            kernel = np.ones((k_size, k_size), np.uint8)
            
            core_mask = cv2.erode(solid_mask, kernel, iterations=1)
            boundary_mask = solid_mask - core_mask
            
            box_pseudo = pseudo_label[y1:y2, x1:x2]
            box_pseudo[boundary_mask == 1] = 1
            box_pseudo[core_mask == 1] = 2
            pseudo_label[y1:y2, x1:x2] = box_pseudo

    np.savez_compressed(out_file, pseudo_label=pseudo_label, gate_mask=gate_mask)
    return True

def pregenerate_dataset(img_dir, label_dir, out_dir, num_workers=32):
    os.makedirs(out_dir, exist_ok=True)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Generating pseudo masks for {len(img_files)} images in {img_dir} using {num_workers} CPU workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_image, f, img_dir, label_dir, out_dir)
            for f in img_files
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {os.path.basename(img_dir)}"):
            pass

if __name__ == "__main__":
    base_dir = '/home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation'
    train_img_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/train/images')
    train_lbl_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/train/labels')
    train_out_dir = os.path.join(base_dir, 'data/cache_pseudo_masks/train')
    
    val_img_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/valid/images')
    val_lbl_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/valid/labels')
    val_out_dir = os.path.join(base_dir, 'data/cache_pseudo_masks/valid')
    
    pregenerate_dataset(train_img_dir, train_lbl_dir, train_out_dir, num_workers=32)
    pregenerate_dataset(val_img_dir, val_lbl_dir, val_out_dir, num_workers=32)
    print("✅ All pseudo masks pre-generated and cached successfully!")
