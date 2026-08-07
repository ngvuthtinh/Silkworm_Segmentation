import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# Disable OpenCV multi-threading inside DataLoader worker processes to prevent deadlocks
cv2.setNumThreads(0)

class SilkwormDataset(Dataset):
    """
    Dataset for Box-Guided 3-Class Watershed Segmentation.
    Loads YOLO format bounding boxes and dynamically generates:
    - 3-Class Pseudo Label: 0 (Background), 1 (Boundary), 2 (Core)
    - Spatial Gate B(x): 1 inside BBox, 0 outside.
    """
    def __init__(self, image_dir, label_dir, cache_dir=None, image_size=(256, 256), max_samples=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.cache_dir = cache_dir
        self.image_size = image_size
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        if max_samples is not None and max_samples > 0:
            self.image_files = self.image_files[:max_samples]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def _generate_pseudo_labels(self, bboxes, image):
        """
        Generates 3-class pseudo labels for full image supervision:
        - 0: Background (Everything outside BBoxes + leaf noise)
        - 1: Boundary (Organic outer ring of each silkworm via Otsu's threshold)
        - 2: Core (Organic inner body of each silkworm via Otsu's threshold + Erosion)
        """
        H, W = self.image_size[1], self.image_size[0]
        pseudo_label = np.zeros((H, W), dtype=np.longlong) # Default all to Class 0 (Background)
        gate_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Note: image is expected to be RGB numpy array [H, W, 3]
        for bbox in bboxes:
            _, x_center, y_center, width, height = bbox
            x1 = max(0, int((x_center - width / 2) * W))
            y1 = max(0, int((y_center - height / 2) * H))
            x2 = min(W, int((x_center + width / 2) * W))
            y2 = min(H, int((y_center + height / 2) * H))
            
            if y2 > y1 and x2 > x1:
                gate_mask[y1:y2, x1:x2] = 1
                
                # Crop RGB image
                crop_rgb = image[y1:y2, x1:x2]
                crop_h, crop_w = crop_rgb.shape[:2]
                
                # Apply GrabCut foreground extraction using GMM color models
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
                
                # Morphological Closing (7x7 Ellipse) to close small gaps & wrinkles
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                closed_mask = cv2.morphologyEx(organic_mask, cv2.MORPH_CLOSE, kernel_close)
                
                # Solid Contour Filling using findContours and drawContours(..., thickness=-1)
                # Guarantees 100% of internal holes/dark wrinkles are filled solid
                solid_mask = np.zeros_like(closed_mask)
                contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(solid_mask, contours, -1, 1, thickness=-1)
                else:
                    solid_mask = closed_mask
                
                # Dynamic kernel size based on bounding box size (e.g. 10% of min dimension)
                k_size = max(3, int(min(x2-x1, y2-y1) * 0.1))
                k_size = k_size if k_size % 2 == 1 else k_size + 1 # Ensure odd
                kernel = np.ones((k_size, k_size), np.uint8)
                
                # Morphological Erosion ONLY AFTER mask is completely solid to find Core
                core_mask = cv2.erode(solid_mask, kernel, iterations=1)
                
                # Boundary is the solid organic shape minus its core
                boundary_mask = solid_mask - core_mask
                
                # Assign to pseudo_label inside the bounding box
                box_pseudo = pseudo_label[y1:y2, x1:x2]
                box_pseudo[boundary_mask == 1] = 1
                box_pseudo[core_mask == 1] = 2
                pseudo_label[y1:y2, x1:x2] = box_pseudo
        
        return pseudo_label, gate_mask

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        
        pseudo_label = None
        gate_mask = None
        
        # 1. Try loading pre-generated pseudo mask from cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, base_name + '.npz')
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                pseudo_label = data['pseudo_label']
                gate_mask = data['gate_mask']
                
        # 2. Fallback to dynamic computation if cache not available
        if pseudo_label is None or gate_mask is None:
            label_path = os.path.join(self.label_dir, base_name + '.txt')
            bboxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            bboxes.append([float(p) for p in parts[:5]])
            pseudo_label, gate_mask = self._generate_pseudo_labels(bboxes, image)
            
        image_tensor = self.transform(image)
        pseudo_label_tensor = torch.from_numpy(pseudo_label).long()
        gate_tensor = torch.from_numpy(gate_mask).float() # [H, W] gate
        
        return image_tensor, pseudo_label_tensor, gate_tensor
