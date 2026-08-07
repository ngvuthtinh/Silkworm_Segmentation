import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from methods.box_guided_watershed.model import AttentionUNet3Class
from methods.box_guided_watershed.postprocessing import constrained_watershed

def load_yolo_bboxes(label_path, img_w, img_h):
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    x_c, y_c, w, h = [float(p) for p in parts[1:5]]
                    bboxes.append((cls_id, x_c, y_c, w, h))
    return bboxes

def create_gate_mask(bboxes, img_size=(256, 256)):
    W, H = img_size
    if len(bboxes) == 0:
        # If no bboxes are provided, set gate mask to FULL IMAGE (1s everywhere)
        return np.ones((H, W), dtype=np.uint8)
        
    gate_mask = np.zeros((H, W), dtype=np.uint8)
    for cls_id, x_c, y_c, w, h in bboxes:
        x1 = max(0, int((x_c - w / 2) * W))
        y1 = max(0, int((y_c - h / 2) * H))
        x2 = min(W, int((x_c + w / 2) * W))
        y2 = min(H, int((y_c + h / 2) * H))
        if x2 > x1 and y2 > y1:
            gate_mask[y1:y2, x1:x2] = 1
    return gate_mask

def run_inference(image_path, label_path, checkpoint_path, output_save_path="inference_result.png"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}...")
    
    # 1. Load Image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = orig_img.shape
    
    img_resized = cv2.resize(orig_img, (256, 256))
    
    # 2. Load BBoxes & Create Gate Mask
    bboxes = load_yolo_bboxes(label_path, w_orig, h_orig)
    gate_mask = create_gate_mask(bboxes, (256, 256))
    
    # 3. Transform Image for Model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # 4. Load Model
    model = AttentionUNet3Class(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 5. Predict Probabilities
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy() # [3, 256, 256]
        
    # 6. Constrained Watershed Post-Processing
    instance_mask = constrained_watershed(img_resized, probs, gate_mask, tau_core=0.4, tau_bound=0.4)
    
    # 7. Visualization
    plt.figure(figsize=(18, 6))
    
    # Subplot 1: Original Image + BBoxes
    plt.subplot(1, 4, 1)
    img_bbox = img_resized.copy()
    class_names = {0: 'Grasserie', 1: 'Healthy'}
    for cls_id, x_c, y_c, w, h in bboxes:
        x1 = max(0, int((x_c - w / 2) * 256))
        y1 = max(0, int((y_c - h / 2) * 256))
        x2 = min(256, int((x_c + w / 2) * 256))
        y2 = min(256, int((y_c + h / 2) * 256))
        color = (255, 0, 0) if cls_id == 0 else (0, 255, 0) # Red for Grasserie, Green for Healthy
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bbox, class_names.get(cls_id, str(cls_id)), (x1, max(y1-5, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    plt.imshow(img_bbox)
    plt.title("1. Input Image + YOLO BBoxes")
    plt.axis('off')
    
    # Subplot 2: Spatial Gate B(x)
    plt.subplot(1, 4, 2)
    plt.imshow(gate_mask, cmap='gray', vmin=0, vmax=1)
    plt.title("2. Spatial Gate Mask B(x)")
    plt.axis('off')
    
    # Subplot 3: Model Predicted Core Probabilities
    plt.subplot(1, 4, 3)
    plt.imshow(probs[2], cmap='jet') # Channel 2 is Core
    plt.title("3. Predicted P(Core)")
    plt.axis('off')
    
    # Subplot 4: Final Instance Watershed Mask
    plt.subplot(1, 4, 4)
    # Colorize instances
    masked_instances = np.ma.masked_where(instance_mask == 0, instance_mask)
    plt.imshow(img_resized)
    plt.imshow(masked_instances, cmap='tab20', alpha=0.7)
    num_instances = len(np.unique(instance_mask)) - 1
    plt.title(f"4. Constrained Watershed ({num_instances} Silkworms)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Inference finished! Visualization saved to: {output_save_path}")
    print(f"Total instances detected: {num_instances}")

if __name__ == "__main__":
    import random
    
    base_dir = '/home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation'
    test_img_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/valid/images')
    test_lbl_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/valid/labels')
    checkpoint_path = os.path.join(base_dir, 'data/checkpoints/box_guided_watershed/best_model.pth')
    
    # Pick a random valid image
    img_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))])
    if len(img_files) > 0:
        sample_img = random.choice(img_files)
        sample_lbl = os.path.splitext(sample_img)[0] + '.txt'
        
        print(f"🎲 Running inference on valid image with BBoxes: {sample_img}")
        
        run_inference(
            image_path=os.path.join(test_img_dir, sample_img),
            label_path=os.path.join(test_lbl_dir, sample_lbl),
            checkpoint_path=checkpoint_path,
            output_save_path=os.path.join(base_dir, "inference_result.png")
        )
