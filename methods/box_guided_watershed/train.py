import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.box_guided_watershed.dataset import SilkwormDataset
from methods.box_guided_watershed.model import AttentionUNet3Class
from methods.box_guided_watershed.losses import CombinedGatedLoss

def train(epochs=10, batch_size=16, lr=1e-4, max_samples=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Using device: {device}")
    
    # Paths (adjust as needed based on actual workspace structure)
    base_dir = '/home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation'
    train_img_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/train/images')
    train_lbl_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/train/labels')
    
    val_img_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/valid/images')
    val_lbl_dir = os.path.join(base_dir, 'data/Silkworm Diseases.v1i.yolo26/valid/labels')
    
    train_cache_dir = os.path.join(base_dir, 'data/cache_pseudo_masks/train')
    val_cache_dir = os.path.join(base_dir, 'data/cache_pseudo_masks/valid')
    
    chkpt_dir = os.path.join(base_dir, 'data/checkpoints/box_guided_watershed')
    os.makedirs(chkpt_dir, exist_ok=True)
    
    # Dataset and Dataloader (Full Dataset with Cached GrabCut Masks)
    train_dataset = SilkwormDataset(train_img_dir, train_lbl_dir, cache_dir=train_cache_dir, image_size=(256, 256), max_samples=max_samples)
    val_dataset = SilkwormDataset(val_img_dir, val_lbl_dir, cache_dir=val_cache_dir, image_size=(256, 256), max_samples=None)
    
    # num_workers=2 with cv2.setNumThreads(0) prevents multiprocessing queue deadlocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = AttentionUNet3Class(in_channels=3, out_channels=3).to(device)
    
    # Loss and Optimizer
    criterion = CombinedGatedLoss(weights=[1.0, 5.0, 2.0], lambda_dice=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    print("\n" + "="*50)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN (TRAINING STARTED)")
    print(f"Tổng số Epochs: {epochs}")
    print(f"Dữ liệu Train: {len(train_dataset)} ảnh")
    print(f"Dữ liệu Val: {len(val_dataset)} ảnh")
    print("Tiến trình sẽ được hiển thị qua thanh % ở bên dưới...")
    print("="*50 + "\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets, gates in pbar:
            images = images.to(device)
            targets = targets.to(device)
            gates = gates.to(device)
            
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, targets, gates)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, targets, gates in pbar_val:
                images = images.to(device)
                targets = targets.to(device)
                gates = gates.to(device)
                
                logits = model(images)
                loss = criterion(logits, targets, gates)
                
                val_loss += loss.item() * images.size(0)
                pbar_val.set_postfix({'val_loss': loss.item()})
                
        val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(chkpt_dir, 'best_model.pth'))
            print("=> Saved best model")

if __name__ == "__main__":
    train(epochs=15, batch_size=16)
