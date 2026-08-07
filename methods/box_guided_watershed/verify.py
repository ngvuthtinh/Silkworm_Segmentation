import torch
import numpy as np
from methods.box_guided_watershed.model import AttentionUNet3Class
from methods.box_guided_watershed.losses import CombinedGatedLoss
from methods.box_guided_watershed.postprocessing import constrained_watershed

def verify():
    print("Verifying Box-Guided Watershed Components...")
    
    # 1. Model test
    model = AttentionUNet3Class(in_channels=3, out_channels=3)
    dummy_input = torch.randn(2, 3, 256, 256)
    try:
        logits = model(dummy_input)
        print(f"Model output shape: {logits.shape} (Expected: [2, 3, 256, 256])")
        assert logits.shape == (2, 3, 256, 256)
    except Exception as e:
        print(f"Model error: {e}")
        
    # 2. Loss test
    criterion = CombinedGatedLoss()
    # Dummy targets [0, 1, 2]
    dummy_targets = torch.randint(0, 3, (2, 256, 256))
    # Dummy gates [0, 1]
    dummy_gates = torch.randint(0, 2, (2, 256, 256)).float()
    
    try:
        loss = criterion(logits, dummy_targets, dummy_gates)
        print(f"Combined Gated Loss: {loss.item():.4f}")
        # Test backward
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Loss error: {e}")
        
    # 3. Postprocessing test
    # image [256, 256, 3]
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # prob_maps [3, 256, 256]
    dummy_probs = torch.softmax(logits[0], dim=0).detach().numpy()
    # gate_mask [256, 256]
    dummy_gate_np = dummy_gates[0].numpy()
    
    try:
        instance_mask = constrained_watershed(dummy_image, dummy_probs, dummy_gate_np)
        print(f"Instance mask shape: {instance_mask.shape} (Expected: (256, 256))")
        print(f"Unique instances found: {len(np.unique(instance_mask)) - 1}") # subtract background
    except Exception as e:
        print(f"Postprocessing error: {e}")

if __name__ == "__main__":
    verify()
