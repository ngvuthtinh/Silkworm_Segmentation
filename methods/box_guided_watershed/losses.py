import torch
import torch.nn as nn
import torch.nn.functional as F

class FullImageWeightedCELoss(nn.Module):
    """
    Computes Cross-Entropy weighted by class weights over the full image.
    Class 0 (Background) is assigned to all non-silkworm pixels (leaves, tray, background).
    This trains the network to recognize and suppress background noise autonomously.
    """
    def __init__(self, weights=[1.0, 5.0, 2.0]):
        super(FullImageWeightedCELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
    def forward(self, logits, targets, gate_mask=None):
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
            
        # Standard Cross Entropy Loss with weights across full image
        return F.cross_entropy(logits, targets, weight=self.weights)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets, gate_mask=None):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot encoding [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=3).permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union for each class
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        # Calculate dice coefficient per class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - mean dice across classes and batch
        return 1 - dice.mean()

class CombinedGatedLoss(nn.Module):
    def __init__(self, weights=[1.0, 5.0, 2.0], lambda_dice=1.0):
        super(CombinedGatedLoss, self).__init__()
        self.ce = FullImageWeightedCELoss(weights=weights)
        self.dice = DiceLoss()
        self.lambda_dice = lambda_dice
        
    def forward(self, logits, targets, gate_mask=None):
        loss_ce = self.ce(logits, targets, gate_mask)
        loss_dice = self.dice(logits, targets, gate_mask)
        
        return loss_ce + self.lambda_dice * loss_dice
