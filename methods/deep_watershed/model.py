"""Attention U-Net architecture & loss functions for boundary prediction.

The network follows the standard encoder-decoder U-Net design with attention
gates applied to each skip connection before concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Down-sampling block: MaxPool2d followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate that filters encoder skip features using decoder context.

    Args:
        skip_channels: Number of channels in the encoder feature map x.
        gate_channels: Number of channels in the decoder gating signal g.
        inter_channels: Internal channel size used to compute compatibility.
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        inter_channels: int,
    ) -> None:
        super().__init__()
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return encoder features x filtered by the attention coefficients."""

        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)

        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=False)

        compatibility = self.relu(theta_x + phi_g)
        attention = self.sigmoid(self.psi(compatibility))
        return x * attention


class Up(nn.Module):
    """Up-sampling block: ConvTranspose2d, attention-gated skip, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            skip_channels=skip_channels,
            gate_channels=out_channels,
            inter_channels=max(out_channels // 2, 1),
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Attention U-Net for 3-channel RGB input and 1-channel boundary output."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        if self.apply_sigmoid:
            return self.activation(x)
        return x


class DiceLoss(nn.Module):
    """Soft Dice loss for binary masks / boundary learning."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()
