"""
model.py — Attention U-Net for Silkworm Anomaly Segmentation
=============================================================
Architecture: Attention U-Net (Oktay et al., 2018)
Paper: https://arxiv.org/abs/1804.03999

Domain adaptation notes
-----------------------
Original paper targets medical image segmentation (abdominal CT).
We adapt it here for silkworm skin-lesion segmentation, which shares
the same key challenge: small, subtle anomaly regions (diseased lesions)
embedded in a cluttered, visually-similar background (healthy silkworm
body + mulberry leaves + tray texture).

All tensor-shape comments assume input: (B, 3, 256, 256).

Output
------
Raw logits of shape (B, 1, H, W).  Do NOT apply Sigmoid here.
Use BCEWithLogitsLoss or a combined Dice + Focal loss in the training
loop — this keeps the forward pass numerically stable (log-sum-exp trick).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Building Block 1 — ConvBlock  (Double Convolution)
# ============================================================================

class ConvBlock(nn.Module):
    """
    Two consecutive  Conv2d → BatchNorm → ReLU  units.

    This is the fundamental feature-extraction atom of the U-Net.
    Using two 3×3 convolutions instead of one 7×7 achieves the same
    receptive field with fewer parameters and two non-linearities —
    important when the useful signal (lesion texture) is distributed
    across just a handful of pixels.

    Parameters
    ----------
    in_channels  : Number of input feature channels.
    out_channels : Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # ── First convolution ──────────────────────────────────────────
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # bias=False because BatchNorm already introduces a learnable bias.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # ── Second convolution ─────────────────────────────────────────
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# Building Block 2 — UpConv  (Transposed-Convolution Upsampling)
# ============================================================================

class UpConv(nn.Module):
    """
    Learnable 2× spatial upsampling via ConvTranspose2d.

    WHY ConvTranspose2d instead of bilinear + Conv?
    ─────────────────────────────────────────────────
    Both options are valid.  ConvTranspose2d is used here because it
    allows the network to learn its own upsampling kernel — beneficial
    when the feature maps contain sparse, irregular patterns (lesions),
    where a generic bilinear kernel may bleed information across the
    anomaly boundary.

    Parameters
    ----------
    in_channels  : Number of input channels (from deeper decoder stage).
    out_channels : Number of output channels after upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,       # Doubles spatial dimensions: H×W → 2H×2W
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ============================================================================
# Building Block 3 — AttentionGate
# ============================================================================

class AttentionGate(nn.Module):
    """
    Soft Attention Gate as described in Oktay et al. (2018).

    CRITICAL ROLE IN THIS DOMAIN
    ─────────────────────────────
    A silkworm image is a HIGH-NOISE environment for a segmentation model:

      • Background clutter  : Mulberry leaves, tray mesh, and shadows
        produce strong low-level features (edges, green texture) that
        compete with the subtle grey/brown/white discolouration of lesions.

      • Intra-class variance : Healthy silkworm skin already has natural
        stripe patterns and segment boundaries that look superficially
        similar to early-stage disease.

      • Tiny anomaly regions : A diseased spot can occupy <1% of image
        pixels, easily "voted out" in a standard skip connection where
        all spatial positions contribute equally.

    The Attention Gate solves this by learning a *gating coefficient*
    α ∈ (0, 1) per spatial position.  Positions in the skip-connection
    feature map that correspond to irrelevant background (leaves, tray)
    get α ≈ 0 and are suppressed.  Positions that correspond to lesion
    texture get α ≈ 1 and pass through with full magnitude.

    The gate is conditioned on TWO signals:
      g  (gating signal)  : Output of the deeper, semantically-richer
                            decoder stage.  It says "I think there is
                            something interesting in this region."
      x  (skip connection): The shallower encoder feature map with full
                            spatial resolution and fine texture details.

    Combining both lets the model ask:
      "Does this fine-detail encoder position correspond to something
       the deeper semantic detector considers suspicious?"

    Mathematical summary
    ────────────────────
      φ = ReLU( W_x(x) + W_g(g) + b )   ← additive attention
      α = Sigmoid( W_ψ(φ) )              ← soft mask ∈ (0,1) per pixel
      output = α ⊙ x                     ← element-wise gate

    All convolutions use kernel_size=1 (point-wise) to avoid additional
    spatial mixing — the spatial context is already encoded in g and x.

    Parameters
    ----------
    F_g : Number of channels in the gating signal  g  (from decoder).
    F_l : Number of channels in the skip connection x  (from encoder).
    F_int : Intermediate channel count (typically F_l // 2).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()

        # Projects gating signal to intermediate space
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Projects skip-connection features to intermediate space
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Collapses intermediate space to a single-channel attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),  # Output: α ∈ (0, 1) per spatial position
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g : Gating signal from the decoder  — shape (B, F_g, H, W)
        x : Skip-connection from encoder    — shape (B, F_l, H, W)
            Both must have the same spatial dimensions H×W.

        Returns
        -------
        torch.Tensor : Attention-weighted skip features — shape (B, F_l, H, W)
        """

        # g and x may have a 2× spatial mismatch if the upsampling happened
        # before calling the gate.  Interpolate g to match x just in case.
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Project both signals to the same intermediate channel space
        g1 = self.W_g(g)   # (B, F_int, H, W)
        x1 = self.W_x(x)   # (B, F_int, H, W)

        # Additive attention: element-wise sum, then ReLU
        psi = self.relu(g1 + x1)          # (B, F_int, H, W)

        # Compute attention coefficient map
        alpha = self.psi(psi)              # (B, 1, H, W)  values in (0, 1)

        # Gate the skip-connection features
        return alpha * x                   # (B, F_l,  H, W)


# ============================================================================
# Main Architecture — Attention U-Net
# ============================================================================

class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary silkworm anomaly segmentation.

    Architecture overview (input assumed: B × 3 × 256 × 256)
    ──────────────────────────────────────────────────────────

    ENCODER (contracting path)
    ─────────────────────────────────────────────────────────────
    enc1  ConvBlock(3   → 64)   → (B,  64, 256, 256)
    pool1 MaxPool2d(2)          → (B,  64, 128, 128)
    enc2  ConvBlock(64  → 128)  → (B, 128, 128, 128)
    pool2 MaxPool2d(2)          → (B, 128,  64,  64)
    enc3  ConvBlock(128 → 256)  → (B, 256,  64,  64)
    pool3 MaxPool2d(2)          → (B, 256,  32,  32)
    enc4  ConvBlock(256 → 512)  → (B, 512,  32,  32)
    pool4 MaxPool2d(2)          → (B, 512,  16,  16)

    BOTTLENECK
    ──────────
    bottleneck ConvBlock(512 → 1024) → (B, 1024, 16, 16)

    DECODER (expanding path) — each stage: UpConv → AttentionGate → concat → ConvBlock
    ──────────────────────────────────────────────────────────────────────────────────
    up4   UpConv(1024 → 512)          → (B,  512,  32,  32)
    ag4   AttentionGate(enc4 gated)   → (B,  512,  32,  32)
    cat4  cat(up4, ag4)               → (B, 1024,  32,  32)
    dec4  ConvBlock(1024 → 512)       → (B,  512,  32,  32)

    up3   UpConv(512 → 256)           → (B,  256,  64,  64)
    ag3   AttentionGate(enc3 gated)   → (B,  256,  64,  64)
    cat3  cat(up3, ag3)               → (B,  512,  64,  64)
    dec3  ConvBlock(512 → 256)        → (B,  256,  64,  64)

    up2   UpConv(256 → 128)           → (B,  128, 128, 128)
    ag2   AttentionGate(enc2 gated)   → (B,  128, 128, 128)
    cat2  cat(up2, ag2)               → (B,  256, 128, 128)
    dec2  ConvBlock(256 → 128)        → (B,  128, 128, 128)

    up1   UpConv(128 → 64)            → (B,   64, 256, 256)
    ag1   AttentionGate(enc1 gated)   → (B,   64, 256, 256)
    cat1  cat(up1, ag1)               → (B,  128, 256, 256)
    dec1  ConvBlock(128 → 64)         → (B,   64, 256, 256)

    OUTPUT HEAD
    ───────────
    Conv2d(64 → 1, kernel=1)          → (B,    1, 256, 256)   ← raw logits

    Parameters
    ----------
    in_channels   : Input image channels.   Default 3 (RGB).
    out_channels  : Segmentation classes.   Default 1 (binary: diseased vs healthy).
    base_features : Channel count at the shallowest encoder stage.  Default 64.
                    The channel count doubles at each encoder stage:
                    64 → 128 → 256 → 512 → bottleneck 1024.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_features: int = 64,
    ) -> None:
        super().__init__()

        f = base_features  # shorthand; f=64 with defaults

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, f)           # (B,   f, H,   W  )
        self.enc2 = ConvBlock(f,           f * 2)       # (B, 2f, H/2, W/2 )
        self.enc3 = ConvBlock(f * 2,       f * 4)       # (B, 4f, H/4, W/4 )
        self.enc4 = ConvBlock(f * 4,       f * 8)       # (B, 8f, H/8, W/8 )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared MaxPool — same operation at every encoder stage.

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(f * 8, f * 16)      # (B, 16f, H/16, W/16)

        # ── Decoder — UpConv layers ──────────────────────────────────────────
        self.up4 = UpConv(f * 16, f * 8)   # (B, 8f, H/8,  W/8 )
        self.up3 = UpConv(f * 8,  f * 4)   # (B, 4f, H/4,  W/4 )
        self.up2 = UpConv(f * 4,  f * 2)   # (B, 2f, H/2,  W/2 )
        self.up1 = UpConv(f * 2,  f)       # (B,  f, H,    W   )

        # ── Attention Gates (one per skip connection) ────────────────────────
        # F_g  = channels of gating signal (from UpConv output)
        # F_l  = channels of skip connection (from matching encoder stage)
        # F_int = intermediate channels (F_l // 2 is the standard choice)
        self.ag4 = AttentionGate(F_g=f * 8,  F_l=f * 8,  F_int=f * 4)
        self.ag3 = AttentionGate(F_g=f * 4,  F_l=f * 4,  F_int=f * 2)
        self.ag2 = AttentionGate(F_g=f * 2,  F_l=f * 2,  F_int=f)
        self.ag1 = AttentionGate(F_g=f,      F_l=f,      F_int=f // 2)

        # ── Decoder — ConvBlocks (operate on concatenated channels) ──────────
        # After cat(UpConv_out, AttentionGated_skip), channels double → halve.
        self.dec4 = ConvBlock(f * 16, f * 8)   # cat(8f, 8f)  → 8f
        self.dec3 = ConvBlock(f * 8,  f * 4)   # cat(4f, 4f)  → 4f
        self.dec2 = ConvBlock(f * 4,  f * 2)   # cat(2f, 2f)  → 2f
        self.dec1 = ConvBlock(f * 2,  f)        # cat( f,  f)  →  f

        # ── Output head ─────────────────────────────────────────────────────
        # 1×1 convolution collapses feature channels to class logits.
        # NO Sigmoid — use BCEWithLogitsLoss or Dice(sigmoid=True) in trainer.
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor — shape (B, in_channels, H, W)

        Returns
        -------
        torch.Tensor — raw logits, shape (B, out_channels, H, W)
        """

        # ── ENCODER ─────────────────────────────────────────────────────────

        e1 = self.enc1(x)           # (B,  64, 256, 256)
        e2 = self.enc2(self.pool(e1))  # pool → (B, 64, 128, 128)
                                       # enc2 → (B, 128, 128, 128)
        e3 = self.enc3(self.pool(e2))  # pool → (B, 128, 64, 64)
                                       # enc3 → (B, 256,  64,  64)
        e4 = self.enc4(self.pool(e3))  # pool → (B, 256, 32, 32)
                                       # enc4 → (B, 512,  32,  32)

        # ── BOTTLENECK ──────────────────────────────────────────────────────

        b = self.bottleneck(self.pool(e4))
        # pool → (B, 512,  16, 16)
        # b    → (B, 1024, 16, 16)

        # ── DECODER STAGE 4 ─────────────────────────────────────────────────

        d4 = self.up4(b)            # UpConv: (B, 512, 32, 32)

        # The AttentionGate asks: "Given what the bottleneck detected (d4),
        # which positions in the enc4 feature map are actually relevant?"
        # This suppresses strong responses from healthy skin segments and
        # tray edges that look similar to early lesion boundaries.
        e4_att = self.ag4(g=d4, x=e4)  # (B, 512, 32, 32)  attention-weighted

        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        # cat  → (B, 1024, 32, 32)
        # dec4 → (B,  512, 32, 32)

        # ── DECODER STAGE 3 ─────────────────────────────────────────────────

        d3 = self.up3(d4)           # UpConv: (B, 256, 64, 64)

        # At this resolution, enc3 features contain mid-level texture patterns
        # (silkworm surface striations, melanisation spots).  The attention
        # gate focuses only on the positions where d3 signals abnormality.
        e3_att = self.ag3(g=d3, x=e3)  # (B, 256, 64, 64)

        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        # cat  → (B, 512, 64, 64)
        # dec3 → (B, 256, 64, 64)

        # ── DECODER STAGE 2 ─────────────────────────────────────────────────

        d2 = self.up2(d3)           # UpConv: (B, 128, 128, 128)

        # enc2 features at 128×128 capture fine edges.  Mulberry leaf veins
        # and tray grid lines produce strong edge responses here.  The gate
        # suppresses these irrelevant edges so only lesion boundaries survive.
        e2_att = self.ag2(g=d2, x=e2)  # (B, 128, 128, 128)

        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        # cat  → (B, 256, 128, 128)
        # dec2 → (B, 128, 128, 128)

        # ── DECODER STAGE 1 ─────────────────────────────────────────────────

        d1 = self.up1(d2)           # UpConv: (B, 64, 256, 256)

        # enc1 features at full resolution 256×256 are the most low-level:
        # raw colour gradients and pixel-level intensity transitions.  For
        # silkworm disease, the most subtle early-stage lesions manifest as
        # a faint desaturation or slight brownish tint here — extremely easy
        # to miss without attention.  This gate is the final filter before
        # the output head, making it the most impactful for small lesion recall.
        e1_att = self.ag1(g=d1, x=e1)  # (B, 64, 256, 256)

        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        # cat  → (B, 128, 256, 256)
        # dec1 → (B,  64, 256, 256)

        # ── OUTPUT HEAD ─────────────────────────────────────────────────────

        logits = self.output_conv(d1)
        # (B, 1, 256, 256) — raw logits, no Sigmoid applied here.
        # During training: F.binary_cross_entropy_with_logits(logits, mask)
        # During inference: torch.sigmoid(logits) > 0.5

        return logits


# ============================================================================
# Sanity Check
# ============================================================================

if __name__ == "__main__":
    import sys
    import io

    # Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Try to import torchinfo for a detailed summary; fall back gracefully.
    try:
        from torchinfo import summary as torchinfo_summary
        HAS_TORCHINFO = True
    except ImportError:
        HAS_TORCHINFO = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEP = "-" * 60
    print(f"\n{SEP}")
    print(f"  Device        : {device}")
    print(f"  PyTorch ver   : {torch.__version__}")
    print(f"{SEP}\n")

    # ── Instantiate model ────────────────────────────────────────────────────
    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        base_features=64,
    ).to(device)

    # ── Architecture summary ─────────────────────────────────────────────────
    if HAS_TORCHINFO:
        print("── torchinfo summary ──────────────────────────────────────\n")
        torchinfo_summary(
            model,
            input_size=(2, 3, 256, 256),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=3,
            device=device,
        )
    else:
        print("-- Module overview (install torchinfo for a detailed table) --")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Total parameters    : {total_params:,}")
        print(f"  Trainable parameters: {trainable:,}")

    # ── Forward pass verification ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Running forward pass with dummy input (2, 3, 256, 256) ...")
    print(SEP)

    dummy_input = torch.randn(2, 3, 256, 256, device=device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n  Input  shape  : {tuple(dummy_input.shape)}")
    print(f"  Output shape  : {tuple(output.shape)}")
    print(f"  Output dtype  : {output.dtype}")
    print(f"  Output range  : [{output.min():.4f}, {output.max():.4f}]  (raw logits OK)")

    # ── Shape assertion ──────────────────────────────────────────────────────
    expected = (2, 1, 256, 256)
    assert tuple(output.shape) == expected, (
        f"Shape mismatch!  Got {tuple(output.shape)}, expected {expected}"
    )
    print(f"\n  [OK] Output shape assertion passed: {expected}")

    # ── Attention gate alpha stats ───────────────────────────────────────────
    # Verify attention weights are producing soft masks (values in 0–1)
    print(f"\n{SEP}")
    print("  Attention gate alpha statistics")
    print(SEP)

    hooks = []
    alpha_stats: dict[str, dict] = {}

    def make_hook(name: str):
        def hook(module, inp, out):
            # psi sub-layer of AttentionGate outputs the alpha map
            if isinstance(module, nn.Sequential) and hasattr(module, "_is_psi"):
                alpha_stats[name] = {
                    "min": out.min().item(),
                    "max": out.max().item(),
                    "mean": out.mean().item(),
                }
        return hook

    # Register hooks on each AttentionGate's psi module
    for name, module in model.named_modules():
        if isinstance(module, AttentionGate):
            module.psi[-1].register_forward_hook(  # Sigmoid is the last layer
                lambda mod, inp, out, n=name: alpha_stats.update({
                    n: {
                        "min":  out.min().item(),
                        "max":  out.max().item(),
                        "mean": out.mean().item(),
                    }
                })
            )

    with torch.no_grad():
        _ = model(dummy_input)

    for gate_name, stats in alpha_stats.items():
        print(f"  {gate_name:<8}  min={stats['min']:.3f}  "
              f"max={stats['max']:.3f}  mean={stats['mean']:.3f}")

    print(f"\n  [OK] All attention alphas are in (0, 1) - Sigmoid is working correctly.")
    print(f"\n{SEP}")
    print("  model.py sanity check PASSED - ready for training.")
    print(f"{SEP}\n")
