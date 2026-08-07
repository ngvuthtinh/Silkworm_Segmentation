import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    Additive Soft Attention Gate.
    Filters skip connections by gating spatial regions.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Additive attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Multiply skip connection by attention coefficients
        return x * psi

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class AttentionUNet3Class(nn.Module):
    """
    3-Class Attention U-Net.
    Output: 3 channels (Background, Boundary, Core)
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(AttentionUNet3Class, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        self.Conv1 = ConvBlock(in_channels, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)
        
        # Decoder
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = ConvBlock(1024, 512)
        
        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = ConvBlock(512, 256)
        
        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ConvBlock(256, 128)
        
        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ConvBlock(128, 64)
        
        # Final Output Layer for 3 classes
        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)
        
        # Decoder
        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        # Output Logits
        out = self.Conv_1x1(d2)
        
        return out
