"""
experiments/solov2 package — SOLOv2 (AdelaiDet, detectron2) cho phân đoạn từng con + chẩn đoán từng con, ba biến thể backbone:
  r50       ResNet-50 + FPN (SOLOv2 gốc, mốc so sánh)
  mamba     cách A: encoder VSSM của VM-UNet thay ResNet, giữ FPN
  vmunet    cách B: cả VM-UNet (encoder + decoder có skip) làm backbone + neck, ra p2..p6
"""
