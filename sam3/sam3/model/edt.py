# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Triton kernel for euclidean distance transform (EDT)"""

import torch
import triton
import triton.language as tl

"""
Disclaimer: This implementation is not meant to be extremely efficient. A CUDA kernel would likely be more efficient.
"""


@triton.jit
def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
    """
    Parallelize over the dimension not being computed.
    Each program computes a full 1D EDT for a row (if horizontal) or column (if vertical).
    """
    row_idx = triton.program_id(0)

    # Offsets for the current row or column
    if horizontal:
        row_offset = row_idx * width
        stride = 1
        n = width
    else:
        row_offset = row_idx
        stride = width
        n = height

    # 1. Initialization
    # k: index of the rightmost parabola in the lower envelope
    k = 0
    v[row_offset + 0] = 0
    z[row_offset + 0] = -1e10
    z[row_offset + 1] = 1e10

    # 2. Compute lower envelope
    for q in range(1, n):
        # Calculate intersection point s of parabolas at q and v[k]
        # s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
        f_q = tl.load(inputs_ptr + row_offset + q * stride)
        v_k = v[row_offset + k]
        f_vk = tl.load(inputs_ptr + row_offset + v_k * stride)

        s = ((f_q + q * q) - (f_vk + v_k * v_k)) / (2 * q - 2 * v_k)

        while s <= z[row_offset + k]:
            k -= 1
            v_k = v[row_offset + k]
            f_vk = tl.load(inputs_ptr + row_offset + v_k * stride)
            s = ((f_q + q * q) - (f_vk + v_k * v_k)) / (2 * q - 2 * v_k)

        k += 1
        v[row_offset + k] = q
        z[row_offset + k] = s
        z[row_offset + k + 1] = 1e10

    # 3. Fill output distances
    k = 0
    for q in range(n):
        while z[row_offset + k + 1] < q:
            k += 1
        v_k = v[row_offset + k]
        f_vk = tl.load(inputs_ptr + row_offset + v_k * stride)
        dist_sq = (q - v_k) * (q - v_k) + f_vk
        tl.store(outputs_ptr + row_offset + q * stride, dist_sq)


def edt_triton(f):
    """
    2D Euclidean Distance Transform using Triton.
    f: 2D torch.Tensor (B, H, W) or (H, W) with initial squared distances (e.g., 0 for points, inf for background).
    """
    if f.dim() == 2:
        f = f.unsqueeze(0)
    B, H, W = f.shape
    device = f.device

    # Output and intermediate tensors
    output = torch.empty_like(f)
    v = torch.empty((B, H, max(H, W)), dtype=torch.int32, device=device)
    z = torch.empty((B, H, max(H, W) + 1), dtype=f.dtype, device=device)

    # Horizontal pass (1D EDT on rows)
    grid_h = (B * H,)
    edt_kernel[grid_h](f, output, v, z, H, W, True)

    # Intermediate output becomes input for vertical pass
    f_v = output.clone()
    v_v = torch.empty((B, W, max(H, W)), dtype=torch.int32, device=device)
    z_v = torch.empty((B, W, max(H, W) + 1), dtype=f.dtype, device=device)

    # Vertical pass (1D EDT on columns)
    grid_v = (B * W,)
    edt_kernel[grid_v](f_v, output, v_v, z_v, H, W, False)

    return output.squeeze(0)
