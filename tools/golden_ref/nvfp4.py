"""NVFP4 dequantization reference (TD-GOLDEN reference tooling).

Adapted from LayerStoRmKernels/tests/test_nvfp4_gemm_reference.py
(ref_nvfp4_dequant) — the golden reference validated against the SM120
NVFP4 GEMM kernels.

NVFP4 format: FP4 E2M1 packed 2 values/byte (low nibble = even index,
high nibble = odd index) + FP8 E4M3 per-group scales (group_size = 16)
+ one F32 global scale (weight_scale_2).

  val = FP4_E2M1_TABLE[idx] * scale_e4m3[group] * weight_scale_2
"""

import torch

# FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit (bias = 1).
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

GROUP_SIZE = 16


def dequant(weight_u8: torch.Tensor, scale_e4m3: torch.Tensor,
            scale_2: torch.Tensor) -> torch.Tensor:
    """Dequant packed NVFP4 weight to FP32.

    Args:
        weight_u8:  [N, K/2] uint8, packed FP4 pairs.
        scale_e4m3: [N, K/16] FP8 E4M3 per-group scales.
        scale_2:    scalar F32 global scale.

    Returns:
        [N, K] float32.
    """
    n, packed_k = weight_u8.shape
    k = packed_k * 2
    assert scale_e4m3.shape == (n, k // GROUP_SIZE), (
        f"scale shape {tuple(scale_e4m3.shape)} != ({n}, {k // GROUP_SIZE})")

    lo = (weight_u8 & 0x0F).to(torch.int64)
    hi = (weight_u8 >> 4).to(torch.int64)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(n, k)

    vals = FP4_E2M1_TABLE[unpacked]
    scale = scale_e4m3.float().repeat_interleave(GROUP_SIZE, dim=1)
    return vals * scale * scale_2.float().item()
