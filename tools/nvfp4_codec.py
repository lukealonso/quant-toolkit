#!/usr/bin/env python3
"""Exact ModelOpt-compatible NVFP4 weight codec primitives."""

from __future__ import annotations

import torch

BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
FP8_MIN = 1.0 / FP8_MAX
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
FP4_ABS_CODEBOOK = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _require_weight_shape(weight: torch.Tensor) -> None:
    if weight.ndim != 2 or weight.shape[1] % BLOCK_SIZE:
        raise ValueError(
            f"NVFP4 weight must be rank two with columns divisible by {BLOCK_SIZE}: "
            f"shape={tuple(weight.shape)}"
        )


def project_block_scales(block_scales: torch.Tensor) -> torch.Tensor:
    projected = block_scales.clamp(min=FP8_MIN).to(torch.float8_e4m3fn)
    projected_f32 = projected.to(torch.float32)
    projected_f32[projected_f32 == 0] = FP8_MIN
    return projected_f32.to(torch.float8_e4m3fn)


def baseline_scales(
    weight: torch.Tensor, *, scale_2: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ModelOpt's scalar and per-16-value FP8 scales."""
    _require_weight_shape(weight)
    weight_f32 = weight.to(torch.float32)
    if scale_2 is None:
        max_abs = weight_f32.abs().amax().clamp_min(FP8_MIN)
        scale_2 = (max_abs / (FP4_MAX * FP8_MAX)).reshape(())
    else:
        scale_2 = scale_2.to(device=weight.device, dtype=torch.float32).reshape(())
        if not torch.isfinite(scale_2) or scale_2 <= 0:
            raise ValueError(
                "NVFP4 secondary-scale override must be finite and positive"
            )
    block_amax = weight_f32.reshape(weight.shape[0], -1, BLOCK_SIZE).abs().amax(-1)
    block_scale = block_amax / (FP4_MAX * scale_2)
    block_scale[block_amax == 0] = 1.0
    return scale_2, project_block_scales(block_scale)


def pack_fp4(normalized_weight: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 values, low nibble first, exactly as ModelOpt exports them."""
    _require_weight_shape(normalized_weight)
    bounds = E2M1_BOUNDS.to(normalized_weight.device)
    magnitude = normalized_weight.abs()
    ordinals = torch.searchsorted(bounds, magnitude, out_int32=True).to(torch.uint8)
    # ModelOpt's ties-to-even convention selects the upper code only at the
    # three boundaries whose lower ordinal is odd.
    odd_boundaries = bounds[[1, 3, 5]]
    equal_to_odd_boundary = torch.any(
        magnitude.unsqueeze(-1) == odd_boundaries, dim=-1
    ).to(torch.uint8)
    code = (
        ((normalized_weight < 0).to(torch.uint8) << 3)
        + ordinals
        + equal_to_odd_boundary
    )
    return ((code[..., 1::2] << 4) | code[..., 0::2]).contiguous()


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError(
            f"packed NVFP4 weight must be rank-two uint8: "
            f"shape={tuple(packed.shape)} dtype={packed.dtype}"
        )
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    codebook = FP4_ABS_CODEBOOK.to(packed.device)
    values = codebook[(codes & 0x07).to(torch.int64)]
    return torch.where((codes & 0x08) != 0, -values, values)


def encode_nvfp4(
    weight: torch.Tensor,
    *,
    scale_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a dense matrix into packed weight, FP8 block scale, and F32 scale."""
    _require_weight_shape(weight)
    weight_f32 = weight.to(torch.float32)
    scale_2, block_scale = baseline_scales(weight_f32, scale_2=scale_2)
    effective = block_scale.to(torch.float32) * scale_2
    normalized = weight_f32.reshape(
        weight.shape[0], -1, BLOCK_SIZE
    ) / effective.unsqueeze(-1)
    packed = pack_fp4(normalized.reshape_as(weight_f32))
    return packed, block_scale, scale_2


def quantize_nvfp4_with_scales(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    scale_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with an explicit runtime-compatible scale representation."""
    _require_weight_shape(weight)
    expected_scale_shape = (weight.shape[0], weight.shape[1] // BLOCK_SIZE)
    if (
        block_scale.dtype != torch.float8_e4m3fn
        or block_scale.shape != expected_scale_shape
    ):
        raise ValueError(
            f"invalid explicit block scales: expected float8_e4m3fn "
            f"{expected_scale_shape}, got {block_scale.dtype} {tuple(block_scale.shape)}"
        )
    if scale_2.dtype != torch.float32 or scale_2.numel() != 1:
        raise ValueError("NVFP4 secondary scale must be scalar float32")
    effective = block_scale.to(torch.float32) * scale_2.reshape(())
    normalized = weight.to(torch.float32).reshape(
        weight.shape[0], -1, BLOCK_SIZE
    ) / effective.unsqueeze(-1)
    packed = pack_fp4(normalized.reshape_as(weight))
    return packed, decode_nvfp4(packed, block_scale, scale_2)


def optimize_block_scales(
    weight: torch.Tensor,
    *,
    scale_2: torch.Tensor | None = None,
    factors: tuple[float, ...] = (
        0.75,
        0.8125,
        0.875,
        0.9375,
        1.0,
        1.0625,
        1.125,
        1.1875,
        1.25,
    ),
    coordinate_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Search nearby FP8 block scales under a diagonal input-second-moment loss.

    The returned tuple is packed weight, selected block scale, secondary scale,
    and decoded float32 weight. Strict comparisons preserve the stock baseline
    on ties, making the result deterministic for an ordered factor grid.
    """
    _require_weight_shape(weight)
    if not factors or 1.0 not in factors:
        raise ValueError("block-scale factor grid must be non-empty and contain 1.0")
    if any(not isinstance(factor, (float, int)) or factor <= 0 for factor in factors):
        raise ValueError("block-scale factors must be positive numbers")

    weight_f32 = weight.to(torch.float32)
    scale_2, baseline_block_scale = baseline_scales(weight_f32, scale_2=scale_2)
    blocks = weight_f32.reshape(weight.shape[0], -1, BLOCK_SIZE)
    if coordinate_weights is None:
        loss_weights = torch.ones(
            (1, blocks.shape[1], BLOCK_SIZE),
            dtype=torch.float32,
            device=weight.device,
        )
    else:
        if (
            coordinate_weights.ndim != 1
            or coordinate_weights.numel() != weight.shape[1]
        ):
            raise ValueError(
                f"coordinate weights must have shape ({weight.shape[1]},): "
                f"{tuple(coordinate_weights.shape)}"
            )
        loss_weights = coordinate_weights.to(
            device=weight.device, dtype=torch.float32
        ).reshape(1, blocks.shape[1], BLOCK_SIZE)
        if not torch.isfinite(loss_weights).all() or bool((loss_weights < 0).any()):
            raise ValueError("coordinate weights must be finite and nonnegative")

    _, baseline_decoded = quantize_nvfp4_with_scales(
        weight_f32, baseline_block_scale, scale_2
    )
    best_error = (
        (baseline_decoded.reshape_as(blocks) - blocks).square() * loss_weights
    ).sum(dim=-1)
    best_scale = baseline_block_scale.clone()
    for factor in factors:
        if float(factor) == 1.0:
            continue
        candidate_scale = project_block_scales(
            baseline_block_scale.to(torch.float32) * float(factor)
        )
        _, reconstructed = quantize_nvfp4_with_scales(
            weight_f32, candidate_scale, scale_2
        )
        error = (
            (reconstructed.reshape_as(blocks) - blocks).square() * loss_weights
        ).sum(dim=-1)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scale = torch.where(improved, candidate_scale, best_scale)

    packed, decoded = quantize_nvfp4_with_scales(weight_f32, best_scale, scale_2)
    return packed, best_scale, scale_2, decoded


def decode_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    scale_2: torch.Tensor,
) -> torch.Tensor:
    """Decode the serving representation to float32 without changing layout."""
    if block_scale.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"NVFP4 block scale must be float8_e4m3fn: {block_scale.dtype}"
        )
    if block_scale.shape != (packed.shape[0], packed.shape[1] * 2 // BLOCK_SIZE):
        raise ValueError(
            f"block-scale shape does not match packed weight: "
            f"packed={tuple(packed.shape)} scale={tuple(block_scale.shape)}"
        )
    if scale_2.dtype != torch.float32 or scale_2.numel() != 1:
        raise ValueError("NVFP4 secondary scale must be scalar float32")
    normalized = unpack_fp4(packed).reshape(
        packed.shape[0], block_scale.shape[1], BLOCK_SIZE
    )
    effective = block_scale.to(torch.float32) * scale_2.reshape(())
    return (normalized * effective.unsqueeze(-1)).reshape(packed.shape[0], -1)
