# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def _get_cached_wo_a_bf16_from_int8(
    wo_a: nn.Module, n_groups: int, o_lora_rank: int
) -> torch.Tensor:
    """Dequantize an INT8 channelwise ``wo_a`` to bf16 once and cache it.

    ``wo_a`` is read directly here (``is_bmm``), bypassing
    ``quant_method.apply()``, so the INT8 linear kernel's scale handling never
    runs. Stopgap: keep the checkpoint INT8 but compute in bf16 (no runtime
    memory/bandwidth win). TODO: replace with a real grouped INT8 GEMM.

    The INT8 kernel's ``process_weights_after_loading`` may have transposed the
    weight to (K, N) (Cutlass does), so the layout is recovered from the shape;
    ``weight_scale`` is per-output-channel, (N, 1).
    """
    cached = getattr(wo_a, "_dsv4_wo_a_bf16", None)
    if cached is not None:
        return cached
    weight = wo_a.weight
    n_out = n_groups * o_lora_rank
    if weight.shape[0] == n_out and weight.shape[1] != n_out:
        transposed = False
    elif weight.shape[1] == n_out and weight.shape[0] != n_out:
        transposed = True
    else:  # square-ish ambiguity: a Cutlass-transposed view is non-contiguous
        transposed = not weight.is_contiguous()
    w = weight.t() if transposed else weight  # (N, K)
    assert w.shape[0] == n_out, (w.shape, n_out)
    scale = wo_a.weight_scale.reshape(-1, 1).float()
    assert scale.shape[0] == n_out, (scale.shape, n_out)
    cached = (w.float() * scale).to(torch.bfloat16).view(n_groups, o_lora_rank, -1)
    wo_a._dsv4_wo_a_bf16 = cached
    return cached


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization.
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=einsum_recipe[2],
        tma_aligned_scales=tma_aligned_scales,
        quantize=use_fp8,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        if wo_a.weight.dtype == torch.int8:
            grouped_weight = _get_cached_wo_a_bf16_from_int8(
                wo_a, n_groups, o_lora_rank
            )
        else:
            grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))
