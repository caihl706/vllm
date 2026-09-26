# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused MoE expert for W4A8 INT8 quantization.

Handles compressed-tensors checkpoints where each expert weight is stored as
signed INT4 values packed eight-per-int32 in little-endian order, with a
per-output-channel BF16 scale.  Activations arrive unquantized (BF16 / FP16 /
FP32) and are dynamically quantized to per-token symmetric INT8 inside the
kernel wrapper (via ``per_token_quant_int8``).  The Triton kernel unpacks the
nibbles on the fly, performs INT8 x INT8 matmul with INT32 accumulation, and
folds the per-token activation scale and per-output-channel weight scale into
the output cast.

The expert reuses vLLM's routed-assignment machinery
(``moe_align_block_size``), so the entire fused MoE runs as three device-side
launches per (w13, w2) GEMM pair (activation + weight quant + GEMM) instead of
the Python-per-expert loop used by the correctness-first ``CUDAExpertsInt4``
fallback.
"""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4W4A8StaticChannelSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _w4a8_int8_moe_kernel(
    # Data pointers.
    a_ptr,  # [num_tokens, K] int8
    b_ptr,  # [E, N, K_packed] int32 (K_packed = K // 8)
    c_ptr,  # [num_tokens, top_k, N] compute_type
    a_scale_ptr,  # [num_tokens, 1] fp32, per-token activation scale
    b_scale_ptr,  # [E, N, 1] fp32/bf16, per-output-channel weight scale
    topk_weights_ptr,  # [num_tokens, top_k] fp32
    sorted_token_ids_ptr,  # [EM] int32
    expert_ids_ptr,  # [num_m_blocks] int32
    num_tokens_post_padded_ptr,  # [1] int32
    # Shape metadata.
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides.
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_bse,
    stride_bsn,
    # Constexprs.
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Grouped INT8 x packed-INT4 MoE GEMM with fused per-token dequant.

    ``a`` holds one INT8-quantized activation row per input token; the router
    replicates every input token ``top_k`` times, so ``offs_token // top_k``
    maps a scheduled slot back to the underlying token row (and to its
    per-token activation scale).  ``b`` holds the checkpoint-native packed
    representation with eight signed INT4 nibbles per int32 word along the K
    axis; the kernel unpacks and biases them ``(nibble - 8)`` inline.  We
    accumulate INT32, then multiply by the per-token activation scale and the
    per-output-channel weight scale before writing ``compute_type`` output.
    """
    # Program id -> tile assignment (grouped ordering, matches fused_moe_kernel).
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    cn_mask = offs_cn < N

    if off_experts == -1:
        # Invalid expert (EP expert_map miss): write zeros for the tile so the
        # downstream reduction sees a defined value.
        zero_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & cn_mask[None, :]
        tl.store(c_ptrs, zero_tile, mask=c_mask)
        return

    offs_bn = offs_cn % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_row = offs_token // top_k
    a_ptrs = a_ptr + (a_row[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # Packed K stride: 8 logical values share one int32.  We fold the
    # bit-shift for a given logical k into the load index.
    packed_k = offs_k // 8
    shift = (offs_k % 8) * 4
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + packed_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask_a = offs_k[None, :] < K - k * BLOCK_SIZE_K
        k_mask_b = offs_k[:, None] < K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask_a,
            other=0,
        )
        b_packed = tl.load(b_ptrs, mask=k_mask_b, other=0)
        b_nibble = (b_packed >> shift[:, None]) & 0xF
        b = (b_nibble - 8).to(tl.int8)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        # b_ptrs advances by BLOCK_SIZE_K logical values, i.e. BLOCK_SIZE_K/8
        # packed int32 words along the K axis.
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

    acc_f = acc.to(tl.float32)
    a_scale = tl.load(
        a_scale_ptr + a_row * stride_asm,
        mask=token_mask,
        other=0.0,
    )
    b_scale = tl.load(
        b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn,
        mask=cn_mask,
        other=0.0,
    ).to(tl.float32)
    out = acc_f * a_scale[:, None] * b_scale[None, :]

    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        out = out * w[:, None]

    out = out.to(compute_type)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & cn_mask[None, :]
    tl.store(c_ptrs, out, mask=c_mask)


def _select_w4a8_config(
    M: int,
    N: int,
    K: int,
    top_k: int,
) -> dict[str, int]:
    """Return a valid ``BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M`` triple.

    Constraints:
    - ``BLOCK_K`` must be a multiple of 8 so the packed nibble loads align with
      whole int32 words; the K loop advances by ``BLOCK_K // 8`` in the packed
      K stride.
    - Tiles must be a power of two so ``tl.dot`` / ``tl.arange`` cooperate.
    """
    if M <= 16:
        block_m = 16
    elif M <= 32:
        block_m = 32
    elif M <= 128:
        block_m = 64
    else:
        block_m = 128

    block_n = 64 if N <= 128 else 128

    # BLOCK_K must divide 8 evenly (packed layout) and should divide K where
    # possible to avoid masked K tails hurting throughput.
    if K % 128 == 0:
        block_k = 128
    elif K % 64 == 0:
        block_k = 64
    elif K % 32 == 0:
        block_k = 32
    else:
        # Any packed K length must be a multiple of 8; fall back to that.
        block_k = 16 if K % 16 == 0 else 8

    # Grouping helps L2 reuse for compute-bound batches; at small M each
    # expert only touches a handful of blocks so grouping is a wash.
    tokens_per_expert_hint = max(M // max(top_k, 1), 1)
    group_m = 4 if tokens_per_expert_hint > 32 else 1

    num_warps = 4 if M <= 128 else 8
    num_stages = 3 if M <= 64 else 2

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _invoke_w4a8_int8_moe_kernel(
    a_int8: torch.Tensor,
    b_packed: torch.Tensor,
    c: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    num_valid_tokens: int,
    compute_type: Any,
    config: dict[str, int],
) -> None:
    """Launch the W4A8 grouped GEMM against a routed-expert assignment.

    ``c`` must be shape ``[num_tokens, top_k, N]`` with ``stride(-1) == 1``; the
    kernel writes each scheduled ``offs_token`` row using ``stride_cm=C.stride(1)``
    and ``stride_cn=C.stride(2)``, mirroring the layout used by
    ``fused_moe_kernel``.  Callers responsible for reducing over ``top_k``.
    """
    assert a_int8.dtype == torch.int8
    assert a_int8.dim() == 2 and a_int8.stride(-1) == 1
    assert b_packed.dtype == torch.int32 and b_packed.dim() == 3
    assert b_packed.stride(-1) == 1
    assert a_scale.dtype == torch.float32
    assert b_scale.dim() == 3 and b_scale.size(-1) == 1
    assert c.dim() == 3 and c.stride(-1) == 1

    K = a_int8.size(-1)
    packed_k = b_packed.size(-1)
    assert packed_k * 8 == K, (
        f"packed K dimension {packed_k} does not match logical K {K}"
    )
    N = b_packed.size(1)
    assert c.size(-1) == N

    EM = sorted_token_ids.size(0)
    # For very small batches only a handful of expert blocks will be visited;
    # clamp the launched program count so we do not spawn empty tiles that
    # early-return.  Same optimisation used by fused_moe_kernel.
    if num_valid_tokens < config["BLOCK_SIZE_M"] and top_k > 0:
        EM = min(EM, num_valid_tokens * top_k * config["BLOCK_SIZE_M"])

    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    group_m = config["GROUP_SIZE_M"]
    num_warps = config.get("num_warps", 4)
    num_stages = config.get("num_stages", 2)

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)

    _w4a8_int8_moe_kernel[grid](
        a_int8,
        b_packed,
        c,
        a_scale,
        b_scale,
        topk_weights if mul_routed_weight else a_int8,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        a_int8.stride(0),
        a_int8.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        c.stride(1),
        c.stride(2),
        a_scale.stride(0),
        b_scale.stride(0),
        b_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _compute_type_for(dtype: torch.dtype) -> Any:
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported W4A8 output dtype: {dtype}")


class CUDATritonExpertsW4A8Int8(mk.FusedMoEExpertsMonolithic):
    """Monolithic Triton W4A8 INT8 experts backed by packed INT4 weights.

    Replaces the correctness-first ``CUDAExpertsInt4`` fallback: the routed
    assignment is prepared once with ``moe_align_block_size``, activations are
    quantized once per matmul, and both expert GEMMs run as a single Triton
    launch that unpacks nibbles inline and folds the scales into the store.
    """

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Triton W4A8 INT8 backend requires a CUDA platform"
        # tl.dot(int8, int8, out_dtype=int32) requires at least Turing tensor
        # cores in practice; the kernel targets SM75+ but is only actively
        # validated on SM80+.  We keep the same gate as the CUDA fallback.
        if not current_platform.has_device_capability((7, 5)):
            return False, "Triton W4A8 INT8 backend requires SM75 or newer"
        if moe_config.in_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            return False, f"unsupported input/output dtype {moe_config.in_dtype}"
        if activation_format != mk.FusedMoEActivationFormat.Standard:
            return False, "batched-experts activation format is not supported"
        if (weight_key, activation_key) != (
            kInt4W4A8StaticChannelSym,
            None,
        ):
            return False, "requires channel-wise W4A8 INT8 quantization"
        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kInt4W4A8StaticChannelSym,
            None,
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in (
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.MiniMax2,
            RoutingMethodType.Sigmoid,
            RoutingMethodType.SigmoidRenorm,
        )

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._w1_scale: torch.Tensor | None = None
        self._w2_scale: torch.Tensor | None = None

    def set_scales(
        self, w1_scale: torch.Tensor, w2_scale: torch.Tensor
    ) -> None:
        # Keep the checkpoint-native BF16 scales; the Triton kernel loads them
        # per-tile and up-casts to fp32 inline for the final dequant.
        self._w1_scale = w1_scale.contiguous()
        self._w2_scale = w2_scale.contiguous()

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        num_expert_group: int | None,
        topk_group: int | None,
        e_score_correction_bias: torch.Tensor | None,
        routed_scaling_factor: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            select_experts,
        )

        renormalize = self.moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.MiniMax2,
            RoutingMethodType.SigmoidRenorm,
        )
        scoring_func = (
            "softmax"
            if self.moe_config.routing_method
            in (
                RoutingMethodType.Default,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            )
            else "sigmoid"
        )
        if scoring_func == "sigmoid" and num_expert_group is None:
            scores = router_logits.float().sigmoid()
            topk_weights, topk_ids = torch.topk(
                scores, k=self.moe_config.experts_per_token, dim=-1, sorted=False
            )
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-20)
            if routed_scaling_factor not in (None, 1.0):
                topk_weights = topk_weights * routed_scaling_factor
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        del a1q_scale
        if hidden_states.dim() != 2:
            raise ValueError("Triton W4A8 INT8 backend expects 2-D hidden states")
        if w1.dim() != 3 or w2.dim() != 3:
            raise ValueError("packed expert weights must have shape [E, N, K/8]")
        if self._w1_scale is None or self._w2_scale is None:
            raise RuntimeError("Triton W4A8 INT8 backend needs set_scales() first")

        # Router.  Weights are fp32, ids int32.  Global ids are remapped to the
        # local expert space by moe_align_block_size when expert_map is set.
        topk_weights, topk_ids = self._select_experts(
            hidden_states,
            router_logits,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            routed_scaling_factor,
        )

        num_tokens, hidden_size = hidden_states.shape
        top_k = topk_ids.size(1)
        local_num_experts = w1.size(0)

        # Shape validation using the packed representation.  The last dim of
        # w1/w2 counts int32 words, each holding 8 nibbles.
        logical_intermediate_size = w2.size(-1) * 8
        expected_w1_out = (
            2 * logical_intermediate_size
            if activation.is_gated
            else logical_intermediate_size
        )
        if w1.size(1) != expected_w1_out:
            raise ValueError(
                f"w13 output dimension {w1.size(1)} does not match "
                f"activation size {expected_w1_out}"
            )
        if w1.size(-1) * 8 != hidden_size:
            raise ValueError(
                f"hidden dimension {hidden_size} does not match packed w13 "
                f"K dimension {w1.size(-1) * 8}"
            )
        if w2.size(1) != hidden_size:
            raise ValueError(
                f"w2 output dimension {w2.size(1)} does not match hidden "
                f"size {hidden_size}"
            )
        if hidden_size % 8 != 0 or logical_intermediate_size % 8 != 0:
            raise ValueError("W4A8 logical K dimensions must be divisible by 8")

        # Router-weight application on the input is a rare optimisation that
        # bakes topk into the input tensor.  Not needed for the standard MoE
        # path, but preserved for parity with the fallback.
        if apply_router_weight_on_input:
            assert top_k == 1, "apply_router_weight_on_input requires top_k=1"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        # Routed assignment.  ``moe_align_block_size`` handles the EP expert_map
        # remap (invalid experts stay as -1 in expert_ids, which the kernel
        # writes as zeros).  ``global_num_experts`` bounds the counting when
        # expert_map is present; otherwise topk_ids are already local.
        block_m = self._pick_block_m(num_tokens, top_k)
        num_experts_for_align = (
            global_num_experts if expert_map is not None else local_num_experts
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_m,
            num_experts_for_align,
            expert_map,
        )
        num_valid_tokens = num_tokens * top_k

        # A8 quant of the input (per-token symmetric int8).  Fp32 scales.
        a1_int8, a1_scale = per_token_quant_int8(hidden_states)

        # First GEMM: A_int8 @ w13_packed -> [num_tokens, top_k, 2 * IN].
        w13_out = torch.empty(
            (num_tokens, top_k, w1.size(1)),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        compute_type = _compute_type_for(hidden_states.dtype)
        cfg_w13 = _select_w4a8_config(num_tokens, w1.size(1), hidden_size, top_k)
        # Reuse the same BLOCK_SIZE_M chosen for the alignment so pid_m maps
        # 1:1 to the expert_ids buffer produced by moe_align_block_size.
        cfg_w13["BLOCK_SIZE_M"] = block_m
        _invoke_w4a8_int8_moe_kernel(
            a1_int8,
            w1,
            w13_out,
            a1_scale,
            self._w1_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            num_valid_tokens=num_valid_tokens,
            compute_type=compute_type,
            config=cfg_w13,
        )

        # Activation (gated or non-gated).  ``apply_moe_activation`` supports
        # every activation type used by the current models routing through
        # this backend.
        act_out_dim = (
            logical_intermediate_size if activation.is_gated else w1.size(1)
        )
        act_out = torch.empty(
            (num_tokens * top_k, act_out_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        apply_moe_activation(
            activation,
            act_out,
            w13_out.view(num_tokens * top_k, -1),
            activation_config=self.activation_config,
        )

        # A8 quant of the activated intermediate.
        a2_int8, a2_scale = per_token_quant_int8(act_out)

        # Second GEMM: A2_int8 @ w2_packed -> [num_tokens, top_k, H].  We fold
        # the routing weight into this store when it was not applied to the
        # input.
        mul_weight = not apply_router_weight_on_input
        # topk_weights is [num_tokens, top_k]; the kernel indexes it by the
        # flat scheduled offs_token, which walks [0, num_tokens * top_k).
        topk_weights_flat = topk_weights.reshape(-1).contiguous()
        # The second GEMM's activation input (a2_int8) already has one row per
        # (token, slot) pair -- it was produced from act_out of shape
        # [num_tokens * top_k, IN].  So the scheduled offs_token indexes it
        # directly and the kernel must use top_k=1 (a_row = offs_token // 1),
        # exactly as fused_experts_impl passes top_k=1 to its second GEMM.
        # (The first GEMM's input is [num_tokens, K], one row per token, so it
        # uses the real top_k to map offs_token // top_k back to the token.)
        w2_out = torch.empty(
            (num_tokens, top_k, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        cfg_w2 = _select_w4a8_config(
            num_tokens, hidden_size, logical_intermediate_size, top_k
        )
        cfg_w2["BLOCK_SIZE_M"] = block_m
        _invoke_w4a8_int8_moe_kernel(
            a2_int8,
            w2,
            w2_out,
            a2_scale,
            self._w2_scale,
            topk_weights_flat if mul_weight else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=mul_weight,
            top_k=1,
            num_valid_tokens=num_valid_tokens,
            compute_type=compute_type,
            config=cfg_w2,
        )

        # Top-k reduction.
        output = torch.empty(
            (num_tokens, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        ops.moe_sum(w2_out, output)
        return output

    @staticmethod
    def _pick_block_m(num_tokens: int, top_k: int) -> int:
        # Match BLOCK_SIZE_M to the config the kernel wrapper would pick for
        # this M, so ``moe_align_block_size`` produces the alignment the kernel
        # expects.  Kept in sync with ``_select_w4a8_config``.
        m = num_tokens * top_k
        if m <= 16 * max(top_k, 1):
            return 16
        if num_tokens <= 32:
            return 32
        if num_tokens <= 128:
            return 64
        return 128
