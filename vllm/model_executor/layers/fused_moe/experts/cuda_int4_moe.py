# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA W4A8 INT8 MoE expert implementation.

This is a correctness-first CUDA backend for compressed-tensors checkpoints that
store signed INT4 values packed eight per int32.  It keeps the checkpoint packed
and uses CUDA int8 matrix multiplication after unpacking one selected expert at
a time.  A dedicated Triton kernel can replace the per-expert unpack step later
without changing the loader or backend contract.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4W4A8StaticChannelSym,
)
from vllm.platforms import current_platform

# Per-device cache for the INT4 unpack shift constants ``[0, 4, ..., 28]``.
# ``torch.arange`` is otherwise recreated on every ``_unpack_int4`` call.  The
# tensor is a handful of bytes and is safe to keep around for the lifetime of
# the process.
_SHIFT_AMOUNTS_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_shift_amounts(device: torch.device) -> torch.Tensor:
    cached = _SHIFT_AMOUNTS_CACHE.get(device)
    if cached is None:
        cached = torch.arange(8, device=device, dtype=torch.int32) * 4
        _SHIFT_AMOUNTS_CACHE[device] = cached
    return cached


class CUDAExpertsInt4(mk.FusedMoEExpertsMonolithic):
    """CUDA W4A8 INT8 experts backed by packed INT4 weights.

    The implementation intentionally keeps the checkpoint representation
    packed.  For each expert selected by the router it unpacks the small set of
    nibbles needed for that expert, quantizes the dispatched tokens per row, and
    uses ``torch._int_mm`` for the INT8 GEMM.  This is slower than a fused
    production kernel but preserves the exact W4A8 arithmetic and is useful as
    a portable CUDA fallback while a fused Triton kernel is tuned.
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
            return False, "CUDA W4A8 INT8 backend requires a CUDA platform"
        if not current_platform.has_device_capability((7, 5)):
            return False, "CUDA W4A8 INT8 backend requires SM75 or newer"
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
            RoutingMethodType.DeepseekV4,
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

    @staticmethod
    def _unpack_int4(weight_packed: torch.Tensor, logical_k: int) -> torch.Tensor:
        """Unpack compressed-tensors' little-endian int32 nibble layout."""
        if weight_packed.dtype != torch.int32:
            raise TypeError(
                f"W4A8 packed weights must be int32, got {weight_packed.dtype}"
            )
        if weight_packed.shape[-1] * 8 < logical_k:
            raise ValueError(
                f"packed K dimension {weight_packed.shape[-1]} cannot hold {logical_k}"
            )
        shifts = _get_shift_amounts(weight_packed.device)
        nibbles = (weight_packed.unsqueeze(-1) >> shifts) & 0xF
        nibbles = nibbles.reshape(
            weight_packed.shape[:-1] + (weight_packed.shape[-1] * 8,)
        )
        # ``nibbles`` sits in ``[0, 15]`` which already fits in int8, so cast
        # first and subtract the bias in place.  This saves the intermediate
        # int16 allocation the original path went through.
        narrow = nibbles[..., :logical_k].to(torch.int8)
        narrow -= 8
        return narrow

    @staticmethod
    def _quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ``x`` is only read for the max/div here, so the fp32 copy is reused
        # in place as the scaled buffer to avoid a chain of temporaries.
        x_float = x.float()
        scale = (
            x_float.abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min_(1e-12)
            .div_(127.0)
        )
        x_q = (
            x_float.div_(scale).round_().clamp_(-127, 127).to(torch.int8)
        )
        return x_q, scale

    @classmethod
    def _int8_linear(
        cls,
        x: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale_t_fp32: torch.Tensor,
        logical_k: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        x_q, x_scale = cls._quantize_activation(x)
        weight_q = cls._unpack_int4(weight_packed, logical_k)
        # CUDA int8 GEMM requires more than 16 rows on the target PyTorch
        # implementation.  MoE routing often sends only a few tokens to an
        # expert, so pad the M dimension and discard the padded results.
        original_m = x_q.shape[0]
        if original_m <= 16:
            x_q = torch.nn.functional.pad(x_q, (0, 0, 0, 32 - original_m))
        # torch._int_mm is available on CUDA SM75+ and returns int32.
        acc = torch._int_mm(x_q, weight_q.transpose(0, 1).contiguous())
        acc = acc[:original_m]
        # Fuse the two scale multiplications and the final dtype cast so
        # PyTorch runs them on a single fp32 buffer and only materialises the
        # target-dtype output once.
        return (
            acc.float().mul_(x_scale).mul_(weight_scale_t_fp32).to(out_dtype)
        )

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        num_expert_group: int | None,
        topk_group: int | None,
        e_score_correction_bias: torch.Tensor | None,
        routed_scaling_factor: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe.router.cpu_router import (
            select_experts,
        )

        if self.moe_config.routing_method == RoutingMethodType.DeepseekV4:
            # DeepSeek V4 routes with weight = sqrt(softplus(logit)) and picks
            # experts by the *bias-corrected* score ("noaux_tc"): the bias
            # steers selection only, the returned weight stays unbiased.
            # ``cpu_router.select_experts`` implements this with SGLang AVX512
            # kernels that cannot run on CUDA, so route through vLLM's own
            # ``fused_topk_bias`` instead.  It is a two-level dispatch: the
            # fast ``dsv4_topk`` Triton kernel when the shape matches (fp32
            # logits, 256/384 experts, top_k=6) and the general
            # ``vllm_topk_softplus_sqrt`` otherwise.  Both levels are needed
            # here -- the 384-expert target model and the much smaller dspark
            # draft model (128 experts, top_k=3) are both routed by the
            # DeepSeek-V4 method, so no single kernel covers them.  Both fold
            # in the renormalization to ``routed_scaling_factor``.
            #
            # Hash-routed layers and vision layers (``bias_vl``) need routing
            # inputs the monolithic ``apply`` signature does not carry, so they
            # are rejected rather than silently mis-routed.
            if num_expert_group is not None or topk_group is not None:
                raise NotImplementedError(
                    "W4A8 INT8 MoE does not support grouped top-k routing"
                )
            if e_score_correction_bias is None:
                raise NotImplementedError(
                    "DeepSeek V4 W4A8 routing requires e_score_correction_bias, "
                    "but the layer provides none"
                )
            return fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=e_score_correction_bias,
                topk=self.moe_config.experts_per_token,
                renormalize=True,
                indices_type=torch.int32,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
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
        del a1q_scale, global_num_experts
        if hidden_states.dim() != 2:
            raise ValueError("CUDA W4A8 INT8 backend expects 2-D hidden states")
        if w1.dim() != 3 or w2.dim() != 3:
            raise ValueError("packed expert weights must have shape [E, N, K/8]")

        topk_weights, topk_ids = self._select_experts(
            hidden_states,
            router_logits,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            routed_scaling_factor,
        )
        if expert_map is not None:
            mapped_ids = expert_map[topk_ids.long()]
        else:
            mapped_ids = topk_ids

        output = torch.zeros(
            hidden_states.shape[0],
            w2.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        logical_intermediate_size = w2.shape[2] * 8
        intermediate_size = (
            logical_intermediate_size
            if activation.is_gated
            else w1.shape[1] * 8
        )
        expected_w1_size = (
            2 * intermediate_size if activation.is_gated else intermediate_size
        )
        if w1.shape[1] != expected_w1_size:
            raise ValueError(
                f"w13 output dimension {w1.shape[1]} does not match "
                f"activation size {expected_w1_size}"
            )
        hidden_size = hidden_states.shape[1]
        if hidden_size != w1.shape[2] * 8:
            raise ValueError(
                f"hidden state dimension {hidden_size} does not match packed "
                f"w13 K dimension {w1.shape[2] * 8}"
            )
        if w2.shape[2] * 8 != intermediate_size:
            raise ValueError(
                f"w2 K dimension {w2.shape[2] * 8} does not match activation "
                f"dimension {intermediate_size}"
            )
        if hidden_size % 8 != 0 or intermediate_size % 8 != 0:
            raise ValueError("W4A8 logical K dimensions must be divisible by 8")

        # Group the dispatched tokens by local expert.  A naive implementation
        # calls ``torch.nonzero(mapped_ids == expert_id)`` inside the Python
        # loop; each of those launches a forced device->host synchronisation
        # to size the returned index tensor, adding one sync per expert per
        # layer per step.  Instead we sort the flattened (token, slot) pairs
        # by their assigned local expert once, so ``argsort`` + ``bincount``
        # produce a compact ``[num_tokens_per_expert]`` layout with a single
        # D2H copy of the per-expert counts.  Empty experts are then skipped
        # entirely on the CPU side, without touching CUDA.  The dynamic Python
        # loop still requires eager execution; a production CUDA-graph path
        # should replace it with a fused routed-experts kernel that uses a
        # preallocated workspace.
        num_local_experts = w1.shape[0]
        flat_ids = mapped_ids.reshape(-1)
        valid = (flat_ids >= 0) & (flat_ids < num_local_experts)
        sort_key = torch.where(
            valid,
            flat_ids,
            torch.full_like(flat_ids, num_local_experts),
        )
        expert_token_counts = torch.bincount(
            sort_key.long(),
            minlength=num_local_experts + 1,
        )[:num_local_experts]
        perm = torch.argsort(sort_key, stable=False)
        counts_cpu = expert_token_counts.tolist()
        topk_width = mapped_ids.shape[1]
        token_ids_all = perm // topk_width
        slot_ids_all = perm % topk_width
        router_weights_flat = topk_weights.reshape(-1)

        offset = 0
        for expert_id in range(num_local_experts):
            count = counts_cpu[expert_id]
            if count == 0:
                continue
            sel = slice(offset, offset + count)
            offset += count
            token_ids = token_ids_all[sel]
            slot_ids = slot_ids_all[sel]
            x = hidden_states.index_select(0, token_ids)
            router_weight = router_weights_flat[
                token_ids * topk_width + slot_ids
            ].to(x.dtype)
            if apply_router_weight_on_input:
                x = x * router_weight[:, None]

            w13 = self._int8_linear(
                x,
                w1[expert_id],
                self._w1_scale_t_fp32[expert_id],
                hidden_size,
                hidden_states.dtype,
            )
            activated = torch.empty(
                w13.shape[0],
                intermediate_size,
                device=w13.device,
                dtype=hidden_states.dtype,
            )
            apply_moe_activation(
                activation,
                activated,
                w13,
                activation_config=self.activation_config,
            )
            y = self._int8_linear(
                activated,
                w2[expert_id],
                self._w2_scale_t_fp32[expert_id],
                intermediate_size,
                output.dtype,
            )
            if not apply_router_weight_on_input:
                y = y * router_weight[:, None]
            output.index_add_(0, token_ids, y)
        return output

    def set_scales(
        self, w1_scale: torch.Tensor, w2_scale: torch.Tensor
    ) -> None:
        self._w1_scale = w1_scale
        self._w2_scale = w2_scale
        # Pre-transpose and cast the per-output-channel weight scales once so
        # every GEMM can reuse the same fp32 buffer instead of rebuilding
        # ``weight_scale.float().transpose(0, 1)`` per expert per call.  The
        # tensors are small ([E, 1, N] fp32) but the allocation churn showed
        # up in profiling of the eager fallback.
        self._w1_scale_t_fp32 = (
            w1_scale.float().transpose(-1, -2).contiguous()
        )
        self._w2_scale_t_fp32 = (
            w2_scale.float().transpose(-1, -2).contiguous()
        )
