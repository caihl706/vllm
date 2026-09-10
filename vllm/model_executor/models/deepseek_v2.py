# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""

import math
import os
import sys
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, RSWAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    DCPGroupColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    fused_indexer_q_rope_quant,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    sequence_parallel_chunk,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from .interfaces import (
    MixtureOfExperts,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    PPMissingLayer,
    get_pp_missing_layer_names,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# Debug logging: only fires after `touch $VLLM_DBG_SENTINEL` (default
# /tmp/vllm_dbg_enable). This lets vLLM's own warmup / dummy runs finish
# silently; the user turns logging on after `vllm serve` is fully up.
_DBG_SENTINEL = os.environ.get("VLLM_DBG_SENTINEL", "/tmp/vllm_dbg_enable")


# Announce once at import so the user can grep the vLLM startup log to
# confirm the patched file is actually loaded and see which sentinel path
# it is checking (useful if VLLM_DBG_SENTINEL was overridden).
logger.info(
    "[deepseek_v2 debug] instrumentation loaded; touch %s (rank 0) "
    "to enable per-layer trace",
    _DBG_SENTINEL,
)


# DeepseekV2Model is decorated with @support_torch_compile, so any function it
# calls transitively is a tracing target for TorchDynamo. Without disabling,
# Dynamo will constant-fold the first `os.path.isfile(sentinel)` return value
# (typically False during warmup) into the compiled graph — after that,
# `touch`ing the sentinel would not re-enable prints because the branch was
# already eliminated. Marking the helpers with @_dynamo_disable forces a graph
# break at each call so they always run in eager Python.
try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:  # pragma: no cover
    def _dynamo_disable(fn):
        return fn


@_dynamo_disable
def _dbg_enabled():
    if not os.path.isfile(_DBG_SENTINEL):
        return False
    try:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
    except Exception:
        rank = 0
    return rank == 0


def _dbg_header(caller_frame):
    caller_file = caller_frame.f_code.co_filename
    return (
        f"##info,{os.path.basename(caller_file)}:{caller_frame.f_lineno},"
        f"pid:{os.getpid()}"
    )


def _get_weight(module):
    """Best-effort fetch of a linear module's underlying weight tensor."""
    if module is None:
        return None
    for attr in ("weight", "qweight"):
        w = getattr(module, attr, None)
        if w is not None:
            return w
    return None


def _get_scale(module):
    """Best-effort fetch of a quantized module's weight scale.

    Covers common vLLM patterns:
      - FP8 block-wise: `weight_scale_inv` (2D)
      - FP8 per-tensor / INT8 per-channel: `weight_scale`
    """
    if module is None:
        return None
    for attr in ("weight_scale_inv", "weight_scale"):
        s = getattr(module, attr, None)
        if s is not None:
            return s
    return None


# vLLM / DeepSeek / GLM-5 FP8 block-wise quant uses fixed 128x128 blocks.
# Inferring block size from `ceil(rows/scale_rows)` is WRONG when rows is not
# a multiple of 128 (e.g. fused_qkv_a_proj with rows=2624 has scale_rows=21,
# but the block is still 128x128 with a 64-row zero pad, not a 125-row block).
_FP8_BLOCK = (128, 128)


def _get_moe_expert_params(experts_module):
    """Extract the actual `w13_weight` / `w2_weight` (+ their scale) from a
    FusedMoEFactory-built runner.

    In this vLLM version `self.experts` is a `MoERunner` wrapper — the real
    stacked expert parameters live on `experts_module.routed_experts` (a
    `RoutedExperts` layer registered by the quant method). A plain
    `getattr(self.experts, "w13_weight")` returns None because the runner
    delegates weight storage to its `routed_experts` child.

    Returns a dict with keys ``w13_weight`` / ``w13_scale`` / ``w2_weight``
    / ``w2_scale`` (values may be None). Handles both FP8 block-wise
    (``weight_scale_inv``, 3D) and W8A8 int-quantized / FP8 per-channel
    (``weight_scale``, 1D or 2D) layouts.
    """
    if experts_module is None:
        return {
            "w13_weight": None,
            "w13_scale": None,
            "w2_weight": None,
            "w2_scale": None,
        }
    # Try the runner's inner RoutedExperts first, then fall back to the
    # object itself for unwrapped implementations.
    for holder in (getattr(experts_module, "routed_experts", None), experts_module):
        if holder is None:
            continue
        w13 = getattr(holder, "w13_weight", None)
        w2 = getattr(holder, "w2_weight", None)
        if w13 is None and w2 is None:
            continue
        # NOTE: cannot use `a or b` here because a/b may be tensors, and
        # `Tensor.__bool__` on numel>1 raises "Boolean value of Tensor with
        # more than one element is ambiguous". Prefer weight_scale_inv
        # (FP8 block-wise) over weight_scale (FP8 per-tensor / INT8 per-ch).
        w13_scale = getattr(holder, "w13_weight_scale_inv", None)
        if w13_scale is None:
            w13_scale = getattr(holder, "w13_weight_scale", None)
        w2_scale = getattr(holder, "w2_weight_scale_inv", None)
        if w2_scale is None:
            w2_scale = getattr(holder, "w2_weight_scale", None)
        return {
            "w13_weight": w13,
            "w13_scale": w13_scale,
            "w2_weight": w2,
            "w2_scale": w2_scale,
        }
    return {
        "w13_weight": None,
        "w13_scale": None,
        "w2_weight": None,
        "w2_scale": None,
    }


@_dynamo_disable
def _dbg_moe_router_topk(
    label_prefix: str,
    gate: nn.Module,
    hidden_states: torch.Tensor,
    top_k: int,
) -> None:
    """Recompute the router logits on a small token slice and print top-k
    expert indices + scores.

    Purpose: locate the layer where FP8 and W8A8 diverge in expert selection.
    We reuse `gate` (a GateLinear replicated projection), so no TP communication
    is triggered. To keep log volume manageable and cost negligible, only the
    first 8 tokens are inspected. Note this is the *naive* top-k on raw
    router_logits — the real FusedMoE routing may apply grouped-topk /
    e_score_correction_bias / norm; but for divergence detection the raw signal
    is sufficient (if raw topk stays aligned, grouped topk almost always does
    too).
    """
    if not _dbg_enabled():
        return
    try:
        with torch.no_grad():
            sample = hidden_states[:8].contiguous()
            router_logits, _ = gate(sample)
            router_logits_f = router_logits.to(torch.float32)
            k = min(top_k, router_logits_f.shape[-1])
            top_val, top_idx = torch.topk(router_logits_f, k=k, dim=-1)
    except Exception as e:
        _dbg_print((f"{label_prefix}.router_topk", None))
        print(f"  {label_prefix}.router_topk: ERR={type(e).__name__}:{e}", flush=True)
        return
    _dbg_print(
        (f"{label_prefix}.router_topk_idx[:8]", top_idx),
        (f"{label_prefix}.router_topk_val[:8]", top_val),
    )


def _dequant_block_sum_2d(w_f: torch.Tensor, s_f: torch.Tensor) -> float:
    """Sum of `dequant(w) = w * scale_block` for FP8 block-wise 2D weight.

    Mirrors `dequant_fp8_block` in build_mini_glm5_3.py:91-120: block size is
    fixed at (128, 128); weight is zero-padded to a multiple of the block
    size before being carved into (row_blocks, col_blocks) tiles. Uses
    "per-block sum first, then multiply by scale" — mathematically identical
    to multiply-then-sum but avoids materialising the dequantised tensor.
    """
    rows, cols = w_f.shape
    br, bc = _FP8_BLOCK
    padded_rows = math.ceil(rows / br) * br
    padded_cols = math.ceil(cols / bc) * bc
    row_blocks = padded_rows // br
    col_blocks = padded_cols // bc
    if tuple(s_f.shape) != (row_blocks, col_blocks):
        # Not a 128x128 block layout — fall back to a naive broadcast so the
        # caller still gets something rather than a silently wrong number.
        try:
            return (w_f * s_f).sum().item()
        except Exception as e:
            return f"ERR:block-shape-mismatch:{type(e).__name__}:{e}"
    if (padded_rows, padded_cols) != (rows, cols):
        padded = torch.zeros(
            padded_rows, padded_cols, dtype=w_f.dtype, device=w_f.device
        )
        padded[:rows, :cols] = w_f
        w_f = padded
    per_block_sum = w_f.reshape(row_blocks, br, col_blocks, bc).sum(dim=(1, 3))
    return (per_block_sum * s_f).sum().item()


def _dequant_block_sum_3d(w_f: torch.Tensor, s_f: torch.Tensor) -> float:
    """Same as `_dequant_block_sum_2d` but with a leading expert axis
    (FusedMoE `w13_weight` / `w2_weight` are [num_experts, out, in]).
    Block size is (128, 128) on the (out, in) plane.
    """
    n_experts, rows, cols = w_f.shape
    br, bc = _FP8_BLOCK
    padded_rows = math.ceil(rows / br) * br
    padded_cols = math.ceil(cols / bc) * bc
    row_blocks = padded_rows // br
    col_blocks = padded_cols // bc
    if tuple(s_f.shape) != (n_experts, row_blocks, col_blocks):
        try:
            return (w_f * s_f).sum().item()
        except Exception as e:
            return f"ERR:block-shape-mismatch:{type(e).__name__}:{e}"
    if (padded_rows, padded_cols) != (rows, cols):
        padded = torch.zeros(
            n_experts, padded_rows, padded_cols, dtype=w_f.dtype, device=w_f.device
        )
        padded[:, :rows, :cols] = w_f
        w_f = padded
    per_block_sum = w_f.reshape(n_experts, row_blocks, br, col_blocks, bc).sum(
        dim=(2, 4)
    )
    return (per_block_sum * s_f).sum().item()


def _dequant_weight_sum(w, scale):
    """Sum of dequantised `w * scale`, matching build_mini_glm5_3.py's
    convention for the schemes actually stored on live vLLM modules:

      - Per-tensor scale (scalar): `dequant = w * scale`.
      - Per-output-channel 1D scale (FP8 per-channel, W8A8 int8):
          2D weight [out, in] -> broadcast on dim 0
          3D MoE weight [num_experts, out, in] -> broadcast on dim 1
      - Block-wise 2D scale (FP8 block, typically 128x128):
          2D weight -> _dequant_block_sum_2d
          3D MoE weight w/ 3D scale -> _dequant_block_sum_3d (per expert)

    Returns a float sum, or an ``ERR:...`` string when shapes don't line up.
    W4A16/W8A16 packed weights (`weight_packed`) are checkpoint-only in vLLM;
    they are unpacked at load time, so we don't handle them here.
    """
    if w is None or scale is None:
        return None
    try:
        w_f = w.detach().to(torch.float32)
        s_f = scale.detach().to(torch.float32)

        # Per-tensor scalar.
        if s_f.numel() == 1:
            return (w_f * s_f).sum().item()

        # Per-output-channel (1D scale).
        if s_f.ndim == 1:
            if w_f.ndim == 2 and s_f.numel() == w_f.shape[0]:
                return (w_f * s_f.unsqueeze(-1)).sum().item()
            if w_f.ndim == 3 and s_f.numel() == w_f.shape[1]:
                return (w_f * s_f.reshape(1, -1, 1)).sum().item()

        # Block-wise 2D scale for a 2D linear weight.
        if s_f.ndim == 2 and w_f.ndim == 2:
            return _dequant_block_sum_2d(w_f, s_f)

        # Block-wise 3D scale for a 3D MoE stacked weight.
        if s_f.ndim == 3 and w_f.ndim == 3 and s_f.shape[0] == w_f.shape[0]:
            return _dequant_block_sum_3d(w_f, s_f)

        # Unknown layout — try a naive broadcast; will raise if it fails.
        return (w_f * s_f).sum().item()
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}"


def _print_tensor(name, t, scale=None):
    if t is None:
        print(f"  {name}: None", flush=True)
        return
    try:
        shape = tuple(t.shape)
        dtype = t.dtype
        if t.numel() == 0:
            raw = 0.0
        elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raw = t.detach().to(torch.float32).sum().item()
        elif dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
            raw = t.detach().to(torch.float64).sum().item()
        else:
            raw = t.detach().float().sum().item()
        extra = ""
        if scale is not None:
            dq = _dequant_weight_sum(t, scale)
            if dq is not None:
                extra = f", dequant_sum={dq}"
        # For small tensors (e.g. router_topk_idx of shape (8, top_k)),
        # emit the actual values too — sum alone won't reveal which experts
        # were selected, and picking the divergence layer needs the indices.
        values_str = ""
        if t.numel() <= 128:
            try:
                values_str = f", values={t.detach().cpu().tolist()}"
            except Exception:
                pass
        print(
            f"  {name}: shape={shape}, dtype={dtype}, sum={raw}{extra}{values_str}",
            flush=True,
        )
    except Exception as e:
        print(f"  {name}: ERR={type(e).__name__}:{e}", flush=True)


def _dbg_dump_one(name, obj):
    """Pretty-print one debug item. Accepts:

    - torch.Tensor: printed as-is
    - nn.Module: auto-extracts .weight and any scale attribute
                 (weight_scale_inv / weight_scale) and prints both
                 the raw and dequantized sums so FP8 vs INT8 vs BF16
                 checkpoints are numerically comparable.
    - None: printed as None.
    """
    if obj is None:
        print(f"  {name}: None", flush=True)
        return
    if isinstance(obj, torch.Tensor):
        _print_tensor(name, obj)
        return
    if isinstance(obj, nn.Module):
        w = _get_weight(obj)
        s = _get_scale(obj)
        _print_tensor(f"{name}.weight", w, scale=s)
        if s is not None:
            _print_tensor(f"{name}.weight_scale", s)
        return
    print(f"  {name}: unsupported type {type(obj).__name__}", flush=True)


@_dynamo_disable
def _dbg_print(*items):
    """Print debug info for tensors/modules on rank 0 (after sentinel).

    Each item is one of:
      - (name, obj) where obj is a torch.Tensor or nn.Module
      - (name, tensor, scale) where `scale` is applied via _dequant_weight_sum
        (use this for MoE experts whose stacked weight+scale aren't packaged
        into a standard nn.Module).
    """
    if not _dbg_enabled():
        return
    caller_frame = sys._getframe(1)
    print(_dbg_header(caller_frame), flush=True)
    for item in items:
        if len(item) == 2:
            name, obj = item
            _dbg_dump_one(name, obj)
        elif len(item) == 3:
            name, tensor, scale = item
            _print_tensor(name, tensor, scale=scale)
        else:
            print(f"  ??: unexpected item arity {len(item)}: {item!r}", flush=True)


@_dynamo_disable
def _dbg_msg(msg: str):
    """Emit a plain ##info header line + message on rank 0 (after sentinel)."""
    if not _dbg_enabled():
        return
    caller_frame = sys._getframe(1)
    print(f"{_dbg_header(caller_frame)} {msg}", flush=True)


def _get_moe_router_dtype(
    config: DeepseekV2Config | DeepseekV3Config,
) -> torch.dtype | None:
    router_dtype = getattr(config, "moe_router_dtype", None)
    if getattr(config, "model_type", None) == "glm_moe_dsa":
        # Older GLM-5/5.2 configs require fp32 routing but do not expose
        # moe_router_dtype yet.
        return torch.float32
    if router_dtype == "float32":
        return torch.float32
    return None


class DeepseekAttention(nn.Module):
    """Normal MHA implementation used by Deepseek v1."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        rswa_window = getattr(vllm_config.model_config.hf_config, "rswa_window", None)
        if rswa_window is not None:
            self.attn = RSWAAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                rswa_window=rswa_window,
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _dbg_print(
            ("DeepseekAttention.input.hidden_states", hidden_states),
            ("DeepseekAttention.qkv_proj", self.qkv_proj),
        )
        qkv, _ = self.qkv_proj(hidden_states)
        _dbg_print(("DeepseekAttention.qkv_proj.output", qkv))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        _dbg_print(
            ("DeepseekAttention.rotary_emb.q", q),
            ("DeepseekAttention.rotary_emb.k", k),
        )
        attn_output = self.attn(q, k, v)
        _dbg_print(
            ("DeepseekAttention.attn.output", attn_output),
            ("DeepseekAttention.o_proj", self.o_proj),
        )
        output, _ = self.o_proj(attn_output)
        _dbg_print(("DeepseekAttention.output", output))
        return output


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        _dbg_print(
            ("DeepseekV2MLP.input.x", x),
            ("DeepseekV2MLP.gate_up_proj", self.gate_up_proj),
        )
        gate_up, _ = self.gate_up_proj(x)
        _dbg_print(("DeepseekV2MLP.gate_up_proj.output", gate_up))
        x = self.act_fn(gate_up)
        _dbg_print(
            ("DeepseekV2MLP.act_fn.output", x),
            ("DeepseekV2MLP.down_proj", self.down_proj),
        )
        x, _ = self.down_proj(x)
        _dbg_print(("DeepseekV2MLP.output", x))
        return x


class DeepseekV2MoE(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config | DeepseekV3Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        if (
            self.is_rocm_aiter_moe_enabled
            and self.gate.e_score_correction_bias is not None
            and self.gate.out_dtype is None
        ):
            # Accumulates in fp32; avoids bf16->fp32 cast.
            self.gate.set_out_dtype(self.gate.weight.dtype)

        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            reduce_results=reduce_results,
            n_shared_experts=config.n_shared_experts
            if self.is_fusion_moe_shared_experts_enabled
            else None,
            router_logits_dtype=self.gate.out_dtype,
        )

        if (
            self.is_rocm_aiter_moe_enabled
            and self.gate.e_score_correction_bias is not None
        ):
            self.gate.e_score_correction_bias.data = (
                self.gate.e_score_correction_bias.data.to(self.gate.out_dtype)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        _moe_params = _get_moe_expert_params(self.experts)
        _dbg_print(
            ("DeepseekV2MoE.input.hidden_states", hidden_states),
            ("DeepseekV2MoE.gate", self.gate),
            (
                "DeepseekV2MoE.gate.e_score_correction_bias",
                getattr(self.gate, "e_score_correction_bias", None),
            ),
            (
                "DeepseekV2MoE.experts.w13_weight",
                _moe_params["w13_weight"],
                _moe_params["w13_scale"],
            ),
            (
                "DeepseekV2MoE.experts.w2_weight",
                _moe_params["w2_weight"],
                _moe_params["w2_scale"],
            ),
        )
        # Emit top-k routing indices/scores over the first 8 tokens BEFORE
        # `self.experts(...)`. This is a naive top-k on raw router_logits
        # (no grouped-topk / e_score_correction_bias applied); good enough
        # to locate the layer where FP8 and W8A8 first disagree on which
        # experts win, which is the usual source of exponential activation
        # drift in deep MoE stacks.
        _dbg_moe_router_topk(
            "DeepseekV2MoE",
            self.gate,
            hidden_states,
            top_k=getattr(
                getattr(self.experts, "moe_config", None),
                "top_k",
                8,
            ),
        )
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
        _dbg_print(
            ("DeepseekV2MoE.experts.output", final_hidden_states),
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        out = final_hidden_states.view(num_tokens, hidden_dim)
        _dbg_print(("DeepseekV2MoE.output", out))
        return out


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    # Broadcast over num_heads and head_dim
    return scaling[..., None, None]


class DeepseekV2Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        assert topk_indices_buffer is None, (
            "topk_indices_buffer is not \
        supported for DeepseekV2Attention"
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        _dbg_print(
            ("DeepseekV2Attention.input.hidden_states", hidden_states),
        )
        if self.q_lora_rank is not None:
            _dbg_print(
                ("DeepseekV2Attention.q_a_proj", self.q_a_proj),
            )
            q = self.q_a_proj(hidden_states)[0]
            _dbg_print(("DeepseekV2Attention.q_a_proj.output", q))
            q = self.q_a_layernorm(q)
            _dbg_print(
                ("DeepseekV2Attention.q_a_layernorm", self.q_a_layernorm),
                ("DeepseekV2Attention.q_a_layernorm.output", q),
                ("DeepseekV2Attention.q_b_proj", self.q_b_proj),
            )
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            _dbg_print(("DeepseekV2Attention.q_b_proj.output", q))
        else:
            _dbg_print(
                ("DeepseekV2Attention.q_proj", self.q_proj),
            )
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            _dbg_print(("DeepseekV2Attention.q_proj.output", q))
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        _dbg_print(
            (
                "DeepseekV2Attention.kv_a_proj_with_mqa",
                self.kv_a_proj_with_mqa,
            ),
        )
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        _dbg_print(
            ("DeepseekV2Attention.kv_a_proj_with_mqa.output", latent_cache),
        )
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        _dbg_print(
            ("DeepseekV2Attention.kv_a_layernorm", self.kv_a_layernorm),
            ("DeepseekV2Attention.kv_a_layernorm.output", kv_a),
            ("DeepseekV2Attention.kv_b_proj", self.kv_b_proj),
        )
        kv = self.kv_b_proj(kv_a)[0]
        _dbg_print(("DeepseekV2Attention.kv_b_proj.output", kv))
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        _dbg_print(
            ("DeepseekV2Attention.rotary_emb.q_pe", q_pe),
            ("DeepseekV2Attention.rotary_emb.k_pe", k_pe),
        )

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        _dbg_print(
            ("DeepseekV2Attention.attn.output", attn_output),
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., : self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        _dbg_print(
            ("DeepseekV2Attention.o_proj", self.o_proj),
        )
        output, _ = self.o_proj(attn_output)
        _dbg_print(("DeepseekV2Attention.output", output))
        return output


class DeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )  # Only has one vector instead of K + V

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV32IndexerBackend


class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
        is_inplace_rope: bool = False,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        assert cache_config is not None
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

        self.is_inplace_rope = is_inplace_rope
        self.n_head_scale = self.n_head**-0.5
        self.use_fused_indexer_q = (
            current_platform.is_cuda()
            and self.quant_block_size == self.head_dim
            and self.head_dim == 128
            and self.rope_dim == 64
            and self.scale_fmt is not None
        )

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        _dbg_print(
            ("Indexer.input.hidden_states", hidden_states),
            ("Indexer.input.qr", qr),
            ("Indexer.wq_b", self.wq_b),
            ("Indexer.wk_weights_proj", self.wk_weights_proj),
            ("Indexer.k_norm", self.k_norm),
        )
        q, _ = self.wq_b(qr)
        _dbg_print(("Indexer.wq_b.output", q))
        q = q.view(-1, self.n_head, self.head_dim)

        if current_platform.is_rocm() and self.is_inplace_rope:
            # This path should works on all platform, will remove extra
            # branches in the future
            # This fast path relies on rotary_emb mutating q and k inplace.
            # On ROCm, this is only valid for kernels used as custom ops.
            # In pytorch-native rope for inductor fusion, rotated q/k tensors
            # are not mutated inplace but returned as new tensors.
            # Fused wk + weights_proj: one GEMM, then split
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]

            k = self.k_norm(k)

            rotary_emb(
                positions, q[..., : self.rope_dim], k[..., : self.rope_dim].unsqueeze(1)
            )
        elif self.use_fused_indexer_q and q.dtype == torch.bfloat16:
            # fused wk + weights_proj: one GEMM, then split
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]

            k = self.k_norm(k)
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )

            q_fp8, weights = fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                weights,
                self.softmax_scale,
                self.n_head_scale,
                rotary_emb.is_neox_style,
            )

            # rotate only the MQA K
            k_pe = k_pe.unsqueeze(1)
            q_dummy = torch.empty_like(k_pe)
            _, k_pe = rotary_emb(positions, q_dummy, k_pe)
            k_pe = k_pe.reshape(-1, self.rope_dim)
            k = torch.cat([k_pe, k_nope], dim=-1)

            _dbg_print(
                ("Indexer.fused.q_fp8", q_fp8),
                ("Indexer.fused.k", k),
                ("Indexer.fused.weights", weights),
            )
            out = self.indexer_op(hidden_states, q_fp8, k, weights)
            _dbg_print(("Indexer.output(fused)", out))
            return out
        else:
            q_pe, q_nope = torch.split(
                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            # Fused wk + weights_proj: one GEMM, then split
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]

            k = self.k_norm(k)
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )

            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
            # Note: RoPE (NeoX) can introduce extra leading dimensions during
            # compilation so we need to reshape back to token-flattened shapes
            q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
            k_pe = k_pe.reshape(-1, self.rope_dim)

            # `rotary_emb` is shape-preserving; `q_pe` is already
            # [num_tokens, n_head, rope_dim].
            q = torch.cat([q_pe, q_nope], dim=-1)
            # `k_pe` is [num_tokens, rope_dim] (MQA).
            k = torch.cat([k_pe, k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head)

        weights = weights * q_scale * self.softmax_scale * self.n_head_scale

        _dbg_print(
            ("Indexer.q_fp8", q_fp8),
            ("Indexer.q_scale", q_scale),
            ("Indexer.k", k),
            ("Indexer.weights", weights),
        )
        out = self.indexer_op(hidden_states, q_fp8, k, weights)
        _dbg_print(("Indexer.output", out))
        return out


def _try_load_fp8_indexer_wk(
    name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
):
    """
    We fuse the WK and weights_proj projections, but in some checkpoints WK is stored
    in FP8 with a separate weight_scale_inv, while weights_proj is stored in BF16.
    Upcasting to BF16 during loading enables the fusion. This function loads the FP8 WK
    weights and scale, and when both are available, dequantizes to BF16 and stores into
    the fused wk_weights_proj.weight parameter.
    """
    if "indexer.wk." not in name or "wk_weights" in name:
        return False  # Weight is not an isolated WK weight for the indexer, ignore.
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale" in name
    if not is_weight and not is_scale:
        return False  # WK is not in FP8 format, ignore.
    # Buffer this tensor (weight or scale) until both have arrived.
    layer_prefix = name.rsplit(".wk.", 1)[0]  # e.g. "model.layers.0.self_attn.indexer"
    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    if any(
        name.startswith(missing_layer_name)
        for missing_layer_name in pp_missing_layer_names
    ):
        return True
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True  # still waiting for the other param

    # We have both weight and scale: dequantize FP8 to BF16.
    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    if scale_inv.ndim == 1:
        # Per-channel scale: one scale per row of [out, in]
        group_shape = GroupShape(1, weight_fp8.shape[1])
    else:
        block_size = weight_fp8.shape[1] // scale_inv.shape[1]
        group_shape = GroupShape(block_size, block_size)
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=group_shape,
        out_dtype=torch.bfloat16,
    )

    # Load the dequantized weight into shard 0 of the fused buffer.
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _min_latency_fused_qkv_a_proj_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Dynamically run min-latency gemm if num_tokens <= 16.
    This must be wrapped in a custom op because our torch.compile integration
    does not support runtime dispatching on num_tokens.
    """
    num_tokens = input_.shape[0]
    if 0 < num_tokens <= 16:
        output = torch.empty(
            num_tokens,
            weight.shape[0],
            dtype=torch.bfloat16,
            device=input_.device,
        )
        ops.dsv3_fused_a_gemm(output, input_, weight.T)
        return output
    else:
        return torch.nn.functional.linear(input_, weight)


def _min_latency_fused_qkv_a_proj_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return input_.new_empty(input_.shape[0], weight.shape[0])


direct_register_custom_op(
    op_name="min_latency_fused_qkv_a_proj",
    op_func=_min_latency_fused_qkv_a_proj_impl,
    mutates_args=[],
    fake_impl=_min_latency_fused_qkv_a_proj_fake,
)


class DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: list[int],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            disable_tp=True,
            prefix=prefix,
        )

        # Check if the DeepSeek V3 fused A GEMM kernel can be used.
        # This kernel supports PDL and is optimized for low batch size.
        self._use_min_latency_gemm = (
            hasattr(self, "weight")
            and self.weight.dtype == torch.bfloat16
            and self.weight.shape[0] == 2112
            and self.weight.shape[1] == 7168
            and current_platform.is_cuda()
            and (
                current_platform.is_device_capability(90)
                or current_platform.is_device_capability_family(100)
            )
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.nn.Parameter | None]:
        if self._use_min_latency_gemm:
            output = torch.ops.vllm.min_latency_fused_qkv_a_proj(input_, self.weight)
            if not self.return_bias:
                return output
            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias
        else:
            # Fallback to the standard forward method when
            # the fused A GEMM kernel cannot be used.
            return super().forward(input_)


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

        For more info see MLACommonImpl in:
        vllm/v1/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        reduce_results: bool = True,
        non_causal_multi_token_decode: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Use input_size for projection input dimensions if provided,
        # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA)
        proj_input_size = input_size if input_size is not None else self.hidden_size

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        qrep_enabled = (
            envs.VLLM_DCP_Q_REPLICATE
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.prefill_context_parallel_size <= 1
        )
        q_proj_cls = (
            DCPGroupColumnParallelLinear if qrep_enabled else ColumnParallelLinear
        )
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = q_proj_cls(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = q_proj_cls(
                proj_input_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.is_v32 = hasattr(config, "index_topk")

        # IndexCache config
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        _skip_topk = False
        is_mtp_layer = False
        if self.is_v32:
            _index_topk_freq = getattr(config, "index_topk_freq", 1)
            _index_topk_pattern = getattr(config, "index_topk_pattern", None)
            _index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
            layer_id = extract_layer_index(prefix)

            if _index_topk_pattern is None:
                _skip_topk = (
                    max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq
                    != 0
                )
            elif 0 <= layer_id < len(_index_topk_pattern):
                _skip_topk = _index_topk_pattern[layer_id] == "S"

            # The skip pattern only governs backbone layers. MTP/nextn
            # layers (layer_id >= num_hidden_layers) always build a full
            # indexer: they compute indices at draft step 0 and toggle
            # at runtime via set_skip_topk
            # (index_share_for_mtp_iteration).
            _num_hidden_layers = getattr(config, "num_hidden_layers", None)
            is_mtp_layer = (
                _num_hidden_layers is not None and layer_id >= _num_hidden_layers
            )

        if self.is_v32 and (not _skip_topk or is_mtp_layer):
            assert q_lora_rank is not None
            assert cache_config is not None
            self.indexer_rope_emb: nn.Module | None
            self.indexer: Indexer | None
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
                is_inplace_rope=self.indexer_rope_emb.enabled(),
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            # MTP layers must never start with skip_topk=True: their indexer
            # computes indices at draft step 0, and the runtime toggle
            # (set_skip_topk, index_share_for_mtp_iteration) only exists in
            # the V1 proposer. A frozen True would leave the draft reading a
            # never-written topk buffer.
            skip_topk=_skip_topk and not is_mtp_layer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
            # Do not skip scoring for MTP layers: their top-k buffer may be
            # reused by later draft iterations through index sharing.
            allow_short_prefill_indexer_scoring_skip=not is_mtp_layer,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        _dbg_print(
            ("DeepseekV2MLAAttention.input.hidden_states", hidden_states),
            (
                "DeepseekV2MLAAttention.fused_qkv_a_proj",
                getattr(self, "fused_qkv_a_proj", None)
                if self.q_lora_rank is not None
                else None,
            ),
            (
                "DeepseekV2MLAAttention.kv_a_proj_with_mqa",
                getattr(self, "kv_a_proj_with_mqa", None)
                if self.q_lora_rank is None
                else None,
            ),
            (
                "DeepseekV2MLAAttention.q_a_layernorm",
                getattr(self, "q_a_layernorm", None)
                if self.q_lora_rank is not None
                else None,
            ),
            (
                "DeepseekV2MLAAttention.q_b_proj",
                getattr(self, "q_b_proj", None)
                if self.q_lora_rank is not None
                else None,
            ),
            (
                "DeepseekV2MLAAttention.q_proj",
                getattr(self, "q_proj", None)
                if self.q_lora_rank is None
                else None,
            ),
            (
                "DeepseekV2MLAAttention.kv_a_layernorm",
                self.kv_a_layernorm,
            ),
            (
                "DeepseekV2MLAAttention.kv_b_proj",
                self.kv_b_proj,
            ),
            (
                "DeepseekV2MLAAttention.o_proj",
                self.o_proj,
            ),
        )
        output = self.mla_attn(positions, hidden_states, llama_4_scaling)
        _dbg_print(("DeepseekV2MLAAttention.output", output))
        return output


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        # verify MLA attention specific fields
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        self.use_mha = use_mha

        attn_cls: typing.Any
        if use_mha:
            attn_cls = DeepseekAttention
        elif model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        # TODO(wentao): enable SP MoE with PP after the PP boundary logic can safely
        # send/receive sequence-parallel hidden_states across stages.
        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )
        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
        )

        if is_moe_layer:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                # aiter applies routed_scaling_factor internally
                apply_routed_scale_to_output=not rocm_aiter_ops.is_fused_moe_enabled(),
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _dbg_msg(
            f">>> DeepseekV2DecoderLayer starting layer_idx={self.layer_idx} "
            f"(use_mha={self.use_mha}, "
            f"mlp={type(self.mlp).__name__}, "
            f"attn={type(self.self_attn).__name__})"
        )
        _dbg_print(
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}].input.hidden_states",
                hidden_states,
            ),
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}].input.residual",
                residual,
            ),
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}].input_layernorm",
                self.input_layernorm,
            ),
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}]."
                "post_attention_layernorm",
                self.post_attention_layernorm,
            ),
        )
        full_num_tokens = positions.shape[0]
        input_is_sequence_parallel = (
            self.use_sequence_parallel_moe
            and residual is not None
            and hidden_states.shape[0] != full_num_tokens
        )

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        _dbg_print(
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}].input_layernorm.output",
                hidden_states,
            ),
        )

        if input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.use_mha:
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
        _dbg_print(
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}].self_attn.output",
                hidden_states,
            ),
        )

        if (
            not isinstance(self.self_attn, DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        if self.use_sequence_parallel_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            # small trick using minus, eg. -17 % 8 = 7
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            # pad if not divisible by world size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                residual = sequence_parallel_chunk(residual)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        _dbg_print(
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}]."
                "post_attention_layernorm.output",
                hidden_states,
            ),
        )
        if self.use_sequence_parallel_moe:
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1.0 / self.routed_scaling_factor

        _dbg_print(
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}].output.hidden_states",
                hidden_states,
            ),
            (
                f"DeepseekV2DecoderLayer[{self.layer_idx}].output.residual",
                residual,
            ),
        )
        return hidden_states, residual


@support_torch_compile
class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                self.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )

        self.aux_hidden_state_layers = tuple[int, ...]()

        # Needed by load_weights
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to DeepseekV2Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # all gather if we need to use the whole states
            if (
                hidden_states.shape[0] != positions.shape[0]
                and not layer.use_sequence_parallel_moe
            ):
                combined_states = torch.cat([hidden_states, residual], dim=-1)
                combined_states = tensor_model_parallel_all_gather(combined_states, 0)
                combined_states = combined_states[: positions.shape[0]]
                hidden_states, residual = combined_states.split(
                    [self.hidden_size, self.hidden_size], dim=-1
                )
            if idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual
                if aux_hidden_state.shape[0] != positions.shape[0]:
                    aux_hidden_state = tensor_model_parallel_all_gather(
                        aux_hidden_state, 0
                    )
                    aux_hidden_state = aux_hidden_state[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != positions.shape[0]:
            combined_states = torch.cat([hidden_states, residual], dim=-1)
            combined_states = tensor_model_parallel_all_gather(combined_states, 0)
            combined_states = combined_states[: positions.shape[0]]
            hidden_states, residual = combined_states.split(
                [self.hidden_size, self.hidden_size], dim=-1
            )

        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        mla_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        mha_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        # Fused indexer wk + weights_proj (shard 0 = wk, shard 1 = weights_proj)
        _pending_wk_fp8 = getattr(self, "_pending_indexer_wk_fp8", None)
        if _pending_wk_fp8 is None:
            self._pending_indexer_wk_fp8 = _pending_wk_fp8 = {}

        indexer_fused_mapping = [
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        stacked_params_mapping.extend(indexer_fused_mapping)

        if self.use_mha:
            stacked_params_mapping.extend(mha_params_mapping)
        else:
            stacked_params_mapping.extend(mla_params_mapping)

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )

        pp_missing_layer_names = get_pp_missing_layer_names(self)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # With index_topk_freq>1 only some layers build an indexer, yet the
        # checkpoint ships indexer weights for all of them; track the built ones.
        indexer_present_prefixes = {
            n.rsplit(".indexer.", 1)[0] for n in params_dict if ".indexer." in n
        }
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            if ".indexer." in name and (
                name.rsplit(".indexer.", 1)[0] not in indexer_present_prefixes
            ):
                continue  # this layer has no indexer; drop its checkpoint weights

            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            )

            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                        if remapped_name is None:
                            continue
                        name = remapped_name

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if name is not None and not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params


class DeepseekV2MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekV2MoE]
    """
    List of MoE MLP layers in the model.
    """

    def extract_moe_parameters(self, example_moe: DeepseekV2MoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("DeepSeekV2: No DeepseekV2MoE layer found in model.layers.")
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV2ForCausalLM(
    nn.Module,
    SupportsPP,
    DeepseekV2MixtureOfExperts,
    SupportsLoRA,
    SupportsEagle,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls = DeepseekV2Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        if self.use_mha:
            self.packed_modules_mapping["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]

        # `packed_modules_mapping` needs to be modified before
        # initializing DeepseekV2Model, as it is passed inplace to
        # quantization config init and may be used to select the
        # quant_method for relevant layers during initialization.
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Set MoE hyperparameters
        self.num_moe_layers = (
            self.config.num_hidden_layers - self.config.first_k_dense_replace
        )
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class DeepseekForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass
