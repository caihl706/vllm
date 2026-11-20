# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MLAAttentionSpec
from vllm.v1.worker.utils import customize_attention_spec

from .attention import HYV4MLAAttention
from .dfly import HYV4DFlyForCausalLM


def build_reuse_indices(
    target_rows: torch.Tensor,
    anchor_positions: torch.Tensor,
    query_width: int,
    local_window: int,
    active: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Build logical context and causal query indices, without a history scan."""
    batch, topk = target_rows.shape
    rows = out[: batch * query_width].view(batch, query_width, -1)
    rows.fill_(-1)
    starts = anchor_positions[:, None] - local_window
    history = torch.where((target_rows >= 0) & (target_rows < starts), target_rows, -1)
    rows[:, :, :topk] = history[:, None, :]
    if local_window:
        window = starts + torch.arange(local_window, device=out.device)
        rows[:, :, topk : topk + local_window] = window.clamp(min=-1)[:, None, :]
    offsets = torch.arange(query_width, device=out.device)
    block = anchor_positions[:, None] + offsets
    causal = offsets[None, :] <= offsets[:, None]
    rows[:, :, topk + local_window : topk + local_window + query_width] = torch.where(
        causal[None, :, :], block[:, None, :], -1
    )
    rows.masked_fill_(~active[:, None, None], -1)


def query_slot_mapping(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    active: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Resolve batch-ordered query slots without writing invalid requests."""
    blocks = torch.div(positions, block_size, rounding_mode="floor")
    valid = active[:, None] & (positions >= 0) & (positions < max_model_len)
    valid &= blocks < block_table.shape[1]
    physical = block_table.gather(1, blocks.clamp(0, block_table.shape[1] - 1).long())
    slots = physical.long() * block_size + positions.remainder(block_size)
    return torch.where(valid, slots, PAD_SLOT_ID).flatten()


class HYV4DFlyTargetReuse:
    def __init__(
        self, target: nn.Module, draft: HYV4DFlyForCausalLM, vllm_config: VllmConfig
    ):
        config = draft.config
        if vllm_config.cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8_ds_mla",
        ):
            raise ValueError("DFly requires BF16 or fp8_ds_mla MLA KV cache")
        language_model = (
            target.get_language_model()
            if hasattr(target, "get_language_model")
            else target
        )
        target_layers = language_model.model.layers
        self.index_layers: dict[int, HYV4MLAAttention] = {}
        self.layer_pairs: list[tuple[HYV4MLAAttention, HYV4MLAAttention]] = []
        self.aliases: dict[str, str] = {}
        self.draft = draft
        self.config = config

        def target_attention(index: int) -> HYV4MLAAttention:
            attn = target_layers[index].self_attn
            if not isinstance(attn, HYV4MLAAttention):
                raise ValueError(f"DFly target layer {index} is not HYV4 MLA")
            return attn

        for index in config.target_index_layer:
            attn = target_attention(index)
            if (
                attn.indexer is None
                or attn.skip_topk
                or attn.topk_indices_buffer is None
            ):
                raise ValueError(
                    f"DFly target index layer {index} must run its own indexer"
                )
            if attn.topk_indices_buffer.shape[-1] != config.target_index_topk:
                raise ValueError(
                    f"DFly target index layer {index} has incompatible top-k width"
                )
            self.index_layers[index] = attn
        for layer, index in zip(draft.model.layers, config.target_kv_layers):
            source, dest = target_attention(index), layer.self_attn
            if type(source.rotary_emb) is not type(dest.rotary_emb):
                raise ValueError(f"DFly target KV layer {index} uses incompatible RoPE")
            if source.num_local_heads != dest.num_local_heads:
                raise ValueError("DFly target and draft must use matching attention TP")
            storage_specs: list[MLAAttentionSpec] = []
            for attn in (source, dest):
                spec = attn.mla_attn.get_kv_cache_config(vllm_config).spec
                if not isinstance(spec, MLAAttentionSpec):
                    raise ValueError("DFly requires MLA KV cache specs")
                customized_spec = customize_attention_spec(
                    attn.mla_attn.get_attn_backend(), spec, attn.mla_attn.layer_name
                )
                if not isinstance(customized_spec, MLAAttentionSpec):
                    raise ValueError("DFly backend must preserve the MLA KV cache spec")
                storage_specs.append(
                    replace(
                        customized_spec,
                        topk_buffer_width=None,
                        is_index_group_leader=False,
                    )
                )
            if storage_specs[0] != storage_specs[1]:
                raise ValueError(
                    f"DFly target KV layer {index} has incompatible cache spec"
                )
            if (
                source.learnable_sink
                and not source.mla_attn.get_attn_backend().supports_sink()
            ):
                raise ValueError("DFly target backend must apply its learnable sink")
            self.layer_pairs.append((source, dest))
            self.aliases[dest.mla_attn.layer_name] = source.mla_attn.layer_name
        if len(set(self.aliases.values())) != len(self.aliases):
            raise ValueError("DFly needs a distinct target KV layer per draft layer")
        for attn in self.index_layers.values():
            attn.topk_snapshot = torch.full_like(attn.topk_indices_buffer, -1)
        self._set_storage_aliases(True)

    def _set_storage_aliases(self, enabled: bool) -> None:
        """Toggle the KV-sharing alias on the draft layers.

        The alias makes KV cache allocation bind each draft layer to its target
        layer's storage, but MLA also skips the KV write for sharing layers. The
        draft must write its query block KV, so the alias is only kept while the
        KV cache is being allocated and bound.
        """
        for source, dest in self.layer_pairs:
            dest.mla_attn.kv_sharing_target_layer_name = (
                source.mla_attn.layer_name if enabled else None
            )

    def validate_groups(self, config: KVCacheConfig) -> None:
        self._set_storage_aliases(True)
        groups = {
            name: i
            for i, group in enumerate(config.kv_cache_groups)
            for name in group.layer_names
        }
        for draft, target in self.aliases.items():
            if draft not in groups or groups[draft] != groups.get(target):
                raise ValueError(
                    f"DFly alias {draft} and {target} must share a KV group"
                )

    def validate_storage(self) -> None:
        for source, dest in self.layer_pairs:
            a, b = source.mla_attn.kv_cache, dest.mla_attn.kv_cache
            if (
                a.numel() == 0
                or a.data_ptr() != b.data_ptr()
                or a.shape != b.shape
                or a.stride() != b.stride()
                or a.dtype != b.dtype
            ):
                raise ValueError(
                    "DFly query and target KV must alias the same local layout"
                )
        self._set_storage_aliases(False)

    def update_indices(
        self,
        source_rows: torch.Tensor,
        anchor_positions: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        for i, index in enumerate(self.config.target_index_layer):
            snapshot = self.index_layers[index].topk_snapshot
            assert snapshot is not None
            build_reuse_indices(
                snapshot[source_rows],
                anchor_positions,
                self.draft.model.query_width,
                self.config.context_local_window or 0,
                active,
                self.draft.model.topk_indices[i],
            )
