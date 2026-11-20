# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from copy import copy
from typing import Any

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    get_target_lm_head,
    maybe_share_target_embed,
    share_target_lm_head,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

from .dfly import HYV4DFlyForCausalLM
from .dfly_target_reuse import HYV4DFlyTargetReuse, query_slot_mapping

logger = init_logger(__name__)


class HYV4DFlyQueryModel(nn.Module):
    """Fixed-shape query computation; attention keeps its existing eager breaks."""

    def __init__(self, model: HYV4DFlyForCausalLM, draft_tokens: torch.Tensor):
        super().__init__()
        self.model = model
        self.query_width = model.model.query_width
        self.register_buffer("draft_tokens", draft_tokens, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        batch = active.shape[0]
        embeddings = self.model.embed_input_ids(input_ids)
        embeddings[:: self.query_width] = self.model.model.mtp_fusion(
            embeddings[:: self.query_width], target_hidden
        )
        hidden = self.model(
            input_ids=input_ids, positions=positions, inputs_embeds=embeddings
        ).view(batch, self.query_width, -1)
        previous = input_ids.view(batch, self.query_width)[:, 0].long()
        for step in range(self.query_width - 1):
            corrected = self.model.apply_hidden_correction(
                hidden[:, step + 1], previous
            )
            previous = greedy_argmax(self.model.compute_logits(corrected))
            self.draft_tokens[:batch, step] = torch.where(active, previous, -1)
        return self.draft_tokens[:batch]


class HYV4DFlySpeculator(DraftModelSpeculator):
    can_reuse_target_attn_metadata = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla"):
            raise ValueError("DFly requires BF16 or fp8_ds_mla MLA KV cache")
        spec = vllm_config.speculative_config
        assert spec is not None
        draft_cache_dtype = spec.kv_cache_dtype or cache_dtype
        target_dtype = "bfloat16" if cache_dtype == "auto" else cache_dtype
        draft_dtype = "bfloat16" if draft_cache_dtype == "auto" else draft_cache_dtype
        if draft_dtype != target_dtype:
            raise ValueError(
                "DFly draft and target must use the same MLA KV cache dtype; "
                "omit speculative_config.kv_cache_dtype to inherit the target"
            )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError("DFly does not support PD disaggregation")
        if (
            vllm_config.lora_config is not None
            or vllm_config.parallel_config.use_ubatching
        ):
            raise ValueError("DFly does not support LoRA or microbatching")
        super().__init__(vllm_config, device)
        self.num_query_per_req = self.num_speculative_steps + 1
        self.query_offsets = torch.arange(self.num_query_per_req, device=device)
        self.request_offsets = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.target_hidden_states = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        self.active_requests = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        self.mrope_positions = (
            torch.zeros(
                self.draft_model_config.mrope_num_dims,
                self.max_num_tokens,
                dtype=torch.int64,
                device=device,
            )
            if self.uses_mrope
            else None
        )
        self.query_model: HYV4DFlyQueryModel | None = None
        self.cudagraph_manager: CudaGraphManager | None = None
        self.reuse: HYV4DFlyTargetReuse | None = None
        self._storage_validated = False
        self._slot_mappings: torch.Tensor | None = None

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        config = super().attn_vllm_config
        # Keep equivalent auto/BF16 overrides identical in the shared cache spec.
        config.cache_config = self.vllm_config.cache_config
        config.model_config = self.draft_model_config
        config.quant_config = VllmConfig.get_quantization_config(
            self.draft_model_config, get_draft_load_config(config)
        )
        config.parallel_config = self.speculative_config.draft_parallel_config
        config.attention_config = copy(config.attention_config)
        config.attention_config.sparse_mla_force_mqa = True
        config.attention_config.use_non_causal = False
        draft = get_model(
            vllm_config=config,
            model_config=self.draft_model_config,
            load_config=get_draft_load_config(config),
            prefix="dfly",
        )
        if not isinstance(draft, HYV4DFlyForCausalLM):
            raise TypeError("DFly speculator requires HYV4DFlyForCausalLM")
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        maybe_share_target_embed(draft, draft.model, language_model.model)
        head = share_target_lm_head(
            draft,
            get_target_lm_head(target_model, language_model),
            quantization=None,
        )
        if head is None or draft.model.embed_tokens is None:
            raise ValueError("DFly requires compatible target embedding and LM head")
        if (
            head.weight.dtype != self.dtype
            or draft.model.embed_tokens.weight.dtype != self.dtype
        ):
            raise ValueError("DFly requires BF16 shared embedding and LM head weights")
        draft.logits_processor.head_dtype = self.vllm_config.model_config.head_dtype
        self.reuse = HYV4DFlyTargetReuse(target_model, draft, self.vllm_config)
        self.query_model = HYV4DFlyQueryModel(draft, self.draft_tokens)
        logger.info(
            "HYV4 DFly: %d layers, N=%d, TP=%d, target index/KV=%s/%s, "
            "%s writable KV aliases, max length=%d, draft quantization=%s",
            len(draft.model.layers),
            self.num_speculative_steps,
            config.parallel_config.tensor_parallel_size,
            draft.config.target_index_layer,
            draft.config.target_kv_layers,
            config.cache_config.cache_dtype,
            self.max_model_len,
            self.draft_model_config.quantization or "none",
        )
        return draft

    @property
    def attn_vllm_config(self) -> VllmConfig:
        config = super().attn_vllm_config
        config.cache_config = self.vllm_config.cache_config
        config.model_config = copy(self.draft_model_config)
        config.quant_config = self.model.model.quant_config
        config.model_config.hf_config = self.model.config
        config.model_config.hf_text_config = self.model.config
        config.attention_config = copy(config.attention_config)
        config.attention_config.sparse_mla_force_mqa = True
        config.attention_config.use_non_causal = False
        return config

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
        kernel_block_sizes: list[int],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
            kernel_block_sizes,
        )
        assert self.reuse is not None
        self.reuse.validate_groups(kv_cache_config)
        self._storage_validated = False
        self._slot_mappings = torch.full(
            (len(kv_cache_config.kv_cache_groups), self.max_num_tokens),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=self.device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        config = copy(self.vllm_config)
        config.compilation_config = copy(config.compilation_config)
        compilation = config.compilation_config
        query_width = self.num_query_per_req
        max_reqs = min(self.max_num_reqs, self.max_num_tokens // query_width)
        request_sizes = set(compilation.decode_cudagraph_capture_request_sizes or ())
        request_sizes.update(
            min(cdiv(size, query_width), max_reqs)
            for size in compilation.prefill_cudagraph_capture_token_sizes or ()
        )
        compilation.prefill_cudagraph_capture_token_sizes = sorted(
            n * query_width for n in request_sizes if 0 < n <= max_reqs
        )
        compilation.use_breakable_cudagraph = True
        mode = (
            CUDAGraphMode.NONE
            if cudagraph_mode == CUDAGraphMode.NONE
            else CUDAGraphMode.PIECEWISE
        )
        self.cudagraph_manager = CudaGraphManager(
            config, self.device, mode, decode_query_lens={query_width: max_reqs}
        )
        if mode != CUDAGraphMode.NONE:
            assert self.query_model is not None
            self.cudagraph_manager.init_breakable_cg_runner(self.query_model)
            logger.info(
                "DFly CUDA graphs: breakable PIECEWISE, query token buckets=%s; "
                "sparse attention and metadata remain eager",
                compilation.prefill_cudagraph_capture_token_sizes,
            )

    @torch.inference_mode()
    def capture(self) -> None:
        manager = self.cudagraph_manager
        if manager is None or not manager.needs_capture():
            return
        slot_mappings = self._slot_mappings
        assert slot_mappings is not None
        assert self.reuse is not None
        self.reuse.validate_storage()

        def create_forward_fn(
            desc: BatchExecutionDescriptor, warmup: bool
        ) -> Callable[[CUDAGraphMode], None]:
            count = desc.num_tokens
            batch = count // self.num_query_per_req
            self._reset_query_buffers(batch)
            self.block_tables.get_dummy_block_tables(batch)
            slot_mappings.fill_(PAD_SLOT_ID)
            self.model.model.topk_indices.fill_(-1)
            self.draft_max_seq_len = self.num_query_per_req
            metadata, slots = self._query_attention_state(
                count, torch.zeros(batch, dtype=torch.int32, device="cpu")
            )

            def forward(cg_mode: CUDAGraphMode) -> None:
                self._run_query_model(count, metadata, slots, cg_mode)

            return forward

        manager.capture(
            create_forward_fn, progress_bar_desc="Capturing DFly CUDA graphs"
        )

    def _reset_query_buffers(self, batch: int) -> None:
        count = batch * self.num_query_per_req
        ids = self.input_buffers.input_ids[:count].view(batch, self.num_query_per_req)
        ids.fill_(self.model.config.mask_token_id)
        ids[:, 0].zero_()
        self.input_buffers.positions[:count].view(batch, -1).copy_(self.query_offsets)
        self.input_buffers.query_start_loc[: batch + 1].copy_(
            self.request_offsets[: batch + 1] * self.num_query_per_req
        )
        self.input_buffers.seq_lens[:batch].fill_(self.num_query_per_req)
        self.input_buffers.is_padding[:count].fill_(True)
        self.target_hidden_states[:batch].zero_()
        self.active_requests[:batch].zero_()
        if self.mrope_positions is not None:
            self.mrope_positions[:, :count].zero_()

    def _query_attention_state(
        self, count: int, seq_lens_cpu_upper_bound: torch.Tensor
    ) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor]]:
        assert self._slot_mappings is not None
        batch = count // self.num_query_per_req
        metadata = self._build_attn_metadata(
            num_reqs=batch,
            batch_desc=BatchExecutionDescriptor(CUDAGraphMode.NONE, count, batch),
            query_start_loc_np=self.arange_np[: batch + 1] * self.num_query_per_req,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=self.num_query_per_req,
            causal=True,
            slot_mappings=self._slot_mappings[:, :count],
        )
        slots = build_slot_mappings_by_layer(
            self._slot_mappings[:, :count], self.kv_cache_config
        )
        return metadata, slots

    def _run_query_model(
        self,
        count: int,
        metadata: dict[str, Any] | None,
        slots: dict[str, torch.Tensor] | None,
        cg_mode: CUDAGraphMode,
    ) -> None:
        assert self.query_model is not None
        batch = count // self.num_query_per_req
        positions = (
            self.input_buffers.positions[:count]
            if self.mrope_positions is None
            else self.mrope_positions[:, :count]
        )
        model_inputs = dict(
            input_ids=self.input_buffers.input_ids[:count],
            positions=positions,
            target_hidden=self.target_hidden_states[:batch],
            active=self.active_requests[:batch],
        )
        with set_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=count,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=BatchDescriptor(num_tokens=count),
            slot_mapping=slots,
            is_padding=self.input_buffers.is_padding[:count],
        ):
            if cg_mode == CUDAGraphMode.PIECEWISE:
                assert self.cudagraph_manager is not None
                self.cudagraph_manager.run_pw_graph(self.query_model, model_inputs)
            else:
                self.query_model(**model_inputs)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        batch = input_batch.num_reqs
        query_width = self.num_query_per_req
        count = batch * query_width
        if count > self.max_num_tokens:
            raise ValueError("DFly query block exceeds max_num_batched_tokens")
        result = self.draft_tokens[:batch]
        if batch == 0 or num_speculative_tokens == 0:
            return result.fill_(-1)
        assert self.reuse is not None
        descriptor = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, batch)
        if not dummy_run and self.cudagraph_manager is not None:
            descriptor = self.cudagraph_manager.dispatch(
                batch, count, uniform_token_count=query_width, num_active_loras=0
            )
        padded_count = descriptor.num_tokens
        assert padded_count % query_width == 0
        padded_batch = padded_count // query_width
        self._reset_query_buffers(padded_batch)
        self.model.model.topk_indices[:, count:padded_count].fill_(-1)
        if dummy_run:
            self.model.model.topk_indices[:, :count].fill_(-1)
        else:
            if not self._storage_validated:
                self.reuse.validate_storage()
                self._storage_validated = True
            source_rows = (
                input_batch.query_start_loc[1 : batch + 1] - num_rejected[:batch] - 1
            ).long()
            anchor_positions = input_batch.positions[source_rows] + 1
            active = (num_sampled[:batch] > 0) & (
                anchor_positions + query_width <= self.max_model_len
            )
            self.active_requests[:batch].copy_(active)
            ids = self.input_buffers.input_ids[:count].view(batch, query_width)
            ids[:, 0] = torch.where(active, last_sampled[input_batch.idx_mapping, 0], 0)
            target_hidden = last_hidden_states[source_rows]
            if target_hidden.shape[-1] != self.hidden_size:
                raise ValueError(
                    "DFly fusion requires the target's final hidden states"
                )
            self.target_hidden_states[:batch].copy_(target_hidden)
            self.reuse.update_indices(source_rows, anchor_positions, active)
            positions = torch.where(
                active[:, None],
                anchor_positions[:, None] + self.query_offsets,
                self.query_offsets[None, :],
            )
            self.input_buffers.positions[:count].copy_(positions.flatten())
            self.input_buffers.seq_lens[:batch].copy_(positions[:, -1] + 1)
            self.input_buffers.is_padding[:count].copy_(
                (~active[:, None]).expand(-1, query_width).flatten()
            )
            if self.mrope_positions is not None:
                delta = async_tensor_h2d(
                    [
                        self.model_state.get_mrope_position_delta(int(i)) or 0
                        for i in input_batch.idx_mapping_np
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                rope = torch.where(active[:, None], positions + delta[:, None], 0)
                self.mrope_positions[:, :count].copy_(rope.flatten())

        metadata = None
        draft_slots = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert self._slot_mappings is not None
            self._slot_mappings[:, :padded_count].fill_(PAD_SLOT_ID)
            positions = self.input_buffers.positions[:count].view(batch, query_width)
            for gid, groups in enumerate(self.attn_groups):
                if groups:
                    self._slot_mappings[gid, :count] = query_slot_mapping(
                        positions,
                        self.block_tables.input_block_tables[gid][:batch],
                        self.block_tables.block_sizes[gid],
                        self.active_requests[:batch],
                        self.max_model_len,
                    )
            upper_bound = torch.zeros(padded_batch, dtype=torch.int32, device="cpu")
            upper_bound[:batch].copy_(input_batch.seq_lens_cpu_upper_bound[:batch])
            self.draft_max_seq_len = min(
                int(upper_bound.max()) + query_width, self.max_model_len
            )
            metadata, draft_slots = self._query_attention_state(
                padded_count, upper_bound
            )

        self._run_query_model(padded_count, metadata, draft_slots, descriptor.cg_mode)
        return result
