# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm import envs
from vllm.config import ModelConfig, ParallelConfig, VllmConfig, replace
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4(.1) draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def _get_dspark_parallel_config(
    parallel_config: ParallelConfig,
    tensor_parallel_size: int,
) -> ParallelConfig:
    if parallel_config.enable_eplb:
        logger.warning_once(
            "EPLB is disabled for the DSpark draft model. EPLB remains enabled "
            "for the target model."
        )

    return replace(
        parallel_config,
        pipeline_parallel_size=1,
        tensor_parallel_size=tensor_parallel_size,
        enable_eplb=False,
        eplb_config=replace(
            parallel_config.eplb_config,
            num_redundant_experts=0,
        ),
        enable_elastic_ep=False,
    )


# Probe names on the *checkpoint* side of the naming divide. ``_remap_dspark_name``
# rewrites ``mtp.{i}.*`` onto ``model.layers.{i}.*``, so these never name a real
# module at runtime -- which is exactly why a config_groups target keyed on them
# is inert, and why the tier can be recovered from the config even though the
# scheme it describes cannot be applied through it.
_MTP_PROBES = tuple(
    f"{root}.0.{stack}.experts.0.{proj}"
    for root in ("mtp", "stages")
    for stack in ("ffn", "mlp")
    for proj in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3")
)

# Probe names on the *runtime* side, i.e. what ``get_scheme_dict`` is actually
# called with: ``<routed-experts module>.<i>.<proj>``.
_EXPERTS_PROBES = tuple(
    f"model.layers.0.ffn.experts.0.{proj}"
    for proj in ("gate_proj", "up_proj", "down_proj")
)


def _matches_any_probe(pattern: str, probes: tuple[str, ...]) -> bool:
    """True if ``pattern`` is a config target/ignore entry matching a probe.

    Uses vllm's own matcher so this cannot drift from the runtime's notion of
    which layers a pattern selects. Compile the pattern and match it -- never
    substring-match the pattern's source text, whose regex escapes and
    alternations do not appear literally in it.
    """
    from vllm.model_executor.layers.quantization.utils.config_utils import (
        is_equal_or_regex_match,
    )

    return any(is_equal_or_regex_match(p, pattern) for p in probes)


def _infer_draft_moe_tier(quant_config) -> str | None:
    """Recover the draft's MTP expert quantization tier from its own config.

    Returns ``"int4"``, ``"int8"``, ``"bf16"``, or ``None`` when the config
    does not say. A checkpoint written by a quantizer that gives the MTP subtree
    its own ``config_groups`` entry records the tier there; one that leaves the
    subtree unquantized records it in ``ignore`` instead.
    """
    tiers: set[str] = set()
    unknown: list[str] = []
    for target, scheme in getattr(quant_config, "target_scheme_map", {}).items():
        if not isinstance(target, str) or not isinstance(scheme, dict):
            continue
        if not _matches_any_probe(target, _MTP_PROBES):
            continue
        num_bits = getattr(scheme.get("weights"), "num_bits", None)
        fmt = scheme.get("format")
        if num_bits == 4 and fmt == "pack-quantized":
            tiers.add("int4")
        elif num_bits == 8 and fmt == "int-quantized":
            tiers.add("int8")
        else:
            unknown.append(f"{target!r} (bits={num_bits}, format={fmt!r})")

    if unknown or len(tiers) > 1:
        # An MTP entry in a format this does not model, or several
        # disagreeing entries. Do not guess.
        return None
    if tiers:
        return tiers.pop()

    # No MTP-targeted scheme at all: the experts are stored unquantized only
    # if the MTP subtree is explicitly ignored.
    ignore = getattr(quant_config, "ignore", None) or ()
    if any(
        isinstance(p, str) and _matches_any_probe(p, _MTP_PROBES) for p in ignore
    ):
        return "bf16"
    return None


def _apply_draft_moe_scheme_override(quant_config) -> None:
    """Give the DSpark draft the expert scheme its own checkpoint was written with.

    The DSpark draft model is built from the *target* checkpoint and
    ``_remap_dspark_name`` rewrites ``mtp.{i}.*`` weights onto
    ``model.layers.{i}.*`` parameters, so the draft's routed-expert modules
    are named exactly like the target's. Compressed-tensors resolves a scheme
    by matching ``config_groups`` targets against those module names and
    taking the first hit, so a per-MTP ``config_groups`` entry keyed on the
    checkpoint's ``mtp.{i}.`` prefix never matches, and the draft silently
    inherits the target's expert scheme. A checkpoint whose MTP experts use
    a different scheme then fails to load, because the expert parameters exist
    under the target scheme's names only.

    The draft has its own ``CompressedTensorsConfig`` instance, so the tie can
    be broken here without touching the target's config. The tier is inferred
    from that config; ``VLLM_DSPARK_MOE_QUANT`` overrides the inference.
    Mutates in place.
    """
    override = envs.VLLM_DSPARK_MOE_QUANT.strip().lower()
    if override and override not in ("int4", "int8", "bf16"):
        raise ValueError(
            f"Invalid VLLM_DSPARK_MOE_QUANT={envs.VLLM_DSPARK_MOE_QUANT!r}; "
            "expected one of 'int4', 'int8', 'bf16', or unset."
        )

    if override:
        tier = override
        logger.info_once(
            "DSpark draft MTP expert tier taken from VLLM_DSPARK_MOE_QUANT: %s",
            tier,
        )
    else:
        tier = _infer_draft_moe_tier(quant_config)
        if tier is None:
            logger.warning_once(
                "Could not infer the DSpark draft's MTP expert quantization "
                "tier: its quant config has no config_groups entry targeting "
                "the MTP experts, and does not ignore them either. Leaving the "
                "draft's expert scheme as inherited from the target "
                "checkpoint. If this checkpoint stores its MTP experts in a "
                "different scheme than the target's, loading will fail with a "
                "parameter-name KeyError; set VLLM_DSPARK_MOE_QUANT to int4, "
                "int8 or bf16 to say so explicitly."
            )
            return
        logger.info_once(
            "Inferred DSpark draft MTP expert tier from its quant config: %s",
            tier,
        )

    # Requiring a literal dot before "experts" keeps this off
    # ``shared_experts``, which the same config may quantize.
    experts_ignore_re = r"re:.*\.experts\.\d+\..*"

    if tier == "bf16":
        ignore = getattr(quant_config, "ignore", None)
        if ignore is None:
            ignore = []
            quant_config.ignore = ignore
        elif not isinstance(ignore, list):
            ignore = list(ignore)
            quant_config.ignore = ignore
        if experts_ignore_re not in ignore:
            ignore.append(experts_ignore_re)
        logger.info_once(
            "DSpark draft routed experts will be loaded as BF16 "
            "(UnquantizedFusedMoEMethod)."
        )
        return

    from compressed_tensors.quantization import QuantizationArgs

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    if not isinstance(quant_config, CompressedTensorsConfig):
        raise ValueError(
            "VLLM_DSPARK_MOE_QUANT only supports the compressed-tensors "
            f"quantization method, got {type(quant_config).__name__}."
        )

    num_bits = 4 if tier == "int4" else 8
    changed: list[str] = []
    for target, scheme in quant_config.target_scheme_map.items():
        if not isinstance(scheme, dict) or "weights" not in scheme:
            continue
        if not isinstance(target, str):
            continue
        if not _matches_any_probe(target, _EXPERTS_PROBES):
            continue
        scheme["weights"] = QuantizationArgs(
            num_bits=num_bits,
            type="int",
            symmetric=True,
            strategy="channel",
            group_size=-1,
            dynamic=False,
        )
        scheme["format"] = "pack-quantized" if num_bits == 4 else "int-quantized"
        changed.append(target)

    if not changed:
        raise ValueError(
            f"DSpark draft MTP tier resolved to {tier!r}, but no config_groups "
            f"target matched the probes {_EXPERTS_PROBES!r}, so nothing was "
            f"rewritten. Targets present: "
            f"{sorted(quant_config.target_scheme_map)!r}"
        )
    logger.info_once(
        "DSpark draft routed experts will be loaded as INT%d; rewrote %d "
        "config_groups target(s): %s",
        num_bits,
        len(changed),
        # *_once wraps an lru_cache, so every argument must be hashable.
        tuple(changed),
    )


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
        maybe_share_target_embed,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        parallel_config=_get_dspark_parallel_config(
            vllm_config.parallel_config,
            speculative_config.draft_parallel_config.tensor_parallel_size,
        ),
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(vllm_config.load_config),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)
    if draft_vllm_config.quant_config is not None:
        _apply_draft_moe_scheme_override(draft_vllm_config.quant_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()

    if draft_model_config.get_vocab_size() <= target_vocab_size:
        maybe_share_target_embed(draft_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
