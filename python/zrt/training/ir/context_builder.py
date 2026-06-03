"""Build TransformContext from ModelSpec + SystemSpec + Strategy.

将路径 B（配置建模）的 spec 类型转换为两条路径共享的 TransformContext，
使路径 A（抓图建模）的 Transform Pipeline 能消费 spec-driven 配置。

Usage::

    from zrt.training.ir.context_builder import build_context
    ctx = build_context(model, system, strategy)
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec

if TYPE_CHECKING:
    from zrt.hardware.spec import HardwareSpec

logger = logging.getLogger(__name__)


def build_context(
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
    *,
    pp_mode: str = "formula",
    fusion_config: Any = None,
    quant_profile: Any = None,
) -> Any:
    """Build a TransformContext from spec types.

    Parameters
    ----------
    model : ModelSpec
        Model architecture specification.
    system : SystemSpec
        Hardware system specification.
    strategy : Strategy
        Parallel training strategy.
    pp_mode : str
        Pipeline modelling mode: ``"formula"`` (default for spec path)
        or ``"trace"`` (grid-based PPStitcher).
    fusion_config : FusionConfig, optional
        Fusion configuration.  ``None`` → default ``FusionConfig()``.
    quant_profile : GraphQuantProfile, optional
        Structured quantization profile.

    Returns
    -------
    TransformContext
    """
    from zrt.transform.context import (
        TransformContext,
        ParallelConfig,
        TrainingConfig,
        FusionConfig,
    )

    hw_spec = _resolve_hw_spec(system)

    parallel = ParallelConfig(
        tp=strategy.tp,
        pp=strategy.pp,
        ep=strategy.ep,
        dp=strategy.dp,
        cp=strategy.cp,
    )

    cp_kind = "none"
    if strategy.cp > 1:
        cp_kind = getattr(strategy, "cp_kind", "ulysses")
        if hasattr(cp_kind, "value"):
            cp_kind = cp_kind.value

    opt_str = "adam"
    if hasattr(strategy, "optimizer"):
        opt = strategy.optimizer
        opt_str = opt.value if hasattr(opt, "value") else str(opt)

    training = TrainingConfig(
        optimizer=opt_str,
        zero_stage=strategy.zero_stage,
        micro_batch=strategy.micro_batch,
        global_batch=strategy.global_batch,
        recompute_policy=getattr(strategy, "recompute_policy", "none"),
        pp_schedule=_resolve_pp_schedule(strategy),
        vpp_chunks=getattr(strategy, "vpp_chunks", 1),
        pp_mode=pp_mode,
        cp_kind=cp_kind,
        seq_len=model.seq_len,
        hidden=model.hidden,
    )

    ctx = TransformContext(
        hw_spec=hw_spec,
        parallel=parallel,
        training=training,
        fusion=fusion_config or FusionConfig(),
        quant_profile=quant_profile,
        model_id="",
    )

    if model.num_experts > 0:
        from types import SimpleNamespace
        ctx.profile = SimpleNamespace(
            num_experts=model.num_experts,
            moe_active=model.top_k,
        )

    ctx._model_spec = model
    ctx._system_spec = system
    ctx._strategy = strategy

    return ctx


def _resolve_hw_spec(system: SystemSpec) -> "HardwareSpec":
    """Resolve HardwareSpec from SystemSpec.

    Priority:
    1. Load from registry by GPU name (exact or fuzzy match).
    2. Build a minimal HardwareSpec from GPU fields.
    """
    gpu = system.gpu

    try:
        from zrt.hardware.registry import load as hw_load
        hw = hw_load(gpu.name)
        return hw
    except (KeyError, Exception):
        pass

    name_lower = gpu.name.lower().replace(" ", "_").replace("-", "_")
    _NAME_HINTS = {
        "h100": "nvidia_h100_sxm",
        "a100": "nvidia_a100_80g",
        "h800": "nvidia_h800",
        "910b": "ascend_910b",
        "910c": "ascend_910c",
    }
    for hint, registry_name in _NAME_HINTS.items():
        if hint in name_lower:
            try:
                from zrt.hardware.registry import load as hw_load
                return hw_load(registry_name)
            except (KeyError, Exception):
                pass

    return _build_minimal_hw_spec(gpu)


def _build_minimal_hw_spec(gpu: Any) -> "HardwareSpec":
    """Build a minimal HardwareSpec from a training GPU object."""
    from zrt.hardware.spec import (
        HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec,
    )

    compute = ComputeSpec(
        bf16_tflops=gpu.flops_bf16,
        fp8_tops=gpu.flops_fp8,
        fp4_tops=getattr(gpu, "flops_fp4", 0.0),
    )
    if getattr(gpu, "cube_tflops", None) is not None:
        compute.cube_bf16_tflops = gpu.cube_tflops
    if getattr(gpu, "vector_tflops", None) is not None:
        compute.vector_bf16_tflops = gpu.vector_tflops
    if getattr(gpu, "sram_kb_per_sm", 0.0) > 0:
        compute.sram_kb_per_sm = gpu.sram_kb_per_sm
    if getattr(gpu, "ep_overlap_waves", 0) > 0:
        compute.ep_overlap_waves = gpu.ep_overlap_waves
    if getattr(gpu, "compute_efficiency", None) is not None:
        compute.compute_efficiency = gpu.compute_efficiency

    memory = MemorySpec(
        capacity_gb=gpu.hbm_gb,
        hbm_bandwidth_gbps=gpu.hbm_bw_gbps,
    )
    if getattr(gpu, "mem_bw_efficiency", None) is not None:
        memory.mem_bw_efficiency = gpu.mem_bw_efficiency

    interconnect = InterconnectSpec()

    return HardwareSpec(
        name=gpu.name,
        vendor="nvidia",
        device_type="gpu",
        compute=compute,
        memory=memory,
        interconnect=interconnect,
    )


def _resolve_pp_schedule(strategy: Strategy) -> str:
    """Extract PP schedule string from Strategy."""
    if hasattr(strategy, "pp_schedule"):
        pp = strategy.pp_schedule
        if hasattr(pp, "value"):
            return pp.value
        return str(pp)
    return "1f1b"
