"""Single-point estimator — the main entry point.

两条路径（参见 docs/architecture_snapshot_zh.md）：
  - 路径 A（抓图建模）：``estimate_via_pipeline()`` → Transform Pipeline
  - 路径 B（配置建模）：``_estimate_legacy()`` → 手工成本模型 + Composer
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from zrt.training.compose.schedules import StepResult, pipeline_step_time
from zrt.training.ir.validate import validate as ir_validate
from zrt.training.models.flops import total_training_flops, forward_backward_flops
from zrt.training.models.memory import MemBreakdown
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import Strategy, rank_product
from zrt.training.spec.system import SystemSpec

if TYPE_CHECKING:
    from zrt.training.spec.capture_config import CaptureConfig

logger = logging.getLogger(__name__)

Report = TrainingReport


def estimate(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
    graph=None,
    capture: "CaptureConfig | None" = None,
) -> TrainingReport:
    """Single-point evaluation of a training config.

    路径 A（capture 不为 None 或 graph 为 None）：走 Transform Pipeline。
    路径 B（graph 传入）：走 Legacy 手工成本模型。
    """
    if graph is not None:
        return _estimate_legacy(model, system, strategy, graph)
    return estimate_via_pipeline(model, system, strategy, capture=capture)


def estimate_via_pipeline(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
    capture: "CaptureConfig | None" = None,
) -> TrainingReport:
    """路径 A（抓图建模）：build_captured_graph → supplement metadata → build_context → pipeline.run → report"""
    from zrt.training.ir.opgraph_builder import build_captured_graph
    from zrt.training.ir.context_builder import build_context
    from zrt.transform.pipeline import build_default_pipeline

    strategy.validate(model, system)
    warnings = ir_validate(model, system, strategy)

    opgraph = build_captured_graph(model, strategy, capture=capture)
    _supplement_metadata(opgraph, model)

    ctx = build_context(model, system, strategy, pp_mode="formula")

    pipe = build_default_pipeline()
    transformed = pipe.run(opgraph, ctx)

    return _build_report_from_transformed(transformed, model, system, strategy, warnings)


def _supplement_metadata(opgraph, model: ModelSpec) -> None:
    """Add ModelSpec metadata that Transform Pipeline passes need."""
    if model.num_experts > 0:
        opgraph.metadata["moe_total_experts"] = model.num_experts
        opgraph.metadata["moe_active_experts"] = model.top_k
        if model.moe_ffn > 0:
            opgraph.metadata["moe_ffn_hidden"] = model.moe_ffn
    if model.vocab > 0:
        opgraph.metadata["vocab_size"] = model.vocab
    layer_counts = Counter(k.value if hasattr(k, "value") else str(k) for k in model.layers)
    opgraph.metadata["layer_type_counts"] = dict(layer_counts)


def _build_report_from_transformed(
    transformed, model: ModelSpec, system: SystemSpec,
    strategy: Strategy, warnings: list[str],
) -> TrainingReport:
    """Build TrainingReport from a pipeline-transformed OpGraph."""
    sr = transformed.metadata.get("step_result", {})
    _d = sr.get

    training_flops = transformed.metadata.get("training_flops", 0.0)
    forward_flops = transformed.metadata.get("forward_flops", 0.0)
    backward_flops = transformed.metadata.get("backward_flops", 0.0)
    total_params = transformed.metadata.get("total_params", model.total_params())

    mem_obj = transformed.metadata.get("memory_breakdown")
    memory = None
    memory_dict = {}
    if mem_obj is not None:
        memory = MemBreakdown(
            weights=getattr(mem_obj, "weights", 0.0),
            grads=getattr(mem_obj, "grads", 0.0),
            opt_state=getattr(mem_obj, "opt_state", 0.0),
            activations=getattr(mem_obj, "activations", 0.0),
            comm_buffers=getattr(mem_obj, "comm_buffers", 0.0),
        )
        memory_dict = mem_obj.to_dict() if hasattr(mem_obj, "to_dict") else {}

    _sched = getattr(strategy.pp_schedule, "value", str(strategy.pp_schedule))
    if strategy.vpp_chunks and strategy.vpp_chunks > 1:
        _sched += f" vpp{strategy.vpp_chunks}"
    config_summary = {
        "model": f"hidden={model.hidden}, layers={len(model.layers)}, heads={model.num_heads}",
        "system": f"{system.gpu.name} x {system.world_size}",
        "strategy": f"TP={strategy.tp} CP={strategy.cp} PP={strategy.pp} EP={strategy.ep} DP={strategy.dp} · {_sched}",
        "parallelism": f"TP*CP*PP*DP = {rank_product(strategy.tp, strategy.cp, strategy.pp, strategy.ep, strategy.dp)} (EP={strategy.ep} shares ranks)",
        "micro_batch": strategy.micro_batch,
        "global_batch": strategy.global_batch,
        "num_microbatches": strategy.num_microbatches(),
        "zero_stage": strategy.zero_stage,
    }

    tokens = (strategy.global_batch * model.seq_len
              if strategy.global_batch > 0
              else strategy.micro_batch * strategy.dp * model.seq_len)
    pipeline_time_s = _d("pipeline_time_ms", 0.0) / 1000.0
    tokens_per_sec = tokens / pipeline_time_s if pipeline_time_s > 0 else 0.0
    flops_per_token = training_flops / tokens if tokens > 0 else 0.0

    return TrainingReport(
        step_time_ms=_d("step_time_ms", 0.0),
        pipeline_time_ms=_d("pipeline_time_ms", 0.0),
        mfu=_d("mfu", 0.0),
        hfu=_d("hfu", 0.0),
        memory=memory,
        memory_breakdown=memory_dict,
        per_stage_ms=_d("per_stage_ms", 0.0),
        total_flops=training_flops,
        forward_flops=forward_flops,
        backward_flops=backward_flops,
        training_flops=training_flops,
        total_params=total_params,
        warnings=warnings,
        config_summary=config_summary,
        bubble_fraction=_d("bubble_fraction", 0.0),
        bubble_time_ms=_d("bubble_time_ms", 0.0),
        schedule_name=_d("schedule_name", "1f1b"),
        warmup_steps=_d("warmup_steps", 0),
        cooldown_steps=_d("cooldown_steps", 0),
        steady_steps=_d("steady_steps", 0),
        warmup_ms=_d("warmup_ms", 0.0),
        steady_ms=_d("steady_ms", 0.0),
        cooldown_ms=_d("cooldown_ms", 0.0),
        dp_exposed_ms=_d("dp_exposed_ms", 0.0),
        optimizer_time_ms=_d("optimizer_time_ms", 0.0),
        optimizer_comm_ms=_d("optimizer_comm_ms", 0.0),
        optimizer_comm_hidden_ms=_d("optimizer_comm_hidden_ms", 0.0),
        warmup_fwd_ms=_d("warmup_fwd_ms", 0.0),
        warmup_bwd_ms=_d("warmup_bwd_ms", 0.0),
        steady_fwd_ms=_d("steady_fwd_ms", 0.0),
        steady_bwd_ms=_d("steady_bwd_ms", 0.0),
        cooldown_fwd_ms=_d("cooldown_fwd_ms", 0.0),
        cooldown_bwd_ms=_d("cooldown_bwd_ms", 0.0),
        steady_fwd_per_mb_ms=_d("steady_fwd_per_mb_ms", 0.0),
        steady_bwd_per_mb_ms=_d("steady_bwd_per_mb_ms", 0.0),
        steady_per_mb_ms=_d("steady_per_mb_ms", 0.0),
        compute_time_ms=_d("compute_time_ms", 0.0),
        fwd_compute_ms=_d("fwd_compute_ms", 0.0),
        bwd_compute_ms=_d("bwd_compute_ms", 0.0),
        recompute_compute_ms=_d("recompute_compute_ms", 0.0),
        recompute_critical_ms=_d("recompute_critical_ms", 0.0),
        recompute_raw_mag_ms=_d("recompute_raw_mag_ms", 0.0),
        exposed_comm_ms=_d("exposed_comm_ms", 0.0),
        tp_exposed_ms=_d("tp_exposed_ms", 0.0),
        cp_exposed_ms=_d("cp_exposed_ms", 0.0),
        ep_exposed_ms=_d("ep_exposed_ms", 0.0),
        pp_exposed_ms=_d("pp_exposed_ms", 0.0),
        hidden_comm_ms=_d("hidden_comm_ms", 0.0),
        dp_hidden_ms=_d("dp_hidden_ms", 0.0),
        tp_hidden_ms=_d("tp_hidden_ms", 0.0),
        ep_hidden_ms=_d("ep_hidden_ms", 0.0),
        pp_hidden_ms=_d("pp_hidden_ms", 0.0),
        total_comm_volume_ms=_d("total_comm_ms", 0.0),
        tp_total_ms=_d("tp_total_ms", 0.0),
        cp_total_ms=_d("cp_total_ms", 0.0),
        ep_total_ms=_d("ep_total_ms", 0.0),
        pp_total_ms=_d("pp_total_ms", 0.0),
        dp_total_ms=_d("dp_total_ms", 0.0),
        tokens_per_sec=tokens_per_sec,
        effective_params=model.effective_params_for_flops(),
        flops_per_token=flops_per_token,
    )


def _estimate_legacy(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
    graph,
) -> TrainingReport:
    """路径 B（配置建模）：直接调用 op_cost / pipeline_step_time 手工成本模型。"""
    strategy.validate(model, system)
    warnings = ir_validate(model, system, strategy)

    total_flops = total_training_flops(graph, model, strategy, system)
    fwd_flops, bwd_flops = forward_backward_flops(graph, model, strategy, system)

    step_result: StepResult = pipeline_step_time(graph, model, system, strategy)

    _sched = getattr(strategy.pp_schedule, "value", str(strategy.pp_schedule))
    if strategy.vpp_chunks and strategy.vpp_chunks > 1:
        _sched += f" vpp{strategy.vpp_chunks}"
    config_summary = {
        "model": f"hidden={model.hidden}, layers={len(model.layers)}, heads={model.num_heads}",
        "system": f"{system.gpu.name} x {system.world_size}",
        "strategy": f"TP={strategy.tp} CP={strategy.cp} PP={strategy.pp} EP={strategy.ep} DP={strategy.dp} · {_sched}",
        "parallelism": f"TP*CP*PP*DP = {rank_product(strategy.tp, strategy.cp, strategy.pp, strategy.ep, strategy.dp)} (EP={strategy.ep} shares ranks)",
        "micro_batch": strategy.micro_batch,
        "global_batch": strategy.global_batch,
        "num_microbatches": strategy.num_microbatches(),
        "zero_stage": strategy.zero_stage,
    }

    s = step_result
    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    pipeline_time = s.pipeline_time
    tokens_per_sec = tokens / pipeline_time if pipeline_time > 0 else 0.0
    flops_per_token = total_flops / tokens if tokens > 0 else 0.0

    return TrainingReport(
        step_time_ms=s.step_time * 1000,
        pipeline_time_ms=s.pipeline_time * 1000,
        mfu=s.mfu,
        hfu=s.hfu,
        mfu_native=s.mfu_native,
        memory=s.memory,
        per_stage=s.per_stage,
        total_flops=total_flops,
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        training_flops=total_flops,
        total_params=model.total_params(),
        warnings=warnings,
        config_summary=config_summary,
        bubble_fraction=s.bubble_fraction,
        bubble_time_ms=(s.warmup + s.cooldown) * 1000,
        schedule_name=s.schedule_name,
        warmup_steps=s.warmup_steps,
        cooldown_steps=s.cooldown_steps,
        steady_steps=max(0, strategy.num_microbatches() - s.warmup_steps - s.cooldown_steps),
        warmup_ms=s.warmup * 1000,
        steady_ms=s.steady * 1000,
        cooldown_ms=s.cooldown * 1000,
        dp_exposed_ms=s.dp_exposed * 1000,
        optimizer_time_ms=s.optimizer_time * 1000,
        optimizer_comm_ms=s.optimizer_comm * 1000,
        optimizer_comm_hidden_ms=s.optimizer_comm_hidden * 1000,
        warmup_fwd_ms=s.warmup_fwd * 1000,
        warmup_bwd_ms=s.warmup_bwd * 1000,
        steady_fwd_ms=s.steady_fwd * 1000,
        steady_bwd_ms=s.steady_bwd * 1000,
        cooldown_fwd_ms=s.cooldown_fwd * 1000,
        cooldown_bwd_ms=s.cooldown_bwd * 1000,
        steady_fwd_per_mb_ms=s.steady_fwd_per_mb * 1000,
        steady_bwd_per_mb_ms=s.steady_bwd_per_mb * 1000,
        steady_per_mb_ms=s.steady_per_mb * 1000,
        compute_time_ms=s.compute_time * 1000,
        fwd_compute_ms=s.fwd_compute * 1000,
        bwd_compute_ms=s.bwd_compute * 1000,
        recompute_critical_ms=s.recompute_critical * 1000,
        recompute_raw_mag_ms=s.recompute_raw_mag * 1000,
        exposed_comm_ms=s.exposed_comm * 1000,
        tp_exposed_ms=s.tp_exposed * 1000,
        cp_exposed_ms=s.cp_exposed * 1000,
        ep_exposed_ms=s.ep_exposed * 1000,
        pp_exposed_ms=s.pp_exposed * 1000,
        hidden_comm_ms=s.hidden_comm * 1000,
        dp_hidden_ms=s.dp_hidden * 1000,
        tp_hidden_ms=s.tp_hidden * 1000,
        ep_hidden_ms=s.ep_hidden * 1000,
        pp_hidden_ms=s.pp_hidden * 1000,
        total_comm_volume_ms=s.total_comm_volume * 1000,
        tp_total_ms=(s.tp_exposed + s.tp_hidden) * 1000,
        cp_total_ms=s.cp_exposed * 1000,
        ep_total_ms=(s.ep_exposed + s.ep_hidden) * 1000,
        pp_total_ms=(s.pp_exposed + s.pp_hidden) * 1000,
        dp_total_ms=(s.dp_exposed + s.dp_hidden) * 1000,
        tokens_per_sec=tokens_per_sec,
        effective_params=model.effective_params_for_flops(),
        flops_per_token=flops_per_token,
        weight_hbm_gb=s.weight_hbm_gb,
        act_hbm_gb=s.act_hbm_gb,
        grad_hbm_gb=s.grad_hbm_gb,
        cast_hbm_gb=s.cast_hbm_gb,
    )


def grid_search(
    model: ModelSpec, system: SystemSpec, space: "SearchSpace",
) -> list[TrainingReport]:
    """Grid search over all valid parallel configurations.

    Returns list of TrainingReports sorted by step_time_ms (ascending).
    Invalid configurations (validation errors) are skipped.
    """
    from zrt.training.search.space import SearchSpace

    strategies = space.strategies(system.world_size)
    reports = []

    for strategy in strategies:
        try:
            strategy.validate(model, system)
        except ValueError:
            continue

        try:
            report = estimate(model, system, strategy)
            if report.memory is not None:
                peak_gb = report.memory.peak_overall / 1e9
                if peak_gb > space.max_memory_gb:
                    continue
            reports.append(report)
        except Exception:
            continue

    reports.sort(key=lambda r: r.step_time_ms)
    return reports


def pareto_frontier(reports: list[TrainingReport]) -> list[TrainingReport]:
    """Extract Pareto frontier (step_time_ms, total_memory) with deterministic ordering.

    A config is on the Pareto frontier if no other config has both:
      - lower step_time_ms AND lower total_memory

    Note: total_memory is the algebraic sum of components (MemBreakdown.total),
    not peak_overall which is the OOM-relevant metric. The frontier uses total
    for consistency with the sorting behavior established in the codebase.

    Deterministic ordering: sort by (step_time_ms, total_memory) to ensure
    reproducible frontier construction. When two configs have identical
    step_time and memory, the first one in sorted order is preferred.

    TODO Phase 3: Pruning rules below depend on CP/EP implementation status:
      - no cross-node TP (requires NVLink topology awareness)
      - CP only when seq_len >= 32768 (depends on CP comm cost model)
      - EP only when num_experts > 1 (requires EP dispatch/all-to-all)
      - ZeRO stage requires dp > 1 (already enforced)

    These should become feature flags in SearchSpace once phase 3 provides
    the missing communication and memory semantics.
    """
    if not reports:
        return []

    # Deterministic sort: by step_time, then by memory (asc)
    sorted_reports = sorted(
        reports,
        key=lambda r: (r.step_time_ms, r.memory.total / 1e9 if r.memory else float("inf"))
    )

    frontier = []
    min_memory = float("inf")

    for report in sorted_reports:
        mem_gb = report.memory.total / 1e9 if report.memory else None
        if not frontier:
            frontier.append(report)
            min_memory = mem_gb if mem_gb is not None else float("inf")
        elif mem_gb is not None and mem_gb < min_memory:
            frontier.append(report)
            min_memory = mem_gb

    return frontier
