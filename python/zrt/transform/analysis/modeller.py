"""Training modeller: estimate training performance from captured computation graphs.

Usage::

    from python.zrt.transform.analysis import estimate_training_from_graphs
    report = estimate_training_from_graphs(
        forward_graph=fwd, backward_graph=bwd,
        hw_spec=hw, tp=8, pp=4, dp=2, ...
    )
    print(report.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

# Import shared TrainingReport type (canonical import path)
from zrt.training.spec.report import TrainingReport

logger = logging.getLogger(__name__)


def estimate_training_from_graphs(
    *,
    forward_graph: "OpGraph",
    backward_graph: "OpGraph | None" = None,
    output_dir: "str | Path | None" = None,
    hw_spec: "HardwareSpec | None" = None,
    total_params: int | None = None,
    hidden: int = 7168,
    num_layers: int = 4,
    num_layers_full: int | None = None,
    seq_len: int = 128,
    batch_size: int = 1,
    tp: int = 1, pp: int = 1, ep: int = 1, dp: int = 1, cp: int = 1,
    cp_kind: str = "ulysses",
    zero_stage: int = 1,
    optimizer: str = "adam",
    muon_rotation: bool = True,
    muon_ns_steps: int | None = None,
    model_type: str | None = None,
    vocab_size: int | None = None,
    micro_batch: int = 1,
    global_batch: int = 32,
    recompute_policy: str = "none",
    mega_moe: bool = False,
    mega_moe_waves: int = 0,
    pp_schedule: str = "1f1b",
    vpp_chunks: int = 1,
    pp_mode: str = "trace",
    tp_coc: bool = False,
    return_transformed: bool = False,
    quant: str | None = None,
    quant_preset: str | None = None,
    quant_config: dict[str, str] | None = None,
    moe_total_experts: int = 0,
    moe_active_experts: int = 1,
    model_id: str = "",
    fusion_config: "FusionConfig | None" = None,
    ffn_hidden: int | None = None,
    moe_ffn_hidden: int | None = None,
    layer_type_counts: dict[str, int] | None = None,
    n_shared_experts: int | None = None,
    num_heads: int | None = None,
    kv_heads: int | None = None,
    head_dim: int | None = None,
    layer_profile: "LayerProfile | None" = None,
) -> "TrainingReport | tuple[TrainingReport, TransformContext, dict[str, OpGraph]]":
    """Estimate training performance from pre-built OpGraph instances.

    Takes already-captured forward and backward computation graphs and
    runs the training analysis pipeline. Use this when the graphs have
    already been captured by ``run_trace_phases``.

    Parameters
    ----------
    return_transformed : bool, default False
        If True, return (TrainingReport, TransformContext, transformed_graphs)
        where transformed_graphs contains the pipeline-processed graphs.
        This enables downstream Excel export via ``export_training_graphs``.
    output_dir : str or Path, optional
        If provided, export each transformed graph as a DOT file to this directory.
    """
    from python.zrt.transform.context import (
        FusionConfig, ParallelConfig, QuantConfig, TrainingConfig, TransformContext,
    )
    from python.zrt.transform.pipeline import build_default_pipeline

    metadata: dict = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers_full or num_layers,
        "num_layers_traced": num_layers,
        "hidden": hidden,
    }
    if moe_total_experts > 0:
        metadata["moe_total_experts"] = moe_total_experts
    if moe_ffn_hidden is not None:
        metadata["moe_ffn_hidden"] = moe_ffn_hidden
    if layer_type_counts is not None:
        metadata["layer_type_counts"] = layer_type_counts
    if vocab_size is not None:
        metadata["vocab_size"] = vocab_size
    if moe_active_experts > 1:
        metadata["moe_active_experts"] = moe_active_experts
    if total_params is not None:
        metadata["total_params"] = int(total_params)
    if model_type is not None:
        metadata["model_type"] = model_type
    
    # Add LayerProfile info for precise scaling
    if layer_profile is not None:
        metadata["layer_profile"] = layer_profile
        metadata["num_dense"] = layer_profile.num_dense
        metadata["num_moe"] = layer_profile.num_moe
        metadata["num_hca_hash"] = layer_profile.num_hca_hash
        metadata["num_hca_topk"] = layer_profile.num_hca_topk
        metadata["num_hca"] = layer_profile.num_hca
        metadata["num_csa_hash"] = layer_profile.num_csa_hash
        metadata["num_csa_topk"] = layer_profile.num_csa_topk
        metadata["num_csa"] = layer_profile.num_csa
        metadata["num_swa_hash"] = layer_profile.num_swa_hash
        metadata["num_swa_topk"] = layer_profile.num_swa_topk
        metadata["num_swa"] = layer_profile.num_swa
        metadata["typical_indices"] = layer_profile.typical_indices
        # Auto-set num_layers from LayerProfile if not provided
        if num_layers_full is None and layer_profile.total_layers > num_layers:
            metadata["num_layers"] = layer_profile.total_layers

    # Force set metadata (overwrite existing keys)
    for key, val in metadata.items():
        forward_graph.metadata[key] = val
    if backward_graph is not None:
        for key, val in metadata.items():
            backward_graph.metadata[key] = val

    quant_cfg = QuantConfig(weight=quant, activation=quant) if quant else None

    # Build structured quant profile from preset, config dict, or scalar quant.
    from python.zrt.transform.context import GraphQuantProfile
    quant_profile = None
    if quant_preset is not None:
        quant_profile = GraphQuantProfile.from_preset(quant_preset)
    elif quant_config is not None:
        quant_profile = GraphQuantProfile.from_dict(quant_config)
    elif quant is not None:
        quant_profile = GraphQuantProfile.from_scalar(quant)

    ctx = TransformContext(
        hw_spec=hw_spec,
        model_id=model_id,
        parallel=ParallelConfig(tp=tp, pp=pp, ep=ep, dp=dp, cp=cp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            muon_rotation=muon_rotation,
            muon_ns_steps=muon_ns_steps,
            micro_batch=micro_batch,
            global_batch=global_batch,
            recompute_policy=recompute_policy,
            mega_moe=mega_moe,
            mega_moe_waves=mega_moe_waves,
            pp_schedule=pp_schedule,
            vpp_chunks=vpp_chunks,
            pp_mode=pp_mode,
            tp_coc=tp_coc,
            seq_len=seq_len,
            hidden=hidden,
            cp_kind=cp_kind,
        ),
        fusion=fusion_config or FusionConfig(),
        quant=quant_cfg,
        quant_profile=quant_profile,
    )

    # Attach MoE profile to ctx so ExpertParallelPass and other MoE-aware
    # passes can read expert counts.
    if moe_total_experts > 0:
        from types import SimpleNamespace
        ctx.profile = SimpleNamespace(
            num_experts=moe_total_experts,
            moe_active=moe_active_experts,
        )

    pipe = build_default_pipeline()
    results: dict[str, "OpGraph"] = {}

    if backward_graph is not None:
        from python.zrt.ir.adapter import stitch_fwd_bwd
        unified = stitch_fwd_bwd(forward_graph, backward_graph)
        for key, val in metadata.items():
            if key not in unified.metadata:
                unified.metadata[key] = val
        results["unified"] = pipe.run(unified, ctx)
    else:
        results["train_forward"] = pipe.run(forward_graph, ctx)

    # DOT export.
    #
    # ``render_dot`` shells out to graphviz ``dot``; layout is super-linear
    # in node count (with ``splines=ortho`` it can take ~15s per 1k-node
    # graph, ~hours for 5k+).  Production runs only need the ``.dot`` text
    # — anyone wanting the SVG can render manually.  We skip rendering for
    # graphs above ``_RENDER_DOT_NODE_BUDGET`` so the e2e stays fast.
    _RENDER_DOT_NODE_BUDGET = 300
    if output_dir is not None:
        from python.zrt.report.dot_exporter import export_dot, render_dot
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_name = forward_graph.name or "model"

        def _maybe_render(graph, dot_path):
            if len(graph.nodes) <= _RENDER_DOT_NODE_BUDGET:
                render_dot(dot_path)  # no-op when graphviz absent

        # Export raw forward and backward graphs separately
        dot_path = export_dot(forward_graph, out / f"{model_name}_train_forward.dot")
        _maybe_render(forward_graph, dot_path)
        if backward_graph is not None:
            dot_path = export_dot(backward_graph, out / f"{model_name}_train_backward.dot")
            _maybe_render(backward_graph, dot_path)
        # Export transformed graphs (unified or forward-only)
        for tag, g in results.items():
            dot_path = export_dot(g, out / f"{model_name}_{tag}.dot")
            _maybe_render(g, dot_path)

        # ── PP Chrome Trace export ──────────────────────────────────────────
        for tag, g in results.items():
            pp_timeline = g.metadata.get("pp_stitched_timeline")
            stage_timelines = g.metadata.get("pp_per_stage_timelines")
            if pp_timeline is not None and stage_timelines is not None:
                from python.zrt.executor.chrome_trace import ChromeTraceExporter
                trace_dir = out / "pp_trace"
                trace_dir.mkdir(parents=True, exist_ok=True)
                exporter = ChromeTraceExporter()
                M = pp_timeline.M

                exporter.export_stitched(pp_timeline, str(trace_dir / "pp_stitched.json"))

                tl_list = [
                    stage_timelines[s] for s in range(pp_timeline.pp)
                    if s in stage_timelines and stage_timelines[s] is not None
                ]
                if tl_list:
                    exporter.export_per_stage(
                        tl_list,
                        str(trace_dir / "pp_per_stage.json"),
                        M=M,
                        pp_stitched=pp_timeline,
                        replicate=True,
                    )
                    exporter.export_stitched_detailed(
                        pp_timeline, tl_list,
                        str(trace_dir / "pp_combined.json"),
                    )
                logger.info("PP Chrome Trace exported to %s", trace_dir)

    if "unified" in results:
        g = results["unified"]
        sr = g.metadata.get("step_result", {})
        memory_breakdown = g.metadata.get("memory_breakdown")
        training_flops = g.metadata.get("training_flops", 0.0)
        forward_flops = g.metadata.get("forward_flops", 0.0)
        backward_flops = g.metadata.get("backward_flops", 0.0)
        total_params = g.metadata.get("total_params", 0)
    else:
        fwd = results["train_forward"]
        sr = fwd.metadata.get("step_result", {})

        memory_breakdown = fwd.metadata.get("memory_breakdown")
        training_flops = fwd.metadata.get("training_flops", 0.0)
        forward_flops = fwd.metadata.get("forward_flops", 0.0)
        backward_flops = fwd.metadata.get("backward_flops", 0.0)
        total_params = fwd.metadata.get("total_params", 0)

    parallel = ctx.parallel
    training = ctx.training
    config_parts: list[str] = []
    if parallel.tp > 1:
        config_parts.append(f"TP{parallel.tp}")
    if parallel.pp > 1:
        config_parts.append(f"PP{parallel.pp}")
    if parallel.ep > 1:
        config_parts.append(f"EP{parallel.ep}")
    if parallel.dp > 1:
        config_parts.append(f"DP{parallel.dp}")
    if training:
        config_parts.append(f"ZeRO-{training.zero_stage}")
        config_parts.append(f"{training.optimizer}")
        config_parts.append(f"micro{training.micro_batch}")
    config_summary = "-".join(config_parts) if config_parts else "default"

    # ── Fused-operator summary ────────────────────────────────────────────────
    fused_ops_summary = _summarise_fused_ops(results)

    _d = sr.get
    _dp_exposed = _d("dp_exposed_ms", 0.0)
    _dp_hidden = _d("dp_hidden_ms", 0.0)

    report = TrainingReport(
        config_summary=config_summary,
        step_time_ms=_d("step_time_ms", 0.0),
        per_stage_ms=_d("per_stage_ms", 0.0),
        mfu=_d("mfu", 0.0),
        hfu=_d("hfu", 0.0),
        training_flops=training_flops,
        forward_flops=forward_flops,
        backward_flops=backward_flops,
        memory_breakdown=memory_breakdown.to_dict() if memory_breakdown else {},
        warmup_steps=_d("warmup_steps", 0),
        cooldown_steps=_d("cooldown_steps", 0),
        steady_steps=_d("steady_steps", 0),
        dp_exposed_ms=_dp_exposed,
        dp_hidden_ms=_dp_hidden,
        dp_total_ms=_dp_exposed + _dp_hidden,
        bubble_fraction=_d("bubble_fraction", 0.0),
        bubble_time_ms=_d("bubble_time_ms", 0.0),
        total_params=total_params,
        fused_ops_summary=fused_ops_summary,
        compute_time_ms=_d("compute_time_ms", 0.0),
        fwd_compute_ms=_d("fwd_compute_ms", 0.0),
        bwd_compute_ms=_d("bwd_compute_ms", 0.0),
        recompute_compute_ms=_d("recompute_compute_ms", 0.0),
        exposed_comm_ms=_d("exposed_comm_ms", 0.0),
        hidden_comm_ms=_d("hidden_comm_ms", 0.0),
        total_comm_volume_ms=_d("total_comm_ms", 0.0),
        optimizer_time_ms=_d("optimizer_time_ms", 0.0),
        optimizer_comm_ms=_d("optimizer_comm_ms", 0.0),
        pipeline_time_ms=_d("pipeline_time_ms", 0.0),
        warmup_ms=_d("warmup_ms", 0.0),
        steady_ms=_d("steady_ms", 0.0),
        cooldown_ms=_d("cooldown_ms", 0.0),
        tp_exposed_ms=_d("tp_exposed_ms", 0.0),
        tp_hidden_ms=_d("tp_hidden_ms", 0.0),
        tp_total_ms=_d("tp_total_ms", 0.0),
        cp_exposed_ms=_d("cp_exposed_ms", 0.0),
        cp_hidden_ms=_d("cp_hidden_ms", 0.0),
        cp_total_ms=_d("cp_total_ms", 0.0),
        ep_exposed_ms=_d("ep_exposed_ms", 0.0),
        ep_hidden_ms=_d("ep_hidden_ms", 0.0),
        ep_total_ms=_d("ep_total_ms", 0.0),
        pp_exposed_ms=_d("pp_exposed_ms", 0.0),
        pp_hidden_ms=_d("pp_hidden_ms", 0.0),
        pp_total_ms=_d("pp_total_ms", 0.0),
        optimizer_comm_hidden_ms=_d("optimizer_comm_hidden_ms", 0.0),
        recompute_critical_ms=_d("recompute_critical_ms", 0.0),
        recompute_raw_mag_ms=_d("recompute_raw_mag_ms", 0.0),
        warmup_fwd_ms=_d("warmup_fwd_ms", 0.0),
        warmup_bwd_ms=_d("warmup_bwd_ms", 0.0),
        steady_fwd_ms=_d("steady_fwd_ms", 0.0),
        steady_bwd_ms=_d("steady_bwd_ms", 0.0),
        cooldown_fwd_ms=_d("cooldown_fwd_ms", 0.0),
        cooldown_bwd_ms=_d("cooldown_bwd_ms", 0.0),
    )

    if return_transformed:
        return report, ctx, results
    return report


# ── Fused-operator summary helper ────────────────────────────────────────────

def _summarise_fused_ops(graphs: dict) -> dict:
    """Aggregate fused-node statistics across all transformed graphs.

    Skips raw aten.* / comm.* nodes so the table focuses on what fusion
    actually produced — module-level units (Linear, RMSNorm, ...) and
    rich-rule outputs (mla_sparse_attn, kv_compressor, rms_norm, ...).

    Returns ``{op_type: {count, sample_names, total_flops, dtype, module_class}}``.
    """
    summary: dict[str, dict] = {}

    for g in graphs.values():
        for node in g.nodes.values():
            op_type = node.op_type or ""
            # Skip primitive aten / comm / optimizer nodes — those aren't
            # the "fused operators" the user wants to see.
            if op_type.startswith("aten.") or op_type.startswith("comm."):
                continue
            if op_type.startswith("optimizer."):
                continue

            entry = summary.setdefault(op_type, {
                "count": 0,
                "sample_names": [],
                "total_flops": 0.0,
                "dtype": None,
                "module_class": None,
            })
            entry["count"] += 1

            # Collect a friendly name from scope tail (e.g.
            # "transformer.layers.0.attn.wq_b" → "wq_b") or from the
            # leaf_attr stored on the node.  Keep up to 8 unique samples.
            name = node.name or (node.scope.rsplit(".", 1)[-1] if node.scope else "")
            if name and name not in entry["sample_names"] and len(entry["sample_names"]) < 8:
                entry["sample_names"].append(name)

            # Prefer the rule-derived sem_flops; fall back to the
            # downstream FlopsPass annotation.
            ann = node.annotations or {}
            flops = ann.get("sem_flops")
            if flops is None:
                flops = ann.get("flops")
            if isinstance(flops, (int, float)):
                entry["total_flops"] += float(flops)

            if entry["dtype"] is None:
                d = ann.get("sem_dtype")
                if d:
                    entry["dtype"] = d
                elif node.inputs:
                    entry["dtype"] = node.inputs[0].dtype.value
                elif node.outputs:
                    entry["dtype"] = node.outputs[0].dtype.value

            if entry["module_class"] is None and node.module_class:
                entry["module_class"] = node.module_class

    return summary
