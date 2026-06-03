from __future__ import annotations

import pytest

from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.compose.stage import stage_time
from zrt.training.ir.builders import build_graph
from zrt.training.models.flops import op_cost
from zrt.training.search.estimator import estimate
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec


def _moe_model(**kwargs) -> ModelSpec:
    base = dict(
        hidden=1024,
        ffn=4096,
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        vocab=32000,
        seq_len=128,
        layers=[LayerKind.MOE],
        num_experts=8,
        moe_ffn=2048,
        top_k=2,
        n_shared_experts=1,
    )
    base.update(kwargs)
    return ModelSpec(**base)


def _system(*, ep_overlap_waves: int = 4) -> SystemSpec:
    link = LinkSpec(
        type="NVLink",
        bandwidth_gbps=900,
        latency_us=1.0,
        topology="all_to_all",
        num_devices=8,
    )
    return SystemSpec(
        gpu=GPU(
            name="h100",
            flops_bf16=989,
            flops_fp8=1979,
            hbm_gb=80,
            hbm_bw_gbps=3350,
            ep_overlap_waves=ep_overlap_waves,
        ),
        host_mem_gb=256,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
        nodes=1,
        gpus_per_node=8,
    )


def _strategy(**kwargs) -> Strategy:
    base = dict(ep=4, dp=8, micro_batch=2)
    base.update(kwargs)
    return Strategy(**base)


def _run_spec_path(
    model: ModelSpec | None = None,
    strategy: Strategy | None = None,
    system: SystemSpec | None = None,
):
    model = model or _moe_model()
    strategy = strategy or _strategy()
    system = system or _system()
    graph = build_graph(model, strategy)
    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    stage = stage_time(graph.ops, graph.collectives, model, system, strategy)
    report = estimate(model, system, strategy, graph=graph)
    return model, strategy, system, graph, op_costs, stage, report


def _ops_for_layer(graph, layer_id: int):
    start, end = graph.layer_index[layer_id]
    return graph.ops[start:end]


def _op_for_layer(
    graph,
    layer_id: int,
    *,
    kind: str | None = None,
    name: str | None = None,
):
    matches = [
        op
        for op in _ops_for_layer(graph, layer_id)
        if (kind is None or op.kind == kind) and (name is None or op.name == name)
    ]
    assert len(matches) == 1
    return matches[0]


def _ep_a2a_collectives(graph):
    return [c for c in graph.collectives if c.group == "EP" and c.kind == "A2A"]


def test_switch_off_preserves_routed_expert_ffn_and_explicit_ep_a2a_end_to_end():
    model, _, _, graph, op_costs, stage, report = _run_spec_path(
        strategy=_strategy(mega_moe=False)
    )

    routed = _op_for_layer(graph, 0, name="L0.routed_expert_ffn")
    ep_a2a = _ep_a2a_collectives(graph)

    assert routed.kind == "matmul"
    assert [op for op in graph.ops if op.kind == "mega_moe"] == []
    assert [c.name for c in ep_a2a] == [
        "a2a_before_L0.routed_expert_ffn",
        "a2a_after_L0.routed_expert_ffn",
    ]
    assert ep_a2a[0].inserted_before == routed.name
    assert ep_a2a[1].inserted_after == routed.name
    assert op_costs[routed.name].fwd_cube_flops > 0
    assert stage.fwd + stage.bwd > 0
    assert report.step_time_ms > 0
    assert report.per_stage[0].fwd + report.per_stage[0].bwd > 0
    assert model.num_experts > 1


def test_switch_on_replaces_routed_expert_ffn_and_external_ep_a2a_end_to_end():
    _, strategy, _, graph, op_costs, stage, report = _run_spec_path(
        strategy=_strategy(mega_moe=True, mega_moe_waves=2)
    )

    mega_moe = _op_for_layer(graph, 0, kind="mega_moe")

    assert mega_moe.name == "L0.mega_moe"
    assert [
        op
        for op in _ops_for_layer(graph, 0)
        if op.kind == "matmul" and op.name == "L0.routed_expert_ffn"
    ] == []
    assert _ep_a2a_collectives(graph) == []
    assert all("routed_expert" not in c.name for c in graph.collectives)
    assert all(c.inserted_before != mega_moe.name for c in graph.collectives)
    assert all(c.inserted_after != mega_moe.name for c in graph.collectives)

    cost = op_costs[mega_moe.name]
    assert cost.fwd_cube_flops > 0
    assert cost.fwd_bytes > 0
    assert stage.fwd + stage.bwd > 0
    assert report.step_time_ms > 0
    assert stage.comm_fwd > 0
    assert stage.comm_bwd > 0
    assert stage.ep_hidden > 0
    assert stage.ep_exposed > 0
    assert report.per_stage[0].ep_exposed == pytest.approx(stage.ep_exposed)
    assert report.per_stage[0].ep_hidden == pytest.approx(stage.ep_hidden)
    assert mega_moe.meta["requested_waves"] == strategy.mega_moe_waves


def test_quant_variants_flow_through_spec_path_and_w4a8_uses_smaller_stored_weights():
    standard_model, _, _, standard_graph, _, _, _ = _run_spec_path(
        strategy=_strategy(mega_moe=True)
    )
    w4a8_model, _, _, w4a8_graph, _, _, _ = _run_spec_path(
        model=_moe_model(
            routed_expert_compute_dtype=Dtype.FP8_E4M3,
            routed_expert_weight_dtype=Dtype.FP4,
            moe_act_dtype=Dtype.FP8_E4M3,
        ),
        strategy=_strategy(mega_moe=True),
    )

    standard = _op_for_layer(standard_graph, 0, kind="mega_moe")
    w4a8 = _op_for_layer(w4a8_graph, 0, kind="mega_moe")
    ln2 = _op_for_layer(w4a8_graph, 0, name="L0.ln2")
    expert_agg = _op_for_layer(w4a8_graph, 0, name="L0.expert_agg")

    assert standard.meta["quant_variant"] == "standard"
    assert w4a8.meta["quant_variant"] == "w4a8"
    assert w4a8.inputs[0].name == ln2.outputs[0].name == "x_ln2"
    assert w4a8.inputs[0].dtype == ln2.outputs[0].dtype
    assert w4a8.outputs[0].dtype == w4a8_model.effective_moe_act_dtype()
    assert expert_agg.inputs[1].dtype == w4a8_model.act_dtype
    assert expert_agg.inputs[1].name.startswith("routed_ffn_out__cast_bf16")
    assert w4a8.meta["moe_act_dtype"] == Dtype.FP8_E4M3
    assert w4a8.meta["moe_act_bytes"] == Dtype.FP8_E4M3.bytes
    assert w4a8.meta["weight_stored_bytes"] < standard.meta["weight_stored_bytes"]
    assert (
        standard.meta["weight_stored_bytes"]
        == standard_model.routed_expert_weight_dtype.stored_bytes
    )
    assert w4a8.meta["weight_stored_bytes"] == Dtype.FP4.stored_bytes


def test_mega_moe_wave_override_and_hardware_default_affect_stage_output():
    model = _moe_model()
    system = _system(ep_overlap_waves=4)

    _, _, _, one_graph, _, one_stage, _ = _run_spec_path(
        model=model,
        system=system,
        strategy=_strategy(mega_moe=True, mega_moe_waves=1),
    )
    _, _, _, default_graph, _, default_stage, _ = _run_spec_path(
        model=model,
        system=system,
        strategy=_strategy(mega_moe=True, mega_moe_waves=0),
    )
    _, _, _, four_graph, _, four_stage, _ = _run_spec_path(
        model=model,
        system=system,
        strategy=_strategy(mega_moe=True, mega_moe_waves=4),
    )

    one = _op_for_layer(one_graph, 0, kind="mega_moe")
    default = _op_for_layer(default_graph, 0, kind="mega_moe")
    four = _op_for_layer(four_graph, 0, kind="mega_moe")

    assert one.meta["requested_waves"] == 1
    assert default.meta["requested_waves"] == 0
    assert four.meta["requested_waves"] == 4
    assert default_stage.ep_exposed == pytest.approx(four_stage.ep_exposed)
    assert default_stage.ep_hidden == pytest.approx(four_stage.ep_hidden)
    assert four_stage.ep_exposed < one_stage.ep_exposed
    assert four_stage.ep_hidden > one_stage.ep_hidden
