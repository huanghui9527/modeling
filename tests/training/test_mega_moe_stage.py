from __future__ import annotations

import pytest

from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.compose import stage as stage_mod
from zrt.training.compose.stage import _ep_gemm_time, stage_time
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec


def _system(*, ep_overlap_waves: int = 4) -> SystemSpec:
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
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(
                type="NVLink",
                bandwidth_gbps=900,
                latency_us=1.0,
                topology="all_to_all",
                num_devices=8,
            ),
            inter_node=LinkSpec(
                type="IB",
                bandwidth_gbps=400,
                latency_us=2.0,
                topology="fat_tree",
            ),
        ),
        nodes=1,
        gpus_per_node=8,
    )


def _moe_model(**kwargs) -> ModelSpec:
    base = dict(
        hidden=4096,
        ffn=8192,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab=32000,
        seq_len=512,
        layers=[LayerKind.MOE],
        num_experts=16,
        moe_ffn=8192,
        top_k=2,
        n_shared_experts=0,
    )
    base.update(kwargs)
    return ModelSpec(**base)


def _ep_a2a_collectives(graph):
    return [c for c in graph.collectives if c.group == "EP" and c.kind == "A2A"]


def test_stage_time_reports_fused_mega_moe_compute_and_internal_ep_comm():
    model = _moe_model()
    strategy = Strategy(ep=4, dp=4, mega_moe=True, mega_moe_waves=4, micro_batch=2)
    graph = build_graph(model, strategy)

    assert [op.kind for op in graph.ops].count("mega_moe") == 1
    assert _ep_a2a_collectives(graph) == []

    st = stage_time(graph.ops, graph.collectives, model, _system(), strategy)

    assert st.fwd > st.comm_fwd
    assert st.bwd > st.comm_bwd
    assert st.comm_fwd > 0.0
    assert st.comm_bwd > 0.0
    assert st.ep_exposed > 0.0
    assert st.ep_hidden > 0.0


def test_more_mega_moe_waves_reduce_exposed_fused_ep_comm():
    model = _moe_model(seq_len=1024)
    system = _system(ep_overlap_waves=4)
    one_wave = Strategy(ep=4, dp=4, mega_moe=True, mega_moe_waves=1, micro_batch=2)
    four_waves = Strategy(ep=4, dp=4, mega_moe=True, mega_moe_waves=4, micro_batch=2)
    graph_one = build_graph(model, one_wave)
    graph_four = build_graph(model, four_waves)

    st_one = stage_time(graph_one.ops, graph_one.collectives, model, system, one_wave)
    st_four = stage_time(graph_four.ops, graph_four.collectives, model, system, four_waves)

    assert st_one.ep_exposed > 0.0
    assert st_four.ep_exposed < st_one.ep_exposed
    assert st_four.ep_hidden > st_one.ep_hidden


def test_unset_mega_moe_waves_auto_selects_best_pipeline_divisor():
    model = _moe_model(seq_len=1024)
    system = _system(ep_overlap_waves=1)
    auto = Strategy(ep=4, dp=4, mega_moe=True, mega_moe_waves=0, micro_batch=2)
    four_waves = Strategy(ep=4, dp=4, mega_moe=True, mega_moe_waves=4, micro_batch=2)
    graph_auto = build_graph(model, auto)
    graph_four = build_graph(model, four_waves)

    st_auto = stage_time(graph_auto.ops, graph_auto.collectives, model, system, auto)
    st_four = stage_time(graph_four.ops, graph_four.collectives, model, system, four_waves)

    assert st_auto.ep_exposed == pytest.approx(st_four.ep_exposed)
    assert st_auto.ep_hidden == pytest.approx(st_four.ep_hidden)


def test_ep_gemm_time_counts_mega_moe_as_routed_expert_compute():
    model = _moe_model()
    system = _system()
    strategy = Strategy(ep=4, dp=4, mega_moe=True, micro_batch=2)
    graph = build_graph(model, strategy)

    layer_ops = [op for op in graph.ops if op.layer_id == 0]

    assert [
        op for op in layer_ops
        if op.kind == "matmul" and "routed_expert" in op.name
    ] == []
    assert _ep_gemm_time(layer_ops, model, system, strategy, system.gpu.name) > 0.0


def test_switch_on_mega_moe_stage_timing_does_not_require_separate_a2a_collectives():
    model = _moe_model()
    strategy = Strategy(ep=4, dp=4, mega_moe=True, mega_moe_waves=4, micro_batch=2)
    graph = build_graph(model, strategy)

    st_without_ep_collectives = stage_time(graph.ops, [], model, _system(), strategy)

    assert _ep_a2a_collectives(graph) == []
    assert st_without_ep_collectives.ep_exposed > 0.0
    assert st_without_ep_collectives.comm_fwd > 0.0
    assert st_without_ep_collectives.comm_bwd > 0.0


def test_mega_moe_fused_ep_comm_uses_ep_imbalance_factor(monkeypatch):
    model = _moe_model()
    strategy = Strategy(ep=4, dp=4, mega_moe=True, mega_moe_waves=4, micro_batch=2)
    graph = build_graph(model, strategy)

    assert _ep_a2a_collectives(graph) == []

    monkeypatch.setattr(stage_mod, "ep_imbalance_factor", lambda *_args: 1.0)
    balanced = stage_time(graph.ops, graph.collectives, model, _system(), strategy)

    monkeypatch.setattr(stage_mod, "ep_imbalance_factor", lambda *_args: 2.0)
    imbalanced = stage_time(graph.ops, graph.collectives, model, _system(), strategy)

    assert imbalanced.comm_fwd == pytest.approx(balanced.comm_fwd * 2.0)
    assert imbalanced.comm_bwd == pytest.approx(balanced.comm_bwd * 2.0)
    assert imbalanced.ep_hidden == pytest.approx(balanced.ep_hidden * 2.0)
    assert imbalanced.ep_exposed > balanced.ep_exposed
