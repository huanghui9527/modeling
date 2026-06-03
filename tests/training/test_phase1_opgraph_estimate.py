"""Phase 1 测试 — Stack A estimate() 走 OpGraph 路径的数值一致性。

验证点：
  1. total_training_flops 对旧 Graph 和新 OpGraph 产出相同值
  2. forward_backward_flops 对旧 Graph 和新 OpGraph 产出相同 fwd/bwd 值
  3. recompute_overhead_flops 对旧 Graph 和新 OpGraph 产出相同值
  4. estimate() 走 OpGraph 路径后 step_time_ms 与旧路径一致
  5. op_cost_from_node 对单个 OpNode 的 cost 与 op_cost(Op) 一致

原理：Phase 1 的核心目标是让 Stack A 的入口 (estimator.py) 切换到
OpGraph，同时保持下游 flops 计算的数值不变。旧 Graph 路径的结果
作为基准值 (oracle)，OpGraph 路径必须精确匹配。
"""

import pytest

from zrt.ir.graph import OpGraph
from zrt.training.ir.builders import build_graph
from zrt.training.ir.opgraph_builder import build_opgraph
from zrt.training.models.flops import (
    OpCost,
    op_cost,
    total_training_flops,
    forward_backward_flops,
    recompute_overhead_flops,
)
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import RecomputePolicy, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec


def _make_dense_model():
    return ModelSpec(
        hidden=4096, ffn=11008, seq_len=2048,
        num_heads=32, num_kv_heads=32, head_dim=128,
        layers=[LayerKind.DENSE] * 4, vocab=32000, act_dtype=Dtype.BF16,
    )


def _make_moe_model():
    return ModelSpec(
        hidden=2048, ffn=8192, moe_ffn=2048,
        seq_len=1024, num_heads=16, num_kv_heads=16, head_dim=128,
        layers=[LayerKind.DENSE] * 2 + [LayerKind.MOE] * 2,
        num_experts=8, top_k=2, n_shared_experts=1,
        vocab=32000, act_dtype=Dtype.BF16,
    )


def _make_system(world_size: int = 1):
    gpu = GPU(
        name="H100", flops_bf16=989.0, flops_fp8=1979.0,
        hbm_gb=80, hbm_bw_gbps=3350,
    )
    return SystemSpec(
        gpu=gpu, host_mem_gb=2048,
        nodes=1, gpus_per_node=world_size,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                                topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                                topology="fat_tree"),
        ),
    )


def _make_strategy(**kwargs):
    defaults = dict(tp=1, pp=1, ep=1, dp=1, cp=1, micro_batch=1, global_batch=32)
    defaults.update(kwargs)
    return Strategy(**defaults)


class TestTotalTrainingFlopsOpGraph:
    """total_training_flops 对旧 Graph 和新 OpGraph 必须产出相同值。

    观测点：6P 规则下只计 compute-bound ops (matmul, attn_core)，
    memory-bound ops (rmsnorm, swiglu, rope, add) 被排除。
    """

    def test_dense_tp1_flops_match(self):
        model = _make_dense_model()
        strategy = _make_strategy(tp=1)
        system = _make_system()

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_flops = total_training_flops(old_graph, model, strategy, system)
        new_flops = total_training_flops(new_graph, model, strategy, system)

        assert new_flops == old_flops, (
            f"OpGraph flops {new_flops:.6e} != Graph flops {old_flops:.6e}"
        )

    def test_dense_tp2_flops_match(self):
        model = _make_dense_model()
        strategy = _make_strategy(tp=2)
        system = _make_system(world_size=2)

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_flops = total_training_flops(old_graph, model, strategy, system)
        new_flops = total_training_flops(new_graph, model, strategy, system)

        assert new_flops == old_flops

    def test_moe_tp1_ep1_flops_match(self):
        model = _make_moe_model()
        strategy = _make_strategy(tp=1, ep=1)
        system = _make_system()

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_flops = total_training_flops(old_graph, model, strategy, system)
        new_flops = total_training_flops(new_graph, model, strategy, system)

        assert new_flops == old_flops

    def test_moe_tp2_ep2_flops_match(self):
        model = _make_moe_model()
        strategy = _make_strategy(tp=2, ep=2)
        system = _make_system(world_size=4)

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_flops = total_training_flops(old_graph, model, strategy, system)
        new_flops = total_training_flops(new_graph, model, strategy, system)

        assert new_flops == old_flops


class TestForwardBackwardFlopsOpGraph:
    """forward_backward_flops 对旧 Graph 和新 OpGraph 的 fwd/bwd 分别匹配。

    观测点：fwd 只含 fwd_cube_flops (compute-bound)，
    bwd = dx_cube_flops + dw_cube_flops。
    """

    def test_dense_fwd_bwd_match(self):
        model = _make_dense_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_fwd, old_bwd = forward_backward_flops(old_graph, model, strategy, system)
        new_fwd, new_bwd = forward_backward_flops(new_graph, model, strategy, system)

        assert new_fwd == old_fwd, f"fwd: {new_fwd:.6e} != {old_fwd:.6e}"
        assert new_bwd == old_bwd, f"bwd: {new_bwd:.6e} != {old_bwd:.6e}"

    def test_moe_fwd_bwd_match(self):
        model = _make_moe_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_fwd, old_bwd = forward_backward_flops(old_graph, model, strategy, system)
        new_fwd, new_bwd = forward_backward_flops(new_graph, model, strategy, system)

        assert new_fwd == old_fwd
        assert new_bwd == old_bwd


class TestRecomputeOverheadFlopsOpGraph:
    """recompute_overhead_flops 对旧 Graph 和新 OpGraph 匹配。

    观测点：selective recompute 只重算 attention 区域的 ops。
    """

    def test_dense_selective_recompute_match(self):
        model = _make_dense_model()
        rc = RecomputePolicy(per_layer={
            LayerKind.DENSE: {"attn_core"},
        })
        strategy = _make_strategy(recompute=rc)
        system = _make_system()

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_rc = recompute_overhead_flops(old_graph, model, strategy, system)
        new_rc = recompute_overhead_flops(new_graph, model, strategy, system)

        assert new_rc == old_rc


class TestOpCostFromNode:
    """op_cost_from_node 对单个 OpNode 的 cost 与 op_cost(Op) 一致。

    原理：通过 build_opgraph 和 build_graph 构建同一个模型，
    逐个对比每个 op 的 OpCost。
    """

    def test_per_op_cost_dense_match(self):
        from zrt.training.models.flops import op_cost_from_node

        model = _make_dense_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_ops = old_graph.ops
        new_nodes = [n for n in new_graph.nodes.values() if not n.is_comm]

        assert len(new_nodes) == len(old_ops), (
            f"节点数不匹配: {len(new_nodes)} vs {len(old_ops)}"
        )

        for op, node in zip(old_ops, new_nodes):
            old_cost = op_cost(op, model, system)
            new_cost = op_cost_from_node(node, model, system)
            assert new_cost.fwd_cube_flops == old_cost.fwd_cube_flops, (
                f"{op.name}: fwd_cube {new_cost.fwd_cube_flops} != {old_cost.fwd_cube_flops}"
            )
            assert new_cost.dx_cube_flops == old_cost.dx_cube_flops, (
                f"{op.name}: dx_cube {new_cost.dx_cube_flops} != {old_cost.dx_cube_flops}"
            )
            assert new_cost.dw_cube_flops == old_cost.dw_cube_flops, (
                f"{op.name}: dw_cube {new_cost.dw_cube_flops} != {old_cost.dw_cube_flops}"
            )
            assert new_cost.fwd_bytes == old_cost.fwd_bytes, (
                f"{op.name}: fwd_bytes {new_cost.fwd_bytes} != {old_cost.fwd_bytes}"
            )

    def test_per_op_cost_moe_match(self):
        from zrt.training.models.flops import op_cost_from_node

        model = _make_moe_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        new_graph = build_opgraph(model, strategy)

        old_ops = old_graph.ops
        new_nodes = [n for n in new_graph.nodes.values() if not n.is_comm]

        assert len(new_nodes) == len(old_ops)

        for op, node in zip(old_ops, new_nodes):
            old_cost = op_cost(op, model, system)
            new_cost = op_cost_from_node(node, model, system)
            assert new_cost.fwd_cube_flops == old_cost.fwd_cube_flops, (
                f"{op.name}: fwd_cube {new_cost.fwd_cube_flops} != {old_cost.fwd_cube_flops}"
            )
            assert new_cost.fwd_bytes == old_cost.fwd_bytes, (
                f"{op.name}: fwd_bytes {new_cost.fwd_bytes} != {old_cost.fwd_bytes}"
            )


class TestEstimateOpGraph:
    """estimate() 走 OpGraph 路径后 step_time_ms 与旧路径一致。

    观测点：step_time_ms 是最终产物，包含 pipeline_time + optimizer_time，
    任何中间环节的不一致都会在这里暴露。
    """

    def test_dense_estimate_step_time_match(self):
        from zrt.training.search.estimator import estimate

        model = _make_dense_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        old_report = estimate(model, system, strategy, graph=old_graph)

        new_graph = build_opgraph(model, strategy)
        new_report = estimate(model, system, strategy, graph=new_graph)

        assert new_report.step_time_ms == pytest.approx(old_report.step_time_ms, rel=1e-9), (
            f"step_time_ms: {new_report.step_time_ms} != {old_report.step_time_ms}"
        )

    def test_dense_estimate_total_flops_match(self):
        from zrt.training.search.estimator import estimate

        model = _make_dense_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        old_report = estimate(model, system, strategy, graph=old_graph)

        new_graph = build_opgraph(model, strategy)
        new_report = estimate(model, system, strategy, graph=new_graph)

        assert new_report.total_flops == old_report.total_flops

    def test_dense_estimate_mfu_match(self):
        from zrt.training.search.estimator import estimate

        model = _make_dense_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        old_report = estimate(model, system, strategy, graph=old_graph)

        new_graph = build_opgraph(model, strategy)
        new_report = estimate(model, system, strategy, graph=new_graph)

        assert new_report.mfu == pytest.approx(old_report.mfu, rel=1e-9)

    def test_moe_estimate_step_time_match(self):
        from zrt.training.search.estimator import estimate

        model = _make_moe_model()
        strategy = _make_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        old_report = estimate(model, system, strategy, graph=old_graph)

        new_graph = build_opgraph(model, strategy)
        new_report = estimate(model, system, strategy, graph=new_graph)

        assert new_report.step_time_ms == pytest.approx(old_report.step_time_ms, rel=1e-9)

    def test_estimate_with_opgraph_auto_builds(self):
        """Phase 2: estimate() 不传 graph 时委托 estimate_via_pipeline()。"""
        from zrt.training.search.estimator import estimate, estimate_via_pipeline

        model = _make_dense_model()
        strategy = _make_strategy()
        system = _make_system()

        auto_report = estimate(model, system, strategy)
        pipeline_report = estimate_via_pipeline(model, system, strategy)

        assert auto_report.step_time_ms == pytest.approx(pipeline_report.step_time_ms, rel=1e-9)
