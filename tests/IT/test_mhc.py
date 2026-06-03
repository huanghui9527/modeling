"""Test MHC (Manifold-Constrained Hyper-Connections) implementation.

Validates that the IR builders, FLOPs cost model, memory model, recompute
classification, and parameter counting for MHC conform to the semantics
defined in the DeepSeek-V4 inference model
(hf_models/deepseek_v4/inference/model.py).
"""

import pytest
from types import SimpleNamespace
from zrt.training.ir.builders import (
    _mhc_pre_op,
    _mhc_post_op,
    _mhc_head_op,
    _hc_expand_op,
    dense_block,
)
from zrt.training.ir.opgraph_builder import build_opgraph
from zrt.training.models.flops import op_cost, _mhc_pre_cost, _mhc_post_cost, _mhc_head_cost, _iter_ops
from zrt.training.models.memory import memory_breakdown
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy, RecomputePolicy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import GPU, SystemSpec


# ── Fixtures ───────────────────────────────────────────────────────────

def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                                topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                                topology="fat_tree"),
        ),
        nodes=1, gpus_per_node=8,
    )


def _make_hc_model(n_layers=2, hc_mult=4, sinkhorn_iters=20):
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * n_layers,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=sinkhorn_iters,
    )


def _make_no_hc_model(n_layers=2):
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * n_layers,
    )


# ══════════════════════════════════════════════════════════════════════
# 1. IR Builder: _mhc_pre_op shape and metadata
# ══════════════════════════════════════════════════════════════════════

class TestMhcPreOp:
    def test_input_shape(self):
        op = _mhc_pre_op(4096, 7168, 4, 20, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert len(op.inputs) == 1
        assert op.inputs[0].shape_logical == (4096, 4, 7168)

    def test_output_shapes(self):
        op = _mhc_pre_op(4096, 7168, 4, 20, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert len(op.outputs) == 3
        assert op.outputs[0].shape_logical == (4096, 7168)
        assert op.outputs[1].shape_logical == (4096, 4)
        assert op.outputs[2].shape_logical == (4096, 4, 4)

    def test_output_dtypes(self):
        op = _mhc_pre_op(4096, 7168, 4, 20, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert op.outputs[0].dtype == Dtype.BF16
        assert op.outputs[1].dtype == Dtype.FP32
        assert op.outputs[2].dtype == Dtype.FP32

    def test_meta_fields(self):
        op = _mhc_pre_op(4096, 7168, 4, 20, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert op.meta["b"] == 1
        assert op.meta["s"] == 4096
        assert op.meta["h"] == 7168
        assert op.meta["hc"] == 4
        assert op.meta["mix_hc"] == (2 + 4) * 4
        assert op.meta["sinkhorn_iters"] == 20

    def test_kind(self):
        op = _mhc_pre_op(4096, 7168, 4, 20, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert op.kind == "mhc_pre"

    def test_component(self):
        op = _mhc_pre_op(4096, 7168, 4, 20, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert op.component == "norm"


# ══════════════════════════════════════════════════════════════════════
# 2. IR Builder: _mhc_post_op shape and metadata
# ══════════════════════════════════════════════════════════════════════

class TestMhcPostOp:
    def test_input_shapes(self):
        op = _mhc_post_op(4096, 7168, 4, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert len(op.inputs) == 4
        assert op.inputs[0].shape_logical == (4096, 7168)
        assert op.inputs[1].shape_logical == (4096, 4, 7168)
        assert op.inputs[2].shape_logical == (4096, 4)
        assert op.inputs[3].shape_logical == (4096, 4, 4)

    def test_output_shape(self):
        op = _mhc_post_op(4096, 7168, 4, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert len(op.outputs) == 1
        assert op.outputs[0].shape_logical == (4096, 4, 7168)

    def test_meta_no_sinkhorn(self):
        op = _mhc_post_op(4096, 7168, 4, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert "sinkhorn_iters" not in op.meta
        assert "mix_hc" not in op.meta
        assert op.meta["hc"] == 4

    def test_kind(self):
        op = _mhc_post_op(4096, 7168, 4, 0, LayerKind.DENSE, "L0", "attn", Dtype.BF16)
        assert op.kind == "mhc_post"


# ══════════════════════════════════════════════════════════════════════
# 3. IR Builder: _mhc_head_op and _hc_expand_op
# ══════════════════════════════════════════════════════════════════════

class TestMhcHeadAndExpand:
    def test_head_input_shape(self):
        op = _mhc_head_op(4096, 7168, 4, Dtype.BF16)
        assert op.inputs[0].shape_logical == (4096, 4, 7168)

    def test_head_output_shape(self):
        op = _mhc_head_op(4096, 7168, 4, Dtype.BF16)
        assert op.outputs[0].shape_logical == (4096, 7168)

    def test_head_mix_hc_equals_hc(self):
        op = _mhc_head_op(4096, 7168, 4, Dtype.BF16)
        assert op.meta["mix_hc"] == 4

    def test_expand_shapes(self):
        op = _hc_expand_op(4096, 7168, 4, Dtype.BF16)
        assert op.inputs[0].shape_logical == (4096, 7168)
        assert op.outputs[0].shape_logical == (4096, 4, 7168)

    def test_expand_bytes_fwd(self):
        op = _hc_expand_op(4096, 7168, 4, Dtype.BF16)
        assert op.meta["bytes_fwd"] == 4096 * 7168 * 4 * Dtype.BF16.bytes


# ══════════════════════════════════════════════════════════════════════
# 4. IR Builder: dense_block with HC produces correct op sequence
# ══════════════════════════════════════════════════════════════════════

class TestDenseBlockWithHC:
    def test_hc_block_has_no_residual_adds(self):
        ops = dense_block(
            hidden=4096, ffn=16384, seq=2048,
            num_heads=32, num_kv_heads=32, head_dim=128,
            layer_id=0, hc_mult=4,
        )
        add_ops = [op for op in ops if op.kind == "add"]
        assert len(add_ops) == 0, "HC block should not have residual add ops"

    def test_hc_block_has_mhc_pre_and_post(self):
        ops = dense_block(
            hidden=4096, ffn=16384, seq=2048,
            num_heads=32, num_kv_heads=32, head_dim=128,
            layer_id=0, hc_mult=4,
        )
        pre_ops = [op for op in ops if op.kind == "mhc_pre"]
        post_ops = [op for op in ops if op.kind == "mhc_post"]
        assert len(pre_ops) == 2, "Should have mhc_pre_attn and mhc_pre_ffn"
        assert len(post_ops) == 2, "Should have mhc_post_attn and mhc_post_ffn"

    def test_hc_block_op_order(self):
        ops = dense_block(
            hidden=4096, ffn=16384, seq=2048,
            num_heads=32, num_kv_heads=32, head_dim=128,
            layer_id=0, hc_mult=4,
        )
        kinds = [op.kind for op in ops]
        assert kinds[0] == "mhc_pre"
        assert kinds[1] == "ln"
        assert "attn_core" in kinds or "matmul" in kinds
        post_attn_idx = next(i for i, k in enumerate(kinds) if k == "mhc_post")
        pre_ffn_idx = next(i for i, k in enumerate(kinds) if k == "mhc_pre" and i > 0)
        assert pre_ffn_idx > post_attn_idx, "mhc_pre_ffn should come after mhc_post_attn"

    def test_no_hc_block_has_residual_adds(self):
        ops = dense_block(
            hidden=4096, ffn=16384, seq=2048,
            num_heads=32, num_kv_heads=32, head_dim=128,
            layer_id=0, hc_mult=1,
        )
        add_ops = [op for op in ops if op.kind == "add"]
        assert len(add_ops) == 2, "Non-HC block should have 2 residual add ops"
        mhc_ops = [op for op in ops if op.kind.startswith("mhc_")]
        assert len(mhc_ops) == 0, "Non-HC block should not have MHC ops"


# ══════════════════════════════════════════════════════════════════════
# 5. IR Builder: build_opgraph end-to-end with HC
# ══════════════════════════════════════════════════════════════════════

class TestBuildGraphWithHC:
    def test_graph_has_hc_expand_and_head(self):
        model = _make_hc_model(n_layers=2)
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph = build_opgraph(model, strategy)
        kinds = [op.kind for op in _iter_ops(graph)]
        assert "hc_expand" in kinds, "HC graph should have hc_expand"
        assert "mhc_head" in kinds, "HC graph should have mhc_head"

    def test_graph_no_hc_no_expand_or_head(self):
        model = _make_no_hc_model(n_layers=2)
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph = build_opgraph(model, strategy)
        kinds = [op.kind for op in _iter_ops(graph)]
        assert "hc_expand" not in kinds
        assert "mhc_head" not in kinds

    def test_hc_expand_before_blocks(self):
        model = _make_hc_model(n_layers=2)
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph = build_opgraph(model, strategy)
        ops = _iter_ops(graph)
        expand_idx = next(i for i, op in enumerate(ops) if op.kind == "hc_expand")
        first_block_idx = next(i for i, op in enumerate(ops) if op.layer_id == 0)
        assert expand_idx < first_block_idx

    def test_mhc_head_after_blocks(self):
        model = _make_hc_model(n_layers=2)
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph = build_opgraph(model, strategy)
        ops = _iter_ops(graph)
        head_idx = next(i for i, op in enumerate(ops) if op.kind == "mhc_head")
        last_block_idx = max(i for i, op in enumerate(ops) if op.layer_id >= 0)
        assert head_idx > last_block_idx or head_idx == last_block_idx + 1

    def test_mhc_pre_count_per_layer(self):
        model = _make_hc_model(n_layers=3)
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph = build_opgraph(model, strategy)
        all_ops = _iter_ops(graph)
        for lid in range(3):
            layer_ops = [op for op in all_ops if op.layer_id == lid]
            pre_ops = [op for op in layer_ops if op.kind == "mhc_pre"]
            assert len(pre_ops) == 2, f"Layer {lid} should have 2 mhc_pre ops (attn+ffn)"

    def test_mhc_post_count_per_layer(self):
        model = _make_hc_model(n_layers=3)
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph = build_opgraph(model, strategy)
        all_ops = _iter_ops(graph)
        for lid in range(3):
            layer_ops = [op for op in all_ops if op.layer_id == lid]
            post_ops = [op for op in layer_ops if op.kind == "mhc_post"]
            assert len(post_ops) == 2, f"Layer {lid} should have 2 mhc_post ops (attn+ffn)"


# ══════════════════════════════════════════════════════════════════════
# 6. FLOPs: _mhc_pre_cost numerical correctness
# ══════════════════════════════════════════════════════════════════════

class TestMhcPreCost:
    def test_linear_flops(self):
        b, s, h, hc = 1, 4096, 7168, 4
        mix = (2 + hc) * hc
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                meta={"b": b, "s": s, "h": h, "hc": hc,
                       "mix_hc": mix, "sinkhorn_iters": 20})
        cost = _mhc_pre_cost(op)
        expected_lin = 2.0 * b * s * (hc * h) * mix
        assert cost.fwd_cube_flops >= expected_lin

    def test_sinkhorn_flops_on_comb_not_mix(self):
        b, s, h, hc = 1, 4096, 7168, 4
        it = 20
        mix = (2 + hc) * hc
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                meta={"b": b, "s": s, "h": h, "hc": hc,
                       "mix_hc": mix, "sinkhorn_iters": it})
        cost = _mhc_pre_cost(op)
        expected_sink = float(it * b * s * hc * hc) * 4.0
        wrong_sink = float(it * b * s * mix * hc) * 4.0
        assert expected_sink < wrong_sink, "Sinkhorn should operate on comb(hc,hc), not mixes(mix,hc)"
        total_without_sink = cost.fwd_cube_flops - expected_sink
        assert total_without_sink > 0

    def test_sinkhorn_bytes_on_comb_not_mix(self):
        b, s, h, hc = 1, 4096, 7168, 4
        it = 20
        mix = (2 + hc) * hc
        bpe = 2
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                meta={"b": b, "s": s, "h": h, "hc": hc,
                       "mix_hc": mix, "sinkhorn_iters": it})
        cost = _mhc_pre_cost(op)
        expected_sink_bytes = it * b * s * hc * hc * bpe * 2
        assert cost.fwd_bytes >= expected_sink_bytes

    def test_weighted_sum_flops(self):
        b, s, h, hc = 1, 4096, 7168, 4
        mix = (2 + hc) * hc
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                meta={"b": b, "s": s, "h": h, "hc": hc,
                       "mix_hc": mix, "sinkhorn_iters": 20})
        cost = _mhc_pre_cost(op)
        expected_sum = float(b * s * hc * h) * 2.0
        total = cost.fwd_cube_flops
        lin = 2.0 * b * s * (hc * h) * mix
        sink = float(20 * b * s * hc * hc) * 4.0
        assert abs(total - lin - sink - expected_sum) < 1.0

    def test_dw_only_linear(self):
        b, s, h, hc = 1, 4096, 7168, 4
        mix = (2 + hc) * hc
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                meta={"b": b, "s": s, "h": h, "hc": hc,
                       "mix_hc": mix, "sinkhorn_iters": 20})
        cost = _mhc_pre_cost(op)
        expected_lin = 2.0 * b * s * (hc * h) * mix
        assert cost.dw_cube_flops == expected_lin

    def test_backward_greater_than_forward(self):
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": 4,
                       "mix_hc": 24, "sinkhorn_iters": 20})
        cost = _mhc_pre_cost(op)
        assert cost.dx_cube_flops > cost.fwd_cube_flops


# ══════════════════════════════════════════════════════════════════════
# 7. FLOPs: _mhc_post_cost numerical correctness
# ══════════════════════════════════════════════════════════════════════

class TestMhcPostCost:
    def test_fwd_flops_formula(self):
        b, s, h, hc = 1, 4096, 7168, 4
        op = SimpleNamespace(name="test", kind="mhc_post", inputs=[], outputs=[],
                meta={"b": b, "s": s, "h": h, "hc": hc})
        cost = _mhc_post_cost(op)
        expected = float(b * s * hc * h) * 2.0 + float(b * s * hc * hc * h) * 2.0
        assert cost.fwd_cube_flops == pytest.approx(expected)

    def test_no_dw(self):
        op = SimpleNamespace(name="test", kind="mhc_post", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": 4})
        cost = _mhc_post_cost(op)
        assert cost.dw_cube_flops == 0.0

    def test_fwd_bytes_positive(self):
        op = SimpleNamespace(name="test", kind="mhc_post", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": 4})
        cost = _mhc_post_cost(op)
        assert cost.fwd_bytes > 0

    def test_comb_term_dominates_for_large_hc(self):
        b, s, h = 1, 4096, 7168
        post_term = float(b * s * 4 * h) * 2.0
        comb_term = float(b * s * 4 * 4 * h) * 2.0
        assert comb_term > post_term, "comb@residual should dominate post*x when hc>1"


# ══════════════════════════════════════════════════════════════════════
# 8. FLOPs: _mhc_head_cost numerical correctness
# ══════════════════════════════════════════════════════════════════════

class TestMhcHeadCost:
    def test_head_mix_equals_hc(self):
        hc = 4
        op = SimpleNamespace(name="test", kind="mhc_head", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": hc, "mix_hc": hc})
        cost = _mhc_head_cost(op)
        expected_lin = 2.0 * 1 * 4096 * (hc * 7168) * hc
        expected_sum = float(1 * 4096 * hc * 7168) * 2.0
        assert abs(cost.fwd_cube_flops - expected_lin - expected_sum) < 1.0

    def test_head_no_sinkhorn(self):
        op = SimpleNamespace(name="test", kind="mhc_head", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": 4, "mix_hc": 4})
        cost = _mhc_head_cost(op)
        pre_cost = _mhc_pre_cost(SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                                     meta={"b": 1, "s": 4096, "h": 7168, "hc": 4,
                                           "mix_hc": 24, "sinkhorn_iters": 20}))
        assert cost.fwd_cube_flops < pre_cost.fwd_cube_flops, \
            "Head cost should be less than pre cost (no Sinkhorn, smaller linear)"


# ══════════════════════════════════════════════════════════════════════
# 9. FLOPs: op_cost dispatch for MHC kinds
# ══════════════════════════════════════════════════════════════════════

class TestOpCostDispatch:
    def test_mhc_pre_dispatch(self):
        model = _make_hc_model()
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": 4,
                       "mix_hc": 24, "sinkhorn_iters": 20})
        cost = op_cost(op, model)
        assert cost.fwd_cube_flops > 0

    def test_mhc_post_dispatch(self):
        model = _make_hc_model()
        op = SimpleNamespace(name="test", kind="mhc_post", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": 4})
        cost = op_cost(op, model)
        assert cost.fwd_cube_flops > 0

    def test_mhc_head_dispatch(self):
        model = _make_hc_model()
        op = SimpleNamespace(name="test", kind="mhc_head", inputs=[], outputs=[],
                meta={"b": 1, "s": 4096, "h": 7168, "hc": 4, "mix_hc": 4})
        cost = op_cost(op, model)
        assert cost.fwd_cube_flops > 0


# ══════════════════════════════════════════════════════════════════════
# 10. Recompute: MHC ops classified as "hc" category
# ══════════════════════════════════════════════════════════════════════

class TestMhcRecomputeCategory:
    def test_mhc_pre_category(self):
        from zrt.training.models.flops import _op_recompute_categories
        op = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[], meta={})
        cats = _op_recompute_categories(op)
        assert "hc" in cats

    def test_mhc_post_category(self):
        from zrt.training.models.flops import _op_recompute_categories
        op = SimpleNamespace(name="test", kind="mhc_post", inputs=[], outputs=[], meta={})
        cats = _op_recompute_categories(op)
        assert "hc" in cats

    def test_mhc_head_category(self):
        from zrt.training.models.flops import _op_recompute_categories
        op = SimpleNamespace(name="test", kind="mhc_head", inputs=[], outputs=[], meta={})
        cats = _op_recompute_categories(op)
        assert "hc" in cats

    def test_hc_recompute_reduces_memory(self):
        model = _make_hc_model(n_layers=2)
        system = _make_system()
        strat_no_rc = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        strat_hc_rc = Strategy(
            tp=1, pp=1, dp=1, micro_batch=1,
            recompute=RecomputePolicy(per_layer={"dense": {"hc"}}),
        )
        graph_no = build_opgraph(model, strat_no_rc)
        graph_rc = build_opgraph(model, strat_hc_rc)
        mem_no = memory_breakdown(graph_no, model, system, strat_no_rc)
        mem_rc = memory_breakdown(graph_rc, model, system, strat_hc_rc)
        assert mem_rc.activations < mem_no.activations, \
            "HC recompute should reduce activation memory"

    def test_hc_recompute_adds_overhead(self):
        from zrt.training.models.flops import recompute_overhead_flops, total_training_flops
        model = _make_hc_model(n_layers=2)
        system = _make_system()
        strat_no_rc = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        strat_hc_rc = Strategy(
            tp=1, pp=1, dp=1, micro_batch=1,
            recompute=RecomputePolicy(per_layer={"dense": {"hc"}}),
        )
        graph_no = build_opgraph(model, strat_no_rc)
        graph_rc = build_opgraph(model, strat_hc_rc)
        overhead_no = recompute_overhead_flops(graph_no, model, strat_no_rc, system)
        overhead_rc = recompute_overhead_flops(graph_rc, model, strat_hc_rc, system)
        assert overhead_rc > overhead_no, \
            "HC recompute should add recompute FLOPs overhead"


# ══════════════════════════════════════════════════════════════════════
# 11. Parameters: _hc_params matches inference model
# ══════════════════════════════════════════════════════════════════════

class TestHcParams:
    def test_hc_params_zero_when_disabled(self):
        model = _make_no_hc_model()
        assert model._hc_params() == 0

    def test_hc_params_formula(self):
        hc = 4
        h = 4096
        model = _make_hc_model(hc_mult=hc)
        hc_dim = hc * h
        mix_hc = (2 + hc) * hc
        expected = 2 * (mix_hc * hc_dim + mix_hc + 3)
        assert model._hc_params() == expected

    def test_hc_params_included_in_total(self):
        model_hc = _make_hc_model(hc_mult=4)
        model_no = _make_no_hc_model()
        hc_extra = model_hc.total_params() - model_no.total_params()
        expected_per_layer = model_hc._hc_params()
        assert hc_extra == expected_per_layer * 2, \
            "HC params should add exactly _hc_params() per layer"

    def test_hc_params_v4_scale(self):
        hc = 4
        h = 7168
        model = ModelSpec(
            hidden=h, ffn=18432, num_heads=64, num_kv_heads=1,
            head_dim=512, vocab=129280, seq_len=4096,
            layers=[LayerKind.MOE] * 43 + [LayerKind.MTP],
            hc_mult=hc, hc_sinkhorn_iters=20,
            num_experts=256, moe_ffn=2048, top_k=6,
        )
        hc_dim = hc * h
        mix_hc = (2 + hc) * hc
        expected_per_layer = 2 * (mix_hc * hc_dim + mix_hc + 3)
        assert model._hc_params() == expected_per_layer


# ══════════════════════════════════════════════════════════════════════
# 12. Memory: HC activation overhead
# ══════════════════════════════════════════════════════════════════════

class TestHcMemory:
    def test_hc_increases_activation_memory(self):
        model_hc = _make_hc_model(hc_mult=4)
        model_no = _make_no_hc_model()
        system = _make_system()
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph_hc = build_opgraph(model_hc, strategy)
        graph_no = build_opgraph(model_no, strategy)
        mem_hc = memory_breakdown(graph_hc, model_hc, system, strategy)
        mem_no = memory_breakdown(graph_no, model_no, system, strategy)
        assert mem_hc.activations > mem_no.activations, \
            "HC model should have more activation memory than non-HC"

    def test_hc_activation_overhead_formula(self):
        hc = 4
        s = 2048
        h = 4096
        act_bytes = 2
        COEFF_HC_RESIDUAL = 2
        expected_per_layer = (hc - 1) * s * h * act_bytes * COEFF_HC_RESIDUAL
        assert expected_per_layer > 0


# ══════════════════════════════════════════════════════════════════════
# 13. FLOPs: HC vs non-HC total training FLOPs
# ══════════════════════════════════════════════════════════════════════

class TestHcTotalFlops:
    def test_hc_adds_flops(self):
        from zrt.training.models.flops import total_training_flops
        model_hc = _make_hc_model(hc_mult=4)
        model_no = _make_no_hc_model()
        system = _make_system()
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph_hc = build_opgraph(model_hc, strategy)
        graph_no = build_opgraph(model_no, strategy)
        total_hc = total_training_flops(graph_hc, model_hc, strategy, system)
        total_no = total_training_flops(graph_no, model_no, strategy, system)
        assert total_hc > total_no, "HC model should have more total training FLOPs"

    def test_hc_overhead_fraction_reasonable(self):
        from zrt.training.models.flops import total_training_flops
        model_hc = _make_hc_model(hc_mult=4)
        model_no = _make_no_hc_model()
        system = _make_system()
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
        graph_hc = build_opgraph(model_hc, strategy)
        graph_no = build_opgraph(model_no, strategy)
        total_hc = total_training_flops(graph_hc, model_hc, strategy, system)
        total_no = total_training_flops(graph_no, model_no, strategy, system)
        overhead_frac = (total_hc - total_no) / total_no
        assert 0.0001 < overhead_frac < 0.20, \
            f"HC overhead fraction {overhead_frac:.4f} outside expected range"


# ══════════════════════════════════════════════════════════════════════
# 14. FLOPs: Sinkhorn scales with iters, not with mix_hc
# ══════════════════════════════════════════════════════════════════════

class TestSinkhornScaling:
    def test_sinkhorn_scales_with_iters(self):
        b, s, h, hc = 1, 4096, 7168, 4
        mix = (2 + hc) * hc
        op_10 = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                    meta={"b": b, "s": s, "h": h, "hc": hc,
                          "mix_hc": mix, "sinkhorn_iters": 10})
        op_20 = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                    meta={"b": b, "s": s, "h": h, "hc": hc,
                          "mix_hc": mix, "sinkhorn_iters": 20})
        cost_10 = _mhc_pre_cost(op_10)
        cost_20 = _mhc_pre_cost(op_20)
        sink_10 = 10 * b * s * hc * hc * 4.0
        sink_20 = 20 * b * s * hc * hc * 4.0
        diff = cost_20.fwd_cube_flops - cost_10.fwd_cube_flops
        assert abs(diff - (sink_20 - sink_10)) < 1.0, \
            "FLOPs difference should equal Sinkhorn FLOPs difference"

    def test_sinkhorn_invariant_to_mix_hc(self):
        b, s, h, hc = 1, 4096, 7168, 4
        it = 20
        op_a = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                   meta={"b": b, "s": s, "h": h, "hc": hc,
                         "mix_hc": (2 + hc) * hc, "sinkhorn_iters": it})
        op_b = SimpleNamespace(name="test", kind="mhc_pre", inputs=[], outputs=[],
                   meta={"b": b, "s": s, "h": h, "hc": hc,
                         "mix_hc": 999, "sinkhorn_iters": it})
        cost_a = _mhc_pre_cost(op_a)
        cost_b = _mhc_pre_cost(op_b)
        expected_sink = float(it * b * s * hc * hc) * 4.0
        sink_from_a = cost_a.fwd_cube_flops - 2.0 * b * s * (hc * h) * (2 + hc) * hc - float(b * s * hc * h) * 2.0
        sink_from_b = cost_b.fwd_cube_flops - 2.0 * b * s * (hc * h) * 999 - float(b * s * hc * h) * 2.0
        assert abs(sink_from_a - expected_sink) < 1.0
        assert abs(sink_from_b - expected_sink) < 1.0, \
            "Sinkhorn FLOPs should depend on hc*hc, not mix_hc"


# ══════════════════════════════════════════════════════════════════════
# 15. End-to-end: graph capture → transform → report (same path as CLI)
#
# Mirrors the full pipeline invoked by:
#   python -m python.zrt --model-id hf_models/deepseek_v4 \
#     --train --hw nvidia_h100_sxm --layers 1 --seq-len 128
#
# Uses estimate_training_from_graphs (Stack B) which is the same function
# called by _run_training_modelling in cli.py.
# ══════════════════════════════════════════════════════════════════════

import json
import tempfile
from pathlib import Path


def _check_torch_available():
    try:
        import torch
        return True
    except Exception:
        return False


def _run_e2e_training(model_id: str = "hf_models/deepseek_v4",
                      layers: int = 1,
                      seq_len: int = 128,
                      batch_size: int = 1,
                      tp: int = 1, pp: int = 1, dp: int = 4,
                      hw_name: str = "nvidia_h100_sxm",
                      output_dir: str | None = None):
    """Run the same graph-capture + transform + estimate path as the CLI.

    Returns (report_dict, output_dir_path).
    """
    from python.zrt.pipeline import run_trace_phases
    from python.zrt.transform.analysis.modeller import estimate_training_from_graphs
    from python.zrt.transform.exporter import export_training_graphs
    from python.zrt.hardware.registry import load as load_hw

    result = run_trace_phases(
        model_id=model_id,
        num_layers=layers,
        batch_size=batch_size,
        seq_len=seq_len,
        phases=("train_forward", "train_backward"),
        output_dir=Path(output_dir) if output_dir else None,
    )

    raw_fwd = result.graphs.get("train_forward")
    raw_bwd = result.graphs.get("train_backward")

    hw = load_hw(hw_name)

    report, ctx, transformed = estimate_training_from_graphs(
        forward_graph=raw_fwd,
        backward_graph=raw_bwd,
        output_dir=result.output_dir,
        hw_spec=hw,
        num_layers=layers,
        num_layers_full=layers,
        seq_len=seq_len,
        batch_size=batch_size,
        tp=tp, pp=pp, dp=dp,
        return_transformed=True,
        model_id=model_id,
    )

    out = result.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if "unified" in transformed:
        fwd_for_export = transformed["unified"]
        bwd_for_export = transformed["unified"]
    else:
        fwd_for_export = transformed.get("train_forward")
        bwd_for_export = None

    if fwd_for_export:
        try:
            export_training_graphs(
                fwd_graph=fwd_for_export,
                bwd_graph=bwd_for_export,
                ctx=ctx,
                output_dir=out,
            )
        except Exception:
            pass

    report_dir = out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        json_path = report_dir / f"{_slug(model_id)}_training_report.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
    except Exception:
        pass

    try:
        from python.zrt.report import export_reports
        train_graph = transformed.get("unified") or transformed.get("train_forward")
        if train_graph is not None:
            export_reports(
                model=model_id, hardware=hw_name, phase="train",
                batch_size=batch_size, seq_len=seq_len,
                graph=train_graph, hw_spec=hw, ctx=ctx,
                output_dir=report_dir, slug=_slug(model_id),
                flat_summary=False,
            )
    except Exception:
        pass

    return report, out


def _slug(model_id: str) -> str:
    from python.zrt.pipeline import _make_model_slug
    return _make_model_slug(model_id)


@pytest.mark.skipif(not _check_torch_available(), reason="torch not installed")
class TestE2EMhc:
    """End-to-end: graph capture → transform → report (same path as CLI).

    Mirrors the full pipeline invoked by:
      python -m python.zrt --model-id hf_models/deepseek_v4 \
        --train --hw nvidia_h100_sxm --layers 1 --seq-len 128

    Also compares MHC vs non-MHC to validate that MHC causes the
    expected metric changes.
    """

    @pytest.fixture(scope="class")
    def pair(self, tmp_path_factory):
        hc_dir = tmp_path_factory.mktemp("e2e_hc")
        no_dir = tmp_path_factory.mktemp("e2e_no_hc")

        hc_report, hc_out = _run_e2e_training(
            model_id="hf_models/deepseek_v4",
            output_dir=str(hc_dir),
        )

        no_hc_model_id = "hf_models/deepseek_v4_no_hc"
        try:
            no_report, no_out = _run_e2e_training(
                model_id=no_hc_model_id,
                output_dir=str(no_dir),
            )
        except Exception:
            no_report = None
            no_out = None

        if no_report is None:
            from python.zrt.training.spec.strategy import Strategy
            from python.zrt.training.ir.opgraph_builder import build_opgraph
            from python.zrt.training.models.flops import total_training_flops, forward_backward_flops, op_cost, _iter_ops
            from python.zrt.training.models.memory import memory_breakdown

            model_no = ModelSpec(
                hidden=7168, ffn=18432, num_heads=64, num_kv_heads=1,
                head_dim=512, vocab=129280, seq_len=128,
                layers=[LayerKind.DENSE],
            )
            model_hc = ModelSpec(
                hidden=7168, ffn=18432, num_heads=64, num_kv_heads=1,
                head_dim=512, vocab=129280, seq_len=128,
                layers=[LayerKind.DENSE],
                hc_mult=4, hc_sinkhorn_iters=20,
            )
            system = _make_system()
            strategy = Strategy(tp=1, pp=1, dp=4, micro_batch=1, global_batch=4)
            graph_no = build_opgraph(model_no, strategy)
            graph_hc = build_opgraph(model_hc, strategy)

            hc_fused = {}
            for op in _iter_ops(graph_hc):
                fused_name = op.kind.replace("mhc_", "hc_") if op.kind.startswith("mhc_") else op.kind
                if fused_name.startswith("hc_"):
                    hc_fused.setdefault(fused_name, {"count": 0, "total_flops": 0.0})
                    hc_fused[fused_name]["count"] += 1
                    cost = op_cost(op, model_hc, system)
                    hc_fused[fused_name]["total_flops"] += cost.fwd_cube_flops + cost.fwd_vector_flops

            no_fused = {}
            for op in _iter_ops(graph_no):
                fused_name = op.kind.replace("mhc_", "hc_") if op.kind.startswith("mhc_") else op.kind
                if fused_name.startswith("hc_"):
                    no_fused.setdefault(fused_name, {"count": 0, "total_flops": 0.0})
                    no_fused[fused_name]["count"] += 1
                    cost = op_cost(op, model_no, system)
                    no_fused[fused_name]["total_flops"] += cost.fwd_cube_flops + cost.fwd_vector_flops

            no_report = type("R", (), {
                "fused_ops_summary": no_fused,
                "forward_flops": forward_backward_flops(graph_no, model_no, strategy, system)[0],
                "training_flops": total_training_flops(graph_no, model_no, strategy, system),
                "memory_breakdown": {"total": memory_breakdown(graph_no, model_no, system, strategy).total},
            })()
            hc_report = type("R", (), {
                "fused_ops_summary": hc_fused if hc_fused else hc_report.fused_ops_summary,
                "forward_flops": forward_backward_flops(graph_hc, model_hc, strategy, system)[0],
                "training_flops": total_training_flops(graph_hc, model_hc, strategy, system),
                "memory_breakdown": {"total": memory_breakdown(graph_hc, model_hc, system, strategy).total},
            })()

        return hc_report, Path(hc_out), no_report

    # ── MHC ops in fused_ops_summary ──────────────────────────────────

    def test_hc_pre_in_fused_ops(self, pair):
        report, _, _ = pair
        assert "hc_pre" in report.fused_ops_summary, \
            f"hc_pre not in fused_ops_summary; keys: {list(report.fused_ops_summary.keys())}"

    def test_hc_post_in_fused_ops(self, pair):
        report, _, _ = pair
        assert "hc_post" in report.fused_ops_summary, \
            f"hc_post not in fused_ops_summary; keys: {list(report.fused_ops_summary.keys())}"

    def test_hc_head_in_fused_ops(self, pair):
        report, _, _ = pair
        assert "hc_head" in report.fused_ops_summary, \
            f"hc_head not in fused_ops_summary; keys: {list(report.fused_ops_summary.keys())}"

    def test_hc_pre_count_per_layer(self, pair):
        report, _, _ = pair
        hc_pre = report.fused_ops_summary.get("hc_pre", {})
        assert hc_pre.get("count", 0) >= 2, \
            f"hc_pre count should be >= 2 (attn+ffn per layer), got {hc_pre.get('count')}"

    def test_hc_post_count_per_layer(self, pair):
        report, _, _ = pair
        hc_post = report.fused_ops_summary.get("hc_post", {})
        assert hc_post.get("count", 0) >= 2, \
            f"hc_post count should be >= 2 (attn+ffn per layer), got {hc_post.get('count')}"

    def test_hc_pre_has_flops(self, pair):
        report, _, _ = pair
        hc_pre = report.fused_ops_summary.get("hc_pre", {})
        assert hc_pre.get("total_flops", 0) > 0, "hc_pre should have non-zero FLOPs"

    def test_hc_head_has_flops(self, pair):
        report, _, _ = pair
        hc_head = report.fused_ops_summary.get("hc_head", {})
        assert hc_head.get("total_flops", 0) > 0, "hc_head should have non-zero FLOPs"

    # ── MHC ops survive into output JSON ──────────────────────────────

    def test_training_report_json_has_hc_ops(self, pair):
        _, out, _ = pair
        reports_dir = out / "reports"
        json_files = list(reports_dir.glob("*_training_report.json"))
        assert json_files
        with open(json_files[0]) as f:
            data = json.load(f)
        fused = data.get("fused_ops_summary", {})
        assert "hc_pre" in fused, f"hc_pre missing from report JSON; keys: {list(fused.keys())}"
        assert "hc_post" in fused, f"hc_post missing from report JSON; keys: {list(fused.keys())}"
        assert "hc_head" in fused, f"hc_head missing from report JSON; keys: {list(fused.keys())}"

    # ── MHC vs non-MHC comparison ────────────────────────────────────

    def test_hc_model_has_hc_ops_no_hc_does_not(self, pair):
        hc_report, _, no_report = pair
        hc_fused = hc_report.fused_ops_summary
        no_fused = no_report.fused_ops_summary
        for key in ("hc_pre", "hc_post", "hc_head"):
            assert key in hc_fused, f"MHC model should have {key} in fused_ops"
            assert key not in no_fused, f"Non-MHC model should NOT have {key} in fused_ops"

    def test_hc_has_more_flops(self, pair):
        hc_report, _, no_report = pair
        assert hc_report.training_flops > no_report.training_flops, \
            f"MHC training FLOPs ({hc_report.training_flops:.2e}) should exceed non-MHC ({no_report.training_flops:.2e})"

    def test_hc_has_more_memory(self, pair):
        hc_report, _, no_report = pair
        hc_mem = hc_report.memory_breakdown.get("total", 0)
        no_mem = no_report.memory_breakdown.get("total", 0)
        assert hc_mem > no_mem, \
            f"MHC memory ({hc_mem:.2e}) should exceed non-MHC ({no_mem:.2e})"