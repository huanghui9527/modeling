"""Tests for cast_pass (Stage D2/D4) and QuantPolicy fusion toggle.

Verifies:
  - BF16 baseline: no cast ops inserted
  - Region quant: cast op inserted at residual-add boundary (FP8/FP4 →
    residual stream)
  - cast op metadata fields populated correctly
  - cast op carries layer_id from consumer (PP-aware)
  - topo_sort succeeds after cast insertion
  - QuantPolicy fused vs unfused changes cast cost only
  - `assume_all_casts_fused=true` (default) returns zero cost
  - `assume_all_casts_fused=false` materializes cast HBM bytes
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from zrt.ir.edge import Edge
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta
from zrt.training.ir.builders import build_opgraph_direct
from zrt.training.ir.cast_pass import insert_cast_pass_opgraph
from zrt.training.models.flops import _cast_cost, op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import QuantPolicy, Strategy


def _moe_model(**kw) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE],
        num_experts=8, moe_ffn=256, top_k=2, n_shared_experts=1,
        **kw,
    )


def _dense_model(**kw) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.DENSE], **kw,
    )


def _count_casts(g: OpGraph) -> int:
    return sum(1 for n in g.nodes.values() if n.attrs.get("spec_kind") == "cast")


def _cast_nodes(g: OpGraph) -> list[OpNode]:
    return [n for n in g.nodes.values() if n.attrs.get("spec_kind") == "cast"]


def _node_as_op(n: OpNode):
    return SimpleNamespace(meta=n.attrs)


# ── BF16 baseline: zero casts ─────────────────────────────────────────────


def test_bf16_baseline_inserts_no_casts():
    m = _moe_model()
    g = build_opgraph_direct(m, Strategy())
    assert _count_casts(g) == 0


def test_dense_block_baseline_inserts_no_casts():
    m = _dense_model()
    g = build_opgraph_direct(m, Strategy())
    assert _count_casts(g) == 0


# ── Region quant triggers cast insertion ─────────────────────────────────


def test_moe_quant_inserts_cast_before_residual_add():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_opgraph_direct(m, Strategy())
    casts = _cast_nodes(g)
    assert len(casts) >= 1
    assert any("residual2" in c.id for c in casts)


def test_attn_quant_inserts_cast_before_residual1():
    m = _moe_model(attn_act_dtype=Dtype.FP8_E4M3)
    g = build_opgraph_direct(m, Strategy())
    assert any(
        n.attrs.get("spec_kind") == "cast" and "residual1" in n.id
        for n in g.nodes.values()
    ), "expected cast before residual1 when attn region is FP8"


# ── Cast op metadata is correct ──────────────────────────────────────────


def test_cast_metadata_records_src_dst_and_amax():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_opgraph_direct(m, Strategy())
    cast = next(n for n in g.nodes.values()
                if n.attrs.get("spec_kind") == "cast" and "residual2" in n.id)
    assert cast.attrs["src_dtype"] is Dtype.FP8_E4M3
    assert cast.attrs["dst_dtype"] is Dtype.BF16
    assert cast.attrs["needs_amax"] is False
    assert cast.attrs["num_elements"] > 0


def test_cast_needs_amax_when_quantizing_to_low_precision():
    """LN1 epilog produces moe_act dtype when MoE quantization is on.
    If we artificially force a quantize cast (BF16 → FP8 somewhere), the
    cast meta should set needs_amax=True.
    """
    prod = OpNode(
        id="prod", op_type="aten.rms_norm.default",
        inputs=[TensorMeta.from_shape_dtype("x", (64, 128), Dtype.BF16)],
        outputs=[TensorMeta.from_shape_dtype("y", (64, 128), Dtype.BF16)],
        attrs={"spec_kind": "rmsnorm", "layer_kind": "moe", "source": "model_spec"},
        scope="prod", category="compute", layer="0", component="norm",
    )
    cons = OpNode(
        id="cons", op_type="aten.mm.default",
        inputs=[TensorMeta.from_shape_dtype("y", (64, 128), Dtype.BF16)],
        outputs=[TensorMeta.from_shape_dtype("z", (64, 256), Dtype.FP8_E4M3)],
        attrs={"spec_kind": "matmul", "m": 64, "n": 256, "k": 128,
               "layer_kind": "moe", "source": "model_spec"},
        scope="cons", category="compute", layer="0", component="routed_expert",
    )
    g = OpGraph(
        name="test_amax", phase="test",
        nodes={"prod": prod, "cons": cons},
        edges=[Edge(src="prod", src_idx=0, dst="cons", dst_idx=0,
                     tensor=prod.outputs[0])],
    )
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3,
                   routed_expert_compute_dtype=Dtype.FP8_E4M3)
    insert_cast_pass_opgraph(g, m, QuantPolicy(assume_all_casts_fused=False))
    casts = _cast_nodes(g)
    assert len(casts) == 1
    c = casts[0]
    assert c.attrs["src_dtype"] is Dtype.BF16
    assert c.attrs["dst_dtype"] is Dtype.FP8_E4M3
    assert c.attrs["needs_amax"] is True


# ── Layer attribution: cast inherits from consumer ───────────────────────


def test_cast_op_inherits_consumer_layer_id():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_opgraph_direct(m, Strategy())
    for cast in _cast_nodes(g):
        assert cast.layer != "-1", f"cast {cast.id} should not be global"


def test_topo_sort_succeeds_after_cast_insertion():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_opgraph_direct(m, Strategy())
    sorted_nodes = g.topo_sort()
    assert len(sorted_nodes) == len(g.nodes)


# ── _cast_cost: fused vs unfused ─────────────────────────────────────────


def test_fused_cast_is_zero_cost():
    meta = {"num_elements": 100_000, "src_dtype": Dtype.BF16,
            "dst_dtype": Dtype.FP8_E4M3, "fused": True,
            "needs_amax": True}
    cost = _cast_cost(SimpleNamespace(meta=meta))
    assert cost.fwd_bytes == 0
    assert cost.dx_bytes == 0
    assert cost.dw_bytes == 0


def test_unfused_cast_fp8_to_bf16_no_amax():
    meta = {"num_elements": 100_000, "src_dtype": Dtype.FP8_E4M3,
            "dst_dtype": Dtype.BF16, "fused": False,
            "needs_amax": False}
    cost = _cast_cost(SimpleNamespace(meta=meta))
    # n * (FP8 + BF16) = 100_000 * (1 + 2) = 300_000
    assert cost.fwd_bytes == 300_000
    assert cost.dx_bytes == 300_000
    assert cost.dw_bytes == 0


def test_unfused_cast_with_amax_adds_extra_read():
    meta = {"num_elements": 100_000, "src_dtype": Dtype.BF16,
            "dst_dtype": Dtype.FP8_E4M3, "fused": False,
            "needs_amax": True}
    cost = _cast_cost(SimpleNamespace(meta=meta))
    # main = 100_000 * (BF16 + FP8) = 100_000 * (2 + 1) = 300_000
    # amax = 100_000 * BF16 = 200_000
    assert cost.fwd_bytes == 500_000


# ── QuantPolicy toggle wired through build_opgraph_direct ─────────────────


def test_quant_policy_assume_fused_makes_all_casts_zero():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(quant=QuantPolicy(assume_all_casts_fused=True))
    g = build_opgraph_direct(m, strat)
    casts = _cast_nodes(g)
    assert len(casts) > 0
    for c in casts:
        assert c.attrs["fused"] is True
        assert _cast_cost(_node_as_op(c)).fwd_bytes == 0


def test_quant_policy_unfused_materializes_cast_bytes():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(quant=QuantPolicy(
        assume_all_casts_fused=False,
        fuse_ln_epilog=False,
        fuse_gemm_epilog=False,
        fuse_attn_internal=False,
    ))
    g = build_opgraph_direct(m, strat)
    casts = _cast_nodes(g)
    assert len(casts) > 0
    total = sum(_cast_cost(_node_as_op(c)).fwd_bytes for c in casts)
    assert total > 0


def test_quant_policy_partial_fusion():
    """fuse_gemm_epilog=True absorbs GEMM-output cast but not residual-add."""
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(quant=QuantPolicy(
        assume_all_casts_fused=False,
        fuse_ln_epilog=True,
        fuse_gemm_epilog=True,
        fuse_attn_internal=True,
    ))
    g = build_opgraph_direct(m, strat)
    cast = next(n for n in g.nodes.values()
                if n.attrs.get("spec_kind") == "cast" and "residual2" in n.id)
    assert cast.attrs["fused"] is False


# ── Idempotency: running cast_pass again is a no-op ──────────────────────


def test_cast_pass_idempotent():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_opgraph_direct(m, Strategy())
    n_casts_first = _count_casts(g)
    insert_cast_pass_opgraph(g, m, QuantPolicy())
    assert _count_casts(g) == n_casts_first


# ── OpGraph cast pass: no cycles with semantically-named tensors ────────


def test_opgraph_cast_pass_no_cycle_with_v4_compressor():
    """Regression: insert_cast_pass_opgraph used rsplit('_',1)[0] to derive
    producer node IDs from tensor names.  Semantically-named tensors like
    ``comp_kv_raw`` produced bogus IDs (``comp_kv``), creating dangling
    edges that broke topo_sort.  The fix builds a proper tensor→producer
    map and replaces old edges instead of appending duplicates.
    """
    m = ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=1, head_dim=32,
        vocab=1000, seq_len=64,
        layers=[LayerKind.MOE, LayerKind.MOE],
        num_experts=8, moe_ffn=256, top_k=2, n_shared_experts=1,
        o_groups=2, o_lora_rank=64,
        compress_ratios=[4, 128],
        index_topk=16, index_n_heads=4, index_head_dim=32,
        q_lora_rank=64, kv_lora_rank=0,
        attn_act_dtype=Dtype.FP8_E4M3,
        moe_act_dtype=Dtype.FP8_E4M3,
    )
    strat = Strategy(tp=2)
    g = build_opgraph_direct(m, strat)
    sorted_nodes = g.topo_sort()
    assert len(sorted_nodes) == len(g.nodes)


def test_opgraph_cast_pass_replaces_old_edge():
    """When a cast is spliced between producer P and consumer C, the old
    edge P→C must be replaced by P→cast, not left in place alongside
    cast→C (which would inflate C's in-degree and block topo_sort).
    """
    m = ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64,
        layers=[LayerKind.MOE],
        num_experts=8, moe_ffn=256, top_k=2, n_shared_experts=1,
        moe_act_dtype=Dtype.FP8_E4M3,
    )
    strat = Strategy()
    g = build_opgraph_direct(m, strat)
    sorted_nodes = g.topo_sort()
    assert len(sorted_nodes) == len(g.nodes)
    cast_nodes = [n for n in g.nodes.values() if n.op_type == "spec.cast"]
    for cn in cast_nodes:
        in_edges = g.in_edges(cn.id)
        assert len(in_edges) == 1, f"cast node {cn.id} should have exactly 1 in-edge, got {len(in_edges)}"
        assert in_edges[0].src in g.nodes, f"cast in-edge src {in_edges[0].src} must be a real node"
