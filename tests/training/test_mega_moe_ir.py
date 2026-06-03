from __future__ import annotations

from zrt.training.ir.builders import build_graph
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy


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


def _ops_for_layer(graph, layer_id: int):
    start, end = graph.layer_index[layer_id]
    return graph.ops[start:end]


def test_mega_moe_switch_off_preserves_routed_expert_matmul():
    model = _moe_model()
    graph = build_graph(model, Strategy(mega_moe=False))

    layer_ops = _ops_for_layer(graph, 0)
    routed = [op for op in layer_ops if op.name == "L0.routed_expert_ffn"]

    assert len(routed) == 1
    assert routed[0].kind == "matmul"
    assert [op for op in layer_ops if op.kind == "mega_moe"] == []


def test_mega_moe_switch_on_emits_single_mega_moe_op_for_layer():
    model = _moe_model()
    graph = build_graph(model, Strategy(mega_moe=True, mega_moe_waves=4, micro_batch=3))

    layer_ops = _ops_for_layer(graph, 0)
    mega_moe_ops = [op for op in layer_ops if op.kind == "mega_moe"]
    routed_matmuls = [
        op
        for op in layer_ops
        if op.kind == "matmul" and op.name == "L0.routed_expert_ffn"
    ]

    assert [op.name for op in mega_moe_ops] == ["L0.mega_moe"]
    assert routed_matmuls == []

    mega_moe = mega_moe_ops[0]
    assert mega_moe.inputs[0].shape_logical == (model.seq_len, model.hidden)
    assert mega_moe.outputs[0].shape_logical == (model.seq_len, model.hidden)

    assert mega_moe.meta == {
        "m": model.seq_len,
        "n": model.hidden,
        "k": model.moe_ffn,
        "micro_batch": 3,
        "num_experts": model.num_experts,
        "top_k": model.top_k,
        "requested_waves": 4,
        "act_bytes": model.act_dtype.bytes,
        "out_bytes": model.act_dtype.bytes,
        "moe_act_bytes": model.effective_moe_act_dtype().bytes,
        "moe_act_dtype": model.effective_moe_act_dtype(),
        "weight_bytes": model.routed_expert_compute_dtype.bytes,
        "weight_stored_bytes": model.routed_expert_weight_dtype.stored_bytes,
        "quant_variant": "standard",
        "fwd_multiplier": 3 * model.top_k,
        "swiglu_clamp": model.swiglu_clamp,
        "fused_dispatch_compute_combine": True,
    }


def test_mega_moe_quant_variant_w4a8_for_fp4_weights_and_fp8_moe_acts():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )
    graph = build_graph(model, Strategy(mega_moe=True))

    layer_ops = _ops_for_layer(graph, 0)
    ln2 = [op for op in layer_ops if op.name == "L0.ln2"][0]
    mega_moe = [op for op in graph.ops if op.kind == "mega_moe"][0]

    assert mega_moe.meta["quant_variant"] == "w4a8"
    assert mega_moe.inputs[0].name == ln2.outputs[0].name == "x_ln2"
    assert mega_moe.inputs[0].dtype == ln2.outputs[0].dtype
    assert mega_moe.outputs[0].dtype == model.effective_moe_act_dtype()
    assert mega_moe.meta["act_bytes"] == mega_moe.inputs[0].dtype.bytes
    assert mega_moe.meta["out_bytes"] == mega_moe.outputs[0].dtype.bytes
    assert mega_moe.meta["moe_act_dtype"] == Dtype.FP8_E4M3
    assert mega_moe.meta["moe_act_bytes"] == Dtype.FP8_E4M3.bytes


def test_mega_moe_w4a8_shared_expert_agg_consumes_routed_output_dtype():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
        n_shared_experts=1,
    )
    graph = build_graph(model, Strategy(mega_moe=True))

    layer_ops = _ops_for_layer(graph, 0)
    mega_moe = [op for op in layer_ops if op.kind == "mega_moe"][0]
    expert_agg = [op for op in layer_ops if op.name == "L0.expert_agg"][0]

    assert mega_moe.outputs[0].name == "routed_ffn_out"
    assert expert_agg.inputs[1].name.startswith("routed_ffn_out__cast_bf16")
    assert mega_moe.outputs[0].shape_logical == (model.seq_len, model.hidden)
    assert expert_agg.inputs[1].shape_logical == (model.seq_len, model.hidden)
    assert mega_moe.outputs[0].dtype == model.effective_moe_act_dtype()
    assert expert_agg.inputs[1].dtype == model.act_dtype
    assert mega_moe.meta["out_bytes"] == mega_moe.outputs[0].dtype.bytes


def test_mega_moe_w4a8_no_shared_expert_agg_consumes_routed_output_dtype():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
        n_shared_experts=0,
    )
    graph = build_graph(model, Strategy(mega_moe=True))

    layer_ops = _ops_for_layer(graph, 0)
    mega_moe = [op for op in layer_ops if op.kind == "mega_moe"][0]
    expert_agg = [op for op in layer_ops if op.name == "L0.expert_agg"][0]
    residual2 = [op for op in layer_ops if op.name == "L0.residual2"][0]

    assert mega_moe.outputs[0].name == "routed_ffn_out"
    assert expert_agg.inputs[0].name.startswith("routed_ffn_out__cast_bf16")
    assert mega_moe.outputs[0].shape_logical == (model.seq_len, model.hidden)
    assert expert_agg.inputs[0].shape_logical == (model.seq_len, model.hidden)
    assert mega_moe.outputs[0].dtype == model.effective_moe_act_dtype()
    assert expert_agg.inputs[0].dtype == model.act_dtype
    assert expert_agg.outputs[0].name == "ffn_out"
    assert residual2.inputs[0].name.startswith("ffn_out__cast_bf16")
    assert expert_agg.outputs[0].dtype == model.effective_moe_act_dtype()
    assert residual2.inputs[0].dtype == model.act_dtype
    assert mega_moe.meta["out_bytes"] == mega_moe.outputs[0].dtype.bytes
