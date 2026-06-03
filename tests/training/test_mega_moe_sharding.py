from __future__ import annotations

from zrt.training.ir.builders import build_graph
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


def _op_for_layer(graph, layer_id: int, *, kind: str | None = None, name: str | None = None):
    matches = [
        op
        for op in _ops_for_layer(graph, layer_id)
        if (kind is None or op.kind == kind) and (name is None or op.name == name)
    ]
    assert len(matches) == 1
    return matches[0]


def _ep_a2a_collectives(graph):
    return [c for c in graph.collectives if c.group == "EP" and c.kind == "A2A"]


def test_mega_moe_off_with_ep_inserts_a2a_around_routed_expert_ffn():
    model = _moe_model()
    graph = build_graph(model, Strategy(ep=2, mega_moe=False))

    routed = _op_for_layer(graph, 0, name="L0.routed_expert_ffn")
    ep_a2a = _ep_a2a_collectives(graph)

    assert routed.kind == "matmul"
    assert [c.name for c in ep_a2a] == [
        "a2a_before_L0.routed_expert_ffn",
        "a2a_after_L0.routed_expert_ffn",
    ]
    assert ep_a2a[0].inserted_before == routed.name
    assert ep_a2a[1].inserted_after == routed.name


def test_mega_moe_on_with_ep_uses_fused_op_without_routed_expert_a2a():
    model = _moe_model()
    graph = build_graph(model, Strategy(ep=2, mega_moe=True))

    mega_moe = _op_for_layer(graph, 0, kind="mega_moe")
    ep_a2a = _ep_a2a_collectives(graph)

    assert mega_moe.name == "L0.mega_moe"
    assert [op for op in _ops_for_layer(graph, 0) if op.name == "L0.routed_expert_ffn"] == []
    assert ep_a2a == []
    assert all("routed_expert" not in c.name for c in graph.collectives)
    assert all(c.inserted_before != mega_moe.name for c in graph.collectives)
    assert all(c.inserted_after != mega_moe.name for c in graph.collectives)


def test_mega_moe_tp_sharding_keeps_input_hidden_and_shards_output_hidden_without_ep():
    model = _moe_model()
    strategy = Strategy(tp=4, mega_moe=True)
    graph = build_graph(model, strategy)

    mega_moe = _op_for_layer(graph, 0, kind="mega_moe")

    assert mega_moe.inputs[0].shape_logical == (model.seq_len, model.hidden)
    assert mega_moe.inputs[0].shape_local == (model.seq_len, model.hidden)
    assert mega_moe.outputs[0].shape_logical == (model.seq_len, model.hidden)
    assert mega_moe.outputs[0].shape_local == (model.seq_len, model.hidden // strategy.tp)
    assert mega_moe.meta["n_local"] == mega_moe.meta["n"] // strategy.tp
    assert mega_moe.meta["k_local"] == mega_moe.meta["k"] // strategy.tp


def test_mega_moe_with_ep_uses_main_tp_for_routed_path():
    model = _moe_model()
    strategy = Strategy(tp=4, ep=4, mega_moe=True)
    graph = build_graph(model, strategy)

    mega_moe = _op_for_layer(graph, 0, kind="mega_moe")

    assert mega_moe.inputs[0].shape_local == (model.seq_len, model.hidden)
    assert mega_moe.outputs[0].shape_local == (model.seq_len, model.hidden // strategy.tp)
    assert mega_moe.meta["n_local"] == mega_moe.meta["n"] // strategy.tp
    assert mega_moe.meta["k_local"] == mega_moe.meta["k"] // strategy.tp


def test_routed_expert_with_ep_uses_main_tp_for_routed_path_and_a2a_bytes():
    model = _moe_model()
    strategy = Strategy(tp=4, ep=4, mega_moe=False)
    graph = build_graph(model, strategy)

    routed = _op_for_layer(graph, 0, name="L0.routed_expert_ffn")
    ep_a2a = _ep_a2a_collectives(graph)

    assert routed.inputs[0].shape_local == (model.seq_len, model.hidden)
    assert routed.outputs[0].shape_local == (model.seq_len, model.hidden // strategy.tp)
    assert routed.meta["k_local"] == routed.meta["k"] // strategy.tp
    expected_bytes = (
        strategy.micro_batch
        * model.seq_len
        * (model.hidden // strategy.tp)
        * model.top_k
        * model.effective_moe_act_dtype().bytes
    )
    assert [c.bytes_ for c in ep_a2a] == [expected_bytes, expected_bytes]


def test_routed_expert_without_ep_still_uses_main_tp_sharding():
    model = _moe_model()
    strategy = Strategy(tp=4, ep=1, mega_moe=False)
    graph = build_graph(model, strategy)

    routed = _op_for_layer(graph, 0, name="L0.routed_expert_ffn")

    assert routed.inputs[0].shape_local == (model.seq_len, model.hidden)
    assert routed.outputs[0].shape_local == (model.seq_len, model.hidden // strategy.tp)
    assert routed.meta["k_local"] == routed.meta["k"] // strategy.tp


def test_mega_moe_cp_sharding_divides_token_dimension_and_meta_m():
    model = _moe_model()
    strategy = Strategy(cp=4, mega_moe=True)
    graph = build_graph(model, strategy)

    mega_moe = _op_for_layer(graph, 0, kind="mega_moe")

    assert mega_moe.inputs[0].shape_local == (model.seq_len // strategy.cp, model.hidden)
    assert mega_moe.outputs[0].shape_local == (model.seq_len // strategy.cp, model.hidden)
    assert mega_moe.meta["m"] == model.seq_len // strategy.cp


def test_mega_moe_ep_sharding_records_local_expert_metadata():
    model = _moe_model()
    strategy = Strategy(ep=4, mega_moe=True)
    graph = build_graph(model, strategy)

    mega_moe = _op_for_layer(graph, 0, kind="mega_moe")

    assert mega_moe.meta["ep"] == strategy.ep
    assert mega_moe.meta["experts_per_rank"] == model.num_experts // strategy.ep
