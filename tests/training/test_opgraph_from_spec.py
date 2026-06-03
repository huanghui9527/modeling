"""Test OpGraph.from_model_spec() factory method.

Phase 3: Updated to use build_opgraph() internally (no longer depends on old IR).
"""

import pytest

from zrt.ir.graph import OpGraph
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy


def _make_model(num_layers=4, **kwargs) -> ModelSpec:
    defaults = dict(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * num_layers,
    )
    defaults.update(kwargs)
    return ModelSpec(**defaults)


def test_opgraph_from_spec_returns_opgraph():
    """from_model_spec() should return a valid OpGraph with compute nodes."""
    model = _make_model()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    assert isinstance(opgraph, OpGraph)
    assert len(opgraph.nodes) > 0
    compute_nodes = [n for n in opgraph.nodes.values() if n.category != "communication"]
    assert len(compute_nodes) > 0
    for node in compute_nodes:
        assert node.op_type


def test_opgraph_from_spec_metadata():
    """from_model_spec() should set source and model metadata."""
    model = _make_model(num_layers=2)
    strategy = Strategy(tp=2, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    assert opgraph.metadata["source"] == "model_spec"
    assert opgraph.metadata["hidden"] == model.hidden
    assert opgraph.metadata["num_layers"] == len(model.layers)
    assert "strategy" in opgraph.metadata


def test_opgraph_from_spec_layer_info():
    """Compute nodes should have layer and layer_kind attributes."""
    model = _make_model(
        num_layers=2,
        layers=[LayerKind.DENSE, LayerKind.MOE],
        num_experts=4, moe_ffn=1024, top_k=1,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    compute_nodes = [n for n in opgraph.nodes.values() if n.category != "communication"]
    layer_nodes = [n for n in compute_nodes if n.layer]
    assert len(layer_nodes) > 0
    for node in layer_nodes:
        assert isinstance(node.layer, str)
        assert "layer_kind" in node.attrs


def test_opgraph_from_spec_edges_exist():
    """OpGraph should have data-flow edges between nodes."""
    model = _make_model(num_layers=3)
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    assert len(opgraph.edges) > 0
    for edge in opgraph.edges:
        assert edge.src in opgraph.nodes
        assert edge.dst in opgraph.nodes


def test_opgraph_from_spec_phase_parameter():
    """Custom phase parameter should be reflected in the graph."""
    model = _make_model(num_layers=1)
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy, phase="inference")

    assert opgraph.phase == "inference"
    assert "inference" in opgraph.name


def test_opgraph_from_spec_tp_produces_comm_nodes():
    """TP > 1 should produce communication nodes."""
    model = _make_model(num_layers=2)
    strategy = Strategy(tp=2, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    comm_nodes = [n for n in opgraph.nodes.values() if n.category == "communication"]
    assert len(comm_nodes) > 0
    for node in comm_nodes:
        assert node.op_type.startswith("comm.")
