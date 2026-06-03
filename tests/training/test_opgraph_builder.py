"""Tests for zrt.training.ir.opgraph_builder — build_opgraph().

Verifies that the spec-driven builder produces OpGraph instances
satisfying the Stage 1 contract (refactor.md §4.1).
"""

import pytest

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.edge import Edge
from zrt.training.ir.opgraph_builder import build_opgraph
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.dtype import Dtype


def _make_llama_spec():
    return ModelSpec(
        hidden=4096, ffn=11008, seq_len=2048,
        num_heads=32, num_kv_heads=32, head_dim=128,
        layers=[LayerKind.DENSE] * 4, vocab=32000, act_dtype=Dtype.BF16,
    )


def _make_moe_spec():
    return ModelSpec(
        hidden=2048, ffn=8192, moe_ffn=2048,
        seq_len=1024, num_heads=16, num_kv_heads=16, head_dim=128,
        layers=[LayerKind.DENSE] * 2 + [LayerKind.MOE] * 2,
        num_experts=8, top_k=2, n_shared_experts=1,
        vocab=32000, act_dtype=Dtype.BF16,
    )


def _make_strategy(**kwargs):
    defaults = dict(tp=1, pp=1, ep=1, dp=1, cp=1, micro_batch=1, global_batch=32)
    defaults.update(kwargs)
    return Strategy(**defaults)


class TestBuildOpGraphContract:
    """Stage 1 contract: build_opgraph output must satisfy OpGraph invariants."""

    def test_returns_opgraph(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        assert isinstance(g, OpGraph)

    def test_phase_is_train_forward(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        assert g.phase == "train_forward"

    def test_node_count_matches_old_ir(self):
        model, strategy = _make_llama_spec(), _make_strategy()
        g_old = build_graph(model, strategy)
        g_new = build_opgraph(model, strategy)
        expected = len(g_old.ops) + len(g_old.collectives)
        assert len(g_new.nodes) == expected

    def test_all_nodes_have_op_type(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        for nid, node in g.nodes.items():
            assert node.op_type, f"Node {nid} has empty op_type"

    def test_edges_connect_valid_nodes(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        for edge in g.edges:
            assert edge.src in g.nodes, f"Edge src {edge.src} not in graph"
            assert edge.dst in g.nodes, f"Edge dst {edge.dst} not in graph"

    def test_aten_op_types(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        aten_nodes = [n for n in g.nodes.values() if n.op_type.startswith("aten.")]
        assert len(aten_nodes) > 0

    def test_source_attr(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        for node in g.nodes.values():
            if not node.op_type.startswith("comm."):
                assert node.attrs.get("source") == "model_spec"

    def test_metadata_populated(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        assert g.metadata["seq_len"] == 2048
        assert g.metadata["hidden"] == 4096
        assert g.metadata["num_layers"] == 4
        assert g.metadata["total_params"] > 0

    def test_layer_annotations_present(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        block_nodes = [n for n in g.nodes.values() if n.layer and n.layer != "-1"]
        assert len(block_nodes) >= 4 * 10

    def test_data_flow_edges_exist(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        assert len(g.edges) > 0
        data_edges = [e for e in g.edges if e.is_data]
        assert len(data_edges) > 0, "Expected data-flow edges with TensorMeta"

    def test_tensor_meta_on_edges(self):
        g = build_opgraph(_make_llama_spec(), _make_strategy())
        for edge in g.edges:
            if edge.is_data:
                assert edge.tensor is not None
                assert edge.tensor.shape, f"Edge {edge} has empty shape"


class TestBuildOpGraphMoE:
    """MoE-specific tests: collectives converted to comm.* OpNodes."""

    def test_ep_produces_comm_nodes(self):
        g = build_opgraph(_make_moe_spec(), _make_strategy(ep=2))
        comm_nodes = [n for n in g.nodes.values() if n.op_type.startswith("comm.")]
        assert len(comm_nodes) > 0

    def test_ep_comm_are_all_to_all(self):
        g = build_opgraph(_make_moe_spec(), _make_strategy(ep=2))
        comm_nodes = [n for n in g.nodes.values() if n.op_type.startswith("comm.")]
        a2a = [n for n in comm_nodes if n.op_type == "comm.all_to_all"]
        assert len(a2a) == len(comm_nodes), "EP collectives should be A2A"

    def test_ep_comm_attrs(self):
        g = build_opgraph(_make_moe_spec(), _make_strategy(ep=2))
        comm = next(n for n in g.nodes.values() if n.op_type.startswith("comm."))
        assert comm.attrs["comm_group"] == "EP"
        assert comm.attrs["comm_bytes"] > 0
        assert comm.category == "communication"

    def test_tp_produces_comm_nodes(self):
        g = build_opgraph(_make_moe_spec(), _make_strategy(tp=2))
        comm_nodes = [n for n in g.nodes.values() if n.op_type.startswith("comm.")]
        assert len(comm_nodes) > 0
        groups = {n.attrs["comm_group"] for n in comm_nodes}
        assert "TP" in groups

    def test_tp_ep_combined(self):
        g = build_opgraph(_make_moe_spec(), _make_strategy(tp=2, ep=2))
        comm_nodes = [n for n in g.nodes.values() if n.op_type.startswith("comm.")]
        groups = {n.attrs["comm_group"] for n in comm_nodes}
        assert "TP" in groups
        assert "EP" in groups

    def test_no_comm_when_all_parallel_1(self):
        g = build_opgraph(_make_moe_spec(), _make_strategy())
        comm_nodes = [n for n in g.nodes.values() if n.op_type.startswith("comm.")]
        assert len(comm_nodes) == 0


class TestBuildOpGraphNodeCount:
    """Verify node count consistency across different configurations."""

    def test_dense_tp1(self):
        model, strategy = _make_llama_spec(), _make_strategy(tp=1)
        g_old = build_graph(model, strategy)
        g_new = build_opgraph(model, strategy)
        assert len(g_new.nodes) == len(g_old.ops) + len(g_old.collectives)

    def test_dense_tp2(self):
        model, strategy = _make_llama_spec(), _make_strategy(tp=2)
        g_old = build_graph(model, strategy)
        g_new = build_opgraph(model, strategy)
        assert len(g_new.nodes) == len(g_old.ops) + len(g_old.collectives)

    def test_moe_ep2(self):
        model, strategy = _make_moe_spec(), _make_strategy(ep=2)
        g_old = build_graph(model, strategy)
        g_new = build_opgraph(model, strategy)
        assert len(g_new.nodes) == len(g_old.ops) + len(g_old.collectives)

    def test_moe_tp2_ep2(self):
        model, strategy = _make_moe_spec(), _make_strategy(tp=2, ep=2)
        g_old = build_graph(model, strategy)
        g_new = build_opgraph(model, strategy)
        assert len(g_new.nodes) == len(g_old.ops) + len(g_old.collectives)
