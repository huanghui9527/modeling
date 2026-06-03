"""Tests for PipelineParallelPass and TrainingPipelinePass.

Organization:
  1. Basic PP functionality (stage_id, P2P insertion, functional semantics)
  2. Schedule-specific tests (1F1B, VPP, DualPipe, DualPipeV)
  3. Cross-schedule comparison
  4. TrainingPipelinePass per-stage metrics
  5. Integration with default pipeline
"""
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.context import (
    TransformContext, ParallelConfig, StreamConfig, TrainingConfig,
)
from python.zrt.transform.parallel.pipeline_parallel import PipelineParallelPass
from python.zrt.transform.analysis.training import TrainingPipelinePass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _t(tid: str, shape=(1, 128, 4096)):
    return TensorMeta.from_shape_dtype(tid, shape, DType.BF16)


def _make_hw():
    import python.zrt.hardware.registry as hw_registry
    return hw_registry.load("nvidia_h100_sxm")


def _make_linear_graph(num_layers: int = 4) -> OpGraph:
    """Build a simple linear graph: one matmul node per transformer layer."""
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []

    prev_out_tensor = _t("input_0")
    for i in range(num_layers):
        node_id = f"mm_layer{i}"
        out_tensor = _t(f"out_layer{i}")
        node = OpNode(
            id=node_id,
            op_type="aten.mm.default",
            inputs=[prev_out_tensor],
            outputs=[out_tensor],
            scope=f"model.layers.{i}.mlp.gate_proj",
            layer=str(i),
            category="compute",
        )
        node.annotations["latency_us"] = 100.0 * (i + 1)
        nodes[node_id] = node

        if i > 0:
            prev_id = f"mm_layer{i-1}"
            edges.append(Edge(
                src=prev_id, src_idx=0,
                dst=node_id, dst_idx=0,
                tensor=prev_out_tensor,
            ))

        prev_out_tensor = out_tensor

    return OpGraph(
        name="test_model",
        phase="train_forward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": 128, "hidden": 4096, "num_layers": num_layers},
    )


def _make_ctx(
    pp: int = 2,
    tp: int = 1,
    dp: int = 1,
    global_batch: int = 8,
    micro_batch: int = 1,
    pp_layer_assignment=None,
    pp_schedule: str = "1f1b",
    vpp_chunks: int = 1,
) -> TransformContext:
    return TransformContext(
        hw_spec=_make_hw(),
        parallel=ParallelConfig(tp=tp, pp=pp, dp=dp),
        stream_config=StreamConfig(),
        training=TrainingConfig(
            micro_batch=micro_batch,
            global_batch=global_batch,
            pp_layer_assignment=pp_layer_assignment,
            pp_schedule=pp_schedule,
            vpp_chunks=vpp_chunks,
        ),
    )


# ── 0. Non-Layer Node Distribution ─────────────────────────────────────────────

def _make_graph_with_nonlayer(phase: str = "train_forward") -> OpGraph:
    """Build a graph with layer nodes + non-layer nodes (embed, norm, lm_head).

    Layer nodes (2 layers, each 100µs load):
      - ``mm_layer0``  (layer="0", scope="model.layers.0.mlp")
      - ``mm_layer1``  (layer="1", scope="model.layers.1.mlp")

    Non-layer forward nodes:
      - ``embed``      (layer="", component="embedding",   load=50µs)
      - ``final_norm`` (layer="", component="final_norm",  load=30µs)
      - ``lm_head``    (layer="", component="lm_head",     load=80µs)
      - ``other_fwd``  (layer="", component="attention",   load=20µs)

    Backward nodes (when phase="train"):
      - ``bwd_op0``, ``bwd_op1``  (layer="", phase="bwd", each 60µs)
    """
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []

    def _make_nid(prefix: str, layer: str, component: str,
                  load: float = 100.0, phase: str = "fwd") -> str:
        nid = f"{prefix}"
        shape = (1, 128, 4096)
        node = OpNode(
            id=nid,
            op_type="aten.mm.default",
            inputs=[_t(f"{nid}_in", shape)],
            outputs=[_t(f"{nid}_out", shape)],
            scope=f"model.{prefix}",
            layer=layer,
            component=component,
            category="compute",
        )
        node.annotations["latency_us"] = load
        if phase == "bwd":
            node.annotations["phase"] = "bwd"
        nodes[nid] = node
        return nid

    # Layer 0
    l0 = _make_nid("mm_layer0", "0", "attention")
    # Layer 1
    l1 = _make_nid("mm_layer1", "1", "ffn")
    edges.append(Edge(src=l0, src_idx=0, dst=l1, dst_idx=0, tensor=_t("e0_l1")))

    # Non-layer forward: embedding
    emb = _make_nid("embed", "", "embedding", load=50.0)
    edges.append(Edge(src=emb, src_idx=0, dst=l0, dst_idx=0, tensor=_t("emb_l0")))

    # Non-layer forward: final norm
    fn = _make_nid("final_norm", "", "final_norm", load=30.0)
    edges.append(Edge(src=l1, src_idx=0, dst=fn, dst_idx=0, tensor=_t("l1_fn")))

    # Non-layer forward: lm_head
    lmh = _make_nid("lm_head", "", "lm_head", load=80.0)
    edges.append(Edge(src=fn, src_idx=0, dst=lmh, dst_idx=0, tensor=_t("fn_lmh")))

    # Non-layer forward: other (e.g. shared attention ops)
    other = _make_nid("other_fwd", "", "attention", load=20.0)

    bwd_will_be_added = phase == "train"
    if bwd_will_be_added:
        bwd0 = _make_nid("bwd_op0", "", "mm", load=60.0, phase="bwd")
        bwd1 = _make_nid("bwd_op1", "", "mm", load=60.0, phase="bwd")
        edges.append(Edge(src=lmh, src_idx=0, dst=bwd0, dst_idx=0, tensor=_t("lmh_b0")))
        edges.append(Edge(src=bwd0, src_idx=0, dst=bwd1, dst_idx=0, tensor=_t("b0_b1")))

    return OpGraph(
        name="test_nonlayer",
        phase=phase,
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": 128, "hidden": 4096, "num_layers": 2},
    )


class TestNonLayerNodeDistribution:
    """layer="" nodes must be distributed to correct stages, not all dumped on 0."""

    def test_embedding_goes_to_stage0(self):
        graph = _make_graph_with_nonlayer("train_forward")
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)
        assert result.nodes["embed"].annotations["stage_id"] == 0

    def test_lm_head_goes_to_last_stage(self):
        graph = _make_graph_with_nonlayer("train_forward")
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)
        assert result.nodes["lm_head"].annotations["stage_id"] == 3

    def test_final_norm_goes_to_last_stage(self):
        graph = _make_graph_with_nonlayer("train_forward")
        ctx = _make_ctx(pp=3)
        result = PipelineParallelPass().run(graph, ctx)
        assert result.nodes["final_norm"].annotations["stage_id"] == 2

    def test_other_fwd_nonlayer_greedy_distributed(self):
        """Other forward non-layer nodes should NOT all be on stage 0."""
        graph = _make_graph_with_nonlayer("train_forward")
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)
        sid = result.nodes["other_fwd"].annotations["stage_id"]
        assert 0 <= sid < 4

    def test_backward_nodes_distributed_proportionally(self):
        """Backward nodes should be spread across stages, not all on 0."""
        graph = _make_graph_with_nonlayer("train")
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)
        bwd_stages = {result.nodes[nid].annotations["stage_id"]
                      for nid in ("bwd_op0", "bwd_op1")}
        # With 2 stages and proportional distribution, at least 1 bwd
        # node should be on a non-zero stage
        assert len(bwd_stages) > 1

    def test_no_nonlayer_node_has_default_stage0_when_stages_available(self):
        """After fix, no non-layer forward node should default to stage 0 just
        because it lacks a layer number — unless it genuinely belongs there."""
        graph = _make_graph_with_nonlayer("train_forward")
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)

        for node in result.nodes.values():
            if node.layer or node.annotations.get("phase") == "bwd":
                continue
            sid = node.annotations["stage_id"]
            comp = node.component
            if comp == "embedding":
                assert sid == 0, f"{node.id} (embedding) should be stage 0, got {sid}"
            elif comp in ("final_norm", "lm_head"):
                assert sid == 3, f"{node.id} ({comp}) should be stage 3, got {sid}"
            else:
                assert 0 <= sid < 4

    def test_all_nodes_have_stage_id(self):
        graph = _make_graph_with_nonlayer("train")
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)
        for node in result.nodes.values():
            assert "stage_id" in node.annotations

    def test_existing_layer_nodes_unchanged(self):
        """Layer nodes should still be assigned via greedy bin-packing."""
        graph = _make_graph_with_nonlayer("train_forward")
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)
        # Layers 0 and 1 should go to different stages with pp=2
        stage0 = result.nodes["mm_layer0"].annotations["stage_id"]
        stage1 = result.nodes["mm_layer1"].annotations["stage_id"]
        assert stage0 != stage1

    def test_pp1_nonlayer_all_stage0(self):
        """pp=1 should keep all nodes on stage 0."""
        graph = _make_graph_with_nonlayer("train")
        ctx = _make_ctx(pp=1)
        result = PipelineParallelPass().run(graph, ctx)
        for node in result.nodes.values():
            assert node.annotations["stage_id"] == 0

    def test_stage0_no_longer_absorbs_all_nonlayer_fwd(self):
        """Old behavior: all non-layer forward nodes defaulted to stage 0.
        New behavior: stage 0 should NOT contain ALL non-layer forward nodes."""
        graph = _make_graph_with_nonlayer("train")
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)

        nonlayer_stages: set[int] = set()
        for node in result.nodes.values():
            if node.layer or node.op_type == "comm.send_recv":
                continue
            if node.annotations.get("phase") == "bwd":
                continue
            nonlayer_stages.add(node.annotations["stage_id"])

        # Non-layer forward nodes (embed, final_norm, lm_head, other_fwd)
        # should be spread across at least 2 different stages
        assert len(nonlayer_stages) >= 2, (
            f"Non-layer fwd nodes all on stages {nonlayer_stages}"
        )

    def test_backward_nodes_spread_across_mutiple_stages(self):
        """Backward nodes should not all be on a single stage."""
        graph = _make_graph_with_nonlayer("train")
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)

        bwd_stages: set[int] = set()
        for node in result.nodes.values():
            if node.annotations.get("phase") != "bwd":
                continue
            bwd_stages.add(node.annotations["stage_id"])

        assert len(bwd_stages) >= 2, (
            f"Backward nodes all on stage(s) {bwd_stages}"
        )

    def test_stage0_embedding_only_nonlayer(self):
        """Stage 0 should only contain embedding as non-layer forward nodes,
        not lm_head, final_norm, or other unrelated forward nodes."""
        graph = _make_graph_with_nonlayer("train")
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)

        for node in result.nodes.values():
            if node.layer or node.op_type == "comm.send_recv":
                continue
            if node.annotations.get("phase") == "bwd":
                continue
            sid = node.annotations["stage_id"]
            if sid == 0:
                assert node.component == "embedding", (
                    f"Stage 0 has non-layer fwd node {node.id} "
                    f"with component '{node.component}', expected 'embedding'"
                )


# ── 1. Basic PP Functionality ──────────────────────────────────────────────────

class TestPipelineParallelPassBasic:
    """Core functionality: stage_id annotation, P2P insertion, functional semantics."""

    def test_pp1_all_stage0(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=1)
        result = PipelineParallelPass().run(graph, ctx)
        for node in result.nodes.values():
            assert node.annotations.get("stage_id") == 0

    def test_pp2_splits_layers(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)
        stage_ids = {
            n.annotations["stage_id"]
            for n in result.nodes.values()
            if not n.op_type.startswith("comm.")
        }
        assert {0, 1} == stage_ids

    def test_pp4_inserts_three_p2p_nodes(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)
        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3

    def test_pp2_does_not_mutate_input(self):
        graph = _make_linear_graph(num_layers=4)
        original_ids = set(graph.nodes.keys())
        ctx = _make_ctx(pp=2)
        _ = PipelineParallelPass().run(graph, ctx)
        assert set(graph.nodes.keys()) == original_ids

    def test_explicit_layer_assignment(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_layer_assignment=[0, 0, 1, 1])
        result = PipelineParallelPass().run(graph, ctx)

        def get_layers_for_stage(stage_id):
            return {
                int(n.layer) for n in result.nodes.values()
                if n.layer and n.op_type != "comm.send_recv"
                and n.annotations.get("stage_id") == stage_id
            }

        assert get_layers_for_stage(0) == {0, 1}
        assert get_layers_for_stage(1) == {2, 3}

    def test_all_nodes_have_stage_id_after_pp2(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)
        for node in result.nodes.values():
            assert "stage_id" in node.annotations


class TestP2PNode:
    """P2P node creation and attributes."""

    def test_pp2_inserts_p2p_node(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_layer_assignment=[0, 0, 1, 1])
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) >= 1
        p2p = p2p_nodes[0]
        assert p2p.attrs["src_stage"] == 0
        assert p2p.attrs["dst_stage"] == 1

    def test_p2p_node_belongs_to_receiver_stage(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p = next(n for n in result.nodes.values() if n.op_type == "comm.send_recv")
        assert p2p.annotations["stage_id"] == 1

    def test_p2p_node_has_positive_message_size(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p = next(n for n in result.nodes.values() if n.op_type == "comm.send_recv")
        assert p2p.attrs["message_size_bytes"] > 0


# ── 2. Schedule-Specific Tests ─────────────────────────────────────────────────

class TestPPMode1F1B:
    """Standard 1F1B: greedy bin-packing by compute load."""

    def test_pp2_greedy_assignment(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_schedule="1f1b")
        result = PipelineParallelPass().run(graph, ctx)

        stage_ids = {n.annotations["stage_id"] for n in result.nodes.values()
                     if n.op_type != "comm.send_recv"}
        assert stage_ids == {0, 1}

    def test_pp2_p2p_count_explicit(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_schedule="1f1b", pp_layer_assignment=[0, 0, 1, 1])
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 1

    def test_pp4_p2p_count(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=4, pp_schedule="1f1b")
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3

    def test_no_virtual_stage_id(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_schedule="1f1b")
        result = PipelineParallelPass().run(graph, ctx)

        for node in result.nodes.values():
            if node.op_type != "comm.send_recv":
                assert node.annotations.get("virtual_stage_id") is None


class TestPPModeVPP:
    """VPP/Interleaved: round-robin layer assignment with virtual stages."""

    def test_pp2_vpp2_interleaved_assignment(self):
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        def get_layers_for_stage(stage_id):
            return {
                int(n.layer) for n in result.nodes.values()
                if n.layer and n.op_type != "comm.send_recv"
                and n.annotations.get("stage_id") == stage_id
            }

        assert get_layers_for_stage(0) == {0, 1, 4, 5}
        assert get_layers_for_stage(1) == {2, 3, 6, 7}

    def test_pp2_vpp2_p2p_count(self):
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3

    def test_virtual_stage_id_annotation(self):
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        compute_nodes = [n for n in result.nodes.values() if n.op_type != "comm.send_recv"]
        virtual_ids = {n.annotations["virtual_stage_id"] for n in compute_nodes}
        assert virtual_ids == {0, 1, 2, 3}

    def test_p2p_has_virtual_stage_attrs(self):
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        for p2p in p2p_nodes:
            assert "src_virtual_stage" in p2p.attrs
            assert "dst_virtual_stage" in p2p.attrs

    def test_vpp_chunks_1_fallback(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=1)
        result = PipelineParallelPass().run(graph, ctx)

        for node in result.nodes.values():
            if node.op_type != "comm.send_recv":
                assert node.annotations.get("virtual_stage_id") is None

    def test_pp4_vpp2_interleaved_assignment(self):
        graph = _make_linear_graph(num_layers=16)
        ctx = _make_ctx(pp=4, pp_schedule="interleaved", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        def get_layers_for_stage(stage_id):
            return {
                int(n.layer) for n in result.nodes.values()
                if n.layer and n.op_type != "comm.send_recv"
                and n.annotations.get("stage_id") == stage_id
            }

        assert get_layers_for_stage(0) == {0, 1, 8, 9}
        assert get_layers_for_stage(1) == {2, 3, 10, 11}
        assert get_layers_for_stage(2) == {4, 5, 12, 13}
        assert get_layers_for_stage(3) == {6, 7, 14, 15}

    def test_vpp_uneven_layer_count_10_layers_pp2_vpp2(self):
        """Regression: chunk_id overflow when layers not divisible by pp*vpp_chunks."""
        graph = _make_linear_graph(num_layers=10)
        ctx = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        def get_layers_for_stage(stage_id):
            return {
                int(n.layer) for n in result.nodes.values()
                if n.layer and n.op_type != "comm.send_recv"
                and n.annotations.get("stage_id") == stage_id
            }

        assert get_layers_for_stage(0) == {0, 1, 4, 5}
        assert get_layers_for_stage(1) == {2, 3, 6, 7, 8, 9}

        compute_nodes = [n for n in result.nodes.values() if n.op_type != "comm.send_recv"]
        virtual_ids = {n.annotations["virtual_stage_id"] for n in compute_nodes}
        assert virtual_ids == {0, 1, 2, 3}
        assert max(virtual_ids) == 3

    def test_vpp_uneven_layer_count_11_layers_pp2_vpp3(self):
        """Regression: 6 total chunks, bounded chunk_id."""
        graph = _make_linear_graph(num_layers=11)
        ctx = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=3)
        result = PipelineParallelPass().run(graph, ctx)

        compute_nodes = [n for n in result.nodes.values() if n.op_type != "comm.send_recv"]
        virtual_ids = {n.annotations["virtual_stage_id"] for n in compute_nodes}
        assert virtual_ids == {0, 1, 2, 3, 4, 5}
        assert max(virtual_ids) == 5


class TestPPModeDualPipe:
    """DualPipe: greedy assignment (same as 1F1B), F/B parallel scheduling."""

    def test_pp2_greedy_assignment_same_as_1f1b(self):
        graph = _make_linear_graph(num_layers=4)
        ctx_dp = _make_ctx(pp=2, pp_schedule="dualpipe")
        ctx_1f1b = _make_ctx(pp=2, pp_schedule="1f1b")

        result_dp = PipelineParallelPass().run(graph, ctx_dp)
        result_1f1b = PipelineParallelPass().run(graph, ctx_1f1b)

        dp_stages = {n.annotations["stage_id"] for n in result_dp.nodes.values()
                     if n.op_type != "comm.send_recv"}
        f1b_stages = {n.annotations["stage_id"] for n in result_1f1b.nodes.values()
                      if n.op_type != "comm.send_recv"}

        assert dp_stages == f1b_stages == {0, 1}

    def test_pp2_p2p_count_same_as_1f1b(self):
        graph = _make_linear_graph(num_layers=4)
        ctx_dp = _make_ctx(pp=2, pp_schedule="dualpipe", pp_layer_assignment=[0, 0, 1, 1])
        ctx_1f1b = _make_ctx(pp=2, pp_schedule="1f1b", pp_layer_assignment=[0, 0, 1, 1])

        result_dp = PipelineParallelPass().run(graph, ctx_dp)
        result_1f1b = PipelineParallelPass().run(graph, ctx_1f1b)

        p2p_dp = [n for n in result_dp.nodes.values() if n.op_type == "comm.send_recv"]
        p2p_1f1b = [n for n in result_1f1b.nodes.values() if n.op_type == "comm.send_recv"]

        assert len(p2p_dp) == len(p2p_1f1b) == 1

    def test_pp4_p2p_count(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=4, pp_schedule="dualpipe")
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3

    def test_no_virtual_stage_id(self):
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_schedule="dualpipe")
        result = PipelineParallelPass().run(graph, ctx)

        for node in result.nodes.values():
            if node.op_type != "comm.send_recv":
                assert node.annotations.get("virtual_stage_id") is None


class TestPPModeDualPipeV:
    """DualPipeV: interleaved assignment (same as VPP), DualPipe scheduling."""

    def test_pp2_vpp2_interleaved_assignment_same_as_vpp(self):
        graph = _make_linear_graph(num_layers=8)
        ctx_dpv = _make_ctx(pp=2, pp_schedule="dualpipev", vpp_chunks=2)
        ctx_vpp = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)

        result_dpv = PipelineParallelPass().run(graph, ctx_dpv)
        result_vpp = PipelineParallelPass().run(graph, ctx_vpp)

        def get_layers_for_stage(result, stage_id):
            return {
                int(n.layer) for n in result.nodes.values()
                if n.layer and n.op_type != "comm.send_recv"
                and n.annotations.get("stage_id") == stage_id
            }

        assert get_layers_for_stage(result_dpv, 0) == get_layers_for_stage(result_vpp, 0) == {0, 1, 4, 5}

    def test_pp2_vpp2_p2p_count(self):
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2, pp_schedule="dualpipev", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3

    def test_virtual_stage_id_annotation(self):
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2, pp_schedule="dualpipev", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        compute_nodes = [n for n in result.nodes.values() if n.op_type != "comm.send_recv"]
        virtual_ids = {n.annotations["virtual_stage_id"] for n in compute_nodes}
        assert virtual_ids == {0, 1, 2, 3}

    def test_p2p_has_virtual_stage_attrs(self):
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2, pp_schedule="dualpipev", vpp_chunks=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        for p2p in p2p_nodes:
            assert "src_virtual_stage" in p2p.attrs
            assert "dst_virtual_stage" in p2p.attrs

    def test_vpp_chunks_1_fallback_to_dualpipe(self):
        graph = _make_linear_graph(num_layers=4)
        ctx_dpv1 = _make_ctx(pp=2, pp_schedule="dualpipev", vpp_chunks=1)
        ctx_dp = _make_ctx(pp=2, pp_schedule="dualpipe")

        result_dpv1 = PipelineParallelPass().run(graph, ctx_dpv1)
        result_dp = PipelineParallelPass().run(graph, ctx_dp)

        dpv1_stages = {n.annotations["stage_id"] for n in result_dpv1.nodes.values()
                       if n.op_type != "comm.send_recv"}
        dp_stages = {n.annotations["stage_id"] for n in result_dp.nodes.values()
                     if n.op_type != "comm.send_recv"}

        assert dpv1_stages == dp_stages == {0, 1}

        for node in result_dpv1.nodes.values():
            if node.op_type != "comm.send_recv":
                assert node.annotations.get("virtual_stage_id") is None


# ── 3. Cross-Schedule Comparison ──────────────────────────────────────────────

class TestPPScheduleComparison:
    """Compare layer assignment and P2P count across schedules."""

    def test_layer_assignment_comparison(self):
        graph = _make_linear_graph(num_layers=8)

        ctx_1f1b = _make_ctx(pp=2, pp_schedule="1f1b")
        ctx_dp = _make_ctx(pp=2, pp_schedule="dualpipe")
        ctx_vpp = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)
        ctx_dpv = _make_ctx(pp=2, pp_schedule="dualpipev", vpp_chunks=2)

        result_1f1b = PipelineParallelPass().run(graph, ctx_1f1b)
        result_dp = PipelineParallelPass().run(graph, ctx_dp)
        result_vpp = PipelineParallelPass().run(graph, ctx_vpp)
        result_dpv = PipelineParallelPass().run(graph, ctx_dpv)

        def get_stage0_layers(result):
            return {
                int(n.layer) for n in result.nodes.values()
                if n.layer and n.op_type != "comm.send_recv"
                and n.annotations.get("stage_id") == 0
            }

        s0_1f1b = get_stage0_layers(result_1f1b)
        s0_dp = get_stage0_layers(result_dp)
        s0_vpp = get_stage0_layers(result_vpp)
        s0_dpv = get_stage0_layers(result_dpv)

        assert s0_1f1b == s0_dp, "1F1B and DualPipe use same greedy assignment"
        assert s0_vpp == s0_dpv, "VPP and DualPipeV use same interleaved assignment"
        assert s0_1f1b != s0_vpp, "Greedy vs interleaved differ"

    def test_p2p_count_comparison(self):
        graph = _make_linear_graph(num_layers=8)

        ctx_1f1b = _make_ctx(pp=2, pp_schedule="1f1b", pp_layer_assignment=[0, 0, 0, 0, 1, 1, 1, 1])
        ctx_dp = _make_ctx(pp=2, pp_schedule="dualpipe", pp_layer_assignment=[0, 0, 0, 0, 1, 1, 1, 1])
        ctx_vpp = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2)
        ctx_dpv = _make_ctx(pp=2, pp_schedule="dualpipev", vpp_chunks=2)

        def get_p2p_count(ctx):
            result = PipelineParallelPass().run(graph, ctx)
            return len([n for n in result.nodes.values() if n.op_type == "comm.send_recv"])

        p2p_1f1b = get_p2p_count(ctx_1f1b)
        p2p_dp = get_p2p_count(ctx_dp)
        p2p_vpp = get_p2p_count(ctx_vpp)
        p2p_dpv = get_p2p_count(ctx_dpv)

        assert p2p_1f1b == p2p_dp
        assert p2p_vpp == p2p_dpv
        assert p2p_vpp > p2p_1f1b


# ── 4. TrainingPipelinePass Per-Stage Metrics ─────────────────────────────────

class TestTrainingPipelinePassPerStage:
    """TrainingPipelinePass computes per-stage metrics from annotated graphs."""

    def _run_pipeline_pass(self, num_layers=4, pp=2, global_batch=8):
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        graph = _make_linear_graph(num_layers=num_layers)
        ctx = _make_ctx(pp=pp, global_batch=global_batch)

        g_pp = PipelineParallelPass().run(graph, ctx) if pp > 1 else graph
        g_flops = TrainingFlopsPass().run(g_pp, ctx)
        result = TrainingPipelinePass().run(g_flops, ctx)
        return result, result.metadata["step_result"]

    def test_pp1_no_bubble(self):
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=1, global_batch=8)

        g_flops = TrainingFlopsPass().run(graph, ctx)
        result = TrainingPipelinePass().run(g_flops, ctx)
        metrics = result.metadata["step_result"]

        assert metrics["warmup_steps"] == 0
        assert metrics["cooldown_steps"] == 0
        assert metrics["bubble_fraction"] == 0.0

    def test_pp2_has_bubble(self):
        _, metrics = self._run_pipeline_pass(pp=2, global_batch=8)
        assert metrics["warmup_steps"] == 1
        assert metrics["cooldown_steps"] == 1
        assert metrics["bubble_fraction"] > 0.0

    def test_per_stage_latency_not_divided_by_pp(self):
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        graph = _make_linear_graph(num_layers=4)
        ctx2 = _make_ctx(pp=2, global_batch=8)
        ctx1 = _make_ctx(pp=1, global_batch=8)

        g_pp2 = PipelineParallelPass().run(graph, ctx2)
        g_pp2 = TrainingFlopsPass().run(g_pp2, ctx2)
        result2 = TrainingPipelinePass().run(g_pp2, ctx2)
        per_stage_pp2 = result2.metadata["step_result"]["per_stage_ms"]

        g1 = TrainingFlopsPass().run(graph, ctx1)
        result1 = TrainingPipelinePass().run(g1, ctx1)
        total_pp1 = result1.metadata["step_result"]["per_stage_ms"]

        assert 0 < per_stage_pp2 <= total_pp1 * 1.1

    def test_stage_timelines_stored_in_metadata(self):
        result, _ = self._run_pipeline_pass(pp=2)
        assert "stage_timelines_fwd" in result.metadata
        timelines = result.metadata["stage_timelines_fwd"]
        assert isinstance(timelines, dict)
        assert 0 in timelines
        assert 1 in timelines

    def test_bubble_fraction_increases_with_pp(self):
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        results = {}
        for pp in (1, 2, 4):
            graph = _make_linear_graph(num_layers=4)
            ctx = _make_ctx(pp=pp, global_batch=8)
            g = PipelineParallelPass().run(graph, ctx) if pp > 1 else graph
            g = TrainingFlopsPass().run(g, ctx)
            r = TrainingPipelinePass().run(g, ctx)
            results[pp] = r.metadata["step_result"]["bubble_fraction"]

        assert results[1] == 0.0
        assert results[2] > results[1]
        assert results[4] > results[2]
        assert all(0.0 <= v < 1.0 for v in results.values())

    def test_vpp_bubble_smaller_than_1f1b(self):
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        graph = _make_linear_graph(num_layers=4)

        ctx_vpp = _make_ctx(pp=2, pp_schedule="interleaved", vpp_chunks=2, global_batch=8)
        ctx_std = _make_ctx(pp=2, pp_schedule="1f1b", vpp_chunks=1, global_batch=8)

        g_pp_vpp = PipelineParallelPass().run(graph, ctx_vpp)
        g_pp_std = PipelineParallelPass().run(graph, ctx_std)

        g_vpp = TrainingFlopsPass().run(g_pp_vpp, ctx_vpp)
        result_vpp = TrainingPipelinePass().run(g_vpp, ctx_vpp)

        g_std = TrainingFlopsPass().run(g_pp_std, ctx_std)
        result_std = TrainingPipelinePass().run(g_std, ctx_std)

        bubble_vpp = result_vpp.metadata["step_result"]["bubble_fraction"]
        bubble_std = result_std.metadata["step_result"]["bubble_fraction"]

        assert bubble_vpp <= bubble_std + 1e-6

        assert "stage_timelines_fwd" in result_vpp.metadata
        assert "stage_timelines_bwd" in result_vpp.metadata


# ── 5. Integration with Default Pipeline ──────────────────────────────────────

class TestPipelineIntegration:
    """PP pass integrated in build_default_pipeline."""

    def test_pp2_in_default_pipeline(self):
        from python.zrt.transform import build_default_pipeline

        graph = _make_linear_graph(num_layers=4)
        ctx = TransformContext(
            hw_spec=_make_hw(),
            parallel=ParallelConfig(pp=2),
            stream_config=StreamConfig(),
        )
        pipe = build_default_pipeline()
        result = pipe.run(graph, ctx)

        compute_nodes = [n for n in result.nodes.values() if n.category == "compute"]
        assert all("stage_id" in n.annotations for n in compute_nodes)

    def test_pp1_pipeline_no_p2p_nodes(self):
        from python.zrt.transform import build_default_pipeline

        graph = _make_linear_graph(num_layers=4)
        ctx = TransformContext(
            hw_spec=_make_hw(),
            parallel=ParallelConfig(pp=1),
            stream_config=StreamConfig(),
        )
        pipe = build_default_pipeline()
        result = pipe.run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 0


# ── 6. Typical Layer Partition ─────────────────────────────────────────────

class TestTypicalLayerPartition:
    """Partition with typical layer costs (LayerType granularity)."""

    def test_typical_layer_greedy_partition_basic(self):
        """Greedy partition balances stage loads better than uniform."""
        from python.zrt.graph.layer_strategy import LayerProfile, LayerType
        from python.zrt.graph.partition import partition_layers_by_strategy
        
        layer_types = [LayerType.HCA_HASH, LayerType.HCA_HASH, LayerType.HCA_TOPK, LayerType.SWA_HASH]
        profile = LayerProfile(layer_types=layer_types, typical_indices=[0, 2])
        
        typical_costs_fwd = {LayerType.HCA_HASH: 1000.0, LayerType.HCA_TOPK: 500.0, LayerType.SWA_HASH: 100.0}
        
        assignment = partition_layers_by_strategy(profile, typical_costs_fwd, pp=2, pp_schedule="1f1b")
        
        # Compute stage fwd manually
        stage_fwd = [0.0, 0.0]
        for i, layer_type in enumerate(layer_types):
            stage_fwd[assignment[i]] += typical_costs_fwd.get(layer_type, 0.0)
        
        uniform_diff = abs(2000.0 - 600.0)
        greedy_diff = abs(stage_fwd[0] - stage_fwd[1])
        
        assert greedy_diff < uniform_diff
        
    def test_typical_layer_explicit_assignment_override(self):
        """Explicit pp_layer_assignment takes precedence over greedy."""
        from python.zrt.graph.layer_strategy import LayerProfile, LayerType
        from python.zrt.graph.partition import partition_layers_by_strategy
        
        layer_types = [LayerType.HCA_HASH, LayerType.HCA_HASH, LayerType.HCA_TOPK, LayerType.SWA_HASH]
        profile = LayerProfile(layer_types=layer_types, typical_indices=[0, 2])
        
        typical_costs_fwd = {LayerType.HCA_HASH: 1000.0, LayerType.HCA_TOPK: 500.0, LayerType.SWA_HASH: 100.0}
        
        explicit_assignment = [0, 0, 0, 1]
        assignment = partition_layers_by_strategy(
            profile, typical_costs_fwd, pp=2, 
            pp_schedule="1f1b",
            pp_layer_assignment=explicit_assignment
        )
        
        assert assignment == explicit_assignment
        
    def test_typical_layer_vpp_interleaved_override(self):
        """VPP interleaved schedule overrides greedy when vpp_chunks > 1."""
        from python.zrt.graph.layer_strategy import LayerProfile, LayerType
        from python.zrt.graph.partition import partition_layers_by_strategy
        
        layer_types = [LayerType.HCA_HASH] * 8
        profile = LayerProfile(layer_types=layer_types, typical_indices=[0])
        
        typical_costs_fwd = {LayerType.HCA_HASH: 1000.0}
        
        assignment = partition_layers_by_strategy(
            profile, typical_costs_fwd, pp=2,
            pp_schedule="interleaved", vpp_chunks=2
        )
        
        assert assignment == [0, 0, 1, 1, 0, 0, 1, 1]