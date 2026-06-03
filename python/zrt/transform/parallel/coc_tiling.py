"""CoCTilePass: split CoC-annotated TP all-reduce nodes into tiled pipelines.

When tp_coc is enabled, a single ``comm_allreduce`` node is replaced by
K tile-level comm nodes on the communication stream.  The predecessor
compute node is also split into K tile compute nodes so that each comm
tile can fire as soon as the corresponding compute tile finishes, rather
than waiting for the entire compute op to complete.

Execution model (K = 3 shown)::

    preds ─→ src_t1 ──→ comm_t1
              │  ↓          │  ↓
              │ src_t2 ──→ comm_t2
              │  ↓          │  ↓
              └── src_t3 ──→ comm_t3 ──→ succs

- src_ti → src_t(i+1): explicit data edge so topological order is fixed.
- src_ti → comm_ti: comm tile needs tile-i compute output.
- comm_ti → comm_t(i+1): explicit edge guarantees sequential execution
  (augments the same-comm-stream serialization in DAGScheduler).
- All src_ti nodes lie on the same compute stream.
- All comm_ti nodes lie on the same comm stream.
- comm_tK reconnects to the original successors (receivers of the reduced
  output).
"""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from python.zrt.ir.edge import Edge
from python.zrt.transform.base import GraphPass

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.transform.context import TransformContext


def _clone_node_as_tile(
    src: "OpNode",
    tile_index: int,
    tile_k: int,
    suffix: str = "_ct",
) -> "OpNode":
    """Clone *src* and adjust latency for tile *tile_index* (0-based).

    The new node id is ``{src.id}_t{tile_index}{suffix}``.
    ``latency_us`` is divided by *tile_k*.
    """
    tile = copy.deepcopy(src)
    tile.id = f"{src.id}_t{tile_index}{suffix}"
    tile.attrs = dict(src.attrs)
    tile.attrs["coc_tile_index"] = tile_index
    tile.attrs["coc_tile_k"] = tile_k

    orig_lat = src.annotations.get("latency_us", 0.0)
    if orig_lat > 0:
        tile.annotations["latency_us"] = orig_lat / tile_k

    return tile


class CoCTilePass(GraphPass):
    """Split CoC-annotated TP all-reduce comm nodes into tiled pipelines.

    Runs after CommInserterPass (which creates the comm nodes and
    attaches ``attrs["coc_tile_k"]``), but before StreamAssignPass
    (which assigns stream_ids) and CommLatencyPass (which sets
    latency_us on comm nodes).
    """

    name = "coc_tiling"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        if ctx.training is None or not ctx.training.tp_coc:
            return graph

        g = graph.clone()
        coc_k = max(2, ctx.training.tp_coc_tile_k if ctx.training else 4)

        coc_comm_nodes = [
            n for n in g.nodes.values()
            if n.category == "communication"
            and n.attrs.get("coc_tile_k", 0) > 0
        ]
        if not coc_comm_nodes:
            return g

        logger.info(
            "CoCTilePass: splitting %d all_reduce nodes into %d-tile pipelines",
            len(coc_comm_nodes), coc_k,
        )
        for comm_node in coc_comm_nodes:
            self._tile_replace(g, comm_node, coc_k)

        logger.info(
            "CoCTilePass: done — graph now has %d nodes (was %d before split)",
            len(g.nodes), len(g.nodes) - (coc_k - 1) * len(coc_comm_nodes) * 2,
        )
        return g

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tile_replace(g: "OpGraph", comm_node: "OpNode", coc_k: int) -> None:
        """Replace *comm_node* and its compute predecessor with a K-tile pipeline.

        Pre-condition: *comm_node* has exactly one compute predecessor
        (the node whose output is being all-reduced).
        """
        pred_ids = g.predecessors(comm_node.id)
        if len(pred_ids) != 1:
            return
        src_id = pred_ids[0]
        src_node = g.nodes.get(src_id)
        if src_node is None or src_node.category == "communication":
            return

        succ_ids = [e.dst for e in g.edges if e.src == comm_node.id]

        tile_k = max(2, int(comm_node.attrs.get("coc_tile_k", coc_k)))
        orig_src_preds = [(e.src, e.src_idx, e.dst_idx, e.tensor)
                          for e in g.edges if e.dst == src_id]
        orig_src_outputs = list(src_node.outputs)

        # ── 1. Create tile compute nodes ─────────────────────────────────────
        ct_nodes: list["OpNode"] = []
        for ti in range(tile_k):
            ct = _clone_node_as_tile(src_node, ti, tile_k, suffix="_ct")
            ct.category = "compute"
            ct.annotations["coc_tile_index"] = ti
            ct.annotations["coc_tile_k"] = tile_k
            g.nodes[ct.id] = ct
            ct_nodes.append(ct)

        # ── 2. Create tile comm nodes ────────────────────────────────────────
        cm_nodes: list["OpNode"] = []
        for ti in range(tile_k):
            cm = copy.deepcopy(comm_node)
            cm.id = f"{comm_node.id}_t{ti}"
            cm.attrs = dict(comm_node.attrs)
            cm.attrs["coc_tile_index"] = ti
            cm.attrs["coc_tile_k"] = tile_k
            cm.attrs["tiled_from"] = comm_node.id
            cm.annotations["overlap_target"] = f"coc:{ct_nodes[ti].id}"
            cm.annotations["inserted_by"] = "tp_pass"
            cm.annotations["overlap_strategy"] = "tp"
            cm.category = "communication"
            g.nodes[cm.id] = cm
            cm_nodes.append(cm)

        # ── 3. Remove old nodes (src + comm) ---------------------------------
        del g.nodes[src_id]
        if comm_node.id in g.nodes:
            del g.nodes[comm_node.id]

        # ── 4. Purge edges touching src or comm ------------------------------
        g.edges = [
            e for e in g.edges
            if e.src != src_id and e.dst != src_id
            and e.src != comm_node.id and e.dst != comm_node.id
        ]

        # ── 5. Wire predecessors → ct_0 --------------------------------------
        for (pred, src_idx, dst_idx, tensor) in orig_src_preds:
            g.edges.append(Edge(
                src=pred, src_idx=src_idx,
                dst=ct_nodes[0].id, dst_idx=dst_idx,
                tensor=tensor,
            ))

        # ── 6. Wire ct_i → ct_{i+1} chain (same compute stream, enforced topo) ─
        for i in range(tile_k - 1):
            g.edges.append(Edge(
                src=ct_nodes[i].id, src_idx=0,
                dst=ct_nodes[i + 1].id, dst_idx=0,
                tensor=orig_src_outputs[0],
            ))

        # ── 7. Wire ct_i → cm_i (data dependency: comm needs tile i of compute) ─
        for i in range(tile_k):
            g.edges.append(Edge(
                src=ct_nodes[i].id, src_idx=0,
                dst=cm_nodes[i].id, dst_idx=0,
                tensor=orig_src_outputs[0],
            ))

        # ── 8. Wire cm_i → cm_{i+1} (comm serialization + topo safety) ─
        for i in range(tile_k - 1):
            g.edges.append(Edge(
                src=cm_nodes[i].id, src_idx=0,
                dst=cm_nodes[i + 1].id, dst_idx=0,
                tensor=orig_src_outputs[0],
            ))

        # ── 9. Wire cm_{last} → original successors ────────────────────────
        last_cm = cm_nodes[-1]
        for dst_id in succ_ids:
            for e_idx in range(len(last_cm.outputs)):
                g.edges.append(Edge(
                    src=last_cm.id, src_idx=e_idx,
                    dst=dst_id, dst_idx=e_idx,
                    tensor=last_cm.outputs[e_idx],
                ))

        g._rebuild_adjacency()
