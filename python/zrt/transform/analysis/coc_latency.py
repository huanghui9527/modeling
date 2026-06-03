"""CoCLatencyPass: divide latency_us by coc_tile_k for CoC tile nodes.

Runs after RooflinePass and CommLatencyPass have set latency_us
annotations on compute and comm nodes respectively.  Tile nodes
created by CoCTilePass carry ``attrs["coc_tile_k"]`` — this pass
divides their ``latency_us`` so each tile accounts for 1/K of the
original work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


class CoCLatencyPass(GraphPass):
    """Post-process CoC tile node latencies.

    Each tile compute or comm node created by CoCTilePass has a
    ``coc_tile_k`` attr.  RooflinePass / CommLatencyPass estimate
    latency for the full op; this pass divides by K.
    """

    name = "coc_latency"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()
        for node in g.nodes.values():
            k = int(node.attrs.get("coc_tile_k", 1))
            if k <= 1:
                continue
            lat = node.annotations.get("latency_us", 0.0)
            if lat > 0:
                node.annotations["latency_us"] = lat / k
        return g
