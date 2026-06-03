"""ExpertGroupedMMPass: fuse routed expert ops into GroupedMatMul.

Runs after ExpertParallelPass (ep_needs_a2a annotation exists) and
before CommInserterPass (which inserts A2A around fused nodes).

For each MoE layer, fuses:
  - gate_proj + up_proj → GroupedMatMul (output 2×FFN)
  - SwiGLU activation → keeps as is, scope adjusted
  - down_proj → GroupedMatMul (output hidden)

Dataflow:
  dispatch → GroupedMM(gate_up) → silu → GroupedMM(down) → combine
"""
from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

from python.zrt.ir.edge import Edge
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


def _is_routed_expert(node: OpNode) -> bool:
    """Check if node is a routed expert op (not shared)."""
    scope = node.scope.lower()
    has_expert = "expert" in scope or "experts." in scope
    is_shared = "shared" in scope
    return has_expert and not is_shared


def _layer_key(node: OpNode) -> str:
    """Extract a stable key grouping all expert nodes in the same MoE layer."""
    scope = node.scope
    for sep in ("/experts/", "/expert_", ".experts.", ".expert_"):
        idx = scope.lower().find(sep)
        if idx >= 0:
            return scope[:idx]
    return scope.rsplit(".", 1)[0] if "." in scope else scope


def _expert_weight_name(scope: str) -> str:
    """Return the weight suffix (w1, w2, w3, gate_proj, up_proj, down_proj)."""
    parts = [p for p in scope.lower().replace("/", ".").split(".") if p]
    # DSv4-style module names: w1, w2, w3. Match path segments only so names
    # like w12_proj are not accidentally classified as w1.
    for w in ("w1", "w2", "w3"):
        if w in parts:
            return w
    # Standard-style module names.
    for pat in ("gate_proj", "up_proj", "down_proj"):
        if pat in parts:
            return pat
    return ""


def _is_expert_matmul(node: OpNode) -> bool:
    op = node.op_type.lower()
    return (
        "mm" in op
        or "matmul" in op
        or "gemm" in op
        or "linear" in op
    ) and bool(node.inputs) and bool(node.outputs)


def _make_grouped_mm(node_id: str, scope: str,
                     inputs: list[TensorMeta],
                     outputs: list[TensorMeta],
                     src_node: OpNode) -> OpNode:
    """Create a GroupedMatMul node inheriting key annotations from src_node."""
    node = OpNode(
        id=node_id,
        op_type="GroupedMatMul",
        inputs=inputs,
        outputs=outputs,
        scope=scope,
        layer=src_node.layer,
        category="compute",
    )
    # Inherit critical annotations from any source expert node
    for key in ("ep_needs_a2a", "ep_experts_local", "recompute", "phase"):
        val = src_node.annotations.get(key)
        if val is not None:
            node.annotations[key] = val
    node.annotations.setdefault("ep_needs_a2a", True)
    node.annotations["fused_by"] = "expert_grouped_mm"
    return node


def _weight_tensor(name: str, shape: tuple[int, ...], dtype: DType = DType.BF16) -> TensorMeta:
    return TensorMeta.from_shape_dtype(name, shape, dtype)


def _dtype_bytes(tensor: TensorMeta | None, default: float = 2.0) -> float:
    return float(tensor.dtype.itemsize) if tensor is not None else default


def _quant_variant(ctx: "TransformContext") -> str:
    quant = getattr(ctx, "quant", None)
    if quant is None:
        return "standard"
    weight = str(getattr(quant, "weight", "")).lower()
    act = str(getattr(quant, "activation", "")).lower()
    if weight in {"int4", "fp4"} and act.startswith("fp8"):
        return "w4a8"
    return "standard"


def _source_op_suffix(node_id: str) -> str | None:
    match = re.search(r"(op_\d+)(?!.*op_\d+)", node_id)
    return match.group(1) if match else None


def _mega_moe_node_id(layer_key: str, src_node: OpNode) -> str:
    suffix = _source_op_suffix(src_node.id)
    if suffix:
        return f"mega_moe_{suffix}"
    return f"{layer_key.replace('.', '_')}_mega_moe"


def _is_routed_expert_add(node: OpNode, layer_key: str, hidden: int) -> bool:
    if node.scope != layer_key:
        return False
    if node.op_type not in {"aten.add.Tensor", "aten.add_.Tensor"}:
        return False
    return bool(node.outputs) and node.outputs[0].shape[-1] == hidden


def _collect_routed_add_chain(
    g: "OpGraph",
    layer_key: str,
    seed_ids: set[str],
    hidden: int,
) -> set[str]:
    collected: set[str] = set()
    queue = list(seed_ids)
    while queue:
        current = queue.pop(0)
        for succ_id in g.successors(current):
            if succ_id in collected:
                continue
            succ = g.nodes.get(succ_id)
            if succ is None or not _is_routed_expert_add(succ, layer_key, hidden):
                continue
            collected.add(succ_id)
            queue.append(succ_id)
    return collected


def _make_mega_moe(node_id: str, scope: str,
                   input_tensor: TensorMeta,
                   output_tensor: TensorMeta,
                   src_node: OpNode,
                   meta: dict[str, object],
                   ctx: "TransformContext",
                   tokens_per_rank: int,
                   tokens_per_expert: int) -> OpNode:
    node = OpNode(
        id=node_id,
        op_type="mega_moe",
        inputs=[input_tensor],
        outputs=[output_tensor],
        scope=scope,
        layer=src_node.layer,
        category="compute",
    )
    node.component = "moe.mega_moe"
    phase = src_node.annotations.get("phase")
    if phase:
        node.annotations["phase"] = phase
    if phase in ("fwd", "forward", "train_forward") and src_node.annotations.get("recompute"):
        node.annotations["recompute"] = True
    node.annotations["fused_by"] = "mega_moe_graph_capture"
    node.annotations["fused_dispatch_compute_combine"] = True
    node.annotations["mega_moe_waves"] = ctx.training.mega_moe_waves if ctx.training else 0
    node.annotations["ep_tokens_per_rank"] = tokens_per_rank
    node.annotations["ep_tokens_per_expert"] = tokens_per_expert
    node.annotations["mega_moe_meta"] = dict(meta)
    node.attrs["mega_moe_meta"] = dict(meta)
    return node


class ExpertGroupedMMPass(GraphPass):
    """Fuse routed expert gate/up/down ops into GroupedMatMul nodes.

    Condition: only runs when ctx.parallel.ep > 1 and MoE profiles exist.
    """

    name = "expert_grouped_mm"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        ep = ctx.parallel.ep
        if ep <= 1:
            return graph

        profile = ctx.profile
        num_experts = getattr(profile, "num_experts", None) if profile else None
        if not num_experts or num_experts <= 1:
            return graph

        g = graph.clone()
        self._fuse_experts(g, num_experts, ep, ctx)
        return g

    def _fuse_experts(self, g: "OpGraph", num_experts: int, ep: int,
                      ctx: "TransformContext") -> None:
        # Group expert nodes by MoE layer and phase. Forward and backward use
        # different report-level GroupedMM shapes and ordering.
        expert_nodes = [n for n in g.topo_sort() if _is_routed_expert(n)]
        if not expert_nodes:
            return

        layers: dict[tuple[str, str], list[OpNode]] = {}
        for n in expert_nodes:
            phase = n.annotations.get("phase", "fwd")
            if phase in ("backward", "train_backward"):
                phase = "bwd"
            layers.setdefault((_layer_key(n), phase), []).append(n)

        experts_per_rank = num_experts // ep
        seq = g.metadata.get("seq_len", 128)
        hidden = g.metadata.get("hidden", 7168)
        batch = ctx.training.micro_batch if ctx.training else 1
        topk = ctx.profile.moe_active if ctx.profile else 6
        total_routed_tokens = batch * seq * topk
        # After EP all-to-all, each target EP rank receives routed tokens from
        # every source rank in the EP group. Under uniform routing, the per-rank
        # receive volume equals local_tokens_per_source_rank * top_k.
        tokens_per_ep_rank = max(1, total_routed_tokens)
        M = max(1, (tokens_per_ep_rank + experts_per_rank - 1) // experts_per_rank)
        G = experts_per_rank

        for (layer_key, phase), nodes in layers.items():
            if phase == "bwd":
                self._fuse_backward_layer(
                    g, layer_key, nodes, G, M, tokens_per_ep_rank, hidden,
                    num_experts, ep, topk, ctx)
                continue

            if phase not in ("fwd", "forward", "train_forward"):
                continue

            gates, ups, downs = [], [], []
            for n in nodes:
                if not _is_expert_matmul(n):
                    continue
                w = _expert_weight_name(n.scope)
                if w in ("w1", "gate_proj"):
                    gates.append(n)
                elif w in ("w3", "up_proj"):
                    ups.append(n)
                elif w in ("w2", "down_proj"):
                    downs.append(n)
            if not gates or not ups or not downs:
                continue

            gate_dim = gates[0].outputs[0].shape[-1] if gates[0].outputs else 2048
            up_dim = ups[0].outputs[0].shape[-1] if ups[0].outputs else gate_dim
            ffn = downs[0].inputs[0].shape[-1] if downs[0].inputs else min(gate_dim, up_dim)
            H_in = gates[0].inputs[0].shape[-1] if gates[0].inputs else hidden
            H_out = downs[0].outputs[0].shape[-1] if (downs and downs[0].outputs) else hidden
            if (
                any(n.outputs and n.outputs[0].shape[-1] != gate_dim for n in gates)
                or any(n.outputs and n.outputs[0].shape[-1] != up_dim for n in ups)
                or any(n.inputs and n.inputs[0].shape[-1] != ffn for n in downs)
                or any(n.inputs and n.inputs[0].shape[-1] != H_in for n in gates + ups)
                or any(n.outputs and n.outputs[0].shape[-1] != H_out for n in downs)
            ):
                continue

            if ctx.training and ctx.training.mega_moe:
                self._fuse_mega_moe_forward_layer(
                    g, layer_key, nodes, gates[0], downs[0],
                    G, M, tokens_per_ep_rank, num_experts, ep, topk,
                    H_in, H_out, ffn, ctx,
                )
                continue

            expert_ids = {n.id for n in nodes}
            old_ids = expert_ids | _collect_routed_add_chain(
                g, layer_key, {n.id for n in downs}, H_out)

            # Collect ALL external edges at once before any mutation
            # external-in: edge from outside → any old node
            in_edges = [e for e in g.edges if e.dst in expert_ids and e.src not in old_ids]
            # external-out: edge from any old node → outside
            out_edges = [e for e in g.edges if e.src in old_ids and e.dst not in old_ids]

            if not in_edges or not out_edges:
                continue

            # Create grouped nodes
            gate_up_id = f"{layer_key.replace('.','_')}_grouped_gate_up"
            gate_up = _make_grouped_mm(
                gate_up_id, f"{layer_key}.moe",
                [
                    TensorMeta.from_shape_dtype("grouped_in", (G, M, H_in), DType.BF16),
                    _weight_tensor("grouped_gate_up_weight", (G, H_in, gate_dim + up_dim), DType.BF16),
                ],
                [TensorMeta.from_shape_dtype("grouped_gate_up_out", (G, M, gate_dim + up_dim), DType.BF16)],
                gates[0],
            )
            gate_up.component = "moe.grouped_gate_up"
            gate_up.annotations["grouped_mm_role"] = "gate_up"
            gate_up.annotations["ep_tokens_per_rank"] = tokens_per_ep_rank
            gate_up.annotations["ep_tokens_per_expert"] = M
            gate_up.annotations["ep_a2a_inserted"] = False

            act_id = f"{layer_key.replace('.','_')}_grouped_silu"
            act_node = OpNode(
                id=act_id,
                op_type="aten.silu",
                inputs=[TensorMeta.from_shape_dtype("silu_in", (G, M, gate_dim + up_dim), DType.BF16)],
                outputs=[TensorMeta.from_shape_dtype("silu_out", (G, M, ffn), DType.BF16)],
                scope=f"{layer_key}.moe",
                layer=gates[0].layer,
                category="compute",
            )
            phase = gates[0].annotations.get("phase")
            if phase:
                act_node.annotations["phase"] = phase
            if gates[0].annotations.get("recompute"):
                act_node.annotations["recompute"] = True
            act_node.component = "moe.grouped_swiglu"

            down_id = f"{layer_key.replace('.','_')}_grouped_down"
            down = _make_grouped_mm(
                down_id, f"{layer_key}.moe",
                [
                    TensorMeta.from_shape_dtype("grouped_down_in", (G, M, ffn), DType.BF16),
                    _weight_tensor("grouped_down_weight", (G, ffn, H_out), DType.BF16),
                ],
                [TensorMeta.from_shape_dtype("grouped_down_out", (G, M, H_out), DType.BF16)],
                downs[0],
            )
            down.component = "moe.grouped_down"
            down.annotations["grouped_mm_role"] = "down"
            down.annotations["ep_tokens_per_rank"] = tokens_per_ep_rank
            down.annotations["ep_tokens_per_expert"] = M

            # Drop old nodes and all edges touching them
            g.edges = [e for e in g.edges if e.src not in old_ids and e.dst not in old_ids]
            for nid in old_ids:
                g.nodes.pop(nid, None)

            # Add new nodes
            g.nodes[gate_up_id] = gate_up
            g.nodes[act_id] = act_node
            g.nodes[down_id] = down

            # Link: gate_up → silu → down for CommInserter
            gate_up.annotations["ep_block_down_id"] = down_id
            # Only gate_up gets ep_needs_a2a (entry point for CommInserter)
            down.annotations.pop("ep_needs_a2a", None)
            down.annotations.pop("ep_experts_local", None)

            # Wire: external-in → gate_up
            seen_src = set()
            for e in in_edges:
                key = (e.src, e.src_idx)
                if key not in seen_src:
                    seen_src.add(key)
                    g.edges.append(Edge(e.src, e.src_idx, gate_up_id, 0, gate_up.inputs[0]))
            # gate_up → silu → down
            g.edges.append(Edge(gate_up_id, 0, act_id, 0, gate_up.outputs[0]))
            g.edges.append(Edge(act_id, 0, down_id, 0, act_node.outputs[0]))
            # down → external-out
            for e in out_edges:
                g.edges.append(Edge(down_id, 0, e.dst, e.dst_idx, down.outputs[0]))

            g._rebuild_adjacency()

    def _fuse_mega_moe_forward_layer(self, g: "OpGraph", layer_key: str,
                                     nodes: list[OpNode],
                                     gate: OpNode, down: OpNode,
                                     experts_per_rank: int,
                                     tokens_per_expert: int,
                                     tokens_per_rank: int,
                                     num_experts: int, ep: int, topk: int,
                                     H_in: int, H_out: int, ffn: int,
                                     ctx: "TransformContext") -> None:
        expert_ids = {n.id for n in nodes}
        old_ids = expert_ids | _collect_routed_add_chain(g, layer_key, {down.id}, H_out)
        in_edges = [e for e in g.edges if e.dst in expert_ids and e.src not in old_ids]
        out_edges = [e for e in g.edges if e.src in old_ids and e.dst not in old_ids]
        if not in_edges or not out_edges:
            return

        weight_bytes = (
            float(ctx.quant.weight_bytes)
            if getattr(ctx, "quant", None) is not None
            else _dtype_bytes(gate.inputs[0] if gate.inputs else None)
        )
        logical_tokens = int(g.metadata.get("seq_len", tokens_per_rank))
        micro_batch = ctx.training.micro_batch if ctx.training else 1
        boundary_tokens = max(1, micro_batch * logical_tokens)
        input_dtype = gate.inputs[0].dtype if gate.inputs else DType.BF16
        output_dtype = down.outputs[0].dtype if down.outputs else input_dtype
        input_tensor = TensorMeta.from_shape_dtype(
            "mega_moe_in", (boundary_tokens, H_in), input_dtype)
        output_tensor = TensorMeta.from_shape_dtype(
            "mega_moe_out", (boundary_tokens, H_out), output_dtype)
        meta = {
            "m": logical_tokens,
            "micro_batch": micro_batch,
            "compute_tokens_per_rank": tokens_per_rank,
            "n": H_out,
            "k": ffn,
            "top_k": topk,
            "num_experts": num_experts,
            "experts_per_rank": experts_per_rank,
            "ep": ep,
            "requested_waves": ctx.training.mega_moe_waves if ctx.training else 0,
            "act_bytes": _dtype_bytes(gate.inputs[0] if gate.inputs else None),
            "moe_act_bytes": _dtype_bytes(gate.inputs[0] if gate.inputs else None),
            "out_bytes": _dtype_bytes(down.outputs[0] if down.outputs else None),
            "weight_bytes": weight_bytes,
            "weight_stored_bytes": weight_bytes,
            "fwd_multiplier": 3 * topk,
            "quant_variant": _quant_variant(ctx),
            "fused_dispatch_compute_combine": True,
        }

        mega_id = _mega_moe_node_id(layer_key, gate)
        mega = _make_mega_moe(
            mega_id, f"{layer_key}.moe", input_tensor, output_tensor,
            gate, meta, ctx, tokens_per_rank, tokens_per_expert,
        )

        g.edges = [e for e in g.edges if e.src not in old_ids and e.dst not in old_ids]
        for nid in old_ids:
            g.nodes.pop(nid, None)
        g.nodes[mega_id] = mega

        seen_src = set()
        for e in in_edges:
            key = (e.src, e.src_idx)
            if key not in seen_src:
                seen_src.add(key)
                g.edges.append(Edge(e.src, e.src_idx, mega_id, 0, mega.inputs[0]))
        for e in out_edges:
            g.edges.append(Edge(mega_id, 0, e.dst, e.dst_idx, mega.outputs[0]))

        g._rebuild_adjacency()

    def _fuse_backward_layer(self, g: "OpGraph", layer_key: str, nodes: list[OpNode],
                             G: int, M: int, tokens_per_rank: int,
                             hidden: int, num_experts: int, ep: int,
                             topk: int, ctx: "TransformContext") -> None:
        gates, ups, downs = [], [], []
        for n in nodes:
            if not _is_expert_matmul(n):
                continue
            w = _expert_weight_name(n.scope)
            if w in ("w1", "gate_proj"):
                gates.append(n)
            elif w in ("w3", "up_proj"):
                ups.append(n)
            elif w in ("w2", "down_proj"):
                downs.append(n)

        if not gates or not ups or not downs:
            return

        # Backward down consumes grad wrt the forward output (H_out) and
        # produces grad wrt the expert activation (ffn).
        H_out = downs[0].inputs[0].shape[-1] if downs[0].inputs else hidden
        ffn = downs[0].outputs[0].shape[-1] if downs[0].outputs else (
            gates[0].inputs[0].shape[-1] if gates[0].inputs else 3072)
        gate_grad_dim = gates[0].outputs[0].shape[-1] if gates[0].outputs else ffn
        up_grad_dim = ups[0].outputs[0].shape[-1] if ups[0].outputs else gate_grad_dim
        gate_up_dim = gate_grad_dim + up_grad_dim
        if (
            any(n.inputs and n.inputs[0].shape[-1] != H_out for n in downs)
            or any(n.outputs and n.outputs[0].shape[-1] != ffn for n in downs)
            or any(n.inputs and n.inputs[0].shape[-1] != ffn for n in gates + ups)
            or any(n.outputs and n.outputs[0].shape[-1] != gate_grad_dim for n in gates)
            or any(n.outputs and n.outputs[0].shape[-1] != up_grad_dim for n in ups)
        ):
            return
        old_ids = {n.id for n in nodes}

        succ: dict[str, list[str]] = {}
        for e in g.edges:
            succ.setdefault(e.src, []).append(e.dst)

        def _reachable_from_old(node_id: str) -> bool:
            stack = list(succ.get(node_id, []))
            seen = {node_id}
            while stack:
                cur = stack.pop()
                if cur in old_ids:
                    return True
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(succ.get(cur, []))
            return False

        in_edges = [
            e for e in g.edges
            if e.dst in old_ids and e.src not in old_ids
        ]
        out_edges = [
            e for e in g.edges
            if e.src in old_ids and e.dst not in old_ids
            and not _reachable_from_old(e.dst)
        ]
        if not in_edges or not out_edges:
            return

        down_id = f"{layer_key.replace('.','_')}_grouped_down_bwd"
        down = _make_grouped_mm(
            down_id, f"{layer_key}.moe",
            [
                TensorMeta.from_shape_dtype("grouped_down_bwd_in", (G, M, H_out), DType.BF16),
                _weight_tensor("grouped_down_bwd_weight", (G, H_out, ffn), DType.BF16),
            ],
            [TensorMeta.from_shape_dtype("grouped_down_bwd_out", (G, M, ffn), DType.BF16)],
            downs[0],
        )
        down.component = "moe.grouped_down_bwd"
        down.annotations["phase"] = "bwd"
        down.annotations["grouped_mm_role"] = "down_bwd"
        down.annotations["ep_tokens_per_rank"] = tokens_per_rank
        down.annotations["ep_tokens_per_expert"] = M
        down.annotations["ep_a2a_inserted"] = False
        down.annotations.pop("recompute", None)

        gate_up_id = f"{layer_key.replace('.','_')}_grouped_gate_up_bwd"
        gate_up = _make_grouped_mm(
            gate_up_id, f"{layer_key}.moe",
            [
                TensorMeta.from_shape_dtype("grouped_gate_up_bwd_in", (G, M, ffn), DType.BF16),
                _weight_tensor("grouped_gate_up_bwd_weight", (G, ffn, gate_up_dim), DType.BF16),
            ],
            [TensorMeta.from_shape_dtype("grouped_gate_up_bwd_out", (G, M, gate_up_dim), DType.BF16)],
            gates[0],
        )
        gate_up.component = "moe.grouped_gate_up_bwd"
        gate_up.annotations["phase"] = "bwd"
        gate_up.annotations["grouped_mm_role"] = "gate_up_bwd"
        gate_up.annotations["ep_tokens_per_rank"] = tokens_per_rank
        gate_up.annotations["ep_tokens_per_expert"] = M
        gate_up.annotations.pop("ep_needs_a2a", None)
        gate_up.annotations.pop("ep_experts_local", None)
        gate_up.annotations.pop("recompute", None)

        g.edges = [e for e in g.edges if e.src not in old_ids and e.dst not in old_ids]
        for nid in old_ids:
            g.nodes.pop(nid, None)

        g.nodes[down_id] = down
        g.nodes[gate_up_id] = gate_up

        if ctx.training and ctx.training.mega_moe:
            weight_bytes = (
                float(ctx.quant.weight_bytes)
                if getattr(ctx, "quant", None) is not None
                else _dtype_bytes(down.inputs[0] if down.inputs else None)
            )
            logical_tokens = int(g.metadata.get("seq_len", tokens_per_rank))
            meta = {
                "m": logical_tokens,
                "micro_batch": ctx.training.micro_batch,
                "n": H_out,
                "k": ffn,
                "top_k": topk,
                "num_experts": num_experts,
                "experts_per_rank": G,
                "ep": ep,
                "requested_waves": ctx.training.mega_moe_waves,
                "act_bytes": _dtype_bytes(down.inputs[0] if down.inputs else None),
                "moe_act_bytes": _dtype_bytes(down.inputs[0] if down.inputs else None),
                "out_bytes": _dtype_bytes(gate_up.outputs[0] if gate_up.outputs else None),
                "weight_bytes": weight_bytes,
                "weight_stored_bytes": weight_bytes,
                "fwd_multiplier": 3 * topk,
                "quant_variant": _quant_variant(ctx),
                "fused_dispatch_compute_combine": True,
            }
            down.annotations.pop("ep_needs_a2a", None)
            down.annotations.pop("ep_experts_local", None)
            down.annotations.pop("ep_a2a_inserted", None)
            down.annotations["mega_moe_internal_comm"] = True
            down.annotations["fused_dispatch_compute_combine"] = True
            down.annotations["mega_moe_waves"] = ctx.training.mega_moe_waves
            down.annotations["mega_moe_meta"] = dict(meta)
            down.attrs["mega_moe_meta"] = dict(meta)
        else:
            down.annotations["ep_block_down_id"] = gate_up_id

        seen_src = set()
        for e in in_edges:
            key = (e.src, e.src_idx)
            if key not in seen_src:
                seen_src.add(key)
                g.edges.append(Edge(e.src, e.src_idx, down_id, 0, down.inputs[0]))

        g.edges.append(Edge(down_id, 0, gate_up_id, 0, down.outputs[0]))
        for e in out_edges:
            g.edges.append(Edge(gate_up_id, 0, e.dst, e.dst_idx, gate_up.outputs[0]))

        g._rebuild_adjacency()
