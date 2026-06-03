"""HTML exporter for spec-based training estimation (--estimate-config).

Produces a self-contained HTML page with:
1. Summary: model / hardware / topology / strategy / performance / memory / bound.
2. Logical cluster topology with PP / DP / TP / EP / CP rank labels.
3. Model-level timing overview in Transformer-Explainer-like blocks.
4. Expandable model -> layer/block -> op timing/formula/bound analysis.
5. Calibration table with public/official reference points.

Security notes:
- JSON payload is injected with JSON.parse(<safe JS string literal>), not as a raw
  object literal.
- HTML title is escaped.
- User/model/op strings are escaped again in browser-side rendering.
"""

from __future__ import annotations

import html as html_lib
import json
import math
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zrt.training.io.operator_time_stats import (
    build_operator_time_stats,
    classify_op_groups,
)

if TYPE_CHECKING:
    from zrt.training.ir.training_graph import Graph, Op
    from zrt.training.models.flops import OpCost


def _iter_ops(graph):
    """Iterate compute ops from either Graph or OpGraph."""
    if hasattr(graph, "ops"):
        return graph.ops
    if hasattr(graph, "nodes"):
        from zrt.training.models.flops import _OpNodeAsOp
        return [_OpNodeAsOp(n) for n in graph.nodes.values() if not n.is_comm]
    return []


def _ops_for_layer(graph, layer_id: int):
    """Get ops for a specific layer from either Graph or OpGraph."""
    if hasattr(graph, "ops_for_layer"):
        return graph.ops_for_layer(layer_id)
    if hasattr(graph, "nodes"):
        from zrt.training.models.flops import _OpNodeAsOp
        lid_str = str(layer_id)
        return [_OpNodeAsOp(n) for n in graph.nodes.values()
                if not n.is_comm and n.layer == lid_str]
    return []
    from zrt.training.spec.model import ModelSpec
    from zrt.training.spec.report import TrainingReport
    from zrt.training.spec.strategy import Strategy
    from zrt.training.spec.system import SystemSpec


def _fmt_shape(shape: tuple[int, ...]) -> str:
    """Format shape tuple as string."""
    if not shape:
        return "()"
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _fmt_tensor(t) -> str:
    """Format a Tensor as 'shape dtype'."""
    dtype = getattr(getattr(t, "dtype", None), "value", getattr(t, "dtype", ""))
    return f"{_fmt_shape(t.shape_local)} {dtype}"


def _tensor_list_info(tensors: list, label: str = "") -> str:
    """Format input/output tensor list."""
    if not tensors:
        return "-"

    parts = []
    for t in tensors:
        flag = "W" if getattr(t, "is_param", False) else "A"
        name = getattr(t, "name", "")
        parts.append(f"[{flag}] {name}: {_fmt_tensor(t)}")
    return " | ".join(parts)


def _bpe_from_op(op) -> int:
    """Bytes per element from op tensors."""
    if getattr(op, "inputs", None):
        return op.inputs[0].dtype.bytes
    if getattr(op, "outputs", None):
        return op.outputs[0].dtype.bytes
    return 2


def _fmt_e(v: float) -> str:
    """Format a number in compact scientific-like units."""
    try:
        v = float(v)
    except Exception:
        return "-"

    if v <= 0:
        return "-"
    if v >= 1e15:
        return f"{v / 1e15:.2f}P"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}G"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.2f}K"
    return f"{v:.0f}"


def _fwd_flops(cost) -> float:
    return cost.fwd_cube_flops + cost.fwd_vector_flops


def _dx_flops(cost) -> float:
    return cost.dx_cube_flops + cost.dx_vector_flops


def _dw_flops(cost) -> float:
    return cost.dw_cube_flops + cost.dw_vector_flops


def _matmul_display_dims(op) -> tuple[int, int, int]:
    """Return the dimensions used by the matmul cost path.

    Formula display must follow ``models.flops._matmul_cost`` exactly:
    most matmuls use local tensor shapes after CP/TP sharding, while fused
    grouped matmuls keep authoritative dimensions in metadata.
    """
    meta = getattr(op, "meta", {}) or {}
    meta_k = meta.get("k", 0)
    inputs = getattr(op, "inputs", None) or []
    outputs = getattr(op, "outputs", None) or []
    use_meta = (
        meta.get("fused_weight_dims", False)
        or not inputs
        or not outputs
        or (meta_k > 0 and inputs[0].shape_logical[-1] != meta_k)
    )

    if use_meta:
        return (
            meta.get("m", 0),
            meta.get("n_local", meta.get("n", 0)),
            meta.get("k_local", meta.get("k", 0)),
        )

    return (
        inputs[0].shape_local[0],
        outputs[0].shape_local[-1],
        inputs[0].shape_local[-1],
    )


def _matmul_bytes_formula(op, cost, model, mm: int, nn: int, kk: int) -> tuple[str, str]:
    if model is None:
        bpe = _bpe_from_op(op)
        fwd_str = (
            f"(m×k+k×n+m×n)×bpe = "
            f"({mm}×{kk}+{kk}×{nn}+{mm}×{nn})×{bpe} = {_fmt_e(cost.fwd_bytes)}"
        )
        bwd_str = (
            f"dx+dw = 2×(m×k+k×n+m×n)×bpe = "
            f"2×({mm}×{kk}+{kk}×{nn}+{mm}×{nn})×{bpe} = "
            f"{_fmt_e(cost.dx_bytes + cost.dw_bytes)}"
        )
        return fwd_str, bwd_str

    from zrt.training.models.quant import resolve_op_dtypes

    d = resolve_op_dtypes(op, model)
    A_b = d.in_act.bytes
    W_b = d.weight.stored_bytes
    C_b = d.out_act.bytes
    dC_b = d.grad_in.bytes
    dA_b = d.grad_act.bytes
    dW_b = d.grad_weight.stored_bytes
    fwd_str = (
        f"m×k×A_b + k×n×W_b + m×n×C_b "
        f"(A_b={A_b}, W_b={W_b}, C_b={C_b}) = "
        f"{mm}×{kk}×{A_b} + {kk}×{nn}×{W_b} + {mm}×{nn}×{C_b} "
        f"= {_fmt_e(cost.fwd_bytes)}"
    )
    bwd_str = (
        f"dx+dw = "
        f"(m×n×dC_b + k×n×W_b + m×k×dA_b) + "
        f"(m×n×dC_b + m×k×A_b + k×n×dW_b) "
        f"(dC_b={dC_b}, W_b={W_b}, dA_b={dA_b}, A_b={A_b}, dW_b={dW_b}) = "
        f"{_fmt_e(cost.dx_bytes + cost.dw_bytes)}"
    )
    return fwd_str, bwd_str


def _attention_bytes_formula(op, cost, model, system) -> tuple[str, str]:
    m = getattr(op, "meta", {}) or {}
    b = m.get("b", 1)
    s = m.get("s", 0)
    h = m.get("heads", 0)
    d = m.get("head_dim", 0)
    topk = m.get("sparse_topk", 0)
    cr = m.get("compress_ratio", 0)
    swa = m.get("swa_window", 0)
    causal = m.get("causal", True)

    if topk > 0:
        kv_len = topk + swa
    elif cr > 0:
        kv_len = max(1, s // cr) + swa
    elif swa > 0:
        kv_len = swa
    else:
        kv_len = s

    bpe = _bpe_from_op(op)
    qkv_b = o_b = grad_b = bpe
    if model is not None:
        from zrt.training.models.quant import resolve_op_dtypes

        bundle = resolve_op_dtypes(op, model)
        qkv_b = bundle.in_act.bytes
        o_b = bundle.out_act.bytes
        grad_b = bundle.grad_in.bytes

    sram_bytes = 0
    if system is not None and getattr(getattr(system, "gpu", None), "sram_kb_per_sm", 0) > 0:
        sram_bytes = int(system.gpu.sram_kb_per_sm * 1024)

    if sram_bytes <= 0:
        fwd_str = (
            f"(Q + 2×KV)×qkv_b + O×o_b "
            f"(qkv_b={qkv_b}, o_b={o_b}) = {_fmt_e(cost.fwd_bytes)}"
        )
        bwd_str = (
            f"(2×Q + 4×KV)×grad_b "
            f"(grad_b={grad_b}) = {_fmt_e(cost.dx_bytes)}"
        )
        return fwd_str, bwd_str

    from zrt.training.models.flops import _fa_tile_shape

    Br, Bc = _fa_tile_shape(d, bpe, sram_bytes)
    Tr = math.ceil(s / Br)
    Tc = math.ceil(kv_len / Bc)
    tc_eff = Tc if topk > 0 else ((Tc + 1) / 2 if causal else Tc)
    q_elems = b * h * s * d
    o_elems = b * h * s * d
    kv_elems = b * h * Tr * tc_eff * Bc * d
    fwd_str = (
        f"(Q + 2×KV_tiles)×qkv_b + O×o_b "
        f"(Br={Br}, Bc={Bc}, Tr={Tr}, Tc_eff={tc_eff}, "
        f"qkv_b={qkv_b}, o_b={o_b}) = "
        f"({q_elems} + 2×{kv_elems})×{qkv_b} + {o_elems}×{o_b} "
        f"= {_fmt_e(cost.fwd_bytes)}"
    )
    bwd_str = (
        f"(2×Q + 4×KV_tiles)×grad_b "
        f"(Br={Br}, Bc={Bc}, Tr={Tr}, Tc_eff={tc_eff}, grad_b={grad_b}) = "
        f"(2×{q_elems} + 4×{kv_elems})×{grad_b} = {_fmt_e(cost.dx_bytes)}"
    )
    return fwd_str, bwd_str


def _op_formula(op, cost, model=None, system=None):
    """Return formula strings for one op.

    Returns:
        tuple:
            fwd_formula,
            bwd_formula,
            fwd_bytes_formula,
            bwd_bytes_formula
    """
    m = getattr(op, "meta", {}) or {}

    ff = _fwd_flops(cost)
    df = _dx_flops(cost)
    wf = _dw_flops(cost)

    if op.kind == "matmul" or op.kind == "lm_head":
        mm, nn, kk = _matmul_display_dims(op)
        mult = m.get("fwd_multiplier", 1.0)
        fwd_val = ff

        if mult != 1.0:
            fwd_str = (
                f"2×m×n×k×{mult} = "
                f"2×{mm}×{nn}×{kk}×{mult} = {_fmt_e(fwd_val)}"
            )
        else:
            fwd_str = (
                f"2×m×n×k = "
                f"2×{mm}×{nn}×{kk} = {_fmt_e(fwd_val)}"
            )

        bwd_str = f"dx+dw = 2×fwd = {_fmt_e(df + wf)}"
        bytes_str, bwd_bytes_str = _matmul_bytes_formula(op, cost, model, mm, nn, kk)
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
        b = m.get("b", 1)
        s = m.get("s", 0)
        h = m.get("heads", 0)
        d = m.get("head_dim", 0)
        topk = m.get("sparse_topk", 0)
        cr = m.get("compress_ratio", 0)
        swa = m.get("swa_window", 0)

        if topk > 0:
            eff = topk + swa
            fwd_str = (
                f"2×b×s×(topk+swa)×h×d = "
                f"2×{b}×{s}×{eff}×{h}×{d} = {_fmt_e(ff)}"
            )
        elif cr > 0:
            c_len = max(1, s // cr)
            eff = c_len + swa
            fwd_str = (
                f"2×b×s×(s/r+swa)×h×d = "
                f"2×{b}×{s}×{eff}×{h}×{d} = {_fmt_e(ff)}"
            )
        elif swa > 0:
            fwd_str = (
                f"2×b×s×swa×h×d = "
                f"2×{b}×{s}×{swa}×{h}×{d} = {_fmt_e(ff)}"
            )
        else:
            fwd_str = (
                f"2×b×s²×h×d = "
                f"2×{b}×{s}²×{h}×{d} = {_fmt_e(ff)}"
            )

        bwd_str = f"2.5×fwd, including attention backward/recompute approximation = {_fmt_e(df)}"

        kv_len = (
            (topk + swa)
            if topk > 0
            else (max(1, s // cr) + swa if cr > 0 else (swa if swa > 0 else s))
        )

        bytes_str, bwd_bytes_str = _attention_bytes_formula(op, cost, model, system)
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind in ("rmsnorm", "ln"):
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fpe = 5 if op.kind == "rmsnorm" else 7
        fwd_str = f"{fpe}×N = {fpe}×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"fwd_bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "rope":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"2×N, sin/cos mul = 2×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "swiglu":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"5×N, sigmoid + mul = 5×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "add":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"1×N, add = {n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "softmax":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"4×N, max/sub/exp/div = 4×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "indexer_topk":
        s = m.get("s", 0)
        kv = m.get("kv_len", s)
        ih = m.get("ih_local", m.get("ih", 0))
        id_ = m.get("id", 0)
        fwd_str = (
            f"2×s×kv_len×ih×id = "
            f"2×{s}×{kv}×{ih}×{id_} = {_fmt_e(ff)}"
        )
        bwd_str = f"2×fwd, indexer query gradient approximation = {_fmt_e(df)}"
        bytes_str = f"fwd_bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = "dx_bytes = - or folded into matmul backward"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "compressor_pool":
        s = m.get("s", 0)
        mm_ = m.get("m", 4)
        co = m.get("coff", 1)
        dd = m.get("d_local", m.get("d", 0))
        fwd_str = (
            f"4×(s/m)×coff×m×d = "
            f"4×({s}//{mm_})×{co}×{mm_}×{dd} = {_fmt_e(ff)}"
        )
        bwd_str = f"bwd = fwd = {_fmt_e(df)}"
        bytes_str = f"bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "embed":
        s = m.get("m", 0)
        h = m.get("n", 0)
        fwd_str = "0, gather-style embedding lookup has no modeled FLOPs"
        bwd_str = "0, scatter-style embedding gradient has no modeled FLOPs"
        bytes_str = f"s×h×bpe = {s}×{h}×bpe = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx+dw bytes = {_fmt_e(cost.dx_bytes + cost.dw_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        fwd_str = f"mHC fused op fwd = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df + wf)}"
        bytes_str = f"fwd_bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx+dw bytes = {_fmt_e(cost.dx_bytes + cost.dw_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "hash_route":
        return "negligible", "negligible", "negligible", "negligible"

    if op.kind == "mega_moe":
        from zrt.training.models.mega_moe import mega_moe_cost_terms

        terms = mega_moe_cost_terms(op)
        mm = m.get("m", 0)
        micro_batch = m.get("micro_batch", 1)
        tokens = terms.tokens
        nn = terms.n
        kk = terms.k_eff
        top_k = terms.top_k
        mult = f"{terms.fwd_multiplier:g}"
        quant = terms.quant_variant
        waves = m.get("requested_waves", 0)
        local_experts = terms.local_experts
        fwd_str = (
            "Mega MoE dispatch+FFN+combine: "
            f"m={mm}, micro_batch={micro_batch}, tokens={tokens}, "
            f"top_k={top_k}, k={kk}, n={nn}, experts/rank={local_experts}, "
            f"mult={mult}, quant={quant}, waves={waves}; "
            f"2*tokens*top_k*k*n*mult = 2*{tokens}*{top_k}*{kk}*{nn}*{mult} = {_fmt_e(ff)}"
        )
        bwd_str = f"dx+dw = 2*fwd = {_fmt_e(df + wf)}"
        bytes_str = (
            "act_in+act_out+stored_weights "
            f"(quant={quant}) = {_fmt_e(cost.fwd_bytes)}"
        )
        bwd_bytes_str = f"same fused traffic per dx/dw = {_fmt_e(cost.dx_bytes + cost.dw_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    return (
        f"fwd = {_fmt_e(ff)}",
        f"bwd = {_fmt_e(df + wf)}",
        f"bytes = {_fmt_e(cost.fwd_bytes)}",
        f"dx+dw bytes = {_fmt_e(cost.dx_bytes + cost.dw_bytes)}",
    )


def _op_detail(op, cost, model=None, system=None):
    """Return a dict with full op info for HTML."""
    inputs_info = _tensor_list_info(op.inputs, "input")
    outputs_info = _tensor_list_info(op.outputs, "output")
    fwd_formula, bwd_formula, fwd_bytes_formula, bwd_bytes_formula = _op_formula(op, cost, model, system)

    return {
        "inputs": inputs_info,
        "outputs": outputs_info,
        "fwd_formula": fwd_formula,
        "bwd_formula": bwd_formula,
        "fwd_bytes_formula": fwd_bytes_formula,
        "bwd_bytes_formula": bwd_bytes_formula,
    }


def _classify_ops_in_layer(ops: list["Op"], layer_kind: str) -> list[dict]:
    """Split a layer's ops into logical blocks.

    This function is intentionally heuristic. It groups common DeepSeek-like
    layer op names into user-facing blocks for the expandable report.
    """
    layer_kind = str(layer_kind).lower()

    blocks = {
        "HyperConnection": [],
        "Attention + Indexer": [],
        "Router + Shared Expert": [],
        "Routed Expert": [],
        "MTP Embed": [],
        "FFN": [],
        "Other Ops": [],
    }

    for op in ops:
        name = op.name.lower()
        kind = op.kind.lower()
        comp = str(getattr(op, "component", "") or "").lower()

        if kind in ("mhc_pre", "mhc_post", "mhc_head") or "mhc" in name or "hc_" in name:
            blocks["HyperConnection"].append(op)
            continue

        if layer_kind == "mtp" and ("mtp" in name or "embed_proj" in name):
            blocks["MTP Embed"].append(op)
            continue

        if (
            "attn" in name
            or "attention" in comp
            or "ln1" in name
            or "wq" in name
            or "wkv" in name
            or "q_norm" in name
            or "kv_norm" in name
            or "rope" in name
            or "compress" in name
            or "indexer" in name
            or "idx_" in name
            or "wo_" in name
            or kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn")
        ):
            blocks["Attention + Indexer"].append(op)
            continue

        if (
            "router" in name
            or "topk" in name
            or "hash_route" in name
            or "shared_" in name
            or "shared" in comp
        ):
            blocks["Router + Shared Expert"].append(op)
            continue

        if (
            "routed" in name
            or "expert_agg" in name
            or "expert" in comp
        ):
            blocks["Routed Expert"].append(op)
            continue

        if (
            "ln2" in name
            or "ffn" in name
            or "up_proj" in name
            or "down_proj" in name
            or "gate_proj" in name
            or "swiglu" in name
        ):
            blocks["FFN"].append(op)
            continue

        blocks["Other Ops"].append(op)

    ordered_names = [
        "MTP Embed",
        "HyperConnection",
        "Attention + Indexer",
        "Router + Shared Expert",
        "Routed Expert",
        "FFN",
        "Other Ops",
    ]

    return [
        {"name": name, "ops": blocks[name]}
        for name in ordered_names
        if blocks[name]
    ]


def _op_to_dict(op: "Op", cost: "OpCost", system, model=None) -> dict:
    """Convert an op + cost to a dict for the HTML template."""
    from zrt.training.compose.stage import (
        _cost_phase_time,
        _resolve_compute_dtype,
        has_heterogeneous_compute,
    )

    gpu_name = system.gpu.name
    overlap = system.gpu.overlap_ratio.get(op.kind, 0.0) if has_heterogeneous_compute(system) else 0.0
    op_dtype = _resolve_compute_dtype(op, model) if model is not None else Dtype.BF16

    fwd_t = _cost_phase_time(cost, "fwd", system, gpu_name, overlap, op_dtype)
    dx_t = _cost_phase_time(cost, "dx", system, gpu_name, overlap, op_dtype)
    dw_t = _cost_phase_time(cost, "dw", system, gpu_name, overlap, op_dtype)

    detail = _op_detail(op, cost, model, system)

    component = str(getattr(op, "component", "") or "")

    return {
        "name": op.name,
        "kind": op.kind,
        "component": component,
        "layer_id": getattr(op, "layer_id", None),
        "layer_kind": str(getattr(op, "layer_kind", "") or ""),
        "fwd_flops": _fwd_flops(cost),
        "dx_flops": _dx_flops(cost),
        "dw_flops": _dw_flops(cost),
        "fwd_bytes": cost.fwd_bytes,
        "dx_bytes": cost.dx_bytes,
        "dw_bytes": cost.dw_bytes,
        "bound": cost.bound,
        "fwd_ms": fwd_t * 1000,
        "dx_ms": dx_t * 1000,
        "dw_ms": dw_t * 1000,
        "total_ms": (fwd_t + dx_t + dw_t) * 1000,
        "cube_flops": cost.fwd_cube_flops + cost.dx_cube_flops + cost.dw_cube_flops,
        "vector_flops": (
            cost.fwd_vector_flops + cost.dx_vector_flops + cost.dw_vector_flops
        ),
        "inputs": detail["inputs"],
        "outputs": detail["outputs"],
        "fwd_formula": detail["fwd_formula"],
        "bwd_formula": detail["bwd_formula"],
        "fwd_bytes_formula": detail["fwd_bytes_formula"],
        "bwd_bytes_formula": detail["bwd_bytes_formula"],
        "meta": _json_safe_meta(getattr(op, "meta", {}) or {}),
    }


def _json_safe_meta(meta: dict) -> dict:
    def convert(value):
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return {k: convert(v) for k, v in meta.items()}


def _enum_value(x) -> str:
    """Return enum.value when present, otherwise str(x)."""
    if x is None:
        return ""
    return str(getattr(x, "value", x))


def _safe_get(obj, name: str, default=None):
    """getattr with callable-property tolerance."""
    v = getattr(obj, name, default)
    try:
        if callable(v):
            return v()
    except TypeError:
        return v
    return v


def _component_of_op(op) -> str:
    """Map IR op to high-level model block for model-level summary."""
    name = op.name.lower()
    kind = op.kind.lower()
    comp = str(getattr(op, "component", "") or "").lower()

    if "input" in name:
        return "input"

    if "embed" in name:
        return "embedding"

    if (
        "attn" in name
        or "attention" in comp
        or "wq" in name
        or "wkv" in name
        or "q_norm" in name
        or "kv_norm" in name
        or "rope" in name
        or "compress" in name
        or "indexer" in name
        or "idx_" in name
        or "wo_" in name
        or kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn")
    ):
        return "attention"

    if (
        "moe" in comp
        or "expert" in name
        or "router" in name
        or "topk" in name
        or "swiglu" in name
        or "shared_" in name
        or "routed_" in name
        or "hash_route" in name
    ):
        return "moe"

    if "lm_head" in name or "final" in name or "logit" in name or "output" in name:
        return "output"

    return "other"


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _build_summary(
    report: "TrainingReport",
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    op_dicts: list[dict],
) -> dict:
    gpu = getattr(system, "gpu", None)

    nodes = _safe_get(system, "nodes", 1) or 1
    gpus_per_node = _safe_get(system, "gpus_per_node", 1) or 1
    world_size = _safe_get(system, "world_size", None)
    if world_size is None:
        world_size = int(nodes) * int(gpus_per_node)

    layer_kinds = [_enum_value(x).lower() for x in getattr(model, "layers", [])]
    layer_count = len(layer_kinds)

    memory = {}
    if getattr(report, "memory", None) is not None and hasattr(report.memory, "to_gb"):
        memory = report.memory.to_gb()
    elif getattr(report, "memory_breakdown", None):
        memory = dict(report.memory_breakdown)

    compute_bound_ms = 0.0
    memory_bound_ms = 0.0

    for od in op_dicts:
        bound = str(od.get("bound", "")).lower()
        total_ms = _to_float(od.get("total_ms", 0.0))
        if "memory" in bound or "hbm" in bound:
            memory_bound_ms += total_ms
        elif "comm" in bound:
            pass
        else:
            compute_bound_ms += total_ms

    comm_ms = (
        _to_float(getattr(report, "exposed_comm_ms", 0.0))
        + _to_float(getattr(report, "optimizer_comm_ms", 0.0))
    )
    bound_total = max(compute_bound_ms + memory_bound_ms + comm_ms, 1e-12)

    return {
        "model": {
            "name": getattr(model, "model_type", "DeepSeek-like"),
            "hidden": getattr(model, "hidden", None),
            "layers": layer_count,
            "moe_layers": layer_kinds.count("moe"),
            "mtp_layers": layer_kinds.count("mtp"),
            "dense_layers": layer_kinds.count("dense"),
            "seq_len": getattr(model, "seq_len", None),
            "num_heads": getattr(model, "num_heads", None),
            "head_dim": getattr(model, "head_dim", None),
            "num_experts": getattr(model, "num_experts", None),
            "top_k": getattr(model, "top_k", None),
            "moe_ffn": getattr(model, "moe_ffn", None),
            "hc_mult": getattr(model, "hc_mult", None),
            "mtp_depth": getattr(model, "mtp_depth", None),
            "param_dtype": _enum_value(getattr(model, "param_dtype", "")),
            "act_dtype": _enum_value(getattr(model, "act_dtype", "")),
            "total_params": getattr(report, "total_params", 0),
            "effective_params": getattr(report, "effective_params", 0),
        },
        "hardware": {
            "name": getattr(gpu, "name", "unknown"),
            "hbm_gb": getattr(gpu, "hbm_gb", None),
            "hbm_bw_gbps": getattr(gpu, "hbm_bw_gbps", None),
            "tflops_bf16": getattr(gpu, "tflops_bf16", None),
            "tflops_fp16": getattr(gpu, "tflops_fp16", None),
            "tops_fp8": getattr(gpu, "tops_fp8", None),
            "cube_tflops": getattr(gpu, "cube_tflops", None),
            "vector_tflops": getattr(gpu, "vector_tflops", None),
        },
        "topology": {
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "world_size": world_size,
            "host_mem_gb": getattr(system, "host_mem_gb", None),
        },
        "strategy": {
            "tp": getattr(strategy, "tp", 1),
            "cp": getattr(strategy, "cp", 1),
            "pp": getattr(strategy, "pp", 1),
            "ep": getattr(strategy, "ep", 1),
            "dp": getattr(strategy, "dp", 1),
            "micro_batch": getattr(strategy, "micro_batch", None),
            "global_batch": getattr(strategy, "global_batch", None),
            "num_microbatches": (
                strategy.num_microbatches()
                if hasattr(strategy, "num_microbatches")
                else None
            ),
            "pp_schedule": _enum_value(getattr(strategy, "pp_schedule", "")),
            "zero_stage": getattr(strategy, "zero_stage", None),
            "optimizer": _enum_value(getattr(strategy, "optimizer", "")),
        },
        "performance": {
            "step_time_ms": getattr(report, "step_time_ms", 0.0),
            "pipeline_time_ms": getattr(report, "pipeline_time_ms", 0.0),
            "compute_time_ms": getattr(report, "compute_time_ms", 0.0),
            "exposed_comm_ms": getattr(report, "exposed_comm_ms", 0.0),
            "hidden_comm_ms": getattr(report, "hidden_comm_ms", 0.0),
            "recompute_critical_ms": getattr(report, "recompute_critical_ms", 0.0),
            "recompute_raw_mag_ms": getattr(report, "recompute_raw_mag_ms", 0.0),
            "optimizer_time_ms": getattr(report, "optimizer_time_ms", 0.0),
            "optimizer_comm_ms": getattr(report, "optimizer_comm_ms", 0.0),
            "tokens_per_sec": getattr(report, "tokens_per_sec", 0.0),
            "mfu": getattr(report, "mfu", 0.0),
            "hfu": getattr(report, "hfu", 0.0),
            "mfu_native": getattr(report, "mfu_native", 0.0),
            "bubble_fraction": getattr(report, "bubble_fraction", 0.0),
            "bubble_time_ms": getattr(report, "bubble_time_ms", 0.0),
            "flops_per_token": getattr(report, "flops_per_token", 0.0),
            "total_flops": getattr(report, "total_flops", 0.0),
            "recompute_critical_ms": getattr(report, "recompute_critical_ms", 0.0),
            "recompute_raw_mag_ms": getattr(report, "recompute_raw_mag_ms", 0.0),
            "bubble_time_ms": getattr(report, "bubble_time_ms", 0.0),
        },
        "memory": memory,
        "bound": {
            "compute_ms": compute_bound_ms,
            "memory_ms": memory_bound_ms,
            "communication_ms": comm_ms,
            "compute_pct": compute_bound_ms / bound_total,
            "memory_pct": memory_bound_ms / bound_total,
            "communication_pct": comm_ms / bound_total,
        },
        "operator_time_share": build_operator_time_stats(
            model=model,
            report=report,
            op_dicts=op_dicts,
            strategy=strategy,
        ),
        "comm_breakdown": {
            "tp_exposed_ms": getattr(report, "tp_exposed_ms", 0.0),
            "cp_exposed_ms": getattr(report, "cp_exposed_ms", 0.0),
            "ep_exposed_ms": getattr(report, "ep_exposed_ms", 0.0),
            "pp_exposed_ms": getattr(report, "pp_exposed_ms", 0.0),
            "dp_exposed_ms": getattr(report, "dp_exposed_ms", 0.0),
            "tp_hidden_ms": getattr(report, "tp_hidden_ms", 0.0),
            "ep_hidden_ms": getattr(report, "ep_hidden_ms", 0.0),
            "pp_hidden_ms": getattr(report, "pp_hidden_ms", 0.0),
            "dp_hidden_ms": getattr(report, "dp_hidden_ms", 0.0),
            "total_comm_volume_ms": getattr(report, "total_comm_volume_ms", 0.0),
        },
    }


def _get_rank_placement_from_config(system, strategy):
    """Best-effort optional hook for real runtime rank placement.

    Supported shapes:
    1. strategy.rank_placement / system.rank_placement as list[dict]:
       [
         {"rank": 0, "node": 0, "local_rank": 0,
          "tp": 0, "pp": 0, "dp": 0, "ep": 0, "cp": 0},
         ...
       ]

    2. dict with key "ranks":
       {"ranks": [ ... ]}

    Existing configs continue to work. If no explicit placement is found, the
    report will fall back to an inferred visualization-only layout.
    """
    candidates = [
        getattr(strategy, "rank_placement", None),
        getattr(strategy, "rank_map", None),
        getattr(system, "rank_placement", None),
        getattr(system, "rank_map", None),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        if isinstance(candidate, dict) and "ranks" in candidate:
            candidate = candidate["ranks"]

        if isinstance(candidate, list):
            normalized = []
            for item in candidate:
                if not isinstance(item, dict):
                    continue
                if "rank" not in item:
                    continue

                rank = _to_int(item.get("rank", 0))
                normalized.append(
                    {
                        "rank": rank,
                        "node": _to_int(item.get("node", 0)),
                        "local_rank": _to_int(item.get("local_rank", rank)),
                        "tp": _to_int(item.get("tp", 0)),
                        "pp": _to_int(item.get("pp", 0)),
                        "dp": _to_int(item.get("dp", 0)),
                        "ep": _to_int(item.get("ep", 0)),
                        "cp": _to_int(item.get("cp", 0)),
                    }
                )

            if normalized:
                return normalized

    return None


def _build_topology_data(system, strategy) -> dict:
    nodes = int(_safe_get(system, "nodes", 1) or 1)
    gpus_per_node = int(_safe_get(system, "gpus_per_node", 1) or 1)

    world_size = _safe_get(system, "world_size", None)
    if world_size is None:
        world_size = nodes * gpus_per_node
    world_size = int(world_size)

    tp = max(1, int(getattr(strategy, "tp", 1) or 1))
    pp = max(1, int(getattr(strategy, "pp", 1) or 1))
    ep = max(1, int(getattr(strategy, "ep", 1) or 1))
    dp = max(1, int(getattr(strategy, "dp", 1) or 1))
    cp = max(1, int(getattr(strategy, "cp", 1) or 1))

    real_placement = _get_rank_placement_from_config(system, strategy)

    if real_placement:
        by_node: dict[int, list[dict]] = {}
        for r in real_placement:
            by_node.setdefault(int(r["node"]), []).append(r)

        rank_nodes = []
        for n in sorted(by_node):
            ranks = sorted(by_node[n], key=lambda x: x["local_rank"])
            rank_nodes.append({"node": n, "ranks": ranks})

        return {
            "nodes": rank_nodes,
            "legend": {
                "mapping_source": "configured_runtime_rank_placement",
                "is_inferred": False,
                "note": (
                    "Rank placement is loaded from configured runtime rank placement. "
                    "It should match the launcher or distributed runtime mapping."
                ),
                "tp": tp,
                "pp": pp,
                "ep": ep,
                "dp": dp,
                "cp": cp,
            },
        }

    rank_nodes = []
    for n in range(nodes):
        ranks = []
        for local in range(gpus_per_node):
            r = n * gpus_per_node + local
            if r >= world_size:
                continue

            # Visualization-only heuristic.
            # Do not treat this as authoritative runtime rank placement.
            tp_idx = r % tp
            pp_idx = (r // tp) % pp
            dp_idx = (r // max(1, tp * pp)) % dp
            ep_idx = r % ep
            cp_idx = r % cp

            ranks.append(
                {
                    "rank": r,
                    "local_rank": local,
                    "node": n,
                    "tp": tp_idx,
                    "pp": pp_idx,
                    "dp": dp_idx,
                    "ep": ep_idx,
                    "cp": cp_idx,
                }
            )

        rank_nodes.append({"node": n, "ranks": ranks})

    return {
        "nodes": rank_nodes,
        "legend": {
            "mapping_source": "inferred_visualization_only",
            "is_inferred": True,
            "note": (
                "This topology is an inferred logical visualization only. "
                "It may not match the real runtime rank placement used by torchrun, "
                "MindSpore, HCCL, Megatron, or a custom launcher. For accurate topology "
                "validation, provide explicit rank_placement data."
            ),
            "tp": tp,
            "pp": pp,
            "ep": ep,
            "dp": dp,
            "cp": cp,
        },
    }


def _build_model_overview(op_dicts: list[dict]) -> list[dict]:
    buckets = {
        "input": {
            "name": "Input",
            "desc": "tokens / batch / sequence",
            "total_ms": 0.0,
            "ops": 0,
        },
        "embedding": {
            "name": "Embedding",
            "desc": "token embedding / hc_expand",
            "total_ms": 0.0,
            "ops": 0,
        },
        "attention": {
            "name": "Attention",
            "desc": "Q/KV, RoPE, CSA/HCA/SWA, output projection",
            "total_ms": 0.0,
            "ops": 0,
        },
        "moe": {
            "name": "MoE",
            "desc": "router, shared expert, routed experts",
            "total_ms": 0.0,
            "ops": 0,
        },
        "output": {
            "name": "Output",
            "desc": "final norm / lm_head",
            "total_ms": 0.0,
            "ops": 0,
        },
        "other": {
            "name": "Other",
            "desc": "residual / misc ops",
            "total_ms": 0.0,
            "ops": 0,
        },
    }

    for od in op_dicts:
        comp = od.get("component_group", "other")
        if comp not in buckets:
            comp = "other"
        buckets[comp]["total_ms"] += _to_float(od.get("total_ms", 0.0))
        buckets[comp]["ops"] += 1

    ordered = [
        buckets[k]
        for k in ("input", "embedding", "attention", "moe", "output", "other")
    ]

    total = max(sum(x["total_ms"] for x in ordered), 1e-12)
    for x in ordered:
        x["pct"] = x["total_ms"] / total

    return ordered


def _build_layer_tree(
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    op_costs: dict[str, "OpCost"],
) -> tuple[dict, list[dict]]:
    """Build hierarchical tree: model -> layer -> block -> op."""
    layers = getattr(model, "layers", [])

    all_ops = _iter_ops(graph)
    tree = {
        "model_name": f"{len(layers)} layers, hidden={getattr(model, 'hidden', '-')}",
        "total_ops": len(all_ops),
        "layers": [],
        "global_ops": [],
    }

    all_op_dicts: list[dict] = []

    from zrt.training.models.flops import op_cost

    # Global ops: embedding / final head etc.
    for op in all_ops:
        if getattr(op, "layer_id", -1) < 0:
            cost = op_costs.get(op.name)
            if cost is None:
                cost = op_cost(op, model, system)

            od = _op_to_dict(op, cost, system, model)
            od["component_group"] = _component_of_op(op)
            od["op_groups"] = classify_op_groups(od)

            tree["global_ops"].append(od)
            all_op_dicts.append(od)

    # Per-layer data.
    for lid in range(len(layers)):
        lk = _enum_value(layers[lid]).lower()
        layer_ops = _ops_for_layer(graph, lid)
        blocks = _classify_ops_in_layer(layer_ops, lk)

        layer_data = {
            "id": lid,
            "kind": lk,
            "op_count": len(layer_ops),
            "total_ms": 0.0,
            "blocks": [],
        }

        captured = set()

        for blk in blocks:
            blk_data = {
                "name": blk["name"],
                "ops": [],
                "total_ms": 0.0,
                "bound": {
                    "compute_ms": 0.0,
                    "memory_ms": 0.0,
                    "other_ms": 0.0,
                },
            }

            for op in blk["ops"]:
                captured.add(op.name)

                cost = op_costs.get(op.name)
                if cost is None:
                    cost = op_cost(op, model, system)

                od = _op_to_dict(op, cost, system, model)
                od["component_group"] = _component_of_op(op)
                od["op_groups"] = classify_op_groups(od)

                blk_data["ops"].append(od)
                all_op_dicts.append(od)

                ms = _to_float(od.get("total_ms", 0.0))
                blk_data["total_ms"] += ms

                b = str(od.get("bound", "")).lower()
                if "memory" in b or "hbm" in b:
                    blk_data["bound"]["memory_ms"] += ms
                elif "compute" in b or "cube" in b or "vector" in b:
                    blk_data["bound"]["compute_ms"] += ms
                else:
                    blk_data["bound"]["other_ms"] += ms

            layer_data["blocks"].append(blk_data)
            layer_data["total_ms"] += blk_data["total_ms"]

        missed = [op for op in layer_ops if op.name not in captured]
        if missed:
            blk_data = {
                "name": "Other Ops",
                "ops": [],
                "total_ms": 0.0,
                "bound": {
                    "compute_ms": 0.0,
                    "memory_ms": 0.0,
                    "other_ms": 0.0,
                },
            }

            for op in missed:
                cost = op_costs.get(op.name)
                if cost is None:
                    cost = op_cost(op, model, system)

                od = _op_to_dict(op, cost, system, model)
                od["component_group"] = _component_of_op(op)
                od["op_groups"] = classify_op_groups(od)

                blk_data["ops"].append(od)
                all_op_dicts.append(od)

                ms = _to_float(od.get("total_ms", 0.0))
                blk_data["total_ms"] += ms

                b = str(od.get("bound", "")).lower()
                if "memory" in b or "hbm" in b:
                    blk_data["bound"]["memory_ms"] += ms
                elif "compute" in b or "cube" in b or "vector" in b:
                    blk_data["bound"]["compute_ms"] += ms
                else:
                    blk_data["bound"]["other_ms"] += ms

            layer_data["blocks"].append(blk_data)
            layer_data["total_ms"] += blk_data["total_ms"]

        tree["layers"].append(layer_data)

    return tree, all_op_dicts


def _build_calibration_data(
    report: "TrainingReport",
    graph: "Graph",
    model: "ModelSpec",
    strategy: "Strategy",
    all_op_dicts: list[dict],
) -> list[dict]:
    """Build reference calibration rows.

    Public references are used as sanity checks, not as automatic correction
    unless the current scenario is known to be comparable.
    """
    rows = []

    hidden = getattr(model, "hidden", None)
    num_experts = getattr(model, "num_experts", None)
    top_k = getattr(model, "top_k", None)
    moe_ffn = getattr(model, "moe_ffn", None)

    is_dsv4_pro_like = (
        hidden == 7168
        and num_experts == 384
        and top_k == 6
        and moe_ffn == 3072
    )

    rows.append(
        {
            "name": "DeepSeek-V4-Pro 架构",
            "official": (
                "DeepSeek-V4-Pro 公开模型卡 / 技术资料：总参数 1.6T，"
                "激活参数 49B，上下文 1M，FP4+FP8 混合精度。"
            ),
            "modeled": (
                f"hidden={hidden}, experts={num_experts}, top_k={top_k}, "
                f"moe_ffn={moe_ffn}, layers={len(getattr(model, 'layers', []))}"
            ),
            "status": "pass" if is_dsv4_pro_like else "check",
            "delta": "结构校验",
            "source": "DeepSeek-V4-Pro 模型卡 / 技术报告",
            "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro",
            "note": (
                "用于确认 YAML / 模型配置与公开的 DeepSeek-V4-Pro 几何结构一致。"
            ),
        }
    )

    routed_ops = [
        od
        for od in all_op_dicts
        if "routed_expert_ffn" in od.get("name", "")
        or "routed" in od.get("component", "").lower()
    ]

    modeled_us = None
    if routed_ops:
        modeled_us = (
            sum(_to_float(x.get("fwd_ms", 0.0)) for x in routed_ops)
            / len(routed_ops)
            * 1000.0
        )

    ep = getattr(strategy, "ep", 1)
    comparable = is_dsv4_pro_like and int(ep) == 8

    official_us = 369.6
    if modeled_us is not None and comparable:
        err_pct = abs(modeled_us - official_us) / official_us * 100.0
        delta = f"{err_pct:.1f}%"
        status = "pass" if err_pct <= 20.0 else "warn"
        modeled = f"routed_expert_fwd 平均 {modeled_us:.1f} us"
    elif modeled_us is not None:
        delta = "口径不一致"
        status = "info"
        modeled = f"routed_expert_fwd 平均 {modeled_us:.1f} us"
    else:
        delta = "未找到 routed expert 算子"
        status = "check"
        modeled = "-"

    rows.append(
        {
            "name": "DeepGEMM Mega MoE 基准",
            "official": (
                "DeepSeek-V4-Pro EP8，512 tokens/rank：369.6 us；"
                "1098 TFLOPS；全局显存 4619 GB/s；互联 182 GB/s。"
            ),
            "modeled": modeled,
            "status": status,
            "delta": delta,
            "source": "DeepGEMM PR #316",
            "url": "https://github.com/deepseek-ai/DeepGEMM/pull/316",
            "note": (
                "这是推理(serving)场景的 MegaMoE kernel 参考值。在本训练报告中，"
                "除非 EP=8、token 形状与 dtype 与该 PR 场景一致，"
                "否则仅作为 routed expert kernel 的合理性校验。"
            ),
        }
    )

    rows.append(
        {
            "name": "整步级校准（占位）",
            "official": (
                "本导出器未内置公开的 Ascend 910C DeepSeek-V4-Pro 整步训练参考值。"
            ),
            "modeled": (
                f"step_time={getattr(report, 'step_time_ms', 0.0):.3f} ms, "
                f"MFU={getattr(report, 'mfu', 0.0) * 100:.2f}%"
            ),
            "status": "check",
            "delta": "需目标集群本地采样",
            "source": "本地基准采样钩子",
            "url": "",
            "note": (
                "建议：从目标集群采集一个实测 step-time 样本并写入 report metadata，"
                "再计算 correction_factor = measured_step_ms / modeled_step_ms。"
            ),
        }
    )

    return rows


def _build_html_data(
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    report: "TrainingReport",
    op_costs: dict[str, "OpCost"],
) -> dict:
    tree, all_op_dicts = _build_layer_tree(graph, model, system, op_costs)
    summary = _build_summary(report, graph, model, system, strategy, all_op_dicts)

    return {
        "summary": summary,
        "topology": _build_topology_data(system, strategy),
        "model_overview": _build_model_overview(all_op_dicts),
        "tree": tree,
        "calibration": _build_calibration_data(
            report,
            graph,
            model,
            strategy,
            all_op_dicts,
        ),
        "warnings": getattr(report, "warnings", []),
    }


def _json_parse_literal_for_script(data) -> str:
    """Return a safe JavaScript string literal for JSON.parse(...).

    This avoids raw object literal injection inside <script>. The data becomes a
    JS string literal argument to JSON.parse, so substrings such as ${...} or
    backticks inside model/op names are data, not executable template syntax.
    """
    payload = json.dumps(_json_safe(data), ensure_ascii=False, separators=(",", ":"))

    # Prevent data from closing the script element.
    payload = payload.replace("</", "<\\/")

    # Defensive escaping for JavaScript string literal compatibility.
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    # Return a quoted JS string literal. ensure_ascii=True makes the outer JS
    # literal ASCII-safe even when the report contains Chinese text.
    return json.dumps(payload, ensure_ascii=True)


def _json_safe(value):
    """Convert exporter data to plain JSON-compatible containers."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {_json_safe(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root {
  --bg: #f5f7fb;
  --panel: #ffffff;
  --panel2: #f8fafc;
  --text: #152033;
  --muted: #637083;
  --line: #d9e1ec;
  --blue: #2563eb;
  --green: #16a34a;
  --amber: #d97706;
  --red: #dc2626;
  --purple: #7c3aed;
  --chip: #eef2ff;
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}
header {
  padding: 28px 36px;
  background: linear-gradient(135deg, #0f172a, #1e3a8a);
  color: white;
}
header h1 { margin: 0 0 8px; font-size: 28px; }
header p { margin: 0; color: #dbeafe; }
main { padding: 24px 36px 64px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 22px;
  margin-bottom: 22px;
  box-shadow: var(--shadow);
}
h2 { margin: 0 0 14px; font-size: 22px; }
h3 { margin: 16px 0 10px; font-size: 16px; }
small, .muted { color: var(--muted); }
.grid { display: grid; gap: 14px; }
.grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  background: var(--panel2);
}
.metric {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.metric .label { font-size: 12px; color: var(--muted); }
.metric .value { font-size: 20px; font-weight: 760; }
.kv {
  display: grid;
  grid-template-columns: 190px 1fr;
  gap: 8px 14px;
  font-size: 13px;
}
.kv div:nth-child(odd) { color: var(--muted); }
.bound-bar {
  display: flex;
  width: 100%;
  height: 18px;
  overflow: hidden;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: #e5e7eb;
}
.bound-bar span { display: block; height: 100%; }
.bound-compute { background: #2563eb; }
.bound-memory { background: #d97706; }
.bound-comm { background: #7c3aed; }
.legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-size: 12px;
}
.dot {
  width: 10px;
  height: 10px;
  display: inline-block;
  border-radius: 999px;
  margin-right: 5px;
}
.topology {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.node {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px;
  background: #fbfdff;
}
.node-title {
  font-weight: 760;
  margin-bottom: 10px;
}
.ranks {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
.rank {
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  padding: 8px;
  background: white;
  font-size: 12px;
}
.rank strong { display: block; margin-bottom: 4px; }
.chips {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.chip {
  border-radius: 999px;
  padding: 2px 6px;
  background: var(--chip);
  color: #3730a3;
  font-size: 10px;
  font-weight: 650;
}
.chip.pp { background: #dbeafe; color: #1d4ed8; }
.chip.tp { background: #dcfce7; color: #166534; }
.chip.ep { background: #fef3c7; color: #92400e; }
.chip.dp { background: #f3e8ff; color: #6b21a8; }
.chip.cp { background: #fee2e2; color: #991b1b; }

.model-flow {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 14px;
  align-items: stretch;
}
.flow-block {
  position: relative;
  border: 2px solid #dbeafe;
  border-radius: 18px;
  padding: 14px;
  background: linear-gradient(180deg, #ffffff, #eff6ff);
  min-height: 132px;
}
.flow-block:not(:last-child)::after {
  content: "→";
  position: absolute;
  right: -18px;
  top: 42%;
  color: #64748b;
  font-size: 22px;
}
.flow-title { font-weight: 800; margin-bottom: 8px; }
.flow-time { font-size: 20px; font-weight: 800; color: #1d4ed8; }
.flow-desc { font-size: 12px; color: var(--muted); min-height: 30px; }
.mini-bar {
  height: 8px;
  border-radius: 999px;
  background: #e5e7eb;
  overflow: hidden;
  margin-top: 10px;
}
.mini-bar span {
  display: block;
  height: 100%;
  background: #2563eb;
}

.toolbar {
  display: flex;
  gap: 10px;
  margin: 10px 0 16px;
}
button, input {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px 10px;
  background: white;
}
button {
  cursor: pointer;
  font-weight: 650;
}
input { min-width: 260px; }
details {
  border: 1px solid var(--line);
  border-radius: 12px;
  margin: 8px 0;
  background: white;
}
details > summary {
  list-style: none;
  cursor: pointer;
  padding: 12px 14px;
  font-weight: 760;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before {
  content: "+";
  width: 22px;
  height: 22px;
  border-radius: 6px;
  margin-right: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #e0ecff;
  color: #1d4ed8;
  font-weight: 800;
}
details[open] > summary::before { content: "−"; }
.summary-left {
  display: flex;
  align-items: center;
  gap: 8px;
}
.summary-right {
  color: var(--muted);
  font-weight: 600;
  font-size: 12px;
}
.op-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin: 0 0 10px;
}
.op-table th, .op-table td {
  border-top: 1px solid var(--line);
  padding: 8px;
  vertical-align: top;
  text-align: left;
}
.op-table th {
  background: #f8fafc;
  color: #475569;
}
.formula {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  white-space: pre-wrap;
  color: #334155;
}
.bound {
  display: inline-block;
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 760;
}
.bound.compute { background: #dbeafe; color: #1e40af; }
.bound.memory { background: #fef3c7; color: #92400e; }
.bound.communication { background: #ede9fe; color: #6d28d9; }
.bound.other { background: #f1f5f9; color: #475569; }
.calib-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.calib-table th, .calib-table td {
  border-top: 1px solid var(--line);
  padding: 10px;
  text-align: left;
  vertical-align: top;
}
.status {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 11px;
}
.status.pass { background: #dcfce7; color: #166534; }
.status.warn { background: #fef3c7; color: #92400e; }
.status.check { background: #fee2e2; color: #991b1b; }
.status.info { background: #dbeafe; color: #1e40af; }
.warning {
  border-left: 4px solid var(--amber);
  background: #fffbeb;
  padding: 10px 12px;
  border-radius: 10px;
  margin: 8px 0;
}
.pie-toolbar {
  display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-end;
  margin: 12px 0 18px;
}
.pie-field { display: flex; flex-direction: column; gap: 7px; }
.pie-flabel {
  font-size: 11px; font-weight: 800; letter-spacing: .03em;
  text-transform: uppercase; color: var(--muted);
}
.pie-toolbar select {
  border: 1px solid var(--line); border-radius: 10px;
  padding: 8px 11px; background: white; font-weight: 650; min-width: 220px;
}
.opshare-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.tog {
  cursor: pointer; user-select: none;
  border: 1px solid var(--line); border-radius: 999px;
  padding: 6px 13px; font-size: 12px; font-weight: 650;
  background: white; color: var(--text); line-height: 1.4;
}
.tog.on { background: #2563eb; border-color: #2563eb; color: white; }
.opshare-head {
  display: flex; gap: 18px; flex-wrap: wrap; align-items: baseline;
  padding: 10px 14px; border: 1px solid var(--line); border-radius: 12px;
  background: #f8fafc; margin-bottom: 14px;
}
.opshare-head b { font-size: 18px; }
.opshare-empty { color: var(--muted); padding: 16px 0; }
.pie-pair {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px;
}
.pie-block {
  border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
  background: #fdfefe; min-width: 0;
}
.pie-title { font-size: 13px; font-weight: 800; margin-bottom: 10px; }
.pie-layout {
  display: flex; gap: 22px; align-items: center; flex-wrap: wrap;
}
.pie-svg { width: 168px; height: 168px; flex: 0 0 auto; }
.pie-center-top { font-size: 13px; font-weight: 800; fill: var(--text); }
.pie-center-sub { font-size: 15px; font-weight: 800; fill: #2563eb; }
.pie-legend { flex: 1; min-width: 180px; display: flex; flex-direction: column; gap: 2px; }
.pie-leg-head {
  font-size: 11px; font-weight: 800; color: var(--muted);
  text-transform: uppercase; letter-spacing: .03em; margin: 8px 0 2px;
}
.pie-leg-head:first-child { margin-top: 0; }
.pie-leg-row {
  display: grid; grid-template-columns: 14px 1fr auto auto auto; gap: 9px;
  align-items: center; padding: 5px 6px; border-radius: 8px; font-size: 13px;
}
.pie-leg-row:hover { background: #f5f7fb; }
.pie-dot { width: 12px; height: 12px; border-radius: 3px; }
.pie-leg-name { font-weight: 650; display: flex; align-items: center; gap: 10px; min-width: 0; }
.leg-txt { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bound-tag {
  flex: 0 0 auto; margin-left: 4px; font-size: 10px; font-weight: 800; line-height: 1.6;
  padding: 0 7px; border-radius: 999px; color: #fff;
}
.pie-leg-n { color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.pie-leg-val { color: var(--muted); white-space: nowrap; }
.pie-leg-pct { font-weight: 800; white-space: nowrap; min-width: 56px; text-align: right; }
@media (max-width: 1100px) {
  .grid-2, .grid-3, .grid-4, .model-flow { grid-template-columns: 1fr; }
  .flow-block:not(:last-child)::after { display: none; }
  .pie-pair { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>ZRT Training Estimate Report</h1>
  <p>Spec-driven IR graph report: summary, topology, model-level timing, expandable op analysis, and calibration.</p>
</header>

<main>
  <section id="summary"></section>
  <section id="topology"></section>
  <section id="model-overview"></section>
  <section id="hierarchy"></section>
  <section id="op-share"></section>
  <section id="calibration"></section>
</main>

<script>
const DATA = JSON.parse(__DATA_JSON__);

const fmt = {
  ms(v) {
    v = Number(v || 0);
    if (Math.abs(v) >= 1000) return (v / 1000).toFixed(2) + "s";
    if (Math.abs(v) >= 1) return v.toFixed(2) + "ms";
    return (v * 1000).toFixed(2) + "us";
  },
  pct(v) { return (Number(v || 0) * 100).toFixed(2) + "%"; },
  num(v) {
    v = Number(v || 0);
    if (Math.abs(v) >= 1e15) return (v / 1e15).toFixed(2) + "P";
    if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(2) + "T";
    if (Math.abs(v) >= 1e9) return (v / 1e9).toFixed(2) + "G";
    if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(2) + "K";
    return String(v.toFixed ? v.toFixed(2) : v);
  },
  gb(v) { return Number(v || 0).toFixed(2) + " GB"; }
};

function esc(x) {
  return String(x ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[ch]));
}

function boundClass(b) {
  b = String(b || "").toLowerCase();
  if (b.includes("memory") || b.includes("hbm")) return "memory";
  if (b.includes("comm")) return "communication";
  if (b.includes("compute") || b.includes("cube") || b.includes("vector")) return "compute";
  return "other";
}

function renderSummary() {
  const s = DATA.summary;
  const p = s.performance;
  const b = s.bound;
  const m = s.memory || {};
  const operatorStats = s.operator_time_share || [];

  document.getElementById("summary").innerHTML = `
    <h2>1. 报告 Summary</h2>

    <div class="grid grid-4">
      <div class="card metric"><div class="label">Step Time</div><div class="value">${fmt.ms(p.step_time_ms)}</div></div>
      <div class="card metric"><div class="label">Tokens / sec</div><div class="value">${fmt.num(p.tokens_per_sec)}</div></div>
      <div class="card metric"><div class="label">MFU</div><div class="value">${fmt.pct(p.mfu)}</div></div>
      <div class="card metric"><div class="label">HFU</div><div class="value">${fmt.pct(p.hfu)}</div></div>
    </div>

    <h3>模型 / 硬件 / 拓扑 / 并行策略</h3>
    <div class="grid grid-2">
      <div class="card">
        <h3>Model</h3>
        <div class="kv">
          <div>name</div><div>${esc(s.model.name)}</div>
          <div>hidden / layers</div><div>${esc(s.model.hidden)} / ${esc(s.model.layers)}</div>
          <div>MoE / MTP / Dense</div><div>${esc(s.model.moe_layers)} / ${esc(s.model.mtp_layers)} / ${esc(s.model.dense_layers)}</div>
          <div>experts / top_k / moe_ffn</div><div>${esc(s.model.num_experts)} / ${esc(s.model.top_k)} / ${esc(s.model.moe_ffn)}</div>
          <div>heads / head_dim</div><div>${esc(s.model.num_heads)} / ${esc(s.model.head_dim)}</div>
          <div>seq_len</div><div>${esc(s.model.seq_len)}</div>
          <div>dtype</div><div>param=${esc(s.model.param_dtype)}, act=${esc(s.model.act_dtype)}</div>
        </div>
      </div>

      <div class="card">
        <h3>Hardware / Topology</h3>
        <div class="kv">
          <div>accelerator</div><div>${esc(s.hardware.name)}</div>
          <div>HBM</div><div>${esc(s.hardware.hbm_gb)} GB, ${esc(s.hardware.hbm_bw_gbps)} GB/s</div>
          <div>compute</div><div>BF16=${esc(s.hardware.tflops_bf16)} TFLOPS, FP8=${esc(s.hardware.tops_fp8)} TOPS</div>
          <div>nodes × ranks/node</div><div>${esc(s.topology.nodes)} × ${esc(s.topology.gpus_per_node)}</div>
          <div>world size</div><div>${esc(s.topology.world_size)}</div>
          <div>parallel</div><div>TP=${esc(s.strategy.tp)}, CP=${esc(s.strategy.cp)}, PP=${esc(s.strategy.pp)}, EP=${esc(s.strategy.ep)}, DP=${esc(s.strategy.dp)}</div>
          <div>schedule / optimizer</div><div>${esc(s.strategy.pp_schedule)} / ${esc(s.strategy.optimizer)}</div>
        </div>
      </div>
    </div>

    <h3>整体性能评估结果</h3>
    <div class="grid grid-3">
      <div class="card kv">
        <div>pipeline time</div><div>${fmt.ms(p.pipeline_time_ms)}</div>
        <div>compute time</div><div>${fmt.ms(p.compute_time_ms)}</div>
        <div>exposed comm</div><div>${fmt.ms(p.exposed_comm_ms)}</div>
        <div>pipeline bubble</div><div>${fmt.ms(p.bubble_time_ms)}</div>
      </div>

      <div class="card kv">
        <div>optimizer time</div><div>${fmt.ms(p.optimizer_time_ms)}</div>
        <div>optimizer comm</div><div>${fmt.ms(p.optimizer_comm_ms)}</div>
        <div>bubble fraction</div><div>${fmt.pct(p.bubble_fraction)}</div>
        <div>FLOPs/token</div><div>${fmt.num(p.flops_per_token)}</div>
      </div>

      <div class="card kv">
        <div>TP exposed</div><div>${fmt.ms(s.comm_breakdown.tp_exposed_ms)}</div>
        <div>EP exposed</div><div>${fmt.ms(s.comm_breakdown.ep_exposed_ms)}</div>
        <div>DP exposed</div><div>${fmt.ms(s.comm_breakdown.dp_exposed_ms)}</div>
        <div>comm volume</div><div>${fmt.ms(s.comm_breakdown.total_comm_volume_ms)}</div>
      </div>
    </div>

    <h3>step time breakdown</h3>
    <div class="card kv">
      <div>Useful Compute</div><div>${fmt.ms(p.compute_time_ms)}</div>
      <div>Recompute (crit. path)</div><div>${fmt.ms(p.recompute_critical_ms || 0)}</div>
      <div>Communication (exposed)</div><div>${fmt.ms(p.exposed_comm_ms)}</div>
      <div>Pipeline Bubble</div><div>${fmt.ms(p.bubble_time_ms)}</div>
      <div>Optimizer (compute)</div><div>${fmt.ms(p.optimizer_time_ms)}</div>
      <div>Optimizer (comm)</div><div>${fmt.ms(p.optimizer_comm_ms)}</div>
      ${operatorStats.length ? `
        <div>Operator Time Share</div><div>${operatorStats.map(row => esc(row.label) + " " + fmt.ms(row.time_ms) + " / step " + fmt.pct(row.pct_of_step) + " / useful compute " + fmt.pct(row.pct_of_useful_compute) + " (" + esc(row.op_count) + " ops)").join("<br/>")}</div>
      ` : ""}
    </div>

    <h3>内存消耗结果</h3>
    <div class="grid grid-4">
      ${Object.entries(m).map(([k, v]) => `
        <div class="card metric">
          <div class="label">${esc(k)}</div>
          <div class="value">${fmt.gb(v)}</div>
        </div>
      `).join("")}
    </div>

    <h3>计算过程 Bound 占比</h3>
    <div class="card">
      <div class="bound-bar">
        <span class="bound-compute" style="width:${b.compute_pct * 100}%"></span>
        <span class="bound-memory" style="width:${b.memory_pct * 100}%"></span>
        <span class="bound-comm" style="width:${b.communication_pct * 100}%"></span>
      </div>
      <div class="legend">
        <span><i class="dot bound-compute"></i>Compute ${fmt.pct(b.compute_pct)} / ${fmt.ms(b.compute_ms)}</span>
        <span><i class="dot bound-memory"></i>Memory ${fmt.pct(b.memory_pct)} / ${fmt.ms(b.memory_ms)}</span>
        <span><i class="dot bound-comm"></i>Communication ${fmt.pct(b.communication_pct)} / ${fmt.ms(b.communication_ms)}</span>
      </div>
    </div>

    ${(DATA.warnings || []).map(w => `<div class="warning">${esc(w)}</div>`).join("")}
  `;
}

function renderTopology() {
  const t = DATA.topology;

  document.getElementById("topology").innerHTML = `
    <h2>2. 集群逻辑拓扑</h2>

    ${t.legend.is_inferred ? `
      <div class="warning">
        <strong>注意：</strong>当前 Rank 拓扑是 inferred visualization only，不代表真实 runtime rank placement。
        如果 launcher 使用自定义 rank order，请在配置中提供 rank_placement 后再用于拓扑校验。
      </div>
    ` : `
      <div class="card">
        <strong>Rank placement source:</strong> ${esc(t.legend.mapping_source)}
      </div>
    `}

    <p class="muted">${esc(t.legend.note)}</p>

    <div class="topology">
      ${t.nodes.map(node => `
        <div class="node">
          <div class="node-title">Node ${node.node}</div>
          <div class="ranks">
            ${node.ranks.map(r => `
              <div class="rank">
                <strong>Rank ${r.rank}</strong>
                <div class="muted">local ${r.local_rank}</div>
                <div class="chips">
                  <span class="chip pp">PP${r.pp}</span>
                  <span class="chip tp">TP${r.tp}</span>
                  <span class="chip ep">EP${r.ep}</span>
                  <span class="chip dp">DP${r.dp}</span>
                  <span class="chip cp">CP${r.cp}</span>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderModelOverview() {
  const blocks = DATA.model_overview || [];

  document.getElementById("model-overview").innerHTML = `
    <h2>3. 模型级计算耗时</h2>
    <p class="muted">Transformer Explainer 风格概览：Input → Embedding → Attention → MoE → Output。耗时来自 IR op 的 fwd + dx + dw 估算。</p>

    <div class="model-flow">
      ${blocks.map(b => `
        <div class="flow-block">
          <div class="flow-title">${esc(b.name)}</div>
          <div class="flow-desc">${esc(b.desc)}</div>
          <div class="flow-time">${fmt.ms(b.total_ms)}</div>
          <div class="muted">${esc(b.ops)} ops · ${fmt.pct(b.pct)}</div>
          <div class="mini-bar"><span style="width:${Math.max(1, b.pct * 100)}%"></span></div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderOpTable(ops) {
  return `
    <table class="op-table">
      <thead>
        <tr>
          <th>Op</th>
          <th>Kind</th>
          <th>Time</th>
          <th>Bound</th>
          <th>Formula / 原理</th>
          <th>Tensor I/O</th>
        </tr>
      </thead>
      <tbody>
        ${ops.map(op => {
          const bc = boundClass(op.bound);
          return `
            <tr class="op-row" data-search="${esc((op.name + " " + op.kind + " " + op.bound).toLowerCase())}">
              <td><strong>${esc(op.name)}</strong></td>
              <td>${esc(op.kind)}</td>
              <td>
                total=${fmt.ms(op.total_ms)}<br/>
                fwd=${fmt.ms(op.fwd_ms)}<br/>
                dx=${fmt.ms(op.dx_ms)}<br/>
                dw=${fmt.ms(op.dw_ms)}
              </td>
              <td><span class="bound ${bc}">${esc(op.bound || bc)}</span></td>
              <td class="formula">
                <b>FWD FLOPs</b>: ${esc(op.fwd_formula)}<br/>
                <b>BWD FLOPs</b>: ${esc(op.bwd_formula)}<br/>
                <b>FWD Bytes</b>: ${esc(op.fwd_bytes_formula)}<br/>
                <b>BWD Bytes</b>: ${esc(op.bwd_bytes_formula)}<br/>
                <b>原理</b>: roofline = max(compute_time, memory_time); bound 表示该 op 当前瓶颈来源。
              </td>
              <td class="formula">
                <b>Inputs</b>: ${esc(op.inputs)}<br/>
                <b>Outputs</b>: ${esc(op.outputs)}
              </td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

function renderHierarchy() {
  const tree = DATA.tree;

  document.getElementById("hierarchy").innerHTML = `
    <h2>4. 分层展示计算耗时</h2>
    <p class="muted">模型 → Layer / Block → 算子。每一层可以展开和收缩，算子层包含耗时、公式、公式原理和 bound 类型。</p>

    <div class="toolbar">
      <button onclick="setAllDetails(true)">expand all</button>
      <button onclick="setAllDetails(false)">collapse all</button>
      <input id="opSearch" placeholder="search op / kind / bound..." oninput="filterOps(this.value)" />
    </div>

    <details open>
      <summary>
        <span class="summary-left">Model: ${esc(tree.model_name)}</span>
        <span class="summary-right">${esc(tree.total_ops)} ops</span>
      </summary>

      ${tree.global_ops && tree.global_ops.length ? `
        <details>
          <summary>
            <span class="summary-left">Global Ops</span>
            <span class="summary-right">${tree.global_ops.length} ops · ${fmt.ms(tree.global_ops.reduce((a,b)=>a+Number(b.total_ms||0),0))}</span>
          </summary>
          ${renderOpTable(tree.global_ops)}
        </details>
      ` : ""}

      ${tree.layers.map(layer => `
        <details>
          <summary>
            <span class="summary-left">Layer ${layer.id} · ${esc(layer.kind)}</span>
            <span class="summary-right">${layer.op_count} ops · ${fmt.ms(layer.total_ms)}</span>
          </summary>

          ${layer.blocks.map(block => `
            <details>
              <summary>
                <span class="summary-left">${esc(block.name)}</span>
                <span class="summary-right">${block.ops.length} ops · ${fmt.ms(block.total_ms)}</span>
              </summary>
              ${renderOpTable(block.ops)}
            </details>
          `).join("")}
        </details>
      `).join("")}
    </details>
  `;
}

function setAllDetails(open) {
  document.querySelectorAll("details").forEach(d => d.open = open);
}

function filterOps(q) {
  q = String(q || "").trim().toLowerCase();
  document.querySelectorAll(".op-row").forEach(row => {
    if (!q) {
      row.style.display = "";
    } else {
      row.style.display = row.dataset.search.includes(q) ? "" : "none";
    }
  });
}

function renderCalibration() {
  const rows = DATA.calibration || [];

  document.getElementById("calibration").innerHTML = `
    <h2>6. 报告校准</h2>
    <p class="muted">校准分为结构校验、公开 kernel benchmark 对齐、以及目标集群本地 step-time 采样。公开 benchmark 不一定与当前训练配置完全同构，需要看 status 和 note。</p>

    <table class="calib-table">
      <thead>
        <tr>
          <th>校准项</th>
          <th>官方数据</th>
          <th>本报告数据</th>
          <th>偏差 / 状态</th>
          <th>引用出处</th>
          <th>说明</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td><strong>${esc(r.name)}</strong></td>
            <td>${esc(r.official)}</td>
            <td>${esc(r.modeled)}</td>
            <td><span class="status ${esc(r.status)}">${esc(r.status)}</span><br/>${esc(r.delta)}</td>
            <td>${r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.source)}</a>` : esc(r.source)}</td>
            <td>${esc(r.note)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

// ── Operator time-share analysis (pie) ──────────────────────────────────────
const OPSHARE = { ops: [], groups: [], scale: 1 };
const OPSHARE_COLORS = [
  "#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2",
  "#ca8a04", "#db2777", "#65a30d", "#0ea5e9", "#7c3aed", "#e11d48",
  "#0d9488", "#a16207", "#4f46e5", "#be123c",
];

function opTopBound(op) {
  // Mirror _build_summary: anything not memory/comm counts as compute.
  const b = String(op.bound || "").toLowerCase();
  if (b.includes("memory") || b.includes("hbm")) return "memory";
  if (b.includes("comm")) return "comm";
  return "compute";
}

// Flatten the layer tree into a single op list. Each op already carries
// op_groups / cube_flops / vector_flops / layer_id / layer_kind from the
// Python exporter, so all analysis happens client-side with no extra fetch.
function flattenOps(tree) {
  const out = [];
  (tree.global_ops || []).forEach(op => out.push(op));
  (tree.layers || []).forEach(layer => {
    (layer.blocks || []).forEach(block => {
      (block.ops || []).forEach(op => out.push(op));
    });
  });
  return out;
}

// Per-op times are raw single-microbatch estimates; OPSHARE.scale lifts them
// into full-step units (matching summary.operator_time_share / compute_time_ms)
// so percentages against step_time / useful_compute are meaningful.
function opPhaseTime(op, fwdOn, bwdOn) {
  return ((fwdOn ? Number(op.fwd_ms || 0) : 0)
    + (bwdOn ? Number(op.dx_ms || 0) + Number(op.dw_ms || 0) : 0)) * OPSHARE.scale;
}

const OPGROUP_LABELS = {
  matmul: "matmul", attention: "attention", attention_matmul: "attn matmul",
  ffn: "ffn / moe", lm_head: "lm_head", mtp_embed: "mtp embed",
  indexer: "indexer", sparse_fa: "sparse FA",
};
const FOCUS_OPTS = [
  ["__all__", "全部算子"],
  ["matmul", "matmul"],
  ["attention", "attention"],
  ["ffn", "ffn / moe"],
  ["attention_matmul", "attn matmul"],
  ["lm_head", "lm_head"],
  ["indexer", "indexer"],
  ["sparse_fa", "sparse FA"],
];
// Left-pie (module view) dimensions only — the right pie is always the
// operator's internal compute-unit breakdown.
const MODULE_DIM_OPTS = [
  ["component_group", "模型组件 (attn / moe / …)"],
  ["kind", "算子 kind"],
  ["layer_kind", "layer 类型 (dense / moe / mtp)"],
  ["layer_id", "layer 编号"],
  ["op_group", "语义分组"],
];

// Time pieces of an op split into (phase × compute-unit), used only by the
// compute_unit / phase breakdowns. Compute-bound ops are apportioned to cube /
// vector by their FLOP ratio; memory / comm ops stay whole.
function opUnitPieces(op, fwdOn, bwdOn) {
  const tb = opTopBound(op);
  const pieces = [];
  const phases = [
    ["forward", (fwdOn ? Number(op.fwd_ms || 0) : 0) * OPSHARE.scale],
    ["backward", (bwdOn ? Number(op.dx_ms || 0) + Number(op.dw_ms || 0) : 0) * OPSHARE.scale],
  ];
  const cf = Number(op.cube_flops || 0);
  const vf = Number(op.vector_flops || 0);
  const tot = cf + vf;
  for (const [phase, pt] of phases) {
    if (pt <= 0) continue;
    if (tb !== "compute") {
      pieces.push({ phase, unit: tb, t: pt });
    } else if (tot > 0) {
      if (cf > 0) pieces.push({ phase, unit: "cube", t: pt * cf / tot });
      if (vf > 0) pieces.push({ phase, unit: "vector", t: pt * vf / tot });
    } else {
      pieces.push({ phase, unit: "compute_other", t: pt });
    }
  }
  return pieces;
}

function uniqSorted(values) {
  return Array.from(new Set(values.filter(v => v !== "" && v != null))).sort();
}

function togStates(sel) {
  const out = {};
  document.querySelectorAll(`[data-sel="${sel}"]`).forEach(el => {
    out[el.dataset.val] = el.classList.contains("on");
  });
  return out;
}

function selectedVals(sel) {
  return Object.entries(togStates(sel)).filter(([, on]) => on).map(([v]) => v);
}

function chip(sel, val, label, on, sub) {
  return `<span class="tog${sub ? " sub" : ""}${on ? " on" : ""}" `
    + `data-sel="${esc(sel)}" data-val="${esc(val)}">${esc(label)}</span>`;
}

function renderOpShare() {
  const tree = DATA.tree || {};
  OPSHARE.ops = flattenOps(tree);
  OPSHARE.groups = new Set([].concat(...OPSHARE.ops.map(o => o.op_groups || [])));

  // Match operator_time_stats._operator_time_scale: lift raw per-op times to
  // full-step units so "全部算子" ≈ useful compute instead of a tiny fraction.
  const allOpMs = OPSHARE.ops.reduce((a, o) => a + Number(o.total_ms || 0), 0);
  const usefulMs = Number((DATA.summary && DATA.summary.performance
    && DATA.summary.performance.compute_time_ms) || 0);
  OPSHARE.scale = (allOpMs > 0 && usefulMs > 0) ? usefulMs / allOpMs : 1;

  // Only offer a focus option when at least one op carries that group.
  const focusOpts = FOCUS_OPTS.filter(
    ([v]) => v === "__all__" || OPSHARE.groups.has(v)
  );

  const moduleOpts = MODULE_DIM_OPTS.filter(
    ([v]) => v !== "op_group" || OPSHARE.groups.size
  );

  document.getElementById("op-share").innerHTML = `
    <h2>5. 算子耗时占比分析</h2>
    <p class="muted">左图为<b>模块视角</b>：算子耗时在各 module / 组件间的占比；
    右图为<b>算子视角</b>：所选算子内部的计算单元构成 (cube / vector / memory / comm)。
    用「聚焦算子」「阶段」同时过滤两图。耗时来自 IR op 的 fwd + dx + dw 估算，全部在浏览器端计算。</p>

    <div class="pie-toolbar">
      <label class="pie-field">
        <span class="pie-flabel">聚焦算子</span>
        <select id="opFocus">
          ${focusOpts.map(([v, l], i) =>
            `<option value="${esc(v)}"${i === 0 ? " selected" : ""}>${esc(l)}</option>`).join("")}
        </select>
      </label>
      <label class="pie-field">
        <span class="pie-flabel">模块维度 (左图)</span>
        <select id="opModuleDim">
          ${moduleOpts.map(([v, l], i) =>
            `<option value="${esc(v)}"${i === 0 ? " selected" : ""}>${esc(l)}</option>`).join("")}
        </select>
      </label>
      <div class="pie-field">
        <span class="pie-flabel">阶段</span>
        <div class="opshare-chips" data-group="phase">
          ${chip("phase", "forward", "前向", true)}
          ${chip("phase", "backward", "反向", true)}
        </div>
      </div>
    </div>

    <div id="opShareResult"></div>
  `;

  const sec = document.getElementById("op-share");
  sec.querySelector('[data-group="phase"]').addEventListener("click", e => {
    const el = e.target.closest(".tog");
    if (!el) return;
    el.classList.toggle("on");
    recomputeOpShare();
  });
  document.getElementById("opFocus").addEventListener("change", recomputeOpShare);
  document.getElementById("opModuleDim").addEventListener("change", recomputeOpShare);

  recomputeOpShare();
}

// Bucket key for one op under the chosen breakdown dimension. Returns a list of
// {key, t} so that multi-valued dimensions (compute_unit, phase, op_group) can
// split a single op across several slices.
function opShareContribs(op, breakdown, fwdOn, bwdOn) {
  if (breakdown === "compute_unit" || breakdown === "phase" || breakdown === "bound") {
    const out = [];
    for (const pc of opUnitPieces(op, fwdOn, bwdOn)) {
      let key;
      if (breakdown === "phase") {
        key = pc.phase === "forward" ? "前向" : "反向";
      } else if (breakdown === "bound") {
        key = (pc.unit === "memory" || pc.unit === "comm") ? pc.unit : "compute";
      } else {
        key = ({ cube: "cube", vector: "vector", memory: "memory", comm: "comm",
                 compute_other: "compute (无 FLOPs)" }[pc.unit] || pc.unit);
      }
      out.push({ key, t: pc.t });
    }
    return out;
  }

  const t = opPhaseTime(op, fwdOn, bwdOn);
  if (t <= 0) return [];

  if (breakdown === "op_group") {
    const tags = op.op_groups || [];
    if (!tags.length) return [{ key: "(无语义分组)", t }];
    return tags.map(tag => ({ key: OPGROUP_LABELS[tag] || tag, t }));
  }

  let key;
  switch (breakdown) {
    case "kind": key = op.kind || "?"; break;
    case "layer_kind":
      key = Number(op.layer_id) < 0 ? "global" : (op.layer_kind || "?"); break;
    case "layer_id":
      key = Number(op.layer_id) < 0 ? "global" : ("Layer " + op.layer_id); break;
    default: key = op.component_group || "other"; // component_group
  }
  return [{ key, t }];
}

// Aggregate focused ops into {key: ms} buckets under one breakdown dimension.
function buildOpShareBuckets(focus, breakdown, fwdOn, bwdOn) {
  // For module-view dimensions, also record each bucket's bound mix so the
  // legend can tag it with its dominant bound (compute / memory / comm).
  const trackBound = !["compute_unit", "phase", "bound"].includes(breakdown);
  const buckets = new Map();
  let sum = 0;
  for (const op of OPSHARE.ops) {
    if (focus !== "__all__" && !(op.op_groups || []).includes(focus)) continue;
    const ob = trackBound ? opTopBound(op) : null;
    for (const c of opShareContribs(op, breakdown, fwdOn, bwdOn)) {
      if (c.t <= 0) continue;
      const b = buckets.get(c.key) || { t: 0, n: 0, bt: {} };
      b.t += c.t;
      b.n += 1;  // occurrences in this category (= op count for single-key dims)
      if (ob) b.bt[ob] = (b.bt[ob] || 0) + c.t;
      buckets.set(c.key, b);
      sum += c.t;
    }
  }
  const overlaps = breakdown === "op_group";
  const denom = (overlaps
    ? Array.from(buckets.values()).reduce((a, b) => a + b.t, 0)
    : sum) || 1e-12;
  const rows = Array.from(buckets.entries())
    .map(([k, v]) => {
      const bounds = Object.entries(v.bt).sort((a, b) => b[1] - a[1]);
      return { k, t: v.t, n: v.n, bound: bounds.length ? bounds[0][0] : null };
    })
    .sort((a, b) => b.t - a.t)
    .map((r, i) => ({ ...r, color: OPSHARE_COLORS[i % OPSHARE_COLORS.length] }));
  return { rows, denom, overlaps };
}

// Fixed colors so bound (outer ring) and compute-unit (inner ring) read
// consistently: cube/vector are blue shades within the blue "compute" bound.
const BOUND_COLORS = {
  compute: "#2563eb", memory: "#d97706", comm: "#0891b2", "compute (无 FLOPs)": "#94a3b8",
};
const UNIT_COLORS = {
  cube: "#2563eb", vector: "#7dd3fc", memory: "#d97706", comm: "#0891b2",
  "compute (无 FLOPs)": "#94a3b8",
};

function legendHtml(rows, denom) {
  return rows.map(r => {
    const tag = r.bound
      ? `<span class="bound-tag" style="background:${BOUND_COLORS[r.bound] || "#94a3b8"}">${esc(r.bound)}</span>`
      : "";
    return `
    <div class="pie-leg-row">
      <span class="pie-dot" style="background:${r.color}"></span>
      <span class="pie-leg-name"><span class="leg-txt">${esc(r.k)}</span>${tag}</span>
      <span class="pie-leg-n">${r.n}个</span>
      <span class="pie-leg-val">${fmt.ms(r.t)}</span>
      <span class="pie-leg-pct">${fmt.pct(r.t / denom)}</span>
    </div>`;
  }).join("");
}

function pieBlock(title, data) {
  if (!data.rows.length) {
    return `<div class="pie-block"><div class="pie-title">${title}</div>`
      + `<div class="opshare-empty">无数据</div></div>`;
  }
  return `
    <div class="pie-block">
      <div class="pie-title">${title}</div>
      ${data.overlaps ? `<p class="muted" style="font-size:11px;margin:0 0 6px">分组可重叠，饼图按各分组之和归一化</p>` : ""}
      <div class="pie-layout">
        ${donutSvg(data.rows, data.denom)}
        <div class="pie-legend">${legendHtml(data.rows, data.denom)}</div>
      </div>
    </div>
  `;
}


function recomputeOpShare() {
  const focus = document.getElementById("opFocus").value;
  const moduleDim = document.getElementById("opModuleDim").value;
  const phaseOn = togStates("phase");
  const fwdOn = !!phaseOn.forward;
  const bwdOn = !!phaseOn.backward;

  // True focused total (no double counting) for the headline percentages.
  let total = 0;
  for (const op of OPSHARE.ops) {
    if (focus !== "__all__" && !(op.op_groups || []).includes(focus)) continue;
    total += opPhaseTime(op, fwdOn, bwdOn);
  }

  const el = document.getElementById("opShareResult");
  if (total <= 0) {
    el.innerHTML = `<div class="opshare-empty">当前筛选无匹配算子。</div>`;
    return;
  }

  const p = (DATA.summary && DATA.summary.performance) || {};
  const stepT = Number(p.step_time_ms || 0);
  const usefulT = Number(p.compute_time_ms || 0);

  const moduleLabel = (MODULE_DIM_OPTS.find(o => o[0] === moduleDim) || [, moduleDim])[1];
  const left = buildOpShareBuckets(focus, moduleDim, fwdOn, bwdOn);
  const right = buildOpShareBuckets(focus, "compute_unit", fwdOn, bwdOn);
  right.rows.forEach(r => { r.color = UNIT_COLORS[r.k] || r.color; });

  el.innerHTML = `
    <div class="opshare-head">
      <span>聚焦合计 <b>${fmt.ms(total)}</b></span>
      <span>占 step time <b>${fmt.pct(stepT > 0 ? total / stepT : 0)}</b></span>
      <span>占 useful compute <b>${fmt.pct(usefulT > 0 ? total / usefulT : 0)}</b></span>
    </div>
    <div class="pie-pair">
      ${pieBlock(`模块视角 · ${esc(moduleLabel)}`, left)}
      ${pieBlock("算子视角 · 计算单元", right)}
    </div>
  `;
}

// Pure-SVG donut (no chart library). Each slice is a stroked circle arc placed
// via stroke-dasharray / dashoffset, starting at 12 o'clock.
function donutSvg(rows, denom) {
  const R = 70, CX = 90, CY = 90, SW = 34, C = 2 * Math.PI * R;
  let acc = 0;
  const arcs = rows.map(r => {
    const frac = Math.max(0, r.t / denom);
    const len = frac * C;
    const seg = `<circle cx="${CX}" cy="${CY}" r="${R}" fill="none" `
      + `stroke="${r.color}" stroke-width="${SW}" `
      + `stroke-dasharray="${len.toFixed(3)} ${(C - len).toFixed(3)}" `
      + `stroke-dashoffset="${(-acc).toFixed(3)}" `
      + `transform="rotate(-90 ${CX} ${CY})"><title>${esc(r.k)} · ${fmt.pct(frac)}</title></circle>`;
    acc += len;
    return seg;
  }).join("");
  const top = rows[0];
  return `
    <svg class="pie-svg" viewBox="0 0 180 180" role="img" aria-label="operator time share">
      <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="#eef1f6" stroke-width="${SW}"></circle>
      ${arcs}
      <text x="${CX}" y="${CY - 4}" text-anchor="middle" class="pie-center-top">${esc(top.k)}</text>
      <text x="${CX}" y="${CY + 16}" text-anchor="middle" class="pie-center-sub">${fmt.pct(top.t / denom)}</text>
    </svg>
  `;
}

renderSummary();
renderTopology();
renderModelOverview();
renderHierarchy();
renderOpShare();
renderCalibration();
</script>
</body>
</html>
"""


def export_estimate_html(
    *,
    report: "TrainingReport",
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    op_costs: dict[str, "OpCost"],
    output_path: str | Path,
) -> Path:
    """Write a self-contained HTML report.

    Layout:
    1. Summary: model/hardware/topology/strategy/perf/memory/bound.
    2. Logical cluster topology with rank-level PP/DP/TP/EP/CP labels.
    3. Model-level timing overview in Transformer-Explainer-like blocks.
    4. Expandable model -> layer/block -> op timing/formula/bound analysis.
    5. Calibration table with public/official reference points.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _build_html_data(graph, model, system, strategy, report, op_costs)

    title = (
        f"ZRT Training Estimate — "
        f"{getattr(model, 'hidden', '-')}d_{len(getattr(model, 'layers', []))}L"
    )
    safe_title = html_lib.escape(title, quote=True)

    json_data = _json_parse_literal_for_script(data)

    html_text = (
        _HTML_TEMPLATE
        .replace("__TITLE__", safe_title)
        .replace("__DATA_JSON__", json_data)
    )

    output_path.write_text(html_text, encoding="utf-8")
    return output_path
