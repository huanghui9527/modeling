"""Per-op dtype resolution for byte and FLOPs accounting.

See ``docs/docs-pri/mixed_quant_v2_op_bytes_zh.md`` §3 for the full design.

This module is the single source of truth for "what dtype does each operand
of an op live in" — replacing the ad-hoc ``inputs[0].dtype`` lookups and
the ``_resolve_compute_dtype`` helper in ``compose/stage.py``.

Downstream callers:
  - ``models/flops.py::_matmul_cost`` / ``_attn_cost`` (per-operand bytes)
  - ``compose/stage.py::_resolve_compute_dtype`` (thin wrapper → ``.compute``)
  - ``ir/cast_pass.py`` (decides dtype-boundary insertion sites)
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zrt.ir.node import OpNode

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec


def _kind(op) -> str:
    if hasattr(op, "attrs"):
        return op.attrs.get("spec_kind", op.op_type)
    return op.kind


@dataclass(frozen=True)
class OpDtypeBundle:
    """Seven-slot dtype resolution for one op.

    Slots:
      in_act      activation read from HBM (A in fwd, dC in bwd-dx input)
      weight      parameter matrix storage form (FP4 includes block-scale via stored_bytes)
      out_act     activation written back to HBM (C in fwd)
      compute     math-unit tile dtype (selects peak_tflops_for)
      grad_in     dC — output-side gradient flowing into bwd
      grad_weight dW write-back dtype (master grad)
      grad_act    dA — input-side gradient produced by bwd-dx
    """
    in_act:      Dtype
    weight:      Dtype
    out_act:     Dtype
    compute:     Dtype
    grad_in:     Dtype
    grad_weight: Dtype
    grad_act:    Dtype


def _meta(op) -> dict:
    return op.attrs if hasattr(op, "attrs") else op.meta


def _cast_bundle(op, model: ModelSpec) -> OpDtypeBundle:
    """Cast op's dtypes live entirely in meta/attrs."""
    m = _meta(op)
    src = m.get("src_dtype", model.act_dtype)
    dst = m.get("dst_dtype", model.act_dtype)
    return OpDtypeBundle(
        in_act=src,
        weight=src,                 # unused; cast has no weight
        out_act=dst,
        compute=dst,
        grad_in=dst,
        grad_weight=dst,
        grad_act=src,
    )


def resolve_op_dtypes(op: OpNode, model: ModelSpec) -> OpDtypeBundle:
    """Return the dtype bundle for one op.

    Spec-specific ops (``kind == 'cast'``) are handled locally; all other
    component buckets delegate to the shared ``dispatch()`` in
    ``quant_dispatch.py``.
    """
    if _kind(op) == "cast":
        return _cast_bundle(op, model)

    from zrt.training.models.quant_dispatch import dispatch
    comp = op.component or "default"
    return dispatch(comp, model)


def expected_input_dtype(op: OpNode, ti: int, model: ModelSpec) -> Dtype:
    """Dtype that ``op.inputs[ti]`` must arrive in.

    Used by ``cast_pass`` to decide where to insert cast ops. Defaults to
    ``bundle.in_act`` but special-cases multi-input ops where operand
    dtypes differ.
    """
    bundle = resolve_op_dtypes(op, model)
    k = _kind(op)

    if k == "add":
        if op.component in {"routed_expert", "shared_expert"}:
            return bundle.in_act
        return model.effective_residual_dtype()

    if k in ("dispatch", "combine"):
        return model.effective_moe_act_dtype()

    if k in ("mhc_pre", "mhc_post", "mhc_head"):
        if 0 <= ti < len(op.inputs):
            return op.inputs[ti].dtype
        return bundle.in_act

    return bundle.in_act
