"""Tests for _attn_cost per-operand byte accounting under quantization (Stage C2)."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from zrt.training.models.flops import op_cost, _attn_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _make_tensor(name, shape_logical, shape_local, dtype, is_activation=True):
    import math
    ne = math.prod(shape_local)
    return SimpleNamespace(
        name=name, shape_logical=shape_logical, shape_local=shape_local,
        dtype=dtype, is_activation=is_activation,
        num_elements=lambda: ne,
        nbytes=lambda: ne * dtype.bytes,
    )


def _make_attn_op(seq: int = 256, heads: int = 4, head_dim: int = 32,
                   in_dtype: Dtype = Dtype.BF16, component: str = "attention"):
    return SimpleNamespace(
        name="L0.attn_core", kind="attn_core",
        inputs=[
            _make_tensor("q", (seq, heads, head_dim), (seq, heads, head_dim), in_dtype),
            _make_tensor("k", (seq, heads, head_dim), (seq, heads, head_dim), in_dtype),
            _make_tensor("v", (seq, heads, head_dim), (seq, heads, head_dim), in_dtype),
        ],
        outputs=[_make_tensor("o", (seq, heads, head_dim), (seq, heads, head_dim), in_dtype)],
        meta={"b": 1, "s": seq, "heads": heads, "head_dim": head_dim,
              "causal": True},
        layer_id=0, layer_kind=LayerKind.MOE, component=component,
    )


def _model(**kw):
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=256, layers=[LayerKind.DENSE], **kw,
    )


def test_bf16_baseline_unchanged():
    """When all attention dtypes are BF16, the result equals the v1 formula."""
    m = _model()
    op = _make_attn_op(in_dtype=Dtype.BF16)
    cost = _attn_cost(op, m, system=None)
    # FA tile sizing falls back to (128, 64) when sram_bytes=0
    # All Q/K/V/O bytes at bpe=2; v1 used uniform bpe = inputs[0].dtype.bytes = 2
    assert cost.fwd_bytes > 0
    # Compute the same way for sanity — bpe identical → v2 formula == v1 formula
    import math
    Br, Bc = 128, 64
    Tr = math.ceil(256 / Br)
    Tc = math.ceil(256 / Bc)
    tc_eff = (Tc + 1) / 2  # causal
    q_elems = 1 * 4 * 256 * 32
    kv_elems = 1 * 4 * Tr * tc_eff * Bc * 32
    o_elems = q_elems
    expected_fwd = (q_elems + 2 * kv_elems + o_elems) * 2  # all BF16
    assert cost.fwd_bytes == pytest.approx(expected_fwd)


def test_fp8_attn_halves_fwd_bytes():
    """FP8 attention activations → Q/K/V/O bytes halve."""
    m_bf16 = _model()
    m_fp8 = _model(attn_act_dtype=Dtype.FP8_E4M3)
    op_bf16 = _make_attn_op(in_dtype=Dtype.BF16)
    op_fp8 = _make_attn_op(in_dtype=Dtype.FP8_E4M3)
    c_bf16 = _attn_cost(op_bf16, m_bf16, system=None)
    c_fp8 = _attn_cost(op_fp8, m_fp8, system=None)
    assert c_fp8.fwd_bytes == pytest.approx(c_bf16.fwd_bytes / 2)
    assert c_fp8.dx_bytes == pytest.approx(c_bf16.dx_bytes / 2)


def test_attn_flops_unchanged_by_dtype():
    """Quantization changes peak selection, not arithmetic count."""
    m_bf16 = _model()
    m_fp8 = _model(attn_act_dtype=Dtype.FP8_E4M3)
    op = _make_attn_op()
    c_bf16 = _attn_cost(op, m_bf16, system=None)
    c_fp8 = _attn_cost(op, m_fp8, system=None)
    assert c_bf16.fwd_cube_flops == c_fp8.fwd_cube_flops
    assert c_bf16.fwd_vector_flops == c_fp8.fwd_vector_flops


def test_attn_cost_with_explicit_compression_ratio():
    """Sparse-attn meta override works without consulting model.attn_compression_ratio."""
    m = _model()
    op = _make_attn_op(in_dtype=Dtype.BF16)
    op.meta["sparse_topk"] = 64   # bypasses model.attn_compression_ratio path
    op.meta["swa_window"] = 0
    cost = _attn_cost(op, m, system=None)
    assert cost.fwd_bytes > 0
    assert cost.fwd_cube_flops > 0


@pytest.mark.parametrize(
    ("kind", "extra_meta"),
    [
        ("attn_core", {}),
        ("sparse_attn", {"sparse_topk": 64, "swa_window": 16}),
        ("hca_attn", {"compress_ratio": 4, "swa_window": 16}),
        ("swa_attn", {"swa_window": 64}),
    ],
)
def test_mqa_kv_bytes_use_kv_heads(kind, extra_meta):
    """MQA shares one K/V head across Q heads instead of rereading K/V per Q head."""
    model = _model()
    mha = _make_attn_op()
    mha.kind = kind
    mha.meta.update(extra_meta)
    mha.meta["kv_heads"] = 4

    mqa = _make_attn_op()
    mqa.kind = kind
    mqa.meta.update(extra_meta)
    mqa.meta["kv_heads"] = 1

    mha_cost = op_cost(mha, model)
    mqa_cost = op_cost(mqa, model)

    assert mqa_cost.fwd_bytes < mha_cost.fwd_bytes
    assert mqa_cost.dx_bytes < mha_cost.dx_bytes
    assert mqa_cost.fwd_cube_flops == mha_cost.fwd_cube_flops
