from __future__ import annotations

import pytest

from zrt.training.ir.builders import build_graph
from zrt.training.ir.training_graph import Op
from zrt.training.models.flops import op_cost
from zrt.training.models.mega_moe import (
    _mega_moe_dispatch_bytes,
    infer_quant_variant,
    mega_moe_cost_terms,
    mega_moe_cost_terms_from_meta,
    resolve_mega_moe_waves,
    select_mega_moe_waves_by_pipeline,
    simulate_wave_pipeline,
)
from zrt.training.spec.dtype import Dtype
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


def _mega_moe_op(**meta) -> Op:
    base = {
        "m": 128,
        "n": 1024,
        "k": 2048,
        "micro_batch": 3,
        "num_experts": 8,
        "top_k": 2,
        "requested_waves": 4,
        "act_bytes": 2,
        "out_bytes": 2,
        "weight_bytes": 2,
        "weight_stored_bytes": 2,
        "quant_variant": "standard",
        "fwd_multiplier": 3,
        "swiglu_clamp": None,
        "fused_dispatch_compute_combine": True,
    }
    base.update(meta)
    return Op(name="L0.mega_moe", kind="mega_moe", meta=base)


def test_infer_quant_variant_standard_for_bf16():
    model = _moe_model()

    assert infer_quant_variant(model) == "standard"


def test_infer_quant_variant_w4a8_for_fp4_weights_and_fp8_moe_acts():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )

    assert infer_quant_variant(model) == "w4a8"


def test_resolve_waves_prefers_valid_divisor_not_above_target():
    assert resolve_mega_moe_waves(requested=6, hardware_waves=0, experts_per_rank=8) == 4
    assert resolve_mega_moe_waves(requested=0, hardware_waves=4, experts_per_rank=8) == 4
    assert resolve_mega_moe_waves(requested=0, hardware_waves=0, experts_per_rank=3) == 3


def test_wave_pipeline_one_wave_is_serial():
    result = simulate_wave_pipeline(waves=1, dispatch_s=1.0, compute_s=3.0, combine_s=2.0)

    assert result.total_s == pytest.approx(6.0)
    assert result.exposed_comm_s == pytest.approx(3.0)
    assert result.hidden_comm_s == pytest.approx(0.0)


def test_wave_pipeline_more_waves_hide_comm_when_compute_dominates():
    one = simulate_wave_pipeline(waves=1, dispatch_s=1.0, compute_s=6.0, combine_s=1.0)
    four = simulate_wave_pipeline(waves=4, dispatch_s=0.25, compute_s=1.5, combine_s=0.25)

    assert four.total_s < one.total_s
    assert four.hidden_comm_s > 0.0


def test_auto_wave_selector_chooses_best_pipeline_divisor():
    waves = select_mega_moe_waves_by_pipeline(
        experts_per_rank=8,
        fwd_compute_s=8.0,
        bwd_compute_s=16.0,
        dispatch_s=4.0,
        combine_s=4.0,
    )

    assert waves == 8


def test_op_cost_returns_nonzero_flops_and_bytes_for_mega_moe():
    cost = op_cost(_mega_moe_op(), _moe_model())

    assert cost.fwd_cube_flops > 0
    assert cost.dx_cube_flops > 0
    assert cost.dw_cube_flops > 0
    assert cost.fwd_bytes > 0
    assert cost.dx_bytes > 0
    assert cost.dw_bytes > 0


def test_mega_moe_flops_scale_with_tokens_topk_dimensions_and_multiplier():
    op = _mega_moe_op(m=11, micro_batch=5, top_k=4, n=13, k=17, fwd_multiplier=7)

    terms = mega_moe_cost_terms(op)

    assert terms.tokens == 55
    assert terms.fwd_flops == 2 * 55 * 4 * 17 * 13 * 7


def test_mega_moe_cost_terms_matches_mapping_helper_for_equivalent_meta():
    op = _mega_moe_op(
        m=11,
        micro_batch=5,
        top_k=4,
        n=13,
        k=17,
        fwd_multiplier=7,
        experts_per_rank=3,
    )

    assert mega_moe_cost_terms(op) == mega_moe_cost_terms_from_meta(op.meta)


def test_mega_moe_cost_terms_from_meta_does_not_mutate_input_mapping():
    meta = dict(
        _mega_moe_op(
            m=8,
            micro_batch=2,
            top_k=3,
            n=16,
            k=32,
            fwd_multiplier=9,
        ).meta
    )
    before = dict(meta)

    mega_moe_cost_terms_from_meta(meta)

    assert meta == before


def test_mega_moe_cost_terms_from_meta_preserves_legacy_multiplier_normalization():
    terms = mega_moe_cost_terms_from_meta(
        _mega_moe_op(m=8, micro_batch=2, top_k=3, n=16, k=32, fwd_multiplier=9).meta
    )

    assert terms.fwd_multiplier == 3
    assert terms.fwd_flops == 2 * 16 * 3 * 32 * 16 * 3


def test_mega_moe_cost_preserves_legacy_builder_multiplier_intent():
    model = _moe_model(top_k=2)
    graph = build_graph(model, Strategy(mega_moe=True, micro_batch=3))
    op = [op for op in graph.ops if op.kind == "mega_moe"][0]

    terms = mega_moe_cost_terms(op)

    assert op.meta["fwd_multiplier"] == 3 * model.top_k
    assert terms.fwd_multiplier == 3
    assert terms.fwd_flops == 2 * 3 * model.seq_len * model.top_k * model.moe_ffn * model.hidden * 3


def test_mega_moe_bytes_include_visible_activations_and_stored_weight_traffic():
    op = _mega_moe_op(
        m=16,
        micro_batch=2,
        n=32,
        k=64,
        num_experts=3,
        act_bytes=2,
        out_bytes=4,
        weight_bytes=2,
        weight_stored_bytes=1,
        fwd_multiplier=3,
    )

    terms = mega_moe_cost_terms(op)

    assert terms.activation_input_bytes == 16 * 2 * 32 * 2
    assert terms.activation_output_bytes == 16 * 2 * 32 * 4
    assert terms.weight_bytes == 3 * 64 * 32 * 3 * 1
    assert terms.fwd_bytes == (
        terms.activation_input_bytes
        + terms.activation_output_bytes
        + terms.weight_bytes
    )


def test_mega_moe_dispatch_uses_internal_moe_activation_bytes_not_boundary_bytes():
    op = _mega_moe_op(
        m=16,
        micro_batch=2,
        n=32,
        top_k=4,
        act_bytes=2,
        moe_act_bytes=1,
        out_bytes=2,
        ep=4,
    )

    terms = mega_moe_cost_terms(op)

    assert terms.activation_input_bytes == 16 * 2 * 32 * 2
    assert terms.moe_activation_input_bytes == 16 * 2 * 32 * 1
    assert _mega_moe_dispatch_bytes(terms, ep=4) == 16 * 2 * 32 * 1 * 4


def test_mega_moe_cost_uses_local_k_and_local_experts_when_sharded():
    terms = mega_moe_cost_terms(
        _mega_moe_op(k=64, k_local=16, num_experts=8, experts_per_rank=2, fwd_multiplier=3)
    )

    assert terms.k_eff == 16
    assert terms.local_experts == 2
    assert terms.fwd_flops == 2 * terms.tokens * 2 * 16 * 1024 * 3
    assert terms.weight_bytes == 2 * 16 * 1024 * 3 * 2


def test_mega_moe_cost_uses_local_hidden_for_activations_not_compute_when_tp_sharded():
    terms = mega_moe_cost_terms(
        _mega_moe_op(
            n=1024,
            n_local=256,
            k=2048,
            k_local=512,
            top_k=2,
            num_experts=8,
            experts_per_rank=2,
            micro_batch=1,
            m=128,
            moe_act_bytes=1,
            act_bytes=2,
            out_bytes=2,
            weight_stored_bytes=2,
            fwd_multiplier=3,
        )
    )

    assert terms.n == 1024
    assert terms.activation_n == 256
    assert terms.k_eff == 512
    assert terms.fwd_flops == 2 * 128 * 2 * 512 * 1024 * 3
    assert terms.weight_bytes == 2 * 512 * 1024 * 3 * 2
    assert terms.activation_input_bytes == 128 * 256 * 2
    assert terms.activation_output_bytes == 128 * 256 * 2
    assert terms.moe_activation_input_bytes == 128 * 256 * 1
    assert _mega_moe_dispatch_bytes(terms, ep=4) == 128 * 256 * 1 * 2


def test_mega_moe_tp_compute_matches_legacy_routed_expert_compute():
    model = _moe_model()
    strategy = Strategy(tp=4, ep=4, mega_moe=True)
    mega_graph = build_graph(model, strategy)
    routed_graph = build_graph(model, Strategy(tp=4, ep=4, mega_moe=False))

    mega_moe = [op for op in mega_graph.ops if op.kind == "mega_moe"][0]
    routed = [op for op in routed_graph.ops if op.name == "L0.routed_expert_ffn"][0]

    mega_cost = op_cost(mega_moe, model)
    routed_cost = op_cost(routed, model)
    terms = mega_moe_cost_terms(mega_moe)

    assert terms.n == model.hidden
    assert terms.activation_n == model.hidden // strategy.tp
    assert mega_cost.fwd_cube_flops == routed_cost.fwd_cube_flops
    assert mega_cost.dx_cube_flops == routed_cost.dx_cube_flops
    assert mega_cost.dw_cube_flops == routed_cost.dw_cube_flops
    assert _mega_moe_dispatch_bytes(terms, strategy.ep) == (
        strategy.micro_batch
        * model.seq_len
        * (model.hidden // strategy.tp)
        * model.top_k
        * model.effective_moe_act_dtype().bytes
    )


def test_mega_moe_w4a8_bytes_use_stored_fp4_weight_bytes():
    op = _mega_moe_op(
        quant_variant="w4a8",
        moe_act_dtype=Dtype.FP8_E4M3,
        moe_act_bytes=Dtype.FP8_E4M3.bytes,
        weight_bytes=Dtype.FP8_E4M3.bytes,
        weight_stored_bytes=Dtype.FP4.stored_bytes,
        num_experts=4,
        n=32,
        k=64,
        fwd_multiplier=3,
    )

    terms = mega_moe_cost_terms(op)

    assert terms.quant_variant == "w4a8"
    assert terms.weight_bytes == 4 * 64 * 32 * 3 * Dtype.FP4.stored_bytes
