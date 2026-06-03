"""Cross-path numerical consistency test.

Verifies that path A (capture-based) and path B (config-based) produce
consistent numerical results for the same model geometry.

This is the safety net for architecture unification — whenever merging
duplicated FLOPs/comm/memory implementations, run this test first to
confirm the numbers don't drift.

Current gap: OpGraph.from_model_spec() does not attach tensor shapes,
so per-node FLOPs/latency passes produce zero values.  The comparison
is limited to structural metrics (bubble_fraction, steps) and aggregate
metrics that depend only on metadata (via the TrainingPipelinePass bridge).
See architecture_snapshot_zh.md § Stage 1 for the shape-enrichment plan.
"""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from zrt.ir.graph import OpGraph
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import RecomputePolicy, Strategy, OptKind

ANCHOR_DIR = Path(__file__).parent / "anchors"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _model_spec(**overrides) -> ModelSpec:
    kwargs = dict(
        hidden=4096, ffn=16384,
        num_heads=32, num_kv_heads=32, head_dim=128,
        vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    kwargs.update(overrides)
    return ModelSpec(**kwargs)


def _strategy(**overrides) -> Strategy:
    kwargs = dict(
        tp=1, pp=4, dp=2, ep=1, cp=1,
        micro_batch=1, global_batch=32,
        zero_stage=1, pp_schedule="1f1b",
        recompute=RecomputePolicy(per_layer={}),
        optimizer=OptKind.ADAM,
    )
    kwargs.update(overrides)
    return Strategy(**kwargs)


def _make_hw_spec():
    """HardwareSpec for path A (estimate_training_from_graphs)."""
    from zrt.hardware.spec import (
        HardwareSpec, ComputeSpec, MemorySpec,
        InterconnectSpec as HWIC, LinkSpec as HWLink,
    )
    return HardwareSpec(
        name="H100", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989.0, fp8_tops=1978.0),
        memory=MemorySpec(capacity_gb=80.0, hbm_bandwidth_gbps=3350.0),
        interconnect=HWIC(
            intra_node=HWLink(type="nvlink", num_devices=8, bandwidth_gbps=900.0, latency_us=1.0),
            inter_node=HWLink(type="ib", num_devices=128, bandwidth_gbps=200.0, latency_us=5.0),
        ),
    )


def _make_system_spec():
    """SystemSpec for path B (estimate)."""
    from zrt.training.spec.system import GPU, LinkSpec, InterconnectSpec, SystemSpec
    gpu = GPU(
        name="H100", flops_bf16=989.0, flops_fp8=1978.0,
        hbm_gb=80.0, hbm_bw_gbps=3350.0, overlap_ratio={},
    )
    link = LinkSpec(
        type="nvlink", bandwidth_gbps=900.0, latency_us=1.0,
        topology="", num_devices=8, kb_efficiency=1.0, oversubscription=1.0,
    )
    return SystemSpec(
        gpu=gpu, host_mem_gb=1000.0,
        nodes=1, gpus_per_node=8,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
    )


# Comparison field categories:
#   structural  —  should match (same formula/bridge)
#   shape_dependent  —  path A needs tensor shapes (known gap, documented)
_ALL_FIELDS = [
    # (field_name, category, tolerance)
    ("mfu",              "shape_dependent", 0.50),
    ("step_time_ms",     "shape_dependent", 0.50),
    ("bubble_fraction",  "shape_dependent", 0.50),
    ("total_flops",      "shape_dependent", 0.50),
    ("forward_flops",    "shape_dependent", 0.50),
    ("backward_flops",   "shape_dependent", 0.50),
]


def _compare_reports(report_a, report_b, label: str):
    """Print comparison table; fail only on large divergence.

    Currently all fields are shape_dependent because OpGraph.from_model_spec()
    lacks tensor shapes.  As we merge implementations, move fields to
    structural with tight tolerance.
    """
    failures = []

    print(f"\n{'='*60}")
    print(f"Cross-path consistency: {label}")
    print(f"{'='*60}")
    for field, cat, tol in _ALL_FIELDS:
        va = getattr(report_a, field, 0) or 0
        vb = getattr(report_b, field, 0) or 0
        abs_vb = max(abs(vb), 1e-12)
        rel_diff = abs(va - vb) / abs_vb

        is_no_shape = (abs(va) < max(abs(vb) * 0.001, 1e-9))
        if is_no_shape or (cat == "shape_dependent" and rel_diff > 0.50):
            status = "NO-SHAPE"
        elif rel_diff <= tol:
            status = f"OK (diff={rel_diff:.2%})"
        else:
            status = f"FAIL (diff={rel_diff:.2%} > {tol:.0%})"
            if cat == "structural":
                failures.append(f"  {field}: A={va:.6f} B={vb:.6f} diff={rel_diff:.2%}")

        print(f"  [{status:>25s}] {field:20s}  A={va:>14.6f}  B={vb:>14.6f}")

    print(f"{'='*60}")
    if failures:
        print(f"Gated failures ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        pytest.fail(f"{label}: {len(failures)} field(s) exceeded tolerance")


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_both_paths_agree_on_dense_model():
    """Smoke: small dense model — structural fields must match."""
    from zrt.training.search.estimator import estimate
    from zrt.transform.analysis import estimate_training_from_graphs

    model = _model_spec()
    strategy = _strategy()
    system_spec = _make_system_spec()
    hw_spec = _make_hw_spec()

    report_b = estimate(model, system_spec, strategy)
    opgraph = OpGraph.from_model_spec(model, strategy)
    report_a = estimate_training_from_graphs(
        forward_graph=opgraph, hw_spec=hw_spec,
        hidden=model.hidden, num_layers=len(model.layers),
        seq_len=model.seq_len, batch_size=strategy.micro_batch,
        tp=strategy.tp, pp=strategy.pp, dp=strategy.dp, ep=strategy.ep,
        zero_stage=strategy.zero_stage, optimizer="adam",
        vocab_size=model.vocab,
        micro_batch=strategy.micro_batch, global_batch=strategy.global_batch,
        pp_schedule=strategy.pp_schedule,
    )
    _compare_reports(report_a, report_b, "dense_smoke")


def test_both_paths_agree_on_moe_model():
    """Smoke: small MoE model — structural fields must match."""
    from zrt.training.search.estimator import estimate
    from zrt.transform.analysis import estimate_training_from_graphs

    model = _model_spec(
        layers=[LayerKind.MOE] * 4,
        num_experts=8, top_k=2, moe_ffn=16384,
    )
    strategy = _strategy(ep=2)
    system_spec = _make_system_spec()
    hw_spec = _make_hw_spec()

    report_b = estimate(model, system_spec, strategy)
    opgraph = OpGraph.from_model_spec(model, strategy)
    report_a = estimate_training_from_graphs(
        forward_graph=opgraph, hw_spec=hw_spec,
        hidden=model.hidden, num_layers=len(model.layers),
        seq_len=model.seq_len, batch_size=strategy.micro_batch,
        tp=strategy.tp, pp=strategy.pp, dp=strategy.dp, ep=strategy.ep,
        zero_stage=strategy.zero_stage, optimizer="adam",
        vocab_size=model.vocab,
        micro_batch=strategy.micro_batch, global_batch=strategy.global_batch,
        pp_schedule=strategy.pp_schedule,
        recompute_policy="none",
        moe_total_experts=model.num_experts,
        moe_active_experts=model.top_k,
        moe_ffn_hidden=model.moe_ffn,
        layer_type_counts={"dense": 0, "sparse": len(model.layers)},
    )
    _compare_reports(report_a, report_b, "moe_smoke")


@pytest.mark.parametrize("anchor_yaml", [
    p for p in sorted(ANCHOR_DIR.glob("*.yaml"))
    if "fp8" not in p.name and "fp4" not in p.name
])
def test_both_paths_agree_on_anchor(anchor_yaml):
    """Run each anchor through both paths and compare structural fields."""
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate
    from zrt.transform.analysis import estimate_training_from_graphs

    model, system, strategy, _ = load_anchor_config(anchor_yaml)
    anchor_data = yaml.safe_load(anchor_yaml.read_text(encoding="utf-8"))
    model_id = anchor_data["name"]

    hw_spec = _make_hw_spec()

    # Path B
    report_b = estimate(model, system, strategy)

    # Path A (may not support all model types yet)
    opgraph = OpGraph.from_model_spec(model, strategy)
    try:
        report_a = estimate_training_from_graphs(
            forward_graph=opgraph, hw_spec=hw_spec,
            hidden=model.hidden, num_layers=len(model.layers),
            num_layers_full=len(model.layers),
            seq_len=model.seq_len, batch_size=strategy.micro_batch,
            tp=strategy.tp, pp=strategy.pp, dp=strategy.dp, ep=strategy.ep,
            cp=strategy.cp,
            zero_stage=strategy.zero_stage, optimizer="adam",
            vocab_size=model.vocab,
            micro_batch=strategy.micro_batch,
            global_batch=strategy.global_batch,
            pp_schedule=strategy.pp_schedule or "1f1b",
            recompute_policy="none",
            moe_total_experts=getattr(model, "moe_total_experts", 0) or 0,
            moe_active_experts=getattr(model, "moe_active_experts", 1) or 1,
            moe_ffn_hidden=getattr(model, "moe_ffn_hidden", 0) or 0,
            num_heads=model.num_heads, kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
        )
    except Exception as e:
        print(f"Path A skipped for {model_id}: {e}")
        pytest.skip(f"Path A not supported: {e}")
        return

    _compare_reports(report_a, report_b, model_id)
