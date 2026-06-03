"""Tests for Phase 2: unified Transform Pipeline path.

Verifies ``estimate_via_pipeline()`` and ``build_context()`` produce
correct results satisfying the Stage 2-4 contracts (refactor.md §4.2-4.4).
"""

import pytest

from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec, GPU
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.report import TrainingReport
from zrt.hardware.spec import InterconnectSpec


def _make_gpu():
    return GPU(
        name="nvidia_h100_sxm", flops_bf16=989.0, flops_fp8=1979.0,
        hbm_gb=80.0, hbm_bw_gbps=3350.0,
    )


def _make_system(gpus=1):
    return SystemSpec(
        gpu=_make_gpu(), host_mem_gb=512.0,
        interconnect=InterconnectSpec(),
        nodes=1, gpus_per_node=gpus,
    )


def _make_model(n_layers=4, **kwargs):
    defaults = dict(
        hidden=4096, ffn=11008, seq_len=2048,
        num_heads=32, num_kv_heads=32, head_dim=128,
        layers=[LayerKind.DENSE] * n_layers,
        vocab=32000, act_dtype=Dtype.BF16,
    )
    defaults.update(kwargs)
    return ModelSpec(**defaults)


def _make_strategy(**kwargs):
    defaults = dict(tp=1, pp=1, ep=1, dp=1, cp=1, micro_batch=1, global_batch=32)
    defaults.update(kwargs)
    return Strategy(**defaults)


class TestContextBuilder:
    """Verify ModelSpec + SystemSpec + Strategy → TransformContext mapping."""

    def test_parallel_config_matches_strategy(self):
        from zrt.training.ir.context_builder import build_context
        ctx = build_context(
            _make_model(), _make_system(),
            _make_strategy(tp=2, pp=4, ep=1, dp=1, cp=1),
        )
        assert ctx.parallel.tp == 2
        assert ctx.parallel.pp == 4
        assert ctx.parallel.ep == 1
        assert ctx.parallel.dp == 1
        assert ctx.parallel.cp == 1

    def test_training_config_matches_strategy(self):
        from zrt.training.ir.context_builder import build_context
        ctx = build_context(
            _make_model(), _make_system(),
            _make_strategy(zero_stage=3, micro_batch=2, global_batch=64),
        )
        assert ctx.training.zero_stage == 3
        assert ctx.training.micro_batch == 2
        assert ctx.training.global_batch == 64

    def test_hw_spec_resolved(self):
        from zrt.training.ir.context_builder import build_context
        ctx = build_context(_make_model(), _make_system(), _make_strategy())
        assert ctx.hw_spec is not None
        assert "H100" in ctx.hw_spec.name or "h100" in ctx.hw_spec.name.lower()

    def test_hw_spec_fallback_for_unknown_gpu(self):
        from zrt.training.ir.context_builder import build_context
        gpu = GPU(name="unknown_gpu", flops_bf16=100.0, flops_fp8=200.0,
                  hbm_gb=40.0, hbm_bw_gbps=1000.0)
        system = SystemSpec(gpu=gpu, host_mem_gb=256.0,
                            interconnect=InterconnectSpec(),
                            nodes=1, gpus_per_node=1)
        ctx = build_context(_make_model(), system, _make_strategy())
        assert ctx.hw_spec is not None

    def test_moe_profile_attached(self):
        from zrt.training.ir.context_builder import build_context
        model = _make_model(
            n_layers=2,
            layers=[LayerKind.DENSE, LayerKind.MOE],
            num_experts=8, top_k=2, moe_ffn=2048,
        )
        ctx = build_context(model, _make_system(), _make_strategy())
        assert ctx.profile is not None
        assert ctx.profile.num_experts == 8
        assert ctx.profile.moe_active == 2

    def test_seq_len_and_hidden_set(self):
        from zrt.training.ir.context_builder import build_context
        ctx = build_context(
            _make_model(seq_len=8192, hidden=8192),
            _make_system(), _make_strategy(),
        )
        assert ctx.training.seq_len == 8192
        assert ctx.training.hidden == 8192


class TestEstimateViaPipeline:
    """Verify estimate_via_pipeline() produces valid TrainingReport."""

    def test_returns_training_report(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        report = estimate_via_pipeline(
            _make_model(), _make_system(), _make_strategy(),
        )
        assert isinstance(report, TrainingReport)

    def test_step_time_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        report = estimate_via_pipeline(
            _make_model(), _make_system(), _make_strategy(),
        )
        assert report.step_time_ms > 0

    def test_mfu_in_range(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        report = estimate_via_pipeline(
            _make_model(), _make_system(), _make_strategy(),
        )
        assert 0 <= report.mfu <= 1

    def test_training_flops_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        report = estimate_via_pipeline(
            _make_model(), _make_system(), _make_strategy(),
        )
        assert report.training_flops > 0

    def test_memory_breakdown_present(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        report = estimate_via_pipeline(
            _make_model(), _make_system(), _make_strategy(zero_stage=0),
        )
        assert bool(report.memory_breakdown)

    def test_pipeline_metrics_present(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        report = estimate_via_pipeline(
            _make_model(), _make_system(), _make_strategy(),
        )
        assert report.bubble_fraction >= 0

    def test_pp_bubble_increases(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        system_4 = _make_system(gpus=4)
        r1 = estimate_via_pipeline(
            _make_model(), system_4,
            _make_strategy(pp=1, dp=4),
        )
        r4 = estimate_via_pipeline(
            _make_model(), system_4,
            _make_strategy(pp=4, dp=1),
        )
        assert r4.bubble_fraction >= r1.bubble_fraction

    def test_tp_reduces_compute(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        system_2 = _make_system(gpus=2)
        r1 = estimate_via_pipeline(
            _make_model(), system_2,
            _make_strategy(tp=1, dp=2),
        )
        r2 = estimate_via_pipeline(
            _make_model(), system_2,
            _make_strategy(tp=2, dp=1),
        )
        assert r2.fwd_compute_ms < r1.fwd_compute_ms

    def test_more_layers_more_flops(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        r4 = estimate_via_pipeline(
            _make_model(n_layers=4), _make_system(), _make_strategy(),
        )
        system_8 = _make_system(gpus=8)
        r8 = estimate_via_pipeline(
            _make_model(n_layers=8), system_8,
            _make_strategy(dp=8),
        )
        assert r8.training_flops > r4.training_flops
