"""Phase 5B UT: verify estimate() and estimate_via_pipeline() produce identical results.

Since Phase 6 unified the paths (estimate() delegates to estimate_via_pipeline()),
both functions now go through the same code path and must produce bit-identical results.

Principle: estimate() IS estimate_via_pipeline() — they are the same function.
"""
import pytest

from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, InterconnectSpec, LinkSpec, SystemSpec


def _make_model():
    return ModelSpec(
        hidden=4096, ffn=11008, seq_len=2048,
        num_heads=32, num_kv_heads=32, head_dim=128,
        layers=[LayerKind.DENSE] * 4, vocab=32000,
    )


def _make_system():
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


def _make_strategy(tp=2, pp=2, dp=2):
    return Strategy(
        tp=tp, pp=pp, ep=1, dp=dp, cp=1,
        micro_batch=1, global_batch=32,
    )


class TestPathUnification:
    """estimate() and estimate_via_pipeline() must produce identical results."""

    @pytest.fixture
    def reports(self):
        from zrt.training.search.estimator import estimate, estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        a = estimate(model, system, strategy)
        b = estimate_via_pipeline(model, system, strategy)
        return a, b

    def test_step_time_identical(self, reports):
        a, b = reports
        assert a.step_time_ms == pytest.approx(b.step_time_ms, rel=1e-9)

    def test_pipeline_time_identical(self, reports):
        a, b = reports
        assert a.pipeline_time_ms == pytest.approx(b.pipeline_time_ms, rel=1e-9)

    def test_mfu_identical(self, reports):
        a, b = reports
        assert a.mfu == pytest.approx(b.mfu, rel=1e-9)

    def test_hfu_identical(self, reports):
        a, b = reports
        assert a.hfu == pytest.approx(b.hfu, rel=1e-9)

    def test_total_flops_identical(self, reports):
        a, b = reports
        assert a.total_flops == pytest.approx(b.total_flops, rel=1e-9)

    def test_forward_flops_identical(self, reports):
        a, b = reports
        assert a.forward_flops == pytest.approx(b.forward_flops, rel=1e-9)

    def test_backward_flops_identical(self, reports):
        a, b = reports
        assert a.backward_flops == pytest.approx(b.backward_flops, rel=1e-9)

    def test_bubble_fraction_identical(self, reports):
        a, b = reports
        assert a.bubble_fraction == pytest.approx(b.bubble_fraction, rel=1e-9)

    def test_compute_time_identical(self, reports):
        a, b = reports
        assert a.compute_time_ms == pytest.approx(b.compute_time_ms, rel=1e-9)

    def test_fwd_compute_identical(self, reports):
        a, b = reports
        assert a.fwd_compute_ms == pytest.approx(b.fwd_compute_ms, rel=1e-9)

    def test_bwd_compute_identical(self, reports):
        a, b = reports
        assert a.bwd_compute_ms == pytest.approx(b.bwd_compute_ms, rel=1e-9)

    def test_exposed_comm_identical(self, reports):
        a, b = reports
        assert a.exposed_comm_ms == pytest.approx(b.exposed_comm_ms, rel=1e-9)

    def test_hidden_comm_identical(self, reports):
        a, b = reports
        assert a.hidden_comm_ms == pytest.approx(b.hidden_comm_ms, rel=1e-9)

    def test_tp_exposed_identical(self, reports):
        a, b = reports
        assert a.tp_exposed_ms == pytest.approx(b.tp_exposed_ms, rel=1e-9)

    def test_pp_exposed_identical(self, reports):
        a, b = reports
        assert a.pp_exposed_ms == pytest.approx(b.pp_exposed_ms, rel=1e-9)

    def test_dp_exposed_identical(self, reports):
        a, b = reports
        assert a.dp_exposed_ms == pytest.approx(b.dp_exposed_ms, rel=1e-9)

    def test_optimizer_time_identical(self, reports):
        a, b = reports
        assert a.optimizer_time_ms == pytest.approx(b.optimizer_time_ms, rel=1e-9)

    def test_warmup_ms_identical(self, reports):
        a, b = reports
        assert a.warmup_ms == pytest.approx(b.warmup_ms, rel=1e-9)

    def test_steady_ms_identical(self, reports):
        a, b = reports
        assert a.steady_ms == pytest.approx(b.steady_ms, rel=1e-9)

    def test_cooldown_ms_identical(self, reports):
        a, b = reports
        assert a.cooldown_ms == pytest.approx(b.cooldown_ms, rel=1e-9)

    def test_tokens_per_sec_identical(self, reports):
        a, b = reports
        assert a.tokens_per_sec == pytest.approx(b.tokens_per_sec, rel=1e-9)

    def test_flops_per_token_identical(self, reports):
        a, b = reports
        assert a.flops_per_token == pytest.approx(b.flops_per_token, rel=1e-9)

    def test_memory_present(self, reports):
        a, b = reports
        assert a.memory is not None
        assert a.memory.total > 0

    def test_schedule_name_identical(self, reports):
        a, b = reports
        assert a.schedule_name == b.schedule_name

    def test_hbm_traffic_identical(self, reports):
        a, b = reports
        assert a.weight_hbm_gb == pytest.approx(b.weight_hbm_gb, rel=1e-9)
        assert a.act_hbm_gb == pytest.approx(b.act_hbm_gb, rel=1e-9)
        assert a.grad_hbm_gb == pytest.approx(b.grad_hbm_gb, rel=1e-9)
        assert a.cast_hbm_gb == pytest.approx(b.cast_hbm_gb, rel=1e-9)


class TestPipelineAccuracy:
    """Verify pipeline produces accurate results across different configurations."""

    def test_pp1_no_bubble(self):
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        strategy = Strategy(tp=2, pp=1, ep=1, dp=4, cp=1,
                            micro_batch=1, global_batch=32)
        report = estimate(model, system, strategy)
        assert report.bubble_fraction == pytest.approx(0.0, abs=1e-6)

    def test_pp2_has_bubble(self):
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        strategy = Strategy(tp=2, pp=2, ep=1, dp=2, cp=1,
                            micro_batch=1, global_batch=32)
        report = estimate(model, system, strategy)
        assert report.bubble_fraction > 0

    def test_tp_produces_valid_step_time(self):
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        s2 = Strategy(tp=2, pp=1, ep=1, dp=4, cp=1,
                      micro_batch=1, global_batch=32)
        r2 = estimate(model, system, s2)
        assert r2.step_time_ms > 0
        assert r2.mfu > 0

    def test_mfu_in_range(self):
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate(model, system, strategy)
        assert 0 < report.mfu <= 1.0

    def test_step_time_positive(self):
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate(model, system, strategy)
        assert report.step_time_ms > 0
