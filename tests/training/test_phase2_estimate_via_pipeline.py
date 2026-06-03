"""路径 A（抓图建模）：estimate_via_pipeline() 走 Transform Pipeline 路径测试。

验证 estimate_via_pipeline() 通过 build_captured_graph → build_context → build_default_pipeline
完整走通 Transform Pipeline，产出有效的 TrainingReport。

原理：
- estimate_via_pipeline() 将 ModelSpec+SystemSpec+Strategy 转为 OpGraph+TransformContext
- 通过 4 阶段 Transform Pipeline (split→fuse→optim→analyze) 变换
- 从变换后 OpGraph.metadata 提取 step_result/training_flops/memory_breakdown
- 构建 TrainingReport 返回

观测点：
- 返回的 TrainingReport 各字段非零且合理
- estimate() 委托到 estimate_via_pipeline()，结果 bit-identical
- Dense 和 MoE 模型均能走通
- 不同并行策略 (TP/PP/DP) 行为正确
"""
from __future__ import annotations

import pytest

from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.system import GPU, InterconnectSpec, LinkSpec, SystemSpec


def _make_gpu():
    return GPU(
        name="nvidia_h100_sxm",
        flops_bf16=989.0, flops_fp8=1979.0,
        hbm_gb=80.0, hbm_bw_gbps=3350.0,
    )


def _make_system(gpus_per_node=8):
    gpu = _make_gpu()
    link = LinkSpec(
        type="nvlink", bandwidth_gbps=900.0, latency_us=1.0,
        topology="", num_devices=8, kb_efficiency=1.0, oversubscription=1.0,
    )
    return SystemSpec(
        gpu=gpu, host_mem_gb=512.0,
        nodes=1, gpus_per_node=gpus_per_node,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
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


def _make_moe_model(n_layers=4, **kwargs):
    defaults = dict(
        hidden=2048, ffn=8192, moe_ffn=2048, seq_len=1024,
        num_heads=16, num_kv_heads=16, head_dim=128,
        layers=[LayerKind.DENSE] * 2 + [LayerKind.MOE] * 2,
        num_experts=8, top_k=2, n_shared_experts=1,
        vocab=32000, act_dtype=Dtype.BF16,
    )
    defaults.update(kwargs)
    return ModelSpec(**defaults)


def _make_strategy(**kwargs):
    defaults = dict(
        tp=1, pp=1, ep=1, dp=8, cp=1,
        micro_batch=1, global_batch=32,
    )
    defaults.update(kwargs)
    return Strategy(**defaults)


class TestEstimateViaPipelineBasic:
    """estimate_via_pipeline() 基础功能验证。"""

    def test_returns_training_report(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert isinstance(report, TrainingReport)

    def test_step_time_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.step_time_ms > 0, f"step_time_ms={report.step_time_ms}"

    def test_mfu_in_range(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert 0 < report.mfu <= 1.0, f"mfu={report.mfu}"

    def test_training_flops_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.training_flops > 0, f"training_flops={report.training_flops}"

    def test_pipeline_time_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.pipeline_time_ms > 0, f"pipeline_time_ms={report.pipeline_time_ms}"

    def test_schedule_name_set(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.schedule_name != "", f"schedule_name={report.schedule_name!r}"

    def test_forward_flops_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.forward_flops > 0, f"forward_flops={report.forward_flops}"

    def test_backward_flops_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.backward_flops > 0, f"backward_flops={report.backward_flops}"


class TestEstimateDelegation:
    """验证 estimate() 委托到 estimate_via_pipeline()，结果 bit-identical。"""

    def test_step_time_identical(self):
        from zrt.training.search.estimator import estimate, estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)
        a = estimate(model, system, strategy)
        b = estimate_via_pipeline(model, system, strategy)
        assert a.step_time_ms == pytest.approx(b.step_time_ms, rel=1e-9), \
            f"estimate={a.step_time_ms}, via_pipeline={b.step_time_ms}"

    def test_mfu_identical(self):
        from zrt.training.search.estimator import estimate, estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)
        a = estimate(model, system, strategy)
        b = estimate_via_pipeline(model, system, strategy)
        assert a.mfu == pytest.approx(b.mfu, rel=1e-9), \
            f"estimate={a.mfu}, via_pipeline={b.mfu}"

    def test_total_flops_identical(self):
        from zrt.training.search.estimator import estimate, estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)
        a = estimate(model, system, strategy)
        b = estimate_via_pipeline(model, system, strategy)
        assert a.total_flops == pytest.approx(b.total_flops, rel=1e-9), \
            f"estimate={a.total_flops}, via_pipeline={b.total_flops}"

    def test_pipeline_time_identical(self):
        from zrt.training.search.estimator import estimate, estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=2, dp=2)
        a = estimate(model, system, strategy)
        b = estimate_via_pipeline(model, system, strategy)
        assert a.pipeline_time_ms == pytest.approx(b.pipeline_time_ms, rel=1e-9), \
            f"estimate={a.pipeline_time_ms}, via_pipeline={b.pipeline_time_ms}"


class TestPipelineBehavior:
    """验证不同并行策略下的行为正确性。"""

    def test_pp1_no_bubble(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)
        report = estimate_via_pipeline(model, system, strategy)
        assert report.bubble_fraction == pytest.approx(0.0, abs=1e-6), \
            f"PP=1 bubble_fraction={report.bubble_fraction}"

    def test_pp2_has_bubble(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=2, dp=2)
        report = estimate_via_pipeline(model, system, strategy)
        assert report.bubble_fraction > 0, \
            f"PP=2 bubble_fraction={report.bubble_fraction}"

    def test_tp2_produces_valid_result(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model(n_layers=16)
        system = _make_system()
        s2 = _make_strategy(tp=2, pp=1, dp=4)
        r2 = estimate_via_pipeline(model, system, s2)
        assert r2.step_time_ms > 0
        assert r2.mfu > 0
        assert r2.tp_exposed_ms > 0 or r2.tp_hidden_ms > 0, \
            "TP=2 should have some TP communication"

    def test_more_layers_more_flops(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        system = _make_system()
        m4 = _make_model(n_layers=4)
        m8 = _make_model(n_layers=8)
        s = _make_strategy()
        r4 = estimate_via_pipeline(m4, system, s)
        r8 = estimate_via_pipeline(m8, system, s)
        assert r8.training_flops > r4.training_flops, \
            f"4L flops={r4.training_flops}, 8L flops={r8.training_flops}"


class TestMoESupport:
    """验证 MoE 模型能走通 Transform Pipeline。"""

    def test_moe_returns_valid_report(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_moe_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert isinstance(report, TrainingReport)
        assert report.step_time_ms > 0

    def test_moe_flops_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_moe_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.training_flops > 0

    def test_moe_with_ep(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_moe_model()
        system = _make_system()
        strategy = _make_strategy(tp=1, pp=1, ep=2, dp=8)
        report = estimate_via_pipeline(model, system, strategy)
        assert report.step_time_ms > 0
        assert report.mfu > 0


class TestConfigSummary:
    """验证 config_summary 和 warnings 字段。"""

    def test_config_summary_is_dict(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=2, dp=2)
        report = estimate_via_pipeline(model, system, strategy)
        assert isinstance(report.config_summary, dict), \
            f"config_summary type={type(report.config_summary)}"

    def test_config_summary_has_strategy(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=2, dp=2)
        report = estimate_via_pipeline(model, system, strategy)
        assert "strategy" in report.config_summary, \
            f"keys={list(report.config_summary.keys())}"

    def test_warnings_is_list(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert isinstance(report.warnings, list)


class TestDerivedMetrics:
    """验证派生指标计算正确。"""

    def test_tokens_per_sec_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.tokens_per_sec > 0, f"tokens_per_sec={report.tokens_per_sec}"

    def test_flops_per_token_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.flops_per_token > 0, f"flops_per_token={report.flops_per_token}"

    def test_effective_params_positive(self):
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        assert report.effective_params > 0, f"effective_params={report.effective_params}"

    def test_tokens_per_sec_formula(self):
        """tokens_per_sec = tokens / pipeline_time_s"""
        from zrt.training.search.estimator import estimate_via_pipeline
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy()
        report = estimate_via_pipeline(model, system, strategy)
        tokens = strategy.global_batch * model.seq_len
        pipeline_time_s = report.pipeline_time_ms / 1000.0
        expected = tokens / pipeline_time_s if pipeline_time_s > 0 else 0.0
        assert report.tokens_per_sec == pytest.approx(expected, rel=1e-6), \
            f"expected={expected}, got={report.tokens_per_sec}"
