"""Phase C: Pipeline 内部统一测试

验证 Stack A 和 Stack B 走同一条 DAGScheduler 路径，不再有 _original_graph 分支。
"""
from __future__ import annotations

import pytest

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec, GPU
from zrt.hardware.spec import InterconnectSpec


def _make_gpu():
    return GPU(
        name="nvidia_h100_sxm", flops_bf16=989.0, flops_fp8=1979.0,
        hbm_gb=80.0, hbm_bw_gbps=3350.0,
    )


def _make_system(gpus=8):
    return SystemSpec(
        gpu=_make_gpu(), host_mem_gb=512.0,
        interconnect=InterconnectSpec(),
        nodes=1, gpus_per_node=gpus,
    )


def _make_model(n_layers=4):
    return ModelSpec(
        hidden=4096, ffn=11008, seq_len=2048,
        num_heads=32, num_kv_heads=32, head_dim=128,
        layers=[LayerKind.DENSE] * n_layers,
        vocab=32000, act_dtype=Dtype.BF16,
    )


def _make_strategy(tp=1, pp=1, ep=1, dp=1):
    return Strategy(tp=tp, pp=pp, ep=ep, dp=dp, cp=1,
                    micro_batch=1, global_batch=32)


class TestNoOriginalGraphMetadata:
    """验证 build_opgraph_direct 不再设置 _original_graph metadata"""

    def test_no_original_graph_in_opgraph(self):
        """build_opgraph 产出的 OpGraph 不应有 _original_graph metadata"""
        from zrt.training.ir.opgraph_builder import build_opgraph
        model = _make_model()
        strategy = _make_strategy()
        g = build_opgraph(model, strategy)
        assert g.metadata.get("_original_graph") is None, \
            "Phase C: _original_graph metadata should be removed"

    def test_no_model_spec_in_opgraph(self):
        """build_opgraph 产出的 OpGraph 不应有 _model_spec metadata"""
        from zrt.training.ir.opgraph_builder import build_opgraph
        model = _make_model()
        strategy = _make_strategy()
        g = build_opgraph(model, strategy)
        assert g.metadata.get("_model_spec") is None, \
            "Phase C: _model_spec metadata should be removed from OpGraph"


class TestUnifiedPipelinePath:
    """验证 Stack A 走 DAGScheduler 路径（与 Stack B 统一）"""

    def test_estimate_produces_valid_report(self):
        """estimate() 应产出有效的 TrainingReport"""
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)
        report = estimate(model, system, strategy)
        assert report.step_time_ms > 0
        assert 0 <= report.mfu <= 1

    def test_estimate_pp2_produces_bubble(self):
        """PP=2 时应有 bubble"""
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        s_pp1 = _make_strategy(tp=2, pp=1, dp=4)
        s_pp2 = Strategy(tp=2, pp=2, ep=1, dp=2, cp=1,
                         micro_batch=1, global_batch=32)
        r1 = estimate(model, system, s_pp1)
        r2 = estimate(model, system, s_pp2)
        assert r2.bubble_fraction >= r1.bubble_fraction

    def test_pipeline_metrics_present(self):
        """Pipeline 应产出 pipeline_metrics"""
        from zrt.training.search.estimator import estimate
        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)
        report = estimate(model, system, strategy)
        assert report.pipeline_time_ms > 0
        assert report.schedule_name != ""

    def test_flops_computed_via_annotations(self):
        """FLOPs 应通过注解路径计算（非 _original_graph 分支）"""
        from zrt.training.ir.opgraph_builder import build_opgraph
        from zrt.training.ir.context_builder import build_context
        from zrt.transform.pipeline import build_default_pipeline

        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)

        g = build_opgraph(model, strategy)
        ctx = build_context(model, system, strategy, pp_mode="formula")
        pipe = build_default_pipeline()
        transformed = pipe.run(g, ctx)

        training_flops = transformed.metadata.get("training_flops", 0)
        assert training_flops > 0, "FLOPs should be computed via annotation path"


class TestEstimateViaTrainingFromGraphs:
    """验证 estimate_via_pipeline 调用 estimate_training_from_graphs"""

    def test_same_result_as_direct_call(self):
        """estimate_via_pipeline 和直接调 estimate_training_from_graphs 应一致"""
        from zrt.training.search.estimator import estimate_via_pipeline
        from zrt.training.ir.opgraph_builder import build_opgraph
        from zrt.training.ir.context_builder import _resolve_hw_spec
        from zrt.transform.analysis.modeller import estimate_training_from_graphs

        model = _make_model()
        system = _make_system()
        strategy = _make_strategy(tp=2, pp=1, dp=4)

        report_a = estimate_via_pipeline(model, system, strategy)

        opgraph = build_opgraph(model, strategy)
        hw_spec = _resolve_hw_spec(system)
        report_b = estimate_training_from_graphs(
            forward_graph=opgraph,
            hw_spec=hw_spec,
            total_params=model.total_params(),
            hidden=model.hidden,
            num_layers=len(model.layers),
            seq_len=model.seq_len,
            batch_size=strategy.micro_batch,
            tp=strategy.tp, pp=strategy.pp, ep=strategy.ep,
            dp=strategy.dp, cp=strategy.cp,
            zero_stage=strategy.zero_stage,
            micro_batch=strategy.micro_batch,
            global_batch=strategy.global_batch,
            pp_schedule=strategy.pp_schedule.name.lower(),
            pp_mode="formula",
        )

        assert report_a.step_time_ms > 0
        assert report_b.step_time_ms > 0
        ratio = report_a.step_time_ms / report_b.step_time_ms
        assert 0.9 <= ratio <= 1.1, \
            f"step_time_ms mismatch: {report_a.step_time_ms} vs {report_b.step_time_ms}"
