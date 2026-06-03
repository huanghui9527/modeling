"""estimate() / estimate_via_pipeline() capture 参数透传测试。

验证 capture 参数从 estimate → estimate_via_pipeline → build_captured_graph 的透传链。
"""
from unittest.mock import MagicMock, patch

import pytest

from zrt.training.spec.capture_config import CaptureConfig
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, InterconnectSpec, LinkSpec, SystemSpec
from zrt.training.spec.dtype import Dtype


def _make_model():
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1024, seq_len=64, layers=[LayerKind.DENSE] * 2, act_dtype=Dtype.BF16,
    )


def _make_gpu():
    return GPU(
        name="nvidia_h100_sxm",
        flops_bf16=989.0, flops_fp8=1979.0,
        hbm_gb=80.0, hbm_bw_gbps=3350.0,
    )


def _make_system():
    gpu = _make_gpu()
    link = LinkSpec(
        type="nvlink", bandwidth_gbps=900.0, latency_us=1.0,
        topology="", num_devices=8, kb_efficiency=1.0, oversubscription=1.0,
    )
    return SystemSpec(
        gpu=gpu, host_mem_gb=512.0,
        nodes=1, gpus_per_node=1,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
    )


def _make_strategy():
    return Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)


class TestEstimateCapturePassthrough:
    """验证 estimate() 将 capture 参数透传给 estimate_via_pipeline()。"""

    @patch("zrt.training.search.estimator.estimate_via_pipeline")
    def test_capture_none_not_passed(self, mock_pipeline):
        """capture=None 时 estimate_via_pipeline 收到 capture=None。"""
        mock_pipeline.return_value = MagicMock()
        from zrt.training.search.estimator import estimate

        estimate(_make_model(), _make_system(), _make_strategy(), capture=None)

        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("capture") is None

    @patch("zrt.training.search.estimator.estimate_via_pipeline")
    def test_capture_passed(self, mock_pipeline):
        """capture 不为 None 时 estimate_via_pipeline 收到正确的 CaptureConfig。"""
        mock_pipeline.return_value = MagicMock()
        from zrt.training.search.estimator import estimate

        cfg = CaptureConfig(model_id="test/model", num_layers=2)
        estimate(_make_model(), _make_system(), _make_strategy(), capture=cfg)

        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("capture") is cfg
        assert kwargs["capture"].model_id == "test/model"

    @patch("zrt.training.search.estimator._estimate_legacy")
    def test_graph_takes_precedence_over_capture(self, mock_legacy):
        """graph 参数优先于 capture：传入 graph 时走 Legacy 路径。"""
        mock_legacy.return_value = MagicMock()
        from zrt.training.search.estimator import estimate

        cfg = CaptureConfig(model_id="test/model")
        mock_graph = MagicMock()
        estimate(_make_model(), _make_system(), _make_strategy(),
                 graph=mock_graph, capture=cfg)

        mock_legacy.assert_called_once()


class TestEstimateViaPipelineCapturePassthrough:
    """验证 estimate_via_pipeline() 将 capture 参数透传给 build_captured_graph()。"""

    @patch("zrt.training.search.estimator._build_report_from_transformed")
    @patch("zrt.transform.pipeline.build_default_pipeline")
    @patch("zrt.training.ir.context_builder.build_context")
    @patch("zrt.training.ir.opgraph_builder.build_captured_graph")
    def test_capture_passed_to_builder(
        self, mock_builder, mock_ctx, mock_pipe, mock_report
    ):
        """build_captured_graph 收到正确的 capture 参数。"""
        mock_builder.return_value = MagicMock(metadata={})
        mock_ctx.return_value = MagicMock()
        mock_pipe_inst = MagicMock()
        mock_pipe_inst.run.return_value = MagicMock(metadata={})
        mock_pipe.return_value = mock_pipe_inst
        mock_report.return_value = MagicMock()

        from zrt.training.search.estimator import estimate_via_pipeline

        cfg = CaptureConfig(model_id="deepseek-ai/DeepSeek-V3", num_layers=4)
        estimate_via_pipeline(_make_model(), _make_system(), _make_strategy(), capture=cfg)

        mock_builder.assert_called_once()
        _, kwargs = mock_builder.call_args
        assert kwargs.get("capture") is cfg

    @patch("zrt.training.search.estimator._build_report_from_transformed")
    @patch("zrt.transform.pipeline.build_default_pipeline")
    @patch("zrt.training.ir.context_builder.build_context")
    @patch("zrt.training.ir.opgraph_builder.build_captured_graph")
    def test_capture_none_passed_to_builder(
        self, mock_builder, mock_ctx, mock_pipe, mock_report
    ):
        """capture=None 时 build_captured_graph 收到 capture=None。"""
        mock_builder.return_value = MagicMock(metadata={})
        mock_ctx.return_value = MagicMock()
        mock_pipe_inst = MagicMock()
        mock_pipe_inst.run.return_value = MagicMock(metadata={})
        mock_pipe.return_value = mock_pipe_inst
        mock_report.return_value = MagicMock()

        from zrt.training.search.estimator import estimate_via_pipeline

        estimate_via_pipeline(_make_model(), _make_system(), _make_strategy())

        mock_builder.assert_called_once()
        _, kwargs = mock_builder.call_args
        assert kwargs.get("capture") is None
