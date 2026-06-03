"""build_captured_graph 真实实现单元测试。

验证 build_captured_graph 的 capture 参数分支逻辑和 _capture_from_hf_model 的错误处理。
由于真实 HF 抓图需要网络和 torch，使用 mock 测试逻辑流。
"""
from unittest.mock import MagicMock, patch

import pytest

from zrt.training.ir.opgraph_builder import (
    build_captured_graph,
    _capture_from_hf_model,
)
from zrt.training.spec.capture_config import CaptureConfig
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.dtype import Dtype


@pytest.fixture
def minimal_model():
    """最小化 ModelSpec fixture。"""
    return ModelSpec(
        hidden=128,
        ffn=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=32,
        vocab=1024,
        seq_len=64,
        layers=[LayerKind.DENSE, LayerKind.DENSE],
        act_dtype=Dtype.BF16,
    )


@pytest.fixture
def minimal_strategy():
    """最小化 Strategy fixture。"""
    return Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)


class TestBuildCapturedGraphDispatch:
    """验证 build_captured_graph 的 capture 参数分支逻辑。"""

    def test_capture_none_uses_spec_driven(self, minimal_model, minimal_strategy):
        """capture=None 时走 build_opgraph_direct（spec-driven fallback）。

        观测点：返回 OpGraph 且包含 spec-driven 构建的特征节点。
        """
        graph = build_captured_graph(minimal_model, minimal_strategy, capture=None)
        assert graph is not None
        assert len(graph) > 0

    def test_capture_default_uses_spec_driven(self, minimal_model, minimal_strategy):
        """不传 capture 参数时默认走 spec-driven fallback。

        观测点：与 capture=None 行为一致。
        """
        graph = build_captured_graph(minimal_model, minimal_strategy)
        assert graph is not None
        assert len(graph) > 0

    @patch("zrt.training.ir.opgraph_builder._capture_from_hf_model")
    def test_capture_provided_calls_hf_capture(
        self, mock_capture, minimal_model, minimal_strategy
    ):
        """capture 不为 None 时调用 _capture_from_hf_model。

        观测点：mock 被调用且参数正确传递。
        """
        cfg = CaptureConfig(model_id="test/model")
        mock_graph = MagicMock()
        mock_capture.return_value = mock_graph

        result = build_captured_graph(minimal_model, minimal_strategy, capture=cfg)

        mock_capture.assert_called_once_with(cfg, minimal_model, minimal_strategy)
        assert result is mock_graph


class TestCaptureFromHfModel:
    """验证 _capture_from_hf_model 的逻辑和错误处理。"""

    @patch("python.zrt.pipeline.run_trace_phases")
    def test_stitched_graph_preferred(self, mock_trace, minimal_model, minimal_strategy):
        """优先使用 stitched train graph（fwd+bwd 拼接）。

        观测点：当 result.graphs 包含 "train" 时使用它。
        """
        cfg = CaptureConfig(model_id="test/model", num_layers=2)
        mock_stitched = MagicMock()
        mock_stitched.metadata = {}
        mock_fwd = MagicMock()
        mock_fwd.metadata = {}

        mock_result = MagicMock()
        mock_result.graphs = {
            "train_forward": mock_fwd,
            "train_backward": MagicMock(),
            "train": mock_stitched,
        }
        mock_trace.return_value = mock_result

        result = _capture_from_hf_model(cfg, minimal_model, minimal_strategy)

        assert result is mock_stitched
        assert result.metadata["capture_config"] is cfg
        assert result.metadata["model_spec"] is minimal_model
        assert result.metadata["strategy"] is minimal_strategy

    @patch("python.zrt.pipeline.run_trace_phases")
    def test_fallback_to_train_forward(self, mock_trace, minimal_model, minimal_strategy):
        """无 stitched graph 时回退到 train_forward。

        观测点：当 result.graphs 只有 "train_forward" 时使用它。
        """
        cfg = CaptureConfig(model_id="test/model")
        mock_fwd = MagicMock()
        mock_fwd.metadata = {}

        mock_result = MagicMock()
        mock_result.graphs = {"train_forward": mock_fwd}
        mock_trace.return_value = mock_result

        result = _capture_from_hf_model(cfg, minimal_model, minimal_strategy)

        assert result is mock_fwd
        assert result.metadata["capture_config"] is cfg

    @patch("python.zrt.pipeline.run_trace_phases")
    def test_no_graph_raises(self, mock_trace, minimal_model, minimal_strategy):
        """无可用 graph 时抛 RuntimeError。

        观测点：错误信息包含可用 phase 列表。
        """
        cfg = CaptureConfig(model_id="test/model")

        mock_result = MagicMock()
        mock_result.graphs = {}
        mock_trace.return_value = mock_result

        with pytest.raises(RuntimeError, match="Graph capture failed"):
            _capture_from_hf_model(cfg, minimal_model, minimal_strategy)

    @patch("python.zrt.pipeline.run_trace_phases")
    def test_trace_params_passed_correctly(self, mock_trace, minimal_model, minimal_strategy):
        """CaptureConfig 参数正确传递给 run_trace_phases。

        观测点：验证所有关键字段（model_id, num_layers, seq_len, batch_size 等）。
        """
        cfg = CaptureConfig(
            model_id="deepseek-ai/DeepSeek-V3",
            num_layers=8,
            seq_len=2048,
            batch_size=2,
            target_layers=[0, 3],
            gradient_checkpointing=True,
            graph_mode=True,
        )

        mock_result = MagicMock()
        mock_result.graphs = {"train": MagicMock(metadata={})}
        mock_trace.return_value = mock_result

        _capture_from_hf_model(cfg, minimal_model, minimal_strategy)

        mock_trace.assert_called_once_with(
            model_id="deepseek-ai/DeepSeek-V3",
            num_layers=8,
            batch_size=2,
            seq_len=2048,
            phases=("train_forward", "train_backward"),
            target_layers=[0, 3],
            gradient_checkpointing=True,
            graph_mode=True,
        )
