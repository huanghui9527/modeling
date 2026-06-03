"""Build OpGraph from ModelSpec + Strategy.

两条路径共享 OpGraph IR（参见 docs/architecture_snapshot_zh.md）：
  - ``build_explicit_graph()`` — 路径 B（配置建模）：从 ModelSpec 手工构造 OpGraph
  - ``build_captured_graph()`` — 路径 A（抓图建模）：支持真实 HF 模型抓图或 spec-driven fallback

Usage::

    from zrt.training.ir.opgraph_builder import build_explicit_graph, build_captured_graph
    from zrt.training.spec.capture_config import CaptureConfig

    # 路径 B: 配置建模 → Legacy 估算
    opgraph = build_explicit_graph(model, strategy)

    # 路径 A: 抓图建模 → Transform Pipeline（spec-driven fallback）
    opgraph = build_captured_graph(model, strategy)

    # 路径 A: 抓图建模 → Transform Pipeline（真实 HF 抓图）
    capture = CaptureConfig(model_id="meta-llama/Llama-3-70B")
    opgraph = build_captured_graph(model, strategy, capture=capture)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zrt.ir.graph import OpGraph
from zrt.training.ir.builders import build_opgraph_direct
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy

if TYPE_CHECKING:
    from zrt.training.spec.capture_config import CaptureConfig


def build_explicit_graph(model: ModelSpec, strategy: Strategy) -> OpGraph:
    """路径 B（配置建模）：从 ModelSpec 手工构建 OpGraph。

    用于 Legacy 估算路径（``_estimate_legacy``），直接调用
    ``build_opgraph_direct()`` 产出 OpGraph，不经过旧 Graph/Op/Tensor IR。

    Parameters
    ----------
    model : ModelSpec
        Model architecture specification.
    strategy : Strategy
        Parallel strategy configuration.

    Returns
    -------
    OpGraph
        Raw computation graph with sharding and collectives applied.
    """
    return build_opgraph_direct(model, strategy)


def build_captured_graph(
    model: ModelSpec,
    strategy: Strategy,
    capture: "CaptureConfig | None" = None,
) -> OpGraph:
    """路径 A（抓图建模）：构建 OpGraph，用于 Transform Pipeline。

    当 ``capture`` 不为 None 时，走真实 HF 模型抓图：
      load_model → run_trace_phases(train_forward + train_backward)
      → stitch_fwd_bwd → OpGraph

    当 ``capture`` 为 None 时，暂用 ``build_opgraph_direct()`` 作为 spec-driven fallback。

    Parameters
    ----------
    model : ModelSpec
        Model architecture specification.
    strategy : Strategy
        Parallel strategy configuration.
    capture : CaptureConfig | None
        Optional capture configuration. When provided, triggers real HF model
        graph capture. When None, falls back to spec-driven construction.

    Returns
    -------
    OpGraph
        Raw computation graph for Transform Pipeline.
    """
    if capture is not None:
        return _capture_from_hf_model(capture, model, strategy)
    return build_opgraph_direct(model, strategy)


def _capture_from_hf_model(
    capture: "CaptureConfig",
    model: ModelSpec,
    strategy: Strategy,
) -> OpGraph:
    """从 HF 模型真实抓图，产出 OpGraph。

    调用 ``run_trace_phases()`` 抓取 train_forward + train_backward 阶段，
    自动拼接为 stitched OpGraph。将 capture/model/strategy 存入 metadata
    供下游 Pass 使用。

    Parameters
    ----------
    capture : CaptureConfig
        Capture configuration with model_id and tracing parameters.
    model : ModelSpec
        Model architecture specification (for metadata injection).
    strategy : Strategy
        Parallel strategy configuration (for metadata injection).

    Returns
    -------
    OpGraph
        Stitched fwd+bwd OpGraph from real HF model capture.
    """
    from python.zrt.pipeline import run_trace_phases

    result = run_trace_phases(
        model_id=capture.model_id,
        num_layers=capture.num_layers,
        batch_size=capture.batch_size,
        seq_len=capture.seq_len,
        phases=("train_forward", "train_backward"),
        target_layers=capture.target_layers,
        gradient_checkpointing=capture.gradient_checkpointing,
        graph_mode=capture.graph_mode,
    )

    stitched = result.graphs.get("train")
    if stitched is None:
        stitched = result.graphs.get("train_forward")

    if stitched is None:
        raise RuntimeError(
            f"Graph capture failed: no train_forward or train graph in result. "
            f"Available phases: {list(result.graphs.keys())}"
        )

    stitched.metadata["capture_config"] = capture
    stitched.metadata["model_spec"] = model
    stitched.metadata["strategy"] = strategy
    return stitched


# 向后兼容别名
build_opgraph = build_explicit_graph
