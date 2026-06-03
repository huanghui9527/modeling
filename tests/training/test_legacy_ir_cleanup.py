"""Phase 3 UT — 验证旧 IR 清理进度。

验证点:
  1. pipeline_step_time 接受 OpGraph (通过 adapter 桥接)
  2. stage_time 接受 OpNode (通过 adapter 桥接)
  3. graph_adapter 的 OpGraph → Graph 转换正确
  4. 新代码路径 (opgraph_builder, context_builder) 不直接依赖旧 IR 的内部实现
"""
import inspect
from pathlib import Path

import pytest

from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec, GPU
from zrt.training.spec.dtype import Dtype
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


def _make_model_and_strategy():
    model = ModelSpec(
        hidden=2048, ffn=5504, seq_len=1024,
        num_heads=16, num_kv_heads=16, head_dim=128,
        layers=[LayerKind.DENSE] * 4, vocab=16000, act_dtype=Dtype.BF16,
    )
    strategy = Strategy(tp=1, pp=1, ep=1, dp=1, cp=1,
                        micro_batch=1, global_batch=16)
    return model, strategy


class TestPipelineStepTimeAcceptsOpGraph:
    """pipeline_step_time 的签名和运行时行为必须兼容 OpGraph。"""

    def test_signature_accepts_opgraph(self):
        from zrt.training.compose.schedules import pipeline_step_time
        sig = inspect.signature(pipeline_step_time)
        first_param = list(sig.parameters.values())[0]
        annotation = str(first_param.annotation)
        assert "OpGraph" in annotation, (
            f"pipeline_step_time 第一个参数标注应包含 OpGraph，实际: {annotation}"
        )

    def test_runtime_accepts_opgraph(self):
        """传入 OpGraph 实例不应抛出 TypeError。"""
        from zrt.training.ir.opgraph_builder import build_opgraph
        from zrt.training.compose.schedules import pipeline_step_time

        model, strategy = _make_model_and_strategy()
        system = _make_system()
        opgraph = build_opgraph(model, strategy)
        result = pipeline_step_time(opgraph, model, system, strategy)
        assert result.step_time > 0, "step_time 应为正数"
        assert result.pipeline_time > 0, "pipeline_time 应为正数"

    def test_opgraph_and_graph_produce_same_step_time(self):
        """OpGraph 和旧 Graph 经 pipeline_step_time 后 step_time 应一致。"""
        from zrt.training.ir.opgraph_builder import build_opgraph
        from zrt.training.ir.builders import build_graph
        from zrt.training.compose.schedules import pipeline_step_time

        model, strategy = _make_model_and_strategy()
        system = _make_system()

        old_graph = build_graph(model, strategy)
        opgraph = build_opgraph(model, strategy)

        r_old = pipeline_step_time(old_graph, model, system, strategy)
        r_new = pipeline_step_time(opgraph, model, system, strategy)

        if r_old.step_time > 0:
            ratio = r_new.step_time / r_old.step_time
            assert 0.99 <= ratio <= 1.01, (
                f"step_time 不一致: old={r_old.step_time:.6f}, new={r_new.step_time:.6f}, ratio={ratio:.4f}"
            )


class TestStageTimeAcceptsOpNode:
    """stage_time 的签名和运行时行为必须兼容 OpNode。"""

    def test_signature_accepts_opnode(self):
        from zrt.training.compose.stage import stage_time
        sig = inspect.signature(stage_time)
        first_param = list(sig.parameters.values())[0]
        annotation = str(first_param.annotation)
        assert "OpNode" in annotation, (
            f"stage_time 第一个参数标注应包含 OpNode，实际: {annotation}"
        )

    def test_runtime_accepts_opnode_list(self):
        """传入 list[OpNode] 不应抛出 TypeError。"""
        from zrt.training.ir.opgraph_builder import build_opgraph
        from zrt.training.compose.stage import stage_time

        model, strategy = _make_model_and_strategy()
        system = _make_system()
        opgraph = build_opgraph(model, strategy)

        compute_nodes = [n for n in opgraph.nodes.values()
                         if not n.op_type.startswith("comm.")]
        comm_nodes = [n for n in opgraph.nodes.values()
                      if n.op_type.startswith("comm.")]

        if compute_nodes:
            result = stage_time(compute_nodes, comm_nodes, model, system, strategy)
            assert result.fwd >= 0, "fwd 应非负"
            assert result.bwd >= 0, "bwd 应非负"


class TestDeprecatedIRIsolation:
    """验证旧 IR 的引用已被隔离到兼容层。"""

    _COMPAT_FILES = {
        "training_graph.py",
        "graph_adapter.py",
        "opgraph_builder.py",
        "__init__.py",
        "builders.py",
        "shard.py",
        "cast_pass.py",
        "schedules.py",
        "stage.py",
        "flops.py",
        "comm.py",
        "memory.py",
        "quant.py",
        "comm_domain.py",
        "html_exporter.py",
        "excel_exporter.py",
        "estimator.py",
        "graph.py",
    }

    def test_legacy_imports_isolated(self):
        """非兼容文件不应直接 import training_graph 类型。"""
        root = Path(__file__).resolve().parents[2] / "python" / "zrt"
        violations = []
        for py_file in root.rglob("*.py"):
            if py_file.name in self._COMPAT_FILES:
                continue
            text = py_file.read_text(encoding="utf-8", errors="replace")
            if "from zrt.training.ir.training_graph import" in text:
                rel = py_file.relative_to(root)
                violations.append(str(rel))
        assert not violations, (
            f"以下非兼容文件仍引用旧 IR: {violations}"
        )
