"""config_loader capture: 段解析测试。

验证 load_specs 和 load_anchor_config 正确解析可选的 capture: 段，
返回 4-tuple (ModelSpec, SystemSpec, Strategy, CaptureConfig | None)。
"""
import tempfile
from pathlib import Path

import pytest
import yaml

from zrt.training.io.config_loader import load_specs, _parse_capture
from zrt.training.spec.capture_config import CaptureConfig


_MINIMAL_YAML = {
    "model": {
        "hidden": 128, "ffn": 256, "num_heads": 4, "num_kv_heads": 4,
        "head_dim": 32, "vocab": 1024, "seq_len": 64,
        "layers": ["dense", "dense"],
    },
    "system": {
        "hw": "nvidia_h100_sxm",
        "nodes": 1, "gpus_per_node": 1,
    },
    "strategy": {
        "tp": 1, "pp": 1, "dp": 1, "micro_batch": 1, "global_batch": 1,
    },
}


def _write_yaml(data: dict) -> Path:
    """Write dict to a temp YAML file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(data, f)
    f.close()
    return Path(f.name)


class TestParseCapture:
    """验证 _parse_capture 函数。"""

    def test_none_input_returns_none(self):
        """None 输入返回 None。"""
        assert _parse_capture(None) is None

    def test_empty_dict_returns_none(self):
        """空 dict 返回 None。"""
        assert _parse_capture({}) is None

    def test_minimal_capture(self):
        """只有 model_id 即可构造 CaptureConfig。"""
        result = _parse_capture({"model_id": "test/model"})
        assert isinstance(result, CaptureConfig)
        assert result.model_id == "test/model"
        assert result.num_layers == 4
        assert result.seq_len == 128

    def test_full_capture(self):
        """所有字段均可显式指定。"""
        result = _parse_capture({
            "model_id": "deepseek-ai/DeepSeek-V3",
            "num_layers": 8,
            "seq_len": 2048,
            "batch_size": 2,
            "target_layers": [0, 3],
            "gradient_checkpointing": True,
            "graph_mode": True,
        })
        assert result.model_id == "deepseek-ai/DeepSeek-V3"
        assert result.num_layers == 8
        assert result.seq_len == 2048
        assert result.batch_size == 2
        assert result.target_layers == [0, 3]
        assert result.gradient_checkpointing is True
        assert result.graph_mode is True


class TestLoadSpecs4Tuple:
    """验证 load_specs 返回 4-tuple。"""

    def test_no_capture_returns_none(self):
        """无 capture: 段时第 4 个元素为 None。"""
        path = _write_yaml(_MINIMAL_YAML)
        model, system, strategy, capture = load_specs(path)
        assert capture is None
        assert model.hidden == 128
        path.unlink()

    def test_with_capture_returns_config(self):
        """有 capture: 段时第 4 个元素为 CaptureConfig。"""
        data = {**_MINIMAL_YAML, "capture": {"model_id": "test/model", "num_layers": 2}}
        path = _write_yaml(data)
        model, system, strategy, capture = load_specs(path)
        assert isinstance(capture, CaptureConfig)
        assert capture.model_id == "test/model"
        assert capture.num_layers == 2
        path.unlink()

    def test_capture_defaults_applied(self):
        """capture: 段中未指定的字段使用默认值。"""
        data = {**_MINIMAL_YAML, "capture": {"model_id": "test/model"}}
        path = _write_yaml(data)
        _, _, _, capture = load_specs(path)
        assert capture.seq_len == 128
        assert capture.batch_size == 1
        assert capture.target_layers is None
        assert capture.gradient_checkpointing is False
        assert capture.graph_mode is False
        path.unlink()
