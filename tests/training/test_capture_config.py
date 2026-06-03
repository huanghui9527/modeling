"""CaptureConfig dataclass 单元测试。

验证 CaptureConfig 的构造、默认值、校验逻辑和序列化行为。
"""
import pytest

from zrt.training.spec.capture_config import CaptureConfig


class TestCaptureConfigDefaults:
    """验证默认值与 run_trace_phases() 当前默认值一致。"""

    def test_minimal_construction(self):
        """只需 model_id 即可构造，其余字段使用默认值。"""
        cfg = CaptureConfig(model_id="meta-llama/Llama-3-70B")
        assert cfg.model_id == "meta-llama/Llama-3-70B"
        assert cfg.num_layers == 4
        assert cfg.seq_len == 128
        assert cfg.batch_size == 1
        assert cfg.target_layers is None
        assert cfg.gradient_checkpointing is False
        assert cfg.graph_mode is False

    def test_full_construction(self):
        """所有字段均可显式指定。"""
        cfg = CaptureConfig(
            model_id="deepseek-ai/DeepSeek-V3",
            num_layers=8,
            seq_len=2048,
            batch_size=2,
            target_layers=[0, 3, 7],
            gradient_checkpointing=True,
            graph_mode=True,
        )
        assert cfg.model_id == "deepseek-ai/DeepSeek-V3"
        assert cfg.num_layers == 8
        assert cfg.seq_len == 2048
        assert cfg.batch_size == 2
        assert cfg.target_layers == [0, 3, 7]
        assert cfg.gradient_checkpointing is True
        assert cfg.graph_mode is True

    def test_local_path_as_model_id(self):
        """本地路径也可作为 model_id。"""
        cfg = CaptureConfig(model_id="./hf_models/deepseek_v3")
        assert cfg.model_id == "./hf_models/deepseek_v3"


class TestCaptureConfigValidation:
    """验证 __post_init__ 校验逻辑。"""

    def test_empty_model_id_raises(self):
        """空字符串 model_id 应抛 ValueError。"""
        with pytest.raises(ValueError, match="model_id is required"):
            CaptureConfig(model_id="")

    def test_zero_num_layers_raises(self):
        """num_layers < 1 应抛 ValueError。"""
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            CaptureConfig(model_id="test", num_layers=0)

    def test_negative_num_layers_raises(self):
        """负数 num_layers 应抛 ValueError。"""
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            CaptureConfig(model_id="test", num_layers=-1)

    def test_zero_seq_len_raises(self):
        """seq_len < 1 应抛 ValueError。"""
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            CaptureConfig(model_id="test", seq_len=0)

    def test_zero_batch_size_raises(self):
        """batch_size < 1 应抛 ValueError。"""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            CaptureConfig(model_id="test", batch_size=0)

    def test_num_layers_one_is_valid(self):
        """num_layers=1 是合法的最小值。"""
        cfg = CaptureConfig(model_id="test", num_layers=1)
        assert cfg.num_layers == 1


class TestCaptureConfigEquality:
    """验证 dataclass 的相等性和哈希行为。"""

    def test_equal_configs(self):
        """相同参数的两个 CaptureConfig 应相等。"""
        a = CaptureConfig(model_id="test", num_layers=4)
        b = CaptureConfig(model_id="test", num_layers=4)
        assert a == b

    def test_different_model_id(self):
        """不同 model_id 应不相等。"""
        a = CaptureConfig(model_id="model_a")
        b = CaptureConfig(model_id="model_b")
        assert a != b

    def test_different_num_layers(self):
        """不同 num_layers 应不相等。"""
        a = CaptureConfig(model_id="test", num_layers=4)
        b = CaptureConfig(model_id="test", num_layers=8)
        assert a != b


class TestCaptureConfigRepr:
    """验证 repr 输出包含关键信息。"""

    def test_repr_contains_model_id(self):
        """repr 应包含 model_id。"""
        cfg = CaptureConfig(model_id="meta-llama/Llama-3-70B")
        r = repr(cfg)
        assert "meta-llama/Llama-3-70B" in r
        assert "CaptureConfig" in r

    def test_repr_contains_all_fields(self):
        """repr 应包含所有字段名。"""
        cfg = CaptureConfig(model_id="test")
        r = repr(cfg)
        assert "num_layers" in r
        assert "seq_len" in r
        assert "batch_size" in r
        assert "target_layers" in r
        assert "gradient_checkpointing" in r
        assert "graph_mode" in r
