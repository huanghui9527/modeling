from __future__ import annotations

from types import SimpleNamespace

from zrt.training.io.html_exporter import _op_detail
from zrt.training.models.flops import OpCost, op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec


class _FakeTensor:
    def __init__(self, name, shape_logical, shape_local, dtype, is_activation=False, is_param=False):
        self.name = name
        self.shape_logical = shape_logical
        self.shape_local = shape_local
        self.dtype = dtype
        self.is_activation = is_activation
        self.is_param = is_param

    def num_elements(self) -> int:
        result = 1
        for d in self.shape_local:
            result *= d
        return result

    def nbytes(self) -> int:
        return self.num_elements() * self.dtype.bytes


def _Op(name, kind, inputs=None, outputs=None, meta=None, component=None, **kw):
    return SimpleNamespace(
        name=name,
        kind=kind,
        inputs=inputs or [],
        outputs=outputs or [],
        meta=meta or {},
        component=component,
        layer_id=kw.get("layer_id", -1),
        layer_kind=kw.get("layer_kind", LayerKind.DENSE),
    )


def _model() -> ModelSpec:
    return ModelSpec(
        hidden=4096,
        ffn=16384,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab=32000,
        seq_len=4096,
        layers=[LayerKind.DENSE],
    )


def _system() -> SystemSpec:
    link = LinkSpec(type="test", bandwidth_gbps=1000, latency_us=1)
    return SystemSpec(
        gpu=GPU(
            name="test-gpu",
            flops_bf16=100,
            flops_fp8=400,
            flops_fp4=800,
            hbm_gb=80,
            hbm_bw_gbps=3000,
            sram_kb_per_sm=228,
        ),
        host_mem_gb=1024,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
        nodes=1,
        gpus_per_node=1,
    )


def test_matmul_formula_uses_local_sharded_dimensions():
    op = _Op(
        name="L0.wq_a",
        kind="matmul",
        inputs=[
            _FakeTensor(
                name="x_ln1",
                shape_logical=(4096, 4096),
                shape_local=(2048, 4096),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        outputs=[
            _FakeTensor(
                name="qr",
                shape_logical=(4096, 1024),
                shape_local=(2048, 128),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        meta={"m": 4096, "n": 1024, "k": 4096},
    )

    detail = _op_detail(op, op_cost(op, _model()))

    assert "2048" in detail["fwd_formula"]
    assert "128" in detail["fwd_formula"]
    assert "1024" not in detail["fwd_formula"]
    assert "1024" not in detail["fwd_bytes_formula"]


def test_meta_authoritative_matmul_formula_keeps_local_meta_dimensions():
    op = _Op(
        name="L0.routed_expert_ffn",
        kind="matmul",
        inputs=[
            _FakeTensor(
                name="x_ln2",
                shape_logical=(4096, 7168),
                shape_local=(4096, 7168),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        outputs=[
            _FakeTensor(
                name="routed_ffn_out",
                shape_logical=(4096, 7168),
                shape_local=(4096, 224),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        meta={
            "m": 4096,
            "n": 7168,
            "k": 7168,
            "k_local": 224,
            "fwd_multiplier": 18,
            "fused_weight_dims": True,
        },
    )

    detail = _op_detail(op, op_cost(op, _model()))

    assert "4096" in detail["fwd_formula"]
    assert "224" in detail["fwd_formula"]
    assert "7168" in detail["fwd_formula"]
    assert "1024" not in detail["fwd_formula"]


def test_matmul_bwd_bytes_formula_matches_dx_plus_dw_bytes():
    op = _Op(
        name="L0.ffn_up",
        kind="matmul",
        inputs=[
            _FakeTensor(
                name="x",
                shape_logical=(128, 256),
                shape_local=(128, 256),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        outputs=[
            _FakeTensor(
                name="y",
                shape_logical=(128, 512),
                shape_local=(128, 512),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        meta={"m": 128, "n": 512, "k": 256},
    )
    cost = op_cost(op, _model())

    detail = _op_detail(op, cost, _model())

    assert detail["bwd_bytes_formula"] != detail["fwd_bytes_formula"]
    assert "dx+dw" in detail["bwd_bytes_formula"]
    assert "458.75K" in detail["fwd_bytes_formula"]
    assert "917.50K" in detail["bwd_bytes_formula"]


def test_mixed_quant_matmul_bytes_formula_shows_per_operand_dtypes():
    model = _model()
    model.moe_act_dtype = Dtype.FP8_E4M3
    model.routed_expert_compute_dtype = Dtype.FP8_E4M3
    model.routed_expert_weight_dtype = Dtype.FP4
    op = _Op(
        name="L0.routed_expert_ffn",
        kind="matmul",
        inputs=[
            _FakeTensor(
                name="x",
                shape_logical=(128, 256),
                shape_local=(128, 256),
                dtype=Dtype.FP8_E4M3,
                is_activation=True,
            )
        ],
        outputs=[
            _FakeTensor(
                name="y",
                shape_logical=(128, 512),
                shape_local=(128, 512),
                dtype=Dtype.FP8_E4M3,
                is_activation=True,
            )
        ],
        meta={"m": 128, "n": 512, "k": 256},
        component="routed_expert",
    )
    cost = op_cost(op, model)

    detail = _op_detail(op, cost, model)

    assert "A_b=1" in detail["fwd_bytes_formula"]
    assert "W_b=0.5625" in detail["fwd_bytes_formula"]
    assert "C_b=1" in detail["fwd_bytes_formula"]
    assert "dW_b=1" in detail["bwd_bytes_formula"]
    assert "172.03K" in detail["fwd_bytes_formula"]
    assert "401.41K" in detail["bwd_bytes_formula"]


def test_embed_bwd_bytes_formula_matches_dx_plus_dw_bytes():
    op = _Op(
        name="embed",
        kind="embed",
        inputs=[
            _FakeTensor(
                name="token_ids",
                shape_logical=(128,),
                shape_local=(128,),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        outputs=[
            _FakeTensor(
                name="x",
                shape_logical=(128, 256),
                shape_local=(128, 256),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        meta={"m": 128, "n": 256},
        component="embedding",
    )
    cost = op_cost(op, _model())

    detail = _op_detail(op, cost, _model())

    assert "dx+dw" in detail["bwd_bytes_formula"]
    assert "131.07K" in detail["bwd_bytes_formula"]


def test_attention_bytes_formula_is_tile_and_dtype_aware():
    op = _Op(
        name="L0.swa_attn",
        kind="swa_attn",
        meta={
            "b": 1,
            "s": 2048,
            "heads": 8,
            "head_dim": 512,
            "causal": True,
            "swa_window": 128,
        },
        component="attention",
    )
    model = _model()
    system = _system()
    cost = op_cost(op, model, system)

    detail = _op_detail(op, cost, model, system)

    assert "KV_tiles" in detail["fwd_bytes_formula"]
    assert "Br=" in detail["fwd_bytes_formula"]
    assert "qkv_b=2.0" in detail["fwd_bytes_formula"]
    assert "83.98M" in detail["fwd_bytes_formula"]
    assert "134.41M" in detail["bwd_bytes_formula"]


def test_formula_cells_do_not_look_like_excel_formulas():
    ops_and_costs = [
        (
            _Op(name="L0.mhc_pre_attn", kind="mhc_pre"),
            OpCost(fwd_bytes=26_460_000, dx_bytes=39_690_000),
        ),
        (
            _Op(name="L0.comp_pool", kind="compressor_pool", meta={"s": 4096, "m": 4, "coff": 1, "d": 512}),
            OpCost(fwd_cube_flops=2_097_152, dx_cube_flops=2_097_152, fwd_bytes=1_000_000, dx_bytes=1_500_000),
        ),
    ]

    for op, cost in ops_and_costs:
        detail = _op_detail(op, cost)
        formulas = (
            detail["fwd_formula"],
            detail["bwd_formula"],
            detail["fwd_bytes_formula"],
            detail["bwd_bytes_formula"],
        )
        assert all(not formula.startswith("=") for formula in formulas)
