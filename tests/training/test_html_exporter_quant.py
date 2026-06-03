from __future__ import annotations

from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.io.html_exporter import export_estimate_html
from zrt.training.ir.builders import build_graph
from zrt.training.models.flops import op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec


def _system() -> SystemSpec:
    link = LinkSpec(type="test", bandwidth_gbps=1000, latency_us=1)
    return SystemSpec(
        gpu=GPU(
            name="test-gpu",
            flops_bf16=100,
            flops_fp8=400,
            flops_fp4=800,
            hbm_gb=80,
            hbm_bw_gbps=1_000_000,
        ),
        host_mem_gb=1024,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
        nodes=1,
        gpus_per_node=1,
    )


def test_html_export_serializes_quant_cast_meta(tmp_path):
    model = ModelSpec(
        hidden=128,
        ffn=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=32,
        vocab=1000,
        seq_len=64,
        layers=[LayerKind.MOE],
        num_experts=8,
        moe_ffn=256,
        top_k=2,
        moe_act_dtype=Dtype.FP8_E4M3,
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
    )
    system = _system()
    strategy = Strategy()
    graph = build_graph(model, strategy)
    assert any(op.kind == "cast" for op in graph.ops)

    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    out = tmp_path / "quant.html"

    export_estimate_html(
        report=TrainingReport(step_time_ms=1.0),
        graph=graph,
        model=model,
        system=system,
        strategy=strategy,
        op_costs=op_costs,
        output_path=out,
    )

    html = out.read_text(encoding="utf-8")
    assert "const DATA = JSON.parse(" in html
    assert "fp8_e4m3" in html
    assert "Dtype.FP8_E4M3" not in html


def test_html_export_includes_operator_time_share(tmp_path):
    model = ModelSpec(
        hidden=128,
        ffn=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=32,
        vocab=1000,
        seq_len=64,
        layers=[LayerKind.MOE],
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        index_topk=16,
        num_experts=8,
        moe_ffn=256,
        top_k=2,
    )
    system = _system()
    strategy = Strategy()
    graph = build_graph(model, strategy)
    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    out = tmp_path / "operator_time_share.html"

    export_estimate_html(
        report=TrainingReport(step_time_ms=100.0, compute_time_ms=50.0),
        graph=graph,
        model=model,
        system=system,
        strategy=strategy,
        op_costs=op_costs,
        output_path=out,
    )

    html = out.read_text(encoding="utf-8")
    assert "Operator Time Share" in html
    assert "Matmul family total" in html
    assert "Attention matmul family" in html
    assert "MoE/FFN matmul family" in html
    assert "LM head matmul" in html
    assert "Sparse FA core (DSA)" in html
    assert "Lightning Indexer" in html
    assert "DSA attention compute" in html
    assert "MLA attention block" in html
    assert "useful compute" in html
    assert "pct_of_useful_compute" in html


def test_html_export_includes_interactive_op_share(tmp_path):
    model = ModelSpec(
        hidden=128,
        ffn=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=32,
        vocab=1000,
        seq_len=64,
        layers=[LayerKind.DENSE, LayerKind.MOE],
        num_experts=8,
        moe_ffn=256,
        top_k=2,
    )
    system = _system()
    strategy = Strategy()
    graph = build_graph(model, strategy)
    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    out = tmp_path / "op_share.html"

    export_estimate_html(
        report=TrainingReport(step_time_ms=100.0, compute_time_ms=50.0),
        graph=graph,
        model=model,
        system=system,
        strategy=strategy,
        op_costs=op_costs,
        output_path=out,
    )

    html = out.read_text(encoding="utf-8")
    # JSON payload still injected via the safe literal path, no raw enum leakage.
    assert "const DATA = JSON.parse(" in html
    assert "Dtype." not in html
    # New interactive section + its controls and client-side helpers are present.
    assert '<section id="op-share">' in html
    assert "5. 算子耗时占比分析" in html
    assert "function renderOpShare()" in html
    assert "renderOpShare();" in html
    assert "opFocus" in html
    assert "opModuleDim" in html
    assert "function donutSvg(" in html
    # Two side-by-side pies: module view + operator (compute-unit) view.
    assert "pie-pair" in html
    assert "模块视角" in html
    assert "算子视角 · 计算单元" in html
    assert "pie-leg-n" in html             # per-bucket operator count
    assert "bound-tag" in html             # bound tag next to module-view buckets
    # Calibration section renumbered to 6 (op-share inserted before it).
    assert "6. 报告校准" in html
    # Per-op enrichment fields the component depends on are embedded in DATA.
    assert "op_groups" in html
    assert "cube_flops" in html
    assert "vector_flops" in html
