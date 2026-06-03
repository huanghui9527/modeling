from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import openpyxl
import pytest

from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.io.excel_exporter import export_estimate_excel
from zrt.training.io.html_exporter import _op_detail, export_estimate_html
from zrt.training.ir.builders import build_graph
from zrt.training.models.flops import op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec


def _moe_model(**kwargs) -> ModelSpec:
    base = dict(
        hidden=1024,
        ffn=4096,
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        vocab=32000,
        seq_len=128,
        layers=[LayerKind.MOE],
        num_experts=8,
        moe_ffn=2048,
        top_k=2,
        n_shared_experts=1,
    )
    base.update(kwargs)
    return ModelSpec(**base)


def _system() -> SystemSpec:
    link = LinkSpec(
        type="NVLink",
        bandwidth_gbps=900,
        latency_us=1.0,
        topology="all_to_all",
        num_devices=8,
    )
    return SystemSpec(
        gpu=GPU(
            name="h100",
            flops_bf16=989,
            flops_fp8=1979,
            hbm_gb=80,
            hbm_bw_gbps=3350,
            ep_overlap_waves=4,
        ),
        host_mem_gb=256,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
        nodes=1,
        gpus_per_node=8,
    )


def _mega_moe_case():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )
    strategy = Strategy(ep=4, dp=2, mega_moe=True, mega_moe_waves=4, micro_batch=3)
    graph = build_graph(model, strategy)
    system = _system()
    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    mega_moe = [op for op in graph.ops if op.kind == "mega_moe"][0]
    return model, strategy, graph, system, op_costs, mega_moe


def _artifact_dir():
    return TemporaryDirectory(dir=Path.cwd())


def test_html_mega_moe_formula_and_export_include_dimensions_quant_and_waves():
    model, strategy, graph, system, op_costs, mega_moe = _mega_moe_case()

    detail = _op_detail(mega_moe, op_costs[mega_moe.name])

    assert "Mega MoE" in detail["fwd_formula"]
    assert "dispatch+FFN+combine" in detail["fwd_formula"]
    assert "m=128" in detail["fwd_formula"]
    assert "micro_batch=3" in detail["fwd_formula"]
    assert "top_k=2" in detail["fwd_formula"]
    assert "k=2048" in detail["fwd_formula"]
    assert "n=1024" in detail["fwd_formula"]
    assert "quant=w4a8" in detail["fwd_formula"]
    assert "waves=4" in detail["fwd_formula"]

    with _artifact_dir() as artifact_dir:
        out = Path(artifact_dir) / "mega_moe.html"
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
        assert "L0.mega_moe" in html
        assert "Mega MoE" in html
        assert "dispatch+FFN+combine" in html


def test_html_mega_moe_formula_terms_match_cost_for_top_k_greater_than_one():
    _, _, _, _, op_costs, mega_moe = _mega_moe_case()

    detail = _op_detail(mega_moe, op_costs[mega_moe.name])
    formula = detail["fwd_formula"]
    match = re.search(
        r"2\*tokens\*top_k\*k\*n\*mult = "
        r"2\*(\d+)\*(\d+)\*(\d+)\*(\d+)\*([0-9.]+)",
        formula,
    )

    assert match is not None
    tokens, top_k, k, n = (int(value) for value in match.groups()[:4])
    mult = float(match.group(5))
    displayed_formula_value = 2 * tokens * top_k * k * n * mult
    actual_cost_value = op_costs[mega_moe.name].fwd_cube_flops + op_costs[mega_moe.name].fwd_vector_flops

    assert mega_moe.meta["top_k"] > 1
    assert mega_moe.meta["fwd_multiplier"] == 3 * mega_moe.meta["top_k"]
    assert "mult=3" in formula
    assert "mult=6" not in formula
    assert displayed_formula_value == pytest.approx(actual_cost_value)


def test_excel_strategy_sheet_exports_mega_moe_fields():
    model, strategy, graph, system, op_costs, _ = _mega_moe_case()
    with _artifact_dir() as artifact_dir:
        out = Path(artifact_dir) / "mega_moe.xlsx"
        export_estimate_excel(
            report=TrainingReport(step_time_ms=1.0),
            graph=graph,
            model=model,
            system=system,
            strategy=strategy,
            op_costs=op_costs,
            output_path=out,
        )

        wb = openpyxl.load_workbook(out, data_only=True)
        rows = {
            row[0].value: row[1].value
            for row in wb["Strategy"].iter_rows()
            if row[0].value not in (None, "")
        }
        wb.close()
        assert rows["Mega MoE"] == "True"
        assert rows["Mega MoE Waves"] == 4


def test_existing_matmul_formula_still_reports_mnk():
    model, strategy, graph, system, op_costs, _ = _mega_moe_case()
    matmul = [op for op in graph.ops if op.kind == "matmul"][0]

    detail = _op_detail(matmul, op_costs[matmul.name])

    assert "2" in detail["fwd_formula"]
    assert "m" in detail["fwd_formula"]
    assert "n" in detail["fwd_formula"]
    assert "k" in detail["fwd_formula"]
    assert detail["bwd_formula"]
