"""
双路径训练估算对比脚本

对比路径 B（配置建模，_estimate_legacy）和路径 A（抓图建模，estimate_via_pipeline）
的数值一致性与执行性能。参见 docs/architecture_snapshot_zh.md。

用法:
    PYTHONPATH=python python scripts/compare_paths.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

CONFIGS = [
    "python/zrt/training/configs/llama3_70b_3d.yaml",
]

THRESHOLD = 0.05
WARMUP = 1
REPEAT = 3

NUMERIC_FIELDS: list[tuple[str, str, str]] = [
    ("step_time_ms", "Step Time", "ms"),
    ("pipeline_time_ms", "Pipeline Time", "ms"),
    ("compute_time_ms", "Compute Time", "ms"),
    ("fwd_compute_ms", "Fwd Compute", "ms"),
    ("bwd_compute_ms", "Bwd Compute", "ms"),
    ("recompute_compute_ms", "Recompute", "ms"),
    ("recompute_critical_ms", "Recompute Critical", "ms"),
    ("bubble_time_ms", "Bubble Time", "ms"),
    ("bubble_fraction", "Bubble Fraction", ""),
    ("exposed_comm_ms", "Exposed Comm", "ms"),
    ("hidden_comm_ms", "Hidden Comm", "ms"),
    ("total_comm_volume_ms", "Total Comm Volume", "ms"),
    ("optimizer_time_ms", "Optimizer Time", "ms"),
    ("optimizer_comm_ms", "Optimizer Comm", "ms"),
    ("mfu", "MFU", ""),
    ("hfu", "HFU", ""),
    ("mfu_native", "MFU Native", ""),
    ("total_flops", "Total FLOPs", ""),
    ("forward_flops", "Forward FLOPs", ""),
    ("backward_flops", "Backward FLOPs", ""),
    ("tokens_per_sec", "Tokens/sec", ""),
    ("flops_per_token", "FLOPs/token", ""),
    ("warmup_ms", "Warmup", "ms"),
    ("steady_ms", "Steady", "ms"),
    ("cooldown_ms", "Cooldown", "ms"),
    ("warmup_fwd_ms", "Warmup Fwd", "ms"),
    ("warmup_bwd_ms", "Warmup Bwd", "ms"),
    ("steady_fwd_ms", "Steady Fwd", "ms"),
    ("steady_bwd_ms", "Steady Bwd", "ms"),
    ("cooldown_fwd_ms", "Cooldown Fwd", "ms"),
    ("cooldown_bwd_ms", "Cooldown Bwd", "ms"),
    ("tp_exposed_ms", "TP Exposed", "ms"),
    ("cp_exposed_ms", "CP Exposed", "ms"),
    ("ep_exposed_ms", "EP Exposed", "ms"),
    ("pp_exposed_ms", "PP Exposed", "ms"),
    ("dp_exposed_ms", "DP Exposed", "ms"),
    ("tp_hidden_ms", "TP Hidden", "ms"),
    ("ep_hidden_ms", "EP Hidden", "ms"),
    ("pp_hidden_ms", "PP Hidden", "ms"),
    ("dp_hidden_ms", "DP Hidden", "ms"),
    ("tp_total_ms", "TP Total", "ms"),
    ("cp_total_ms", "CP Total", "ms"),
    ("ep_total_ms", "EP Total", "ms"),
    ("pp_total_ms", "PP Total", "ms"),
    ("dp_total_ms", "DP Total", "ms"),
    ("weight_hbm_gb", "Weight HBM", "GB"),
    ("act_hbm_gb", "Activation HBM", "GB"),
    ("grad_hbm_gb", "Gradient HBM", "GB"),
    ("cast_hbm_gb", "Cast HBM", "GB"),
]


def _get_val(report: Any, field: str) -> float:
    v = getattr(report, field, None)
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _fmt_val(v: float, unit: str) -> str:
    if unit in ("ms", "GB"):
        if abs(v) < 0.001:
            return "0.000"
        if abs(v) >= 1e6:
            return f"{v:.2e}"
        return f"{v:.3f}"
    if abs(v) < 1e-6:
        return "0.000"
    if abs(v) >= 1e12:
        return f"{v:.3e}"
    if abs(v) < 0.01:
        return f"{v:.6f}"
    return f"{v:.4f}"


def _diff_color(diff_pct: float) -> str:
    if diff_pct <= 0.001:
        return "green"
    if diff_pct <= THRESHOLD:
        return "yellow"
    return "red"


def run_legacy(model, system, strategy) -> tuple[Any, float]:
    from zrt.training.ir.opgraph_builder import build_explicit_graph
    from zrt.training.search.estimator import estimate

    t0 = time.perf_counter()
    graph = build_explicit_graph(model, strategy)
    report = estimate(model, system, strategy, graph=graph)
    elapsed = time.perf_counter() - t0
    return report, elapsed


def run_pipeline(model, system, strategy) -> tuple[Any, float]:
    from zrt.training.search.estimator import estimate_via_pipeline

    t0 = time.perf_counter()
    report = estimate_via_pipeline(model, system, strategy)
    elapsed = time.perf_counter() - t0
    return report, elapsed


def run_capture(model, system, strategy, capture_config) -> tuple[Any, float]:
    """路径 A（真实抓图）：通过 CaptureConfig 走 estimate_via_pipeline。"""
    from zrt.training.search.estimator import estimate_via_pipeline

    t0 = time.perf_counter()
    report = estimate_via_pipeline(model, system, strategy, capture=capture_config)
    elapsed = time.perf_counter() - t0
    return report, elapsed


def print_comparison(
    report_legacy, report_pipeline,
    time_legacy: float, time_pipeline: float,
    config_label: str,
):
    speed_ratio = f"{time_legacy/time_pipeline:.2f}x" if time_pipeline > 0 else "N/A"
    console.print()
    console.print(Panel(
        f"[bold]{config_label}[/bold]\n"
        f"路径 B: {time_legacy*1000:.1f}ms  |  路径 A: {time_pipeline*1000:.1f}ms  |  "
        f"路径 B/A: {speed_ratio}",
        title="[bold cyan]执行性能[/bold cyan]",
        border_style="cyan",
    ))

    table = Table(
        title=f"数值对比 (阈值: {THRESHOLD:.0%})",
        show_header=True,
        header_style="bold magenta",
        show_lines=False,
        row_styles=["", "dim"],
    )
    table.add_column("指标", style="cyan", min_width=22)
    table.add_column("路径 B", justify="right", min_width=14)
    table.add_column("路径 A", justify="right", min_width=14)
    table.add_column("差异", justify="right", min_width=10)
    table.add_column("差异%", justify="right", min_width=8)

    n_pass = n_warn = n_fail = n_skip = 0

    for field_name, label, unit in NUMERIC_FIELDS:
        v_leg = _get_val(report_legacy, field_name)
        v_pipe = _get_val(report_pipeline, field_name)

        if v_leg == 0.0 and v_pipe == 0.0:
            n_skip += 1
            continue

        denom = max(abs(v_leg), abs(v_pipe), 1e-12)
        diff_abs = abs(v_leg - v_pipe)
        diff_pct = diff_abs / denom

        color = _diff_color(diff_pct)
        if diff_pct <= 0.001:
            n_pass += 1
        elif diff_pct <= THRESHOLD:
            n_warn += 1
        else:
            n_fail += 1

        diff_str = f"{diff_pct:.1%}" if diff_pct < 100 else f"{diff_pct:.0%}"

        table.add_row(
            f"{label} ({unit})" if unit else label,
            _fmt_val(v_leg, unit),
            _fmt_val(v_pipe, unit),
            _fmt_val(diff_abs, unit),
            Text(diff_str, style=color),
        )

    console.print(table)

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column()
    summary.add_row(
        Text("PASS", style="bold green"), Text(str(n_pass)),
        Text("WARN", style="bold yellow"), Text(str(n_warn)),
        Text("FAIL", style="bold red"), Text(str(n_fail)),
        Text("SKIP(both=0)", style="dim"), Text(str(n_skip)),
    )
    console.print(summary)
    console.print()

    return n_fail


def main():
    from zrt.training.io.config_loader import load_specs

    total_failures = 0

    for config_path in CONFIGS:
        console.rule(f"[bold blue]{config_path}")

        model, system, strategy, capture_cfg = load_specs(config_path)
        console.print(
            f"  模型: hidden={model.hidden}, layers={len(model.layers)}, "
            f"heads={model.num_heads}, vocab={model.vocab}"
        )
        console.print(
            f"  策略: TP={strategy.tp} CP={strategy.cp} PP={strategy.pp} "
            f"EP={strategy.ep} DP={strategy.dp}  "
            f"MB={strategy.micro_batch} GB={strategy.global_batch}"
        )
        console.print(f"  系统: {system.gpu.name} x {system.world_size} GPUs")
        if capture_cfg:
            console.print(f"  抓图: model_id={capture_cfg.model_id}, layers={capture_cfg.num_layers}")

        for _ in range(WARMUP):
            run_legacy(model, system, strategy)
            run_pipeline(model, system, strategy)

        times_leg = []
        times_pipe = []
        report_leg = report_pipe = None
        for _ in range(REPEAT):
            rl, tl = run_legacy(model, system, strategy)
            rp, tp = run_pipeline(model, system, strategy)
            times_leg.append(tl)
            times_pipe.append(tp)
            report_leg = rl
            report_pipe = rp

        times_leg.sort()
        times_pipe.sort()
        median_leg = times_leg[len(times_leg) // 2]
        median_pipe = times_pipe[len(times_pipe) // 2]

        nf = print_comparison(
            report_leg, report_pipe,
            median_leg, median_pipe,
            config_path,
        )
        total_failures += nf

        if capture_cfg:
            console.rule(f"[bold green]路径 A (真实抓图) vs 路径 B")
            for _ in range(WARMUP):
                run_capture(model, system, strategy, capture_cfg)

            times_cap = []
            report_cap = None
            for _ in range(REPEAT):
                rc, tc = run_capture(model, system, strategy, capture_cfg)
                times_cap.append(tc)
                report_cap = rc

            times_cap.sort()
            median_cap = times_cap[len(times_cap) // 2]

            nf2 = print_comparison(
                report_leg, report_cap,
                median_leg, median_cap,
                f"{config_path} (capture: {capture_cfg.model_id})",
            )
            total_failures += nf2

    if total_failures > 0:
        console.print(f"[bold red]共 {total_failures} 个指标超出阈值[/bold red]")
    else:
        console.print("[bold green]所有配置的所有指标均在阈值内[/bold green]")


if __name__ == "__main__":
    main()
