"""C6: FusionPass 聚合验证脚本。

验证 aten-level OpGraph（抓图产出）经过 Transform Pipeline 的 FusionPass 后
能否正确聚合为 block-level OpNode。

用法:
    PYTHONPATH=python python scripts/verify_c6_fusion.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import Counter

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    from python.zrt.pipeline import run_trace_phases
    from python.zrt.transform.pipeline import build_default_pipeline
    from python.zrt.transform.context import TransformContext, ParallelConfig
    from python.zrt.hardware.spec import HardwareSpec

    model_id = "./hf_models/llama3_8b"
    num_layers = 2

    print(f"=== C6 FusionPass 聚合验证 ===")
    print(f"模型: {model_id}")
    print(f"层数: {num_layers}")
    print()

    print("Step 1: 抓取 train_forward + train_backward ...")
    t0 = time.perf_counter()
    result = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        batch_size=1,
        seq_len=64,
        phases=("train_forward", "train_backward"),
    )
    capture_time = time.perf_counter() - t0
    print(f"  抓图耗时: {capture_time:.2f}s")

    stitched = result.graphs.get("train")
    if stitched is None:
        stitched = result.graphs.get("train_forward")
    if stitched is None:
        print("ERROR: 无可用 graph")
        return

    raw_node_count = len(stitched)
    raw_op_types = Counter(n.op_type for n in stitched.nodes.values())
    print(f"  Raw OpGraph: {raw_node_count} 节点")
    print(f"  Raw op_type 分布 (top 10):")
    for op, cnt in raw_op_types.most_common(10):
        print(f"    {op}: {cnt}")
    print()

    print("Step 2: 运行 Transform Pipeline ...")
    from python.zrt.training.spec.model import ModelSpec, LayerKind
    from python.zrt.training.spec.system import GPU, SystemSpec
    from python.zrt.training.spec.strategy import Strategy
    from python.zrt.training.spec.dtype import Dtype
    from python.zrt.training.ir.context_builder import build_context

    model = ModelSpec(
        hidden=4096, ffn=11008, num_heads=32, num_kv_heads=8, head_dim=128,
        vocab=128256, seq_len=64, layers=[LayerKind.DENSE] * num_layers,
        act_dtype=Dtype.BF16,
    )
    gpu = GPU(
        name="nvidia_h100_sxm", flops_bf16=989.0, flops_fp8=1979.0,
        hbm_gb=80.0, hbm_bw_gbps=3350.0,
    )
    from python.zrt.hardware import registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    system = SystemSpec(
        gpu=gpu, host_mem_gb=512.0, interconnect=hw.interconnect,
        nodes=1, gpus_per_node=8,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    ctx = build_context(model, system, strategy, pp_mode="formula")

    pipe = build_default_pipeline()
    t0 = time.perf_counter()
    transformed = pipe.run(stitched, ctx)
    pipeline_time = time.perf_counter() - t0
    print(f"  Pipeline 耗时: {pipeline_time:.2f}s")

    transformed_node_count = len(transformed)
    transformed_op_types = Counter(n.op_type for n in transformed.nodes.values())
    print(f"  Transformed OpGraph: {transformed_node_count} 节点")
    print(f"  Transformed op_type 分布 (top 15):")
    for op, cnt in transformed_op_types.most_common(15):
        print(f"    {op}: {cnt}")
    print()

    print("Step 3: 聚合效果评估 ...")
    reduction = raw_node_count - transformed_node_count
    reduction_pct = reduction / raw_node_count * 100 if raw_node_count > 0 else 0
    print(f"  节点减少: {raw_node_count} → {transformed_node_count} ({reduction_pct:.1f}%)")

    block_level_ops = {"mm", "attn_core", "swiglu", "ln", "rope", "embed", "lm_head",
                       "add", "softmax", "router", "dispatch", "combine"}
    found_block_ops = block_level_ops & set(transformed_op_types.keys())
    print(f"  发现的 block-level ops: {sorted(found_block_ops)}")

    aten_ops = {"aten.mm", "aten.addmm", "aten.bmm", "aten.linear", "aten.rms_norm",
                "aten.scaled_dot_product_attention", "aten.silu", "aten.mul"}
    remaining_aten = aten_ops & set(transformed_op_types.keys())
    print(f"  残留的 aten ops: {sorted(remaining_aten)}")

    if found_block_ops and not remaining_aten:
        print("\n[OK] FusionPass 成功聚合 aten -> block-level")
    elif found_block_ops and remaining_aten:
        print(f"\n[WARN] FusionPass 部分聚合: 有 block-level ops 但残留 {len(remaining_aten)} 种 aten ops")
    else:
        print("\n[FAIL] FusionPass 未能聚合 aten ops -> 需要 GraphCoarsenPass")

    print(f"\n=== C6 验证完成 ===")


if __name__ == "__main__":
    main()
