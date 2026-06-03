# Mega MoE Spec Path Design

日期: 2026-05-19

## 背景

Spec-based 训练建模路径当前将 MoE Expert Parallelism 表达为:

```text
a2a_before + routed_expert_ffn + a2a_after
```

其中 `routed_expert_ffn` 是一个融合的 routed expert FFN 计算 Op，EP dispatch/combine 通过 `Collective(group="EP", kind="A2A")` 独立插入。这个结构适合普通 EP 建模，但无法把 vLLM-Ascend `dispatch_ffn_combine` 一类通信-计算融合 kernel 表达成网站/外部实现所称的 `mega_moe` 单算子，也无法在 IR 层清楚表达 wave 级专家调度。

本设计将 `mega_moe` 作为一个显式开关。关闭时保留现有逻辑和数值行为；打开时，在 spec IR 中真正替换原来的 `routed_expert_ffn + EP A2A collectives` 为单个 `mega_moe` Op。

参考:

- vLLM-Ascend: `csrc/mc2/dispatch_ffn_combine`
  <https://github.com/vllm-project/vllm-ascend/tree/main/csrc/mc2/dispatch_ffn_combine>
- vLLM-Ascend W4A8 variant: `csrc/mc2/dispatch_ffn_combine_w4_a8`
  <https://github.com/vllm-project/vllm-ascend/tree/main/csrc/mc2/dispatch_ffn_combine_w4_a8>
- DeepSeek-V4 technical report, Section 3.1, fine-grained communication-computation overlap in expert parallelism
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf>
- Existing local analysis: `mega_moe_v2.md`, `docs/ep_spec_path_analysis.md`, `docs/dualpipe_ep_overlap_analysis.md`

## Goals

1. 在 spec path 中新增可开关的 `mega_moe` 单 Op 表达。
2. 开关关闭时，当前 `routed_expert_ffn + EP A2A collectives + ep_overlap` 逻辑完全保留。
3. 开关打开时，不再为该 MoE routed expert 插入独立 EP A2A collectives，而是在 `mega_moe.meta` 中携带 dispatch/combine payload、expert compute、wave 划分和融合调度参数。
4. 在 stage time 中使用 wave 级 dispatch/compute/combine 流水模型估算 `mega_moe` 时间。
5. 报告和调试输出中能看到算子名 `mega_moe`，用于和外部 fused kernel 名称对齐。
6. 支持普通 fused 版本和 W4A8 fused 版本两种量化变体；当前通过已有 model dtype/quant preset 自动推导。

## Non-Goals

1. 不实现真实 NPU/GPU kernel。
2. 不改变 graph-capture 路径。
3. 不强行重标定所有 DeepSeek V4 anchor；仅新增打开开关后的测试和必要 anchor。
4. 不把 shared expert 合进 `mega_moe`。本设计只融合 routed expert dispatch/compute/combine。

## Configuration

在 `Strategy` 中增加开关:

```python
mega_moe: bool = False
mega_moe_waves: int = 0
```

语义:

- `mega_moe=False`: 默认值，保留现有行为。
- `mega_moe=True`: 对 MoE routed expert 使用 `mega_moe` 单 Op。
- `mega_moe_waves=0`: 自动选择 wave 数，优先使用硬件配置中的推荐值；没有推荐值时使用保守默认。
- `mega_moe_waves>0`: 用户显式指定 wave 数。

YAML 映射:

```yaml
strategy:
  ep: 8
  mega_moe: true
  mega_moe_waves: 4
```

该开关与 `ep_overlap` 的关系:

- `mega_moe=false`: `ep_overlap` 继续控制现有 A2A 与 GEMM overlap。
- `mega_moe=true`: `mega_moe` 自带 wave 流水建模，忽略该 Op 上的 `ep_overlap` 后处理，避免重复隐藏通信。

量化配置不新增重复开关。`mega_moe` 从现有 `ModelSpec` dtype 字段推导量化变体:

- `quant_variant="w4a8"`: `routed_expert_weight_dtype == Dtype.FP4` 且 `effective_moe_act_dtype()` 为 FP8。
- `quant_variant="standard"`: 其他组合，覆盖 BF16/BF16、FP8/BF16、BF16/FP4 等非 W4A8 fused 建模。

现有 YAML 示例:

```yaml
model:
  base: deepseek_v4_pro
  quant_preset: deepseek_v4_fp8_fp4
strategy:
  ep: 8
  mega_moe: true
```

其中 `deepseek_v4_fp8_fp4` 已展开为:

```text
routed_expert_compute_dtype = fp8_e4m3
routed_expert_weight_dtype  = fp4
moe_act_dtype               = fp8_e4m3
```

因此自动选择 W4A8 `mega_moe` 计时路径。

## IR Design

### Off Path

开关关闭时 IR 保持:

```text
Collective(a2a_before_Lx.routed_expert_ffn)
Op(kind="matmul", name="Lx.routed_expert_ffn")
Collective(a2a_after_Lx.routed_expert_ffn)
```

`stage.py` 继续用当前 EP raw comm + `_wave_overlap_saved()` 逻辑。

### On Path

开关打开时 IR 改为:

```text
Op(kind="mega_moe", name="Lx.mega_moe")
```

不再生成:

```text
a2a_before_Lx.routed_expert_ffn
a2a_after_Lx.routed_expert_ffn
```

`mega_moe` 的 inputs/outputs 与原 routed expert 计算保持同一语义:

```text
input:  x_ln2           shape=(seq, hidden)
output: routed_ffn_out  shape=(seq, hidden)
```

`expert_agg` 继续保留，用于聚合 shared expert 和 routed expert 输出。

### Meta Fields

`mega_moe.meta` 至少包含:

```python
{
    "m": seq,
    "n": hidden,
    "k": moe_ffn,
    "num_experts": model.num_experts,
    "top_k": model.top_k,
    "ep": strategy.ep,
    "experts_per_rank": model.num_experts // strategy.ep,
    "waves": resolved_waves,
    "act_bytes": model.effective_moe_act_dtype().bytes,
    "out_bytes": model.act_dtype.bytes,
    "weight_bytes": model.routed_expert_weight_dtype.bytes,
    "weight_stored_bytes": model.routed_expert_weight_dtype.stored_bytes,
    "quant_variant": "standard" | "w4a8",
    "fwd_multiplier": 3 * model.top_k,
    "fused_dispatch_compute_combine": True,
}
```

TP/CP sharding continues to write `k_local`, sequence-local shapes, and tensor shapes as it does today. EP must not multiply down routed expert compute, matching the existing invariant: under uniform routing, per-rank routed expert GEMM time is approximately invariant as EP changes.

## Sharding And Collective Insertion

`insert_collectives()` keeps its current order:

```text
TP -> CP -> EP
```

For EP:

- If `strategy.mega_moe` is false, keep `_insert_ep_collectives()` unchanged.
- If `strategy.mega_moe` is true:
  - skip EP A2A collective insertion for `mega_moe` ops;
  - apply EP metadata to `mega_moe`;
  - keep router local expert shape adjustment;
  - leave shared expert ops untouched.

The skip should be op-local, not global. If a future model contains both legacy routed expert ops and `mega_moe` ops, only `mega_moe` skips A2A insertion.

## Cost Model

Add an `op.kind == "mega_moe"` branch in `flops.py` and `stage.py`.

### Quantization Variants

`mega_moe` uses one IR kind for all fused variants and selects the internal cost path through `op.meta["quant_variant"]`.

`standard` variant:

- follows the current routed expert compute dtype for peak FLOPs;
- uses `effective_moe_act_dtype().bytes` for dispatch activation payload;
- uses `routed_expert_weight_dtype.stored_bytes` for weight traffic;
- covers the non-W4A8 fused `dispatch_ffn_combine` path.

`w4a8` variant:

- selected when routed expert weights are FP4 and MoE activations are FP8;
- uses FP4 stored weight bytes, including block scale overhead from `Dtype.stored_bytes`;
- uses FP8 activation bytes for dispatch and expert GEMM input;
- uses the FP4/FP8 peak path already supported by `perf_tables.peak_tflops_for`;
- covers the vLLM-Ascend `dispatch_ffn_combine_w4_a8` path.

The first implementation should not introduce new dtype enums for W4A8. It is a compound kernel variant inferred from existing weight and activation dtypes.

### FLOPs

Forward expert FFN FLOPs:

```text
F_l1 = 4 * seq * top_k * hidden * moe_ffn
F_l2 = 2 * seq * top_k * hidden * moe_ffn
F_swiglu = 5 * seq * top_k * moe_ffn
F_total = F_l1 + F_swiglu + F_l2
```

This is per rank after dispatch under uniform routing, not divided by EP. TP and CP sharding still apply through `k_local` and local sequence.

Backward uses the same convention as current matmul cost:

```text
dx ~= fwd
dw ~= fwd
```

or preserves the existing `OpCost` phase ratios if the codebase standardizes them elsewhere.

### HBM Bytes

The fused path should count:

- input activation read;
- final routed output write;
- routed expert weight read for local experts touched by the rank;
- quantization scale-factor reads when dtype metadata supports it;
- reduced intermediate activation HBM traffic compared with unfused up/gate/swiglu/down.

Intermediate up/gate/SwiGLU tensors should not be counted as full HBM read/write traffic in `mega_moe`, because the fused operator keeps those within the fused pipeline.

For W4A8, weight traffic uses `routed_expert_weight_dtype.stored_bytes`, not `.bytes`, so FP4 block-scale overhead is included. Activation traffic uses `effective_moe_act_dtype().bytes`, which is FP8 for the existing `deepseek_v4_fp8_fp4` preset.

### Communication Bytes

Dispatch:

```text
dispatch_bytes =
  micro_batch * seq_local * hidden_local * top_k * moe_act_bytes * remote_fraction
```

Combine:

```text
combine_bytes =
  micro_batch * seq_local * hidden_local * top_k * out_bytes * remote_fraction
```

where:

```text
remote_fraction = (ep - 1) / ep
```

This is intentionally separated from the existing A2A payload convention, which divides the total payload by EP for each A2A collective. To keep switch-on timing comparable with the current EP path, `mega_moe` timing uses the existing collective-time-compatible payload. Reporting also carries remote-total payload for diagnostics.

`mega_moe.meta` should carry both:

- `dispatch_bytes_per_rank` for reporting;
- `dispatch_bytes_effective` for timing, using the same per-rank convention as the current EP A2A collective path.

Tests must assert both fields so report volume and timing do not drift silently.

## Wave Pipeline Model

Resolve:

```text
experts_per_rank = num_experts / ep
waves = resolved_waves
experts_per_wave = experts_per_rank / waves
```

Constraints:

- `waves >= 1`
- `waves <= experts_per_rank`
- prefer values that divide `experts_per_rank`;
- if not divisible, allow the last wave to be smaller or round down to a valid divisor.

For each wave:

```text
d = dispatch_time_per_wave
c = compute_time_per_wave
r = combine_time_per_wave
```

The scheduler models dependencies:

```text
compute(w) starts after dispatch(w)
combine(w) starts after compute(w)
```

Steady state allows communication and compute to overlap:

```text
dispatch(w+1) overlaps compute(w)
combine(w-1) overlaps compute(w)
```

Initial implementation can use a deterministic two-resource event simulator:

- one compute resource for expert FFN;
- one communication resource for dispatch/combine, unless hardware metadata explicitly allows separate send/recv engines;
- FIFO wave order.

This is more faithful than applying `_wave_overlap_saved()` to a layer-level aggregate, because it preserves startup, drain, and combine tail effects.

## Interaction With A2/A3

For Ascend A2/A3 style fused MC2/MoE modeling, `mega_moe` should be hardware-parameterized rather than hardcoding chip behavior:

- link bandwidth and latency still come from `SystemSpec.interconnect`;
- compute throughput still comes from `GPU`/accelerator peak and efficiency tables;
- `mega_moe_waves` can be set by config for A2/A3-specific scheduling;
- optional future fields can model separate dispatch/combine engines or HCCS/MC2 efficiency without changing IR shape.

This keeps the operator name aligned with the external fused kernel while preserving the simulator's hardware-independent design.

## Reporting

When `mega_moe=true`, reports should show:

- op name: `Lx.mega_moe`;
- op kind: `mega_moe`;
- quant variant: `standard` or `w4a8`;
- activation dtype and routed expert weight dtype;
- dispatch bytes;
- combine bytes;
- waves;
- exposed communication time;
- hidden communication time;
- total fused op time;
- bottleneck: `compute`, `memory`, or `comm_tail`.

Existing EP A2A communication summaries should not double-count skipped collectives.

## Testing

Add focused tests:

1. Config parsing:
   - default `mega_moe` is false;
   - YAML can set `mega_moe: true`;
   - YAML can set `mega_moe_waves`.
   - `quant_preset: deepseek_v4_fp8_fp4` plus `mega_moe: true` resolves `quant_variant="w4a8"`.

2. IR shape:
   - off path contains `routed_expert_ffn` and two EP A2A collectives;
   - on path contains `mega_moe`;
   - on path contains no EP A2A collectives around that op;
   - shared expert ops and `expert_agg` remain.
   - `mega_moe.meta` records `quant_variant`, activation dtype bytes, and stored weight bytes.

3. Cost behavior:
   - `mega_moe` has nonzero forward/backward compute;
   - EP compute is not divided by EP under uniform routing;
   - communication is not double-counted in `StageTime.ep_exposed`.

4. Wave model:
   - `waves=1` behaves like mostly serial dispatch/compute/combine;
   - increasing valid waves reduces exposed communication when compute can cover comm;
   - invalid wave counts are normalized or rejected deterministically.

5. Quantization:
   - standard and W4A8 variants both produce nonzero compute and memory cost;
   - W4A8 weight bytes are lower than BF16 standard weight bytes;
   - W4A8 activation dispatch bytes use FP8 bytes.

6. Backward compatibility:
   - with `mega_moe=false`, existing EP tests and anchors remain unchanged.

## Risks

1. Communication payload convention may differ between current A2A collectives and fused-kernel reporting. Mitigation: encode both timing payload and reporting payload explicitly in meta.
2. Current `stage.py` applies EP imbalance after base timing. `mega_moe` must still apply imbalance to expert compute and dispatch/combine traffic, but not to unrelated attention/shared expert compute.
3. Anchor changes can be noisy if `mega_moe` is enabled in existing configs. Mitigation: default off and add dedicated mega_moe configs/tests.
4. Backward pass for a fused inference-oriented kernel may not map 1:1 to training. Mitigation: keep phase-specific meta and start with conservative current backward conventions.
5. W4A8 kernels may have backend-specific scale handling. Mitigation: initial model uses existing `Dtype.stored_bytes` and reports scale-overhead assumptions explicitly.

## Decisions

1. `mega_moe` is exposed through `Strategy` and YAML `strategy` only. Model presets may enable it later, but this change does not alter existing model YAMLs.
2. `mega_moe=False` is the default.
3. `mega_moe_waves=0` resolves to `system.gpu.ep_overlap_waves` when positive, else `4`, clamped to valid experts-per-rank divisors.
4. The initial scheduler uses one communication resource for dispatch/combine. Separate send/recv engines can be added later as a hardware capability flag without changing IR shape.
5. Timing payload uses the current EP A2A per-rank convention. Reporting payload also includes remote-total bytes.
6. Quantization variant is inferred from existing model dtypes. `FP4` routed expert weights plus FP8 MoE activations select `w4a8`; all other combinations select `standard`.

## Approval State

User approved the recommended true IR replacement approach and added the requirement that mega_moe must be controlled by a switch:

```text
mega_moe=false -> existing logic
mega_moe=true  -> replace routed_expert_ffn + A2A collectives with mega_moe Op
```
