# Graph Capture 处理全流程：EP、Recompute 与报表输出

## 1. 图处理全流程

### 1.1 CLI 入口

```
python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --ep 8
```

MegaMoE 开启样式：

```
python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --ep 8 --mega-moe --mega-moe-waves 0
```

- `--mega-moe`：在 graph-capture 训练建模路径中，把 routed expert 的 dispatch / expert compute / combine 作为一个 `mega_moe` 融合算子建模。
- `--mega-moe-waves 0`：未显式指定 wave 时，枚举合法的 experts-per-rank divisor，并用内部流水模型自动选择最优 wave。
- `--mega-moe-waves N`：显式指定 wave 数，用于复现实验或对齐外部 kernel 配置。

执行链路：

```
main()                                          # cli.py:250
│
├── [1] run_trace_phases()                      # cli.py:312 — 模型捕获
│   ├── load_model() → FakeTensorMode           # 加载 DSv4 4 层
│   ├── train_forward: model.train() → forward  # 捕获前向 aten ops
│   ├── train_backward: loss.backward()         # 捕获反向梯度 ops
│   └── 返回 TracePhaseResult (graphs + records)
│
├── [2] _run_training_modelling()               # cli.py:331 — 建模
│   ├── estimate_training_from_graphs()         # modeller.py:26
│   │   ├── stitch_fwd_bwd() → unified OpGraph  # 拼接正反向
│   │   ├── TransformContext(ep, tp, training, profile)
│   │   ├── build_default_pipeline().run()      # ⬇ 全管线
│   │   └── 返回 (TrainingReport, ctx, {"unified": OpGraph})
│   ├── print(report.summary())                 # 控制台输出
│   └── export_training_graphs() + export_reports()  # Excel/JSON/HTML
```

### 1.2 Transform Pipeline 四阶段

`build_default_pipeline()` 注册的 pass 及其顺序 (`pipeline.py:101-150`)：

```
Stage 1 (split)
  ├── DataParallelPass         (条件: dp>1 + training)
  ├── TensorParallelPass       (条件: tp>1)
  ├── ExpertParallelPass       (条件: ep>1)           ← EP 在这里
  ├── ContextParallelPass      (条件: cp>1)
  ├── RecomputePass            (条件: is_training)     ← Recompute 在这里
  ├── OffloadPass              (条件: offload.pct>0)
  ├── CommInserterPass         (条件: tp>1 or ep>1 or cp>1) ← EP 的 A2A 在这里
  └── PipelineParallelPass     (条件: pp>1)

Stage 2 (fuse)
  └── FusionPass               # 算子融合 → 合并 aten ops

Stage 3 (optim)
  ├── ZeroFSDPPass             (条件: training)
  └── OptimizerPass            (条件: training)

Stage 4 (analyze)
  ├── FlopsPass                # FLOPs、read/write bytes 标注
  ├── RooflinePass             # compute/memory/latency bound
  ├── CommLatencyPass          # 通信延迟
  ├── StreamAssignPass         # 流分配
  ├── TrainingFlopsPass        (条件: training)  # 训练 FLOPs 汇总
  ├── TrainingMemoryPass       (条件: training)  # 内存 breakdown
  └── TrainingPipelinePass     (条件: training)  # step_time/MFU/HFU
```

**关键顺序**：EP 标注 (split) → 通信插入 (split) → 融合 (fuse) → FLOPs 计算 (analyze)

---

## 2. EP（Expert Parallelism）

### 2.1 执行路径

```
ExpertParallelPass (split 阶段第 3 个 pass)
├── 条件: ctx.parallel.ep > 1 且 ctx.profile.num_experts > 1
├── 动作: 遍历所有节点，_is_expert_scope(scope) → True 的节点设置:
│   ├── annotations["ep_needs_a2a"] = True
│   └── annotations["ep_experts_local"] = num_experts // ep
└── 注意: 不修改 tensor shape，不修改 FLOPs

CommInserterPass (split 阶段第 7 个 pass)
├── 条件: ep > 1
├── 动作: 读取 ep_needs_a2a 标注，对每个 MoE block:
│   ├── 插入 dispatch A2A (comm.all_to_all, role=dispatch)
│   │   放在第一个 expert 节点之前
│   ├── 插入 combine A2A (comm.all_to_all, role=combine)
│   │   放在最后一个 expert 节点之后
│   └── 设置 ep_a2a_inserted = True（防重复）
└── msg_bytes = micro_batch × seq_len × hidden × topk × dtype_bytes // ep
```

### 2.2 当前 EP 对计算结果的影响

```
统一图 (stitched)
│
├── EP=1 (无 EP)
│   ├── 0 个 ep_needs_a2a 标注
│   ├── 0 个 A2A 节点
│   └── 所有节点正常计算
│
├── EP=8 (有 EP)
│   ├── N 个 expert 节点有 ep_needs_a2a + ep_experts_local 标注
│   ├── 2×MoE层数 个 A2A 节点 (dispatch + combine)
│   ├── expert 节点 output shape 不变 ⚠️ (应有缩减)
│   ├── router output 尾维不变 ⚠️ (应为 experts_per_rank)
│   ├── A2A 通信量 = B×S×H×topk×2÷EP ⚠️ (未考虑 TP 分片)
│   ├── expert FLOPs 不变 ⚠️ (应有 1/EP 缩减)
│   ├── TP all_reduce 不受影响 ✓
│   └── TrainingReport:
│       ├── step_time_ms 不同 (EP 引入通信开销)
│       └── MFU/HFU 不同
```

### 2.3 理想的 EP 行为（XFAIL 清单）

| 项目 | 当前 | 理想 |
|------|------|------|
| A2A phase 标注 | 无 | `phase="both"`（正反向对称） |
| A2A overlap 标记 | 无 | `overlap_target` 指向后续算子 |
| A2A msg_bytes | 用完整 hidden | 用 `hidden ÷ TP` |
| Expert FLOPs | 不变 | 缩 1/EP（per-rank） |
| Router output | 尾维 = num_experts | 尾维 = experts_per_rank |
| Expert token 数 | 不变 | dispatch 后 seq 维缩为 B×S×topk÷EP |
| GroupedMM | 无 | gate_up + down 各自融合为 GroupedMatMul |
| 图结构 | router → 散落 experts → shared | dispatch → GroupedMM(gate_up) → silu → GroupedMM(down) → combine → +shared |

### 2.4 EP 开启后的算子图（当前）

```
Dense Layer (L0)
  attn → mlp → ...

MoE Layer (L1)
  router ──→ [dispatch A2A]               ← comm.all_to_all, role=dispatch
          → expert_0_gate → expert_0_down
          → expert_1_gate → expert_1_down
          → ... (Nexperts/EP 个专家本地计算)
          → [combine A2A]                  ← comm.all_to_all, role=combine
          → +shared_expert_gate → ... → shared_expert_down
```

### 2.5 EP + GroupedMM 的理想图（XFAIL 目标）

```
MoE Layer (L1)
  router ──→ [dispatch A2A]
          → GroupedMM(gate_up_proj)        ← (G, M, H) @ (G, H, 2×ffn)
          → SwiGLU activation
          → GroupedMM(down_proj)           ← (G, M, ffn) @ (G, ffn, H)
          → [combine A2A]
          → +shared_expert (单独 aten, 不进 GroupedMM)
```

---

## 3. Recompute（激活重计算）

### 3.1 执行路径

```
RecomputePass (split 阶段第 5 个 pass)
├── 条件: ctx.training is not None
├── 策略: ctx.training.recompute_policy (默认 "none")
├── "none":   直接返回，不标注
├── "full":   所有 fwd-phase 节点标注 recompute=True
└── "selective": 仅 softmax / o_proj / flash_attn scope 的节点标注

FlopsPass (analyze 阶段)
├── 读取 recompute 标注
├── rec_mult = 2.0 if recompute=True and not is_bwd else 1.0
├── node.annotations["flops_fwd"]  = train_flops × rec_mult
├── node.annotations["flops"]      = train_flops  (不受 recompute 影响)
└── node.annotations["flops_dx/dw"] = grads        (不受 recompute 影响)

TrainingFlopsPass (analyze 阶段)
├── recompute_flops = Σ(flops_fwd // 2) for recompute-annotated fwd nodes
└── g.metadata["recompute_flops"] = recompute_flops

TrainingPipelinePass (analyze 阶段)
├── MFU = (training_flops - recompute_flops) / PP / peak / time
└── HFU = training_flops / PP / peak / time
```

### 3.2 Recompute 对计算结果的影响

```
原始前向: 1× fwd + ~2× bwd = ~3× 总计算量

无 Recompute (none)
├── flops_fwd = flops (1×)
├── training_flops = flops_fwd + flops_dx + flops_dw
├── recompute_flops = 0
└── MFU = HFU (无重算开销)

有 Recompute (full)
├── flops_fwd = flops × 2 (1× 原始 + 1× 重跑)
├── flops 不变 (原始前向 pass 本身不变)
├── flops_dx/dw 不变 (反向梯度计算不变)
├── training_flops = Σ(flops_fwd) + Σ(flops_dx) + Σ(flops_dw)
├── recompute_flops = Σ(flops_fwd // 2) for recompute nodes
├── MFU = (training_flops - recompute_flops) / PP / peak / time  ── 不含重算
└── HFU = training_flops / PP / peak / time                        ── 含重算
    → HFU > MFU

有 Recompute (selective)
├── 仅 attention 相关算子 flops_fwd × 2
├── FFN/ln 等保持 1×
├── recompute_flops 介于 none 和 full 之间
└── MFU 在 none 和 full 之间
```

### 3.3 FLOPs 核心语义总结

```
                    ┌─────────┬──────────┬──────────┐
                    │  none   │  full    │ selective│
├──────────────────┼─────────┼──────────┼──────────┤
│ flops (原始前向)  │ 1×      │ 1×       │ 1×       │  ← 永不变
│ flops_fwd        │ 1×      │ 2× (标注) │ 1× or 2× │  ← 标注节点乘 2
│ flops_dx         │ 不变    │ 不变     │ 不变     │  ← 永不变
│ flops_dw         │ 不变    │ 不变     │ 不变     │
│ recompute_flops  │ 0       │ >0       │ 0..full  │
│ MFU              │ = HFU   │ < HFU    │ ≤ HFU    │
└──────────────────┴─────────┴──────────┴──────────┘
```

---

## 4. 报表输出

### 4.1 运行命令

```bash
# EP=8, TP=8, 无 Recompute（默认 none）
python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --ep 8

# EP=1（无 EP）, TP=8
python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --ep 1
```

> 注：CLI 尚无 `--recompute-policy` 参数。要在全管线中测试 Recompute，需通过代码调用：
> ```python
> estimate_training_from_graphs(recompute_policy="full", ...)
> ```
> （`modeller.py` 已新增 `recompute_policy` 参数，默认 `"none"`，对现有 CLI 行为无影响）

### 4.2 输出文件

```
output/<model_slug>/
├── <slug>_train_forward_ops.xlsx      # 前向导出
├── <slug>_train_backward_ops.xlsx     # 反向导出
├── <slug>_training.xlsx               # 训练报表 (Transformed Operators sheet)
└── reports/
    ├── <slug>_train_hier.html          # 层级 HTML 报告
    ├── <slug>_train_trace.json         # Chrome Trace
    └── <slug>_training_report.json     # JSON 报告
```

### 4.3 核心报表列（Transformed Operators sheet）

每行一个算子，关键列：

| 列 | 来源 | EP 开启后 |
|----|------|----------|
| Op Type | 算子类型 | expert 节点仍为 aten/fused；A2A 为 `comm.all_to_all` |
| Category | compute/communication | A2A 为 communication |
| Scope | 模块路径 | 不变 |
| Input/Output Shapes | tensor shape | A2A 为 (batch, seq, hidden) |
| FLOPs | FlopsPass | expert FLOPs 不变 ⚠️ |
| Comm Volume (B) | A2A msg_bytes | = B×S×H×topk×2÷EP |
| Annotations | 所有标注 | ep_needs_a2a, recompute, ep_a2a_inserted, etc. |

---

## 5. 测试文件索引

```
tests/IT/
├── conftest.py              # 共享模型捕获 (captured_model)
├── test_ep_e2e.py           # EP: UT 14 + E2E 27 = 41 tests
└── test_recompute_e2e.py    # Recompute: UT 16 + FLOPs UT 4 + E2E 13 = 33 tests
```

| 测试文件 | 运行命令 | 需 PYTHONPATH |
|----------|---------|--------------|
| EP E2E | `py -m pytest tests/IT/test_ep_e2e.py -v` | 无（UT 部分）/ `PYTHONPATH=python`（E2E 部分） |
| Recompute E2E | `py -m pytest tests/IT/test_recompute_e2e.py -v` | `PYTHONPATH=python`（全部） |
| 全量 | `PYTHONPATH=python py -m pytest tests/IT/ -v` | `PYTHONPATH=python` |
