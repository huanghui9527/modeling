# ZRT-Sim 架构快照（2026-06-01）

> **来源**：本文档来自真实代码扫描（`wc -l` / `grep import` / 目录树），不依赖任何已有 markdown 或 docstring。
> **图渲染**：所有架构图已预渲染为 SVG（位于 [`docs/diagrams/`](diagrams/)），在任何 markdown viewer 中都能稳定显示，**不依赖任何 mermaid 插件**。
> **图编辑流程**：源码 `.mmd` 文件就在 SVG 旁边——改完后跑 `bash docs/diagrams/regenerate.sh` 一键重渲染。
> **分层方式**：参考 [C4 模型](https://c4model.com/) — L1 Context → L2 Container → L3 Component，逐层下钻。
> **重构状态**：§ 12 记录了 2026-05-30 至今的架构归一化重构进展（Phase 1-6 + B1-B2 已完成）。

---

## 0. 一句话结论

> **重构前（2026-05-29）**：仓库**事实上是两个并存的产品**：路径 A（抓图建模）和路径 B（配置建模）各有一套 IR、成本模型、调度与报告。两者并非干净分离，而是靠 **28 处反向 import（跨 10 个文件）**把路径 A 接到路径 B——路径 A 在内部直接调用了路径 B 的代码。
>
> **重构后（2026-06-01）**：已选择 **OpGraph 为唯一 IR**（§ 11.2 选项 B），完成 Phase 1-6 + B1-B2。两条路径在 OpGraph 层汇合，共享 Transform Pipeline。旧 IR（`Graph`/`Op`/`Tensor`）仅作为兼容层保留，主路径不再经过。详见 **§ 12 架构归一化重构**。

**当前状态**：

1. **✅ 已解决**：Graph → OpGraph → Graph 往返转换已消除（Phase B2）
2. **✅ 已解决**：Stack A 独立计算路径已统一到 Transform Pipeline（Phase 2+6）
3. **✅ 已解决**：`_report_from_transformed()` 字段对齐（Phase 5A）
4. **✅ 已解决**：Pipeline 精度对齐（Phase 5B）
5. **⏳ 待解决**：198 处测试仍引用旧 `build_graph`，`training_graph.py` + `graph_adapter.py` 保留为兼容层（Phase B3）
6. **⏳ 待解决**：26 个测试失败（14 个 Strategy validation + 4 个新增回归 + 8 个预存问题）

**历史遗留问题**（详见 § 5、§ 7）：

1. **28 处隐式桥**：路径 A 的 `transform/`、甚至最底层的 `ir/`，反向 import 路径 B 的 `training/`。其中 `ir/types.py` 顶层 `from zrt.training.spec.dtype import Dtype`——最底层的 IR 依赖最上层的 training，循环依赖风险极高。
2. **DType 有 3 份定义**，且 canonical 落在了最上层的 `training/spec/dtype.py`，`ir/types.py` 只是 re-export，`layers/op_base.py` 还有一份独立的简陋版。依赖方向是反的。
3. **Chrome Trace 有 3 份实现**：`report/chrome_trace.py`、`executor/chrome_trace.py`、`training/trace/exporter.py`。
4. **1 处真冗余调用**：`html_writer.py:1304` 每次导 HTML 都重跑一遍 simulator，而结果其实已经算在节点上了（见 § 5.2）。

---

## 1. L1 系统上下文

![L1 系统上下文](diagrams/01_system_context.svg)

<sub>📐 源码: [`docs/diagrams/01_system_context.mmd`](diagrams/01_system_context.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

---

## 2. L2 全景视图（核心图，先看这张）

### 2.1 组件全景图（带职责说明）

> 每个组件 3 行：**名称 · 目录 LOC · 一句话职责**

![L2 组件全景图](diagrams/02_panorama.svg)

<sub>📐 源码: [`docs/diagrams/02_panorama.mmd`](diagrams/02_panorama.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

**怎么看这张图**

- 同色 = 同路径。**橙色（A）和紫色（B）几乎是镜像**：各自一套 IR + 成本模型 + 调度 + 报告
- 深绿色 `hardware/` 是**真共享**；浅绿色其他几个其实只服务路径 A
- 两条虚线 = **桥**：粗的一条是路径 A 反向调用路径 B 成本模型（28 处），细的一条是最底层 `ir/` 反向依赖 `training.spec.dtype`——这是当前架构的病灶
- B 没有调度执行层（无 executor）：spec → 成本模型 → composer 直接算时间，不展开 DAG

### 2.2 功能 × 模块矩阵（一眼看重复）

> 看每一行：**有 ≥2 个 ✓ 的 = 当前有重复实现**。✦ 标记 = 不是平行实现而是桥（path A 调 path B）

| 能力 | A: transform/analysis | A: memory/ | A: report/ | A: simulator/ | B: training/models | B: training/compose | B: training/io | B: training/trace | 共享: ir/ | 共享: layers/ | 共享: hardware/ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **per-op FLOPs 公式** | ✓ FlopsPass | | | ✓ roofline 内嵌 | ✓ op_cost | | | | | | |
| **整图 FLOPs 聚合** | ✓ TrainingFlopsPass（聚合 FlopsPass 结果） | | | | ✓ total_training_flops | | | | | | |
| **通信耗时** | ✓ CommLatencyPass | | | | ✓ collective_time | | | | | | |
| **per-op latency** | ✓ RooflinePass | | | ✓ RooflineSimulator | | | | | | | |
| **整图内存峰值** | ✓ TrainingMemoryPass（独立公式） | ✓ activation / budget | | | ✓ memory_breakdown | | | | | | |
| **PP 调度建模** | ✦ TrainingPipelinePass（**调 B**） | | | | | ✓ 6 Composer | | | | | |
| **Chrome Trace 导出（3 份）** | | | ✓ report/chrome_trace + ✓ executor/chrome_trace | | | | | ✓ trace/exporter | | | |
| **HTML 报告** | | | ✓ html_writer | | | | ✓ html_exporter | | | | |
| **Excel 报告** | | | ✓ excel_writer（在 exporter.py） | | | | ✓ excel_exporter | | | | |
| **DType 定义（3 份）** | | | | | spec/dtype（canonical，在 B） | | | | re-export ir/types | ✓ op_base（简陋版） | |
| **HardwareSpec 加载** | | | | | | | ✓ config_loader | | | | ✓ registry |
| **Op / Tensor IR** | | | | | | | | | ✓ OpGraph | | |
| **训练侧 IR (Op/Coll)** | | | | | ✓ training/ir/Graph | | | | | | |
| **DAG 调度** | （路径 A 有 executor/） | | | | | ✓（隐式，含在 Composer 里） | | | | | |
| **YAML 配置加载** | | | | | | | ✓ config_loader | | | | ✓（hardware 自己加载） |

**重复实现热点**（按合并优先级，详见 § 7）

| 热点 | 位置 A | 位置 B | LOC | 性质 | 建议 |
|---|---|---|---:|---|---|
| 1. per-op FLOPs 公式 | `FlopsPass` (passes.py) | `training/models/flops.py::op_cost` | 900+ | 真重复 | 统一到 `OpCost` |
| 2. 通信耗时 | `transform/analysis/comm_latency.py` (143) | `training/models/comm.py` (635) | 778 | 真重复 | 保留 B，A 改薄 wrapper |
| 3. 内存估算 | `memory/*` + `TrainingMemoryPass`（独立） | `training/models/memory.py` | 1000+ | 真重复 | 待决策 |
| 4. PP 调度 | `TrainingPipelinePass`（其实是桥，调 B） | `training/compose/schedules.py` | 1279 | **不是重复，是桥** | 保留桥即可 |
| 5. Chrome Trace（3 份） | `report/chrome_trace.py` (234) + `executor/chrome_trace.py` (550) | `training/trace/exporter.py` (90) | 874 | 真重复 | 三合一 |
| 6. DType（3 份，方向反） | `layers/op_base.py:5`（简陋版）+ `ir/types.py`（re-export） | `training/spec/dtype.py`（canonical） | — | 方向反了 | 见 § 5.1 |
| 7. **html_writer 真冗余调用** | `report/html_writer.py:1304` 每次都重跑 simulator | — | 1 行 | **同路径内浪费** | 改读 annotations，详见 § 5.2 |

---

## 3. L3-A 路径 A 组件视图（抓图建模）

### 3.0 路径 A 全景图（8 阶段 · 所有 pass 可见）

> 节点颜色：橙=普通组件，深橙描边=训练 pass（桥），红=与路径 B 重复实现

![路径 A 全景](diagrams/03_path_a.svg)

<sub>📐 源码: [`docs/diagrams/03_path_a.mmd`](diagrams/03_path_a.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

**怎么看这张图**

- **8 个阶段从上到下**：HF 输入 → 抓图 → Parallel → Fusion → Training pass → Analysis → 调度 → 成本 → 报告
- **同阶段内的组件是平行的 pass**（实际顺序在 § 3.4 / 3.5 表格里）
- **深橙色描边 = 桥**（反向 import 路径 B）：图中标出的 `OptimizerPass` + `TrainingPipelinePass` + `FlopsPass` + `graph_resolve_op_dtypes`(quant.py) 是主要桥点；`RecomputePass / OffloadPass / ZeroFSDPPass / TrainingFlopsPass / TrainingMemoryPass` 是纯路径 A（无反向 import）。全仓 28 处桥见 § 5.1
- **红色 = 重复**：`CommLatencyPass` 和 `chrome_trace` 在路径 B 里各有一份
- **fallback 链**：simulator 的 6 个后端通过 `SimulatorHub` 按优先级降序选择

下面展开 8 个模块的更细节内部（类 / 关键函数 / 支持的具体选项，全部来自代码）。

### 3.1 graph/ — 抓图入口（3,142 LOC, 12 files）

**职责**：HF 模型 + `FakeTensorMode` + `TorchDispatchMode` → 拦截 aten 算子 → 构建 OpGraph

| 类 / 函数 | 位置 | 职责 |
|---|---|---|
| `load_model(model_id, num_hidden_layers, training)` | model_loader.py | 返回 `(model, config, fake_mode)`，3 层 fallback：Hub → 本地 registry → 异常 |
| `TensorTracker` | dispatch.py:79 | 给每个张量分配稳定唯一 ID |
| `RecordingDispatch` | dispatch.py:108 | `TorchDispatchMode` 子类，拦截 aten op 记录元数据 |
| `ModuleTracker` | tracker.py:9 | 追踪 module path / class / call_id |
| `apply_compat_patches()` | patches.py | 注入 transformers 缺失符号 |
| `patch_moe_for_fake(model)` | patches.py | 替换 MoE forward，避免 `.cpu().numpy()` 崩溃 |
| `patch_indexer_for_fake(model)` | patches.py | 修正 DeepSeek-V3.2 Indexer 的 3D transpose 错误 |
| `patch_for_training_capture(model)` | patches.py | 启用反向，升级 fp4/fp8 kernel 为可微版本 |
| `build_op_graph(records)` | graph_builder.py | records → `nx.DiGraph` |
| `infer_layer_types(config)` | classifier.py | 返回 `{"dense": [...], "sparse": [...]}` |
| `HCBoundMethodModule` 等 5 个 | patches.py:298-327 | 包装 H-Curve 回调为 nn.Module |

**支持的模型**：HF Hub 任意 causal LM；本地 registry `_LOCAL_REGISTRY`：DeepSeek-V3 / V3.2 / V4
**已知 patch 目标**：MoE（DeepSeek/Mixtral/Qwen3-MoE）、Indexer（V3.2）、V4 自定义 kernel
**版本 shim**（compat.py）：`is_flash_attn_greater_or_equal_2_10`、`is_torch_fx_available`、`DynamicCache.{from,to}_legacy_cache`、`get_usable_length`、`seen_tokens`、`get_max_length`
**Fake kernel**（v4_fake_kernels.py，5 个）：`act_quant`、`fp4_gemm`、`fp8_gemm`、`sparse_attn`、`hc_split_sinkhorn`

---

### 3.2 transform/ 顶层 — 编排 + Context + 导出

**职责**：4 阶有序管道（split → fuse → optim → analyze）编排所有 graph pass

| 类 | 位置 | 职责 |
|---|---|---|
| `TransformPipeline` | pipeline.py:34 | pass 调度器，4 阶固定顺序 |
| `ParallelConfig` | context.py:12 | TP/PP/EP/DP/CP/SP 度数 + property `total_devices` |
| `TransformContext` | context.py:237 | 统一配置容器（parallel / training / quant / offload / fusion） |
| `StreamConfig` | context.py:36 | 多流并发（`num_compute_streams`, `num_comm_streams`） |
| `TransformedGraphExcelWriter` | exporter.py:241 | 通用 Excel 导出器 |
| `TrainingGraphExcelWriter` | exporter.py:1219 | 训练专用 Excel 导出器（fwd / bwd 分 sheet） |

**入口函数**：`build_pipeline(fusion="v2")` → `TransformPipeline`，`export_transformed_graph(graph, ctx, path, format)`，`export_html_report()` / `export_hierarchical_html_report()`

**导出格式**（exporter.py 内）：Excel / JSON / HTML / Dot / ONNX / Chrome Trace

---

### 3.3 transform/fusion/ — 算子融合（3,238 LOC, 29 files）

![fusion 流水](diagrams/04_fusion.svg)

<sub>📐 源码: [`docs/diagrams/04_fusion.mmd`](diagrams/04_fusion.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

| 类 | 位置 | 职责 |
|---|---|---|
| `FusionPass` | fusion/api.py:13 | 主入口，桶式融合 + 富规则匹配 |
| `Pattern` | fusion/core/pattern.py | 模式 AST（op_type + 条件谓词） |
| `Rule` | fusion/core/rule.py | 融合规则（pattern → output op_type） |
| `SlidingWindow` | matching/sliding_window.py | 滑动窗口匹配器 |
| `RuleRegistry` | registry/rule_registry.py | 规则注册表 |
| `CallIdBucketer` | bucketing/call_id_bucketer.py | 按 call_id 分桶 |
| `IOResolver` | building/io_resolver.py | 输入输出张量解析 |
| `SemanticsAnnotator` | semantics/annotator.py | 语义注解 |

**支持平台**：cuda / ascend_npu / cpu / generic（`fusion/rules/` 下分目录）
**匹配策略**：v2（MRO 桶） / v3（torch.export DAG，需 fx_graphmodule）
**YAML 规则目录**：`transform/fusion/configs/`

---

### 3.4 transform/parallel/ — 并行变换（2,030 LOC, 8 files）

| 类 | 位置 | 职责 |
|---|---|---|
| `TensorParallelPass` | tensor_parallel.py:55 | 列性（q/k/v/gate/w1/w3） vs 行性（o_proj/down_proj/w2）切分 |
| `ExpertParallelPass` | expert_parallel.py:24 | 路由专家全切，shared expert 保留；注入 A2A |
| `ExpertGroupedMMPass` | expert_grouped_mm.py:100 | 消费 EP 标注，融合专家分组 mm |
| `DataParallelPass` | data_parallel.py:18 | 梯度同步（仅训练） |
| `ContextParallelPass` | context_parallel.py:38 | 序列并行 |
| `PipelineParallelPass` | pipeline_parallel.py:46 | 阶段划分 |
| `CommInserterPass` | comm_inserter.py:81 | 插入 AllReduce / AllGather / ReduceScatter / All-to-All |

**支持的并行维度**

| 维度 | 切分方式 | 通信原语 |
|---|---|---|
| TP | 列（输出 dim=-1） / 行（输入 dim=-1） | AllReduce |
| EP | 路由专家分散 | All-to-All |
| PP | 任意深度分阶 | P2P |
| DP | 梯度全 reduce | AllReduce |
| CP | 序列分片 | AllGather / RingP2P |
| SP | 开关切换 | AllGather + ReduceScatter |

---

### 3.5 transform/analysis/ — 性能分析（2,443 LOC, 6 files）★ 桥最集中的子包

| 类 / 文件 | 位置 | 写到哪 | 职责 | 是否桥 |
|---|---|---|---|:---:|
| `FlopsPass` | passes.py | `node.annotations["flops_fwd/dx/dw"]` | per-op FLOPs 公式（2 处函数内 import training） | ✅ 桥 |
| `RooflinePass` | passes.py | `node.annotations["latency_us", "bound"]` | per-op 延迟（内部实例化 `RooflineSimulator`） | |
| `CommLatencyPass` | comm_latency.py | `node.annotations["latency_us"]` | AR / AG / A2A 延迟（1 处 import spec.dtype） | 重复 + 桥 |
| `StreamAssignPass` | passes.py | `node.annotations["stream_id"]` | 流分配 | |
| `graph_resolve_op_dtypes` | quant.py | OpNode → dtype bundle | 调 `training.models.quant_dispatch` 解析量化 dtype | ✅ 桥 |
| `TrainingFlopsPass` | training.py | `graph.metadata["training_flops"]` | **聚合** per-op FLOPs；6P fallback | ❌ 纯 A |
| `TrainingMemoryPass` | training.py | `graph.metadata["memory_breakdown"]` | Korthikanti 公式算内存峰值 | ❌ 纯 A |
| `TrainingPipelinePass` | training.py | `graph.metadata["pipeline_metrics"]` | 9 处函数内 import `training.compose / spec / models / io` 算 step_time | ✅ **最大桥** |

**两点说明**

- `TrainingFlopsPass` / `TrainingMemoryPass` 不是桥：它们没有反向 import，是纯路径 A 的聚合 pass——输入是 `FlopsPass` 写好的 per-op annotations，输出是 `graph.metadata` 里的整图汇总值，与路径 B 的 `op_cost` 不重叠。
- 桥在本子包内最密集：`training.py` 9 处 + `quant.py` 3 处 + `passes.py` 2 处 + `comm_latency.py` 1 处 + `modeller.py` 1 处 = **16 处，占全仓 28 处桥的一半多**。详见 § 5.1。

**Roofline 支持的 op type**（71+，roofline.py:1231）：`mm` / `addmm` / `bmm` / `linear` / `lm_head` / `conv2d` / `conv3d` / `sdpa` / `flash_attn` / `rms_norm` / `layer_norm` / `softmax` / `sort` / `topk` / `elementwise` / `activation` / ...

---

### 3.6 transform/training/ — 训练专用 pass（1,085 LOC, 4 files）★ 含 1 个桥

| 类 | 位置 | 支持的选项 | 是否桥 |
|---|---|---|:---:|
| `RecomputePass` | recompute.py:33 | none / full / selective（softmax + attn proj） | ❌ 纯 A |
| `OptimizerPass` | optimizer.py:24 | adam / adamw / muon（含 NS 迭代数 + 参数占比 + 旋转优化） | ✅ **桥** |
| `OffloadPass` | offload.py:10 | pct[0-1] + opt_state / grads / params | ❌ 纯 A |
| `ZeroFSDPPass` | zero_fsdp.py:10 | ZeRO-0 / 1 / 2 / 3 | ❌ 纯 A |

只有 `OptimizerPass` 是桥：`optimizer.py:12, 19` 顶层 `import training.models.optimizer / memory`。其余三个 pass 不调路径 B。

---

### 3.7 executor/ + simulator/ — 调度与成本模型

**executor/**（1,912 LOC, 6 files）

| 类 / 模块 | 文件 | 行 | 职责 |
|---|---|---:|---|
| `DAGScheduler` | scheduler.py | 207 | 拓扑排序 + 贪心多流分配 |
| `PPStitchedTimeline` 等 | pp_stitcher.py | 917 | 拓扑驱动的 PP 调度拼接：从 per-stage Timeline 拼出 1F1B / DualPipe 网格 |
| `ChromeTraceExporter` | chrome_trace.py | 550 | Chrome Trace 导出（pipeline 调度结果 → JSON），全仓 3 份之一 |
| `OverlapAnalyzer` 等 | overlap.py | 208 | 计算-通信 overlap 分析（per-strategy） |
| `Stream` | stream.py | 15 | 流抽象 |

> executor 不只做"DAG 调度"——还承担 **PP 调度拼接 + Chrome Trace 导出 + overlap 分析**，职责偏重。`pp_stitcher.py` 与路径 B 的 `compose/` 调度逻辑存在概念重叠，待评审。

**simulator/**（2,270 LOC, 10 files，fallback 链）

| 后端 | 位置 | 职责 |
|---|---|---|
| `RooflineSimulator` | backends/roofline.py:1231 | 71+ 算子公式，**通用 fallback** |
| `RegressionSimulator` | backends/regression.py | 回归模型 |
| `ProfileSimulator` | backends/profile.py | 真机 profile 数据 |
| `TilingSimulator` | backends/tiling.py | tile 级模拟 |
| `LookupSimulator` | backends/lookup.py | 查表 |
| `TileSimSimulator` | backends/tilesim.py | 外部工具集成 |
| `SimulatorHub` | hub.py:35 | 后端路由器，优先级降序 + 内容哈希缓存 |

**返回**：`SimResult`（延迟 / FLOPs / 字节 / `Bound: computation | memory | latency`）

---

### 3.8 report/ — 报告导出（5,865 LOC, 13 files）

| 入口函数 | 文件 | 用途 |
|---|---|---|
| `build_report_context(graph, ctx, hw)` | report_builder.py | 元数据聚合 |
| `build_summary(graph, ctx, hw)` → `E2ESummary` | summary.py:56 | 推理摘要（FLOPs / 延迟 / 利用率） |
| `build_training_summary(graph, ctx, hw)` → `TrainingSummary` | summary.py:372 | 训练摘要（fwd / bwd / 优化器 / 通信） |
| `export_transformed_graph(...)` | report_builder.py | 多格式分发 |
| `export_html_report()` | html_writer.py | HTML 可视化 |
| `export_hierarchical_html_report()` | html_writer.py | 分层 HTML |
| `export_chrome_trace()` | chrome_trace.py | Chrome Tracing JSON |
| `export_full_report()` / `export_full_training_report()` | report_builder.py | 批量导出 |

**支持的输出格式**：Excel / JSON / HTML / Dot / ONNX / Chrome Trace

---

## 4. L3-B 路径 B 组件视图（配置建模）

### 4.0 路径 B 全景图（8 阶段 · 所有组件可见）

> 节点颜色：紫=普通组件，深紫描边=被桥调用，红=与路径 A 重复，灰=Phase 3 TODO 未实现

![路径 B 全景](diagrams/05_path_b.svg)

<sub>📐 源码: [`docs/diagrams/05_path_b.mmd`](diagrams/05_path_b.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

**怎么看这张图**

- **8 个阶段从上到下**：YAML 加载 → Spec 定义 → IR 构建 → 通信拓扑 → 成本模型 → Pipeline 调度 → Pareto 搜索 → 报告
- **同阶段内组件平行**（详细的字段/方法在 § 4.1-4.9 表格里）
- **深紫色 = 被桥调用**：路径 A 28 处桥（详见 § 5.1）调进来用，集中在 `training/spec/dtype` + `training/models/{optimizer,quant,quant_dispatch,promotion}` + `training/compose` + `training/spec/{report,strategy}` + `training/ir/builders` + `training/io/{config_loader,perf_tables}`
- **红色 = 重复**：`collective_time` / `memory_breakdown` / `export_chrome_trace` 在路径 A 也有
- **灰色 = Phase 3 TODO**：`insert_cast_pass`、`SearchEstimator`、`export_chrome_trace` 三处未完成
- B 没有"调度执行层"：spec → 成本模型 → Composer 直接算出 step_time，**不展开 DAG**（这是 B 与 A 的核心架构差异）

下面展开 9 个子包的内部细节。

### 4.1 training/spec/ — Spec 定义（1,308 LOC, 6 files）

| Dataclass | 文件 | 关键字段 |
|---|---|---|
| `ModelSpec` | model.py | 27 字段：hidden / ffn / num_heads / vocab / seq_len；DENSE / MOE / MTP 层；DeepSeek-V3/V3.2 MLA 字段；V4 attention + MoE + HC 扩展；per-component dtype |
| `Strategy` | strategy.py | TP/CP/PP/EP/DP，micro/global_batch，PPSched，ZeRO 0-3，recompute / offload / quant 策略，TP/EP/PP 重叠开关 |
| `SystemSpec` | system.py | GPU（TFLOPS / HBM / bw），节点 / GPU-per-node，intra / inter-node 互连 |
| `Dtype` | dtype.py | FP32 / BF16 / FP16 / FP8_E4M3 / FP8_E5M2 / FP4（含 `stored_bytes`） |
| `TrainingReport` | report.py | 34 字段：step_time_ms、bubble_fraction、FLOPs / 内存 / 通信细分、MFU/HFU、per-stage 指标 |

**枚举**

| Enum | 取值 |
|---|---|
| `PPSched` | `ONE_F_ONE_B` / `INTERLEAVED` / `ZERO_BUBBLE` / `DUALPIPE` / `DUALPIPE_V` |
| `CPKind` | `NONE` / `ULYSSES` / `RING` / `HYBRID` / `COMPRESSED` |
| `TPOverlap` | `NONE` / `COC` / `MC2` |
| `OptKind` | `ADAM` / `MUON` |
| `LayerKind` | `DENSE` / `MOE` / `MTP` |
| `RecomputeCategory` | `full` / `attn_core` / `attn_block` / `ffn_swiglu` / `ln` / `hc` |

**关键函数**
- `ModelSpec.total_params()` — 总参数（含 embedding / lm_head）
- `ModelSpec.effective_params_for_flops()` — MoE 仅计 active params
- `ModelSpec.get_layer_cp_type(layer_id)` — 返回 CSA / HCA / SWA / none
- `Strategy.validate()` — TP/CP/PP/EP/DP 整除性 + batch 校验
- `TrainingReport.summary()` — 人类可读 md 摘要

---

### 4.2 training/ir/ — 第二套 IR（2,348 LOC, 6 files）

| Dataclass | 文件 | 字段 |
|---|---|---|
| `Tensor` | graph.py | shape_logical / shape_local, dtype, is_activation, is_param |
| `Op` | graph.py | name, kind, inputs / outputs, meta, layer_id, layer_kind, component |
| `Collective` | graph.py | kind, group, bytes_, inserted_after / before, overlap, phase |
| `Graph` | graph.py | ops list, collectives list, layer_index |

**Op.kind 完整清单**（20+）：`matmul` / `attn_core` / `sparse_attn` / `hca_attn` / `swa_attn` / `softmax` / `ln` / `rope` / `swiglu` / `router` / `dispatch` / `combine` / `embed` / `lm_head` / `add` / `compressor_pool` / `indexer_topk` / `hash_route` / `cast`

**Collective.kind**：`AG` / `RS` / `AR` / `A2A` / `P2P`
**Collective.group**：`TP` / `CP` / `EP` / `DP` / `PP`
**Collective.phase**：`fwd` / `bwd` / `both`
**Op.component**：`attention` / `routed_expert` / `shared_expert` / `embedding` / `norm`

**关键函数**（builders.py 共 20 个 `build_*`）
- `build_graph(model, strategy)` → `Graph`
- `insert_collectives(graph, model, strategy)` — 按并行度插入集合通信
- `_apply_tp_sharding()` / `_apply_cp_sharding()` / `_apply_ep_sharding()`
- `insert_cast_pass(graph, model, strategy)` — dtype 边界插入 cast（**Phase 4 TODO**）
- `validate(model, system, strategy)` → `list[error]`

---

### 4.3 training/models/ — 成本模型（3,095 LOC, 7 files）

![成本模型数据流](diagrams/06_models_dataflow.svg)

<sub>📐 源码: [`docs/diagrams/06_models_dataflow.mmd`](diagrams/06_models_dataflow.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

| Dataclass / 函数 | 文件 | 职责 |
|---|---|---|
| `OpCost` | flops.py | fwd / dx / dw 各自的 bytes + cube_flops + vector_flops + bound 诊断 |
| `MemBreakdown` | memory.py | weights / grads / opt_state / activations / comm_buffers / hc_overhead / muon_ns_buffer；peak_{forward, backward, optimizer, overall} |
| `OpDtypeBundle` | flops.py | 各区域 param / grad / activation / compute dtype |
| `CompressedCPConfig` 等 3 类 | compressed_cp.py | V4 压缩 CP 推理 |
| `op_cost(op, model, system)` → `OpCost` | flops.py | 20+ 种 op 类型 FLOPs / 字节公式 |
| `total_training_flops(...)` | flops.py | 总 FLOPs（含 recompute overhead） |
| `forward_backward_flops(...)` | flops.py | fwd + bwd FLOPs（不含 recompute） |
| `collective_time(c, group_size, link)` | comm.py | AG / RS / AR / A2A 时间（树形 / 环形 / 阶层） |
| `total_comm_time(...)` | comm.py | 所有通信总时间 |
| `memory_breakdown(...)` → `MemBreakdown` | memory.py | 全 breakdown |
| `muon_step_flops()` / `moonshot_optimizer_hiding()` | optimizer.py | Muon FLOPs 及在 NS 计算下的隐藏 |
| `adam_state_bytes()` / `adam_step_traffic_bytes()` | optimizer.py | Adam 内存 / 流量 |

**Op kind → FLOPs 公式**：`matmul: 2×M×N×K`、`attn_core: seq²×h×batch`、`sparse_attn: ratio×seq²`、`hca_attn` / `swa_attn` 含压缩比
**通信原语**：AG / RS / AR / A2A / P2P，支持阶层（intra / inter-node）
**优化器**：Adam（4 state） / Muon（2×state + NS pair 旋转，可选隐藏）

---

### 4.4 training/compose/ — Pipeline 调度（1,909 LOC, 3 files）

**6 个 Composer**（schedules.py）

| Composer | PP 算法 | Bubble | 备注 |
|---|---|---|---|
| `PipelineComposer` (abstract) | 接口基类 | — | `compose(graph, model, system, strategy) → StepResult` |
| `OneF1BComposer` | 标准 1F1B（Megatron） | `~2×pp / pp_depth` | |
| `Interleaved1F1BComposer` | 交错 1F1B（VPP） | `~2×pp / (pp×vpp_chunks)` | 需 `vpp_chunks > 1` |
| `ZeroBubbleComposer` | Zero Bubble（Saavedra 2024） | 理想 0 | fwd / bwd 分离调度 |
| `DualPipeComposer` | Dual-stream PP | — | fwd / bwd 独立流，P2P 可隐藏 |
| `DualPipeVComposer` | DualPipe 变体 | — | 可选 dualbatch + pp_overlap |

| 类 | 文件 | 字段 |
|---|---|---|
| `StepResult` | pipeline.py | step_time / pipeline_time, bubble / warmup / steady / cooldown, compute / comm / optimizer (ms) |
| `Stage` | stage.py | per-stage ops + sharding |
| `StageTime` | stage.py | fwd / bwd / recompute 时间 |

**关键函数**
- `stage_time(stage_ops, model, system, strategy, graph)` → `StageTime`
- 共享时间公式：`cooldown_compute = f(pp, vpp, schedule)`、`bubble = f(schedule, stages)`

**支持的具体选项**
- `vpp_chunks`：1 / 2 / 4 / ...
- `pp_overlap`：bool（DualPipe/V 专用，P2P 隐藏在 bwd_dw）
- `dualbatch`：bool（DualPipeV 专用，增加 batch 但保持 step 宽度）
- `dp_overlap_in_bubble`：bool
- `dp_steady_overlap_ratio`：0.0–1.0
- `recompute_policy`：per-LayerKind（dense / moe / mtp）的 category 集合

---

### 4.5 training/search/ — Pareto 搜索（1,725 LOC, 5 files）

| 类 / 函数 | 文件 | 职责 |
|---|---|---|
| `SearchSpace` | space.py | tp/cp/pp/ep/dp/zero/schedules/recompute/vpp/optimizer 取值列表 + 剪枝标志 |
| `SearchEstimator` | estimator.py | 单点估计入口 |
| `TrainingConfigManager` | training_search_util.py | YAML 配置管理，内存可行性检验 |
| `SearchReport` | report.py | 搜索结果 |
| `estimate(config, model, system)` → `TrainingReport` | estimator.py | 单配置估计 |
| `grid_search(configs, model, system)` | training_search_util.py | 并行评估多配置 |
| `pareto_frontier(reports)` → list | training_search_util.py | Pareto 前沿 |
| `run_training_search_parallel()` | training_search_util.py | 多进程 pool |

**默认搜索空间**：TP / PP / EP / DP ∈ {1, 2, 4, 8}；CP=1；ZeRO ∈ {0,1,2,3}；Schedules ∈ {`1F1B`, `INTERLEAVED`, `DUALPIPE`}；Recompute ∈ {none, selective, full}

**剪枝规则**
- TP > gpus_per_node 跨节点 → 剪
- CP 需 `seq_len ≥ 32K`
- EP 需 `num_experts > threshold`

**Phase 3 TODO**：`search/space.py:22`、`search/estimator.py:196`

---

### 4.6 training/io/ — 配置与导出（2,955 LOC, 5 files）

| 函数 | 文件 | 职责 |
|---|---|---|
| `load_specs(yaml_path)` → `(ModelSpec, SystemSpec, Strategy)` | config_loader.py | 解析单文件 YAML |
| `load_anchor_config(yaml_path)` | config_loader.py | 同上别名 |
| `_parse_model / _parse_system / _parse_strategy` | config_loader.py | 递归解析子对象 |
| `_parse_layers(spec)` → `list[LayerKind]` | config_loader.py | "dense"/"moe"/"mtp" → enum |
| `_expand_quant_preset(dict)` | config_loader.py | 量化预置展开 |
| `export_estimate_html(report, ...)` | html_exporter.py | HTML（交互式树 + op 公式 + 拓扑） |
| `export_estimate_excel(reports, configs, path)` | excel_exporter.py | 多 sheet：summary / per-config / per-layer |
| `achieved_flops_efficiency()` | perf_tables.py | GPU FLOPs 效率曲线（roofline） |
| `achieved_bandwidth_efficiency()` | perf_tables.py | GPU 带宽效率曲线 |

**YAML schema**：`model`（name 或 inline ModelSpec） + `system`（gpu + nodes + gpus_per_node + interconnect） + `strategy`（tp/cp/pp/ep/dp/zero_stage/pp_schedule/recompute_policy/...）

**量化预置**：`fp8_routed` / `fp8_shared` / `mixed_quant_v4`

---

### 4.7 training/topology/ — 通信拓扑（855 LOC, 3 files）

| 类 | 文件 | 职责 |
|---|---|---|
| `ParallelGroups` | process_groups.py | tp/cp/ep/dp/pp 各轴 process group，含 group_size、rank_in_group |
| `CommDomain` | comm_domain.py | 5 层级通信域：global → dp → pp → tp → cp |
| `GroupTierAssignment` | comm_domain.py | 每轴的 tier 归类（intra / inter-node） |

**关键函数**
- `build_process_groups(system, strategy)` → `ParallelGroups`
- `build_comm_domain(system, strategy)` → `CommDomain`
- `comm_domain_report(domain)` — 人类可读
- `tier_for_group(group, strategy, system)` → `"intra" | "inter" | None`

---

### 4.8 training/anchor/ + training/trace/ — 锚点 + Chrome Trace（187 LOC 合计）

| 类 / 函数 | 文件 | 职责 |
|---|---|---|
| `Anchor` | anchor/validate.py | name、step_time_ms / mfu / total_flops、tolerance（默认 15%）、strict_mfu_check |
| `validate_anchor(report, anchor)` | anchor/validate.py | 对比 report vs anchor，返回 warning / `[STRICT]` / `[CALIBRATION]` |
| `export_chrome_trace(timeline, path)` → `Path` | trace/exporter.py | Timeline → chrome://tracing JSON（**Phase 3 TODO** 在 exporter.py:34） |

**Anchor YAML fixtures**（`tests/training/anchors/`）：GPT-3 175B / LLaMA-3 70B / DeepSeek-V3 / V3.2 / V4 各变体

---

### 4.9 training/configs/ — 配置文件清单

**单文件 3D 配置**（16 个，`training/configs/*.yaml`）
- `deepseek_v3_2_3d_{h100, h800, ascend_910c}.yaml`
- `deepseek_v4_flash_3d_{h100, h800, ascend_910c}.yaml`
- `deepseek_v4_pro_3d_{ascend_910c, h100, h800}` 各带 `_tp4` / `_mc2` / `_none` 后缀变体
- `llama3_70b_3d.yaml`

**模型 profile**（`training/configs/models/`，5 个）
- `deepseek_v3.yaml` — 671B MLA
- `deepseek_v3_2.yaml` — 236B MLA + CSA / HCA / SWA
- `deepseek_v4_flash.yaml` — 236B 标准 attention（无 MLA）
- `deepseek_v4_pro.yaml` — 236B V4 attention（grouped o_proj） + HC（`hc_mult > 1`） + hash-routed MoE
- `llama3_70b.yaml` — 70B 标准 MHA

---

## 5. 真实跨包依赖图（基于 `grep import`）

![跨包依赖图](diagrams/07_dependency.svg)

<sub>📐 源码: [`docs/diagrams/07_dependency.mmd`](diagrams/07_dependency.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

### 5.1 桥（反向依赖）精确清单

> **方法**：`grep -rnE "(from\|import) (python\.)?zrt\.training" python/zrt/{transform,ir,memory,report}`，含函数内 import。
> **结论：共 28 处桥，跨 10 个文件，其中 7 处是顶层 import（import 该模块即拉起整个 training 包）。**

**桥 = path A → path B**（28 处 import，10 个文件）

| 文件 | 处数 | 顶层? | 调路径 B 的什么 | 用途 |
|---|:---:|:---:|---|---|
| `transform/analysis/training.py` | 9 | 函数内 | `compose.{schedules,stage}` + `spec.strategy` + `spec.dtype` + `models.optimizer` + `io.perf_tables` | PP 调度建模（最大桥） |
| `transform/context.py` | 4 | 1 顶层 + 3 函数内 | `spec.dtype.Dtype`（顶层）+ `io.config_loader._QUANT_PRESETS / _expand_quant_preset` + `spec.strategy._MUON_NS_STEPS_DEFAULTS` | DType + 量化预置 |
| `transform/optim/passes.py` | 3 | 函数内 | `spec.dtype.Dtype` ×3 | 量化 pass 取 dtype |
| `transform/analysis/quant.py` | 3 | 2 顶层 + 1 函数内 | `models.quant.OpDtypeBundle`（顶层）+ `spec.dtype.Dtype`（顶层）+ `models.quant_dispatch.dispatch` | 量化 dtype 解析 |
| `transform/training/optimizer.py` | 2 | 2 顶层 | `models.optimizer` + `models.memory` | 优化器内存/FLOPs |
| `transform/analysis/passes.py` | 2 | 函数内 | `models.promotion.ln_softmax_input_byte_multiplier` + `spec.dtype.Dtype` | FP32 promotion 字节系数 |
| `ir/graph.py` | 2 | 函数内 | `ir.builders.build_graph` + `ir.training_graph.Graph` | OpGraph → TrainingGraph 转换 |
| `transform/analysis/modeller.py` | 1 | 顶层 | `spec.report.TrainingReport` | 训练报告数据类 |
| `transform/analysis/comm_latency.py` | 1 | 函数内 | `spec.dtype.Dtype` | 通信延迟取 dtype |
| **`ir/types.py`** | 1 | **顶层** | **`spec.dtype.Dtype as DType`** | **最底层 IR 反向依赖最上层 training** |

**反向 = 下层 → 上层**（1 处）

| 文件 | 调谁 | 用途 |
|---|---|---|
| `memory/model.py:11` | `transform.context.ParallelConfig, QuantConfig` | 只用配置类（边界耦合） |

**三个最值得警惕的点**

1. **`ir/types.py:11` 顶层 `from zrt.training.spec.dtype import Dtype as DType`**：`ir/` 是被 47 处引用的最底层包，却顶层依赖最上层、最大的 `training/`。**import `ir.types` 就会拉起整个 training 包**，循环依赖风险极高。DType 的 canonical 源现在在 `training/spec/dtype.py`，`ir/types.py` 只是 re-export——依赖方向是反的。

2. **量化逻辑形成一类桥**：`transform/analysis/quant.py` + `transform/optim/passes.py` + `transform/analysis/passes.py` 共 8 处，都 import `training.{models.quant, models.quant_dispatch, models.promotion, spec.dtype}`。量化 dtype 解析的共享逻辑被放在了 `training/models/` 下，由 transform 反向调用。

3. **DType 共有 3 个定义点**：`training/spec/dtype.py`（canonical）、`ir/types.py`（re-export）、`layers/op_base.py:5`（独立的简陋版）。后者从未与前两者统一。

---

### 5.2 RooflineSimulator 被调用几次（"是不是 bug"的澄清）

> 一个常见疑问：路径 A 在 transform 内部已算过一次成本，DAG 调度之后又算一次，是不是 bug？下面是完整调用清单。

| # | 调用点 | 行 | 触发条件 | 做什么 | 冗余? |
|---|---|---:|---|---|:---:|
| 1 | `transform/analysis/passes.py::RooflinePass` | 208 | 每次 analyze 阶段 | 算每个 OpNode 的 latency_us，写 annotations | ✅ 必要 |
| 2 | 同上 `RooflinePass._fmr(node)` fallback | 同 | 仅当 op 没有 flops/bytes 时 | 估算缺失的 flops/bytes | ✅ 必要 fallback |
| 3 | `executor/scheduler.py::_latency` | 184-195 | 仅当 node.annotations 没有 latency_us | fallback 算 latency | ✅ 必要 fallback |
| 4 | `report/html_writer.py:1304` | — | **每次导 HTML 都跑** | 全图重新跑 `hub.simulate_graph(graph, hw_spec)` | ⚠️ **真冗余** |
| 5 | 测试 / 各 simulator backend 内部 | — | 测试或对比 | — | — |

**结论**

- **`TrainingPipelinePass` 算的是 graph 级 step_time + bubble**，写到 `graph.metadata["pipeline_metrics"]`
- **`RooflinePass` 算的是 per-op latency**，写到 `node.annotations["latency_us"]`
- 两者**算的不是一回事**，不重叠
- DAG 调度（scheduler）**优先读 annotations，不会重算**
- **唯一真冗余在 `html_writer.py:1304`**——节点上已有 annotations，却又跑一次 `hub.simulate_graph()`，应改为直接读 annotations

→ 已加入 § 7 重复实现表 + § 8 阶段 0 快修

---

## 6. 模块代码量分布

![LOC 占比饼图](diagrams/08_loc_pie.svg)

<sub>📐 源码: [`docs/diagrams/08_loc_pie.mmd`](diagrams/08_loc_pie.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

| 包 | 文件数 | LOC | 角色 |
|---|---:|---:|---|
| `training/` | 44 | **13,511** | 🅱 配置建模全栈（自带 IR/模型/调度/搜索/导出） |
| `transform/` | 55 | **10,145** | 🅰 抓图后变换 pipeline（含 fusion/parallel/analysis/training） |
| `report/` | 13 | 5,089 | 🅰 报告导出 |
| `graph/` | 12 | 2,641 | 🅰 抓图入口 |
| `simulator/` | 10 | 1,912 | 共享 — 成本模型后端 |
| `ir/` | 10 | 1,861 | 共享 — OpGraph IR（被 47 处引用；DType 现顶层 import training.spec.dtype） |
| `executor/` | 6 | 1,659 | 🅰 DAG 调度 + PP 拼接 + chrome_trace + overlap |
| `layers/` | 11 | 1,431 | 共享 — 算子抽象层 |
| `fusion/` | 9 | 1,246 | fusion-rule YAML 草稿生成器（独立于 transform/fusion） |
| `hardware/` | 3 | 484 | ✅ 真共享层 |
| `memory/` | 4 | 324 | 🅰 内存估算（B 用自己的） |
| `policy_model/` | 8 | 117 | 共享策略分发 |

---

## 7. 重复实现对照表

### 7.1 跨路径重复实现

| 能力 | 路径 A 实现 | 路径 B 实现 | LOC 合计 | 建议 |
|---|---|---|---:|---|
| **per-op FLOPs 公式** | `FlopsPass` (passes.py, 几十行 dispatch) | `training/models/flops.py::op_cost` (875) | 900+ | 统一到 `OpCost` |
| **通信耗时** | `transform/analysis/comm_latency.py` (143) | `training/models/comm.py` (635) | 778 | 保留 B，A 改薄 wrapper |
| **Memory** | `memory/*` (373) + `TrainingMemoryPass` (聚合) | `training/models/memory.py` (628) | 1001 | 待评审 |
| **Chrome Trace（3 份）** | `report/chrome_trace.py` (234) + `executor/chrome_trace.py` (550) | `training/trace/exporter.py` (90) | 874 | 三份合一，保留 executor 那份（最完整） |
| **DType（3 份，方向反）** | `layers/op_base.py:5`（简陋版）+ `ir/types.py`（re-export） | `training/spec/dtype.py`（canonical） | — | canonical 落在了 B，方向反，见 § 5.1 |
| **量化 dtype 解析** | `transform/analysis/quant.py`（调 B） | `training/models/{quant,quant_dispatch,promotion}` | ~250 | 是桥，非重复（见 § 5.1） |
| **Pipeline 调度** | `TrainingPipelinePass`（实际是桥，调 B） | `training/compose/schedules.py` (1279) | 1279 | 是桥，非重复（见 § 5.1） |
| **HTML 导出** | `report/html_writer.py` (1387) | `training/io/html_exporter.py` (1945) | 3332 | 格式不同，先各自拆解再评估 |

### 7.2 同路径内的真冗余调用（不是重复实现，是浪费 CPU）

| 位置 | 现象 | 修复 |
|---|---|---|
| `report/html_writer.py:1304` | 每次导 HTML 都 `hub.simulate_graph(graph, hw_spec)` 全图重跑 simulator——node.annotations 里已有 `latency_us`（RooflinePass 写入），却扔掉重算 | 改读 `node.annotations["latency_us"]`，删那一行 |

### 7.3 容易误判为重复、但其实不是

| 看似重复的一对 | 真实情况 |
|---|---|
| `TrainingFlopsPass` 与 `training/models/flops.py` | 不是重复。前者是**聚合 pass**（写 graph.metadata 整图汇总），后者是 **per-op 公式**（写 OpCost），是两层不同的数据 |
| `TrainingMemoryPass` 与 `training/models/memory.py` | 不是重复。前者用 `count_params` + Korthikanti 公式独立计算，不 import 路径 B |
| `TrainingPipelinePass` 与 `training/compose` | 不是重复。前者**就是调用后者**（即桥），不是平行实现 |

---

## 8. 沉淀热点优先级（评审使用）

按 **架构影响 × 工作量** 给出执行顺序：

![阶段路线图](diagrams/09_stage_roadmap.svg)

<sub>📐 源码: [`docs/diagrams/09_stage_roadmap.mmd`](diagrams/09_stage_roadmap.mmd) · 编辑后运行 `bash docs/diagrams/regenerate.sh` 重渲染</sub>

| # | 热点 | 位置 | 量级 | 阶段 |
|---|---|---|---:|:---:|
| 1 | A/B/C IR 决策 | — | 顶层决策 | 0 |
| 2 | 清理根目录 `tmp_*` 与 `python/zrt/deepseek_v4_training.xlsx` (532KB) | 根目录 | 7 文件 | 0 |
| 3 | 4 处 Phase 3 TODO 盘点 | `training/{trace,search,anchor}` | 4 处 | 0 |
| 3.5 | **html_writer simulator 真冗余**（删 1 行） | `report/html_writer.py:1304` | 1 行 | 0 |
| 4 | DType 收口（现 3 份，方向反）：决定 canonical 放哪 + 删 `layers/op_base.py:5` 简陋版 + 解开 `ir/types.py`→training 顶层依赖 | 见 § 5.1 | 3 处 | 1 |
| 5 | Chrome trace 合并（**现 3 份**） | `report/` + `executor/` + `training/trace/` | ~870 | 1 |
| 6 | **通信成本 single source** | `transform/analysis/comm_latency.py` → 薄 wrapper | 778 | 1 |
| 7 | **FLOPs single source（OpCost）** | `transform/analysis/training.py` + `training/models/flops.py` | 1940 | 1 |
| 8 | 拆 `transform/exporter.py` 2085 | → chrome_trace / yaml / json | 2085 | 2 |
| 9 | 拆 `simulator/backends/roofline.py` 1866 | → op_catalog / engine / dtype_table | 1866 | 2 |
| 10 | 拆 `training/io/html_exporter.py` 1933 | → css / template / aggregator / writer | 1933 | 2 |
| 11 | 拆 `training/compose/schedules.py` 1279 | 6 Composer 各一文件 | 1279 | 2 |
| 12 | 拆 `training/ir/builders.py` 1128 | 按 IR entity | 1128 | 2 |
| 13 | 拆 `report/html_writer.py` 1387 | — | 1387 | 2 |
| 14 | 拆 `transform/analysis/training.py` 1082 | → FlopsPass / MemoryPass / PipelinePass | 1082 | 2 |
| 15 | 拆 `graph/patches.py` 928 | 按 patch 类型 | 928 | 2 |
| 16 | 补 tests/ — executor/simulator/memory/policy_model/layers/hardware | — | 6 包 | 3 |
| 17 | 补 tests/ — training/{io,anchor,trace} | — | 3 子包 | 3 |
| 18 | `pipeline.py` 882 拆 config_io + orchestrator | — | 882 | 4 |
| 19 | 定义 `python/zrt/__init__.py` public API 清单 | — | — | 4 |
| 20 | CI 加文件大小阈值检查（>800 行 fail） | — | — | 4 |

---

## 9. 目录树（完整代码地图）

> 本树用于看整体结构；per-file LOC 仅供参考，顶层 LOC 以 § 6 表格为准。

<details>
<summary>展开查看完整目录树</summary>

```text
python/zrt/
├── cli.py                       849   ★ 两条路径分发器
├── pipeline.py                  882   ★ 抓图路径顶层编排
├── __main__.py / runtime_config / tensor_base / input_param
│
├── ir/                          2,043 LOC, 9 files  [真底层]
│   ├── graph.py                 353   OpGraph / OpNode / OpEdge
│   ├── adapter.py               779   records → OpGraph
│   ├── types.py                 182   DType / TensorMeta
│   ├── serde.py                 198
│   ├── hierarchy.py             171
│   └── ...
│
├── hardware/                    590 LOC, 3 files   [✅ 真共享]
│   ├── spec.py 338  registry.py 212  configs/*.yaml
│
├── layers/                      1,664 LOC, 11 files
│   ├── op_base / op_mm / op_attention / op_communication / op_quant / op_fused / ...
│
├── graph/                       3,142 LOC, 12 files   [🅰 抓图入口]
│   ├── patches.py               928   ⚠ MoE/Indexer/V4 patch 混在一起
│   ├── model_loader.py          328
│   ├── compat.py                278
│   ├── v4_fake_kernels.py       257
│   ├── dispatch.py              226
│   ├── pattern_extractor.py     217
│   ├── classifier.py            184
│   ├── graph_builder.py         306
│   └── transform_runner.py
│
├── transform/                   10,908 LOC, 54 files  [🅰 中端]
│   ├── exporter.py              2,085  ⚠ 超大
│   ├── context.py               256    ⚠ import training/
│   ├── pipeline.py / base.py / debug_pass.py
│   ├── analysis/                1,955 (5 files)
│   │   ├── training.py          1,082  ⚠ FLOPs+Memory+Pipeline 混
│   │   ├── modeller.py          345    ⚠ import training/
│   │   ├── comm_latency.py      143    ↘ 与 training/models/comm 重复
│   │   └── passes.py / flops.py
│   ├── parallel/                2,030 (8 files)
│   │   ├── comm_inserter.py     712
│   │   ├── expert_grouped_mm.py 399
│   │   └── tp/ep/pp/dp/cp/...
│   ├── training/                1,031 (4 files)  ★ 桥的另一半
│   │   ├── recompute / optimizer 402 / offload 273 / zero_fsdp 233
│   ├── optim/                   63   (近空)
│   └── fusion/                  3,238 (29 files)
│       ├── core / matching / building / bucketing
│       ├── loading / rules / pipeline / registry / semantics / configs
│
├── fusion/                      1,445 LOC, 9 files  [独立工具]
│   └── discover/                fusion-rule YAML 草稿生成
│
├── memory/                      373 LOC, 4 files   [🅰]
├── executor/                    335 LOC, 4 files   [🅰]
│
├── simulator/                   2,255 LOC, 10 files
│   ├── hub.py / base.py / cache.py
│   └── backends/
│       ├── roofline.py          1,866  ⚠ 70+ op 公式硬编码
│       └── regression / profile / tiling / lookup / tilesim
│
├── policy_model/                141 LOC, 8 files
│
├── report/                      5,865 LOC, 13 files  [🅰 输出]
│   ├── html_writer.py           1,387  ⚠ 超大
│   ├── report_builder.py        1,112
│   ├── summary.py               719
│   ├── formula_registry.py      456
│   ├── onnx_exporter.py         408
│   ├── structure_renderer.py    362
│   ├── chrome_trace.py          234    ↘ 与 training/trace 重复
│   └── compare / shape_desc / topology_renderer / dot_exporter
│
└── training/                    14,387 LOC, 40 files  [🅱 全栈]
    ├── cli.py / __main__.py
    ├── spec/                    1,308 (6 files)
    │   ├── model 462  strategy 282  report 439
    │   └── system / dtype / enums
    ├── ir/                      2,348 (6 files)  ★ 第二套 IR
    │   ├── builders.py          1,128  ⚠ 超大
    │   ├── shard.py             767
    │   └── cast_pass 204 / validate / graph
    ├── models/                  3,095 (7 files)  ★ 成本模型
    │   ├── flops.py             858    ↘ 与 transform/analysis/training 重复
    │   ├── comm.py              635    ↘ 与 transform/analysis/comm_latency 重复
    │   ├── memory.py            628    ↘ 与 memory/ 重复
    │   ├── compressed_cp.py     441
    │   ├── optimizer.py         322
    │   └── quant.py
    ├── compose/                 1,909 (3 files)
    │   ├── schedules.py         1,279  ⚠ 6 Composer 混
    │   ├── stage.py             625
    │   └── pipeline.py
    ├── search/                  1,725 (5 files)
    │   ├── training_search_util.py  1,125  ⚠ 超大
    │   ├── estimator.py         226
    │   └── space / report / ...
    ├── io/                      2,955 (5 files)
    │   ├── html_exporter.py     1,933  ⚠ 超大
    │   ├── config_loader.py     458
    │   ├── excel_exporter.py    433
    │   └── perf_tables.py
    ├── topology/                855 (3 files)
    │   ├── process_groups.py    451
    │   └── comm_domain.py       361
    ├── anchor/                  97  (2 files)
    ├── trace/                   90  (2 files)
    └── configs/                 YAML 配置 + models/
```

</details>

---

## 10. 评审使用建议

按下列顺序看这份文档：

1. **§ 0 一句话结论** — 30 秒理解病灶
2. **§ 2 L2 容器视图** — 看清"两条路径 + 28 处桥"的全景
3. **§ 5 跨包依赖图 + § 5.1 桥的精确清单 + § 5.2 RooflineSimulator 调用关系** — 看清 28 处桥（10 文件）+ 1 处真冗余
4. **§ 7 重复实现对照表** — 看清要合并什么
5. **§ 8 沉淀热点优先级** + 象限图 — 评审执行顺序
6. **§ 11 问题清单 + 行动路线图** — 30 分钟拿走完整执行方案

如需现场标注，建议把本文件 fork 到评审分支，直接在 mermaid 节点 ID 上加 `[已评审]` / `[阶段调整]` 等标记。

---

## 11. 问题清单 + 行动路线图（评审决策版）

> 这是 § 0 - § 9 全文的执行汇总。如果只有 30 分钟，直接看这一节。

### 11.1 问题清单（按 4 层级）

#### 第 1 层 · 架构（路径之间）

| 问题 | 真相 | 影响 | 详情 |
|---|---|---|---|
| 两条建模路径并存 | 抓图（A）15.5k + 配置（B）11.8k LOC 各自独立 | 事实上是 2 个产品 | § 2 |
| **28 处隐式桥**（path A 反向 import path B，10 个文件） | 含量化桥；`ir/types.py` 顶层依赖 training | A 直接调 B，循环依赖风险 | § 5.1 |
| **1 处反向依赖** | `memory/model.py` → `transform.context` | 形成环 | § 5.1 |

#### 第 2 层 · 模块（重复实现）

| 能力 | A 侧实现 | B 侧实现 | LOC | 性质 |
|---|---|---|---:|---|
| per-op FLOPs 公式 | `FlopsPass` (passes.py) | `op_cost` (training/models/flops.py 875) | 900+ | 真重复 |
| 通信耗时 | `transform/analysis/comm_latency.py` (143) | `training/models/comm.py` (635) | 778 | 真重复 |
| 内存估算 | `memory/*` (373) + `TrainingMemoryPass` | `training/models/memory.py` (628) | 1000+ | 真重复 |
| Chrome Trace（3 份） | `report/chrome_trace.py` (234) + `executor/chrome_trace.py` (550) | `training/trace/exporter.py` (90) | 874 | 真重复 |
| DType 定义（3 份，方向反） | `layers/op_base.py:5`（简陋版）+ `ir/types.py`（re-export） | `training/spec/dtype.py`（canonical） | — | 方向反了 |
| HTML 报告 | `report/html_writer.py` (1387) | `training/io/html_exporter.py` (1945) | 3332 | 格式不同（保留） |
| **html_writer:1304 重跑 simulator** | `hub.simulate_graph()` 每次重算 | — | 1 行 | **同路径真冗余** |

#### 第 3 层 · 文件（超大）

13 个文件 > 800 行：

| 文件 | LOC | 拆解建议 |
|---|---:|---|
| `transform/exporter.py` | 2085 | chrome_trace / yaml_export / json_export |
| `training/io/html_exporter.py` | 1945 | css / template / aggregator / writer |
| `simulator/backends/roofline.py` | 1881 | op_catalog（70+ 公式）+ roofline_engine + dtype_table |
| `training/search/training_search_util.py` | 1639 | 搜索 / 剪枝 / 报表分离 |
| `transform/analysis/training.py` | 1420 | FlopsPass / MemoryPass / PipelinePass 各一文件 |
| `report/html_writer.py` | 1387 | 同上拆 |
| `training/compose/schedules.py` | 1279 | 6 Composer 各一文件 |
| `training/ir/builders.py` | 1128 | 按 IR entity 类型拆 |
| `graph/patches.py` | 928 | 按 patch 类型（MoE / Indexer / V4） |
| `executor/pp_stitcher.py` | 917 | 与 compose/ 概念重叠，待评审 |
| `pipeline.py` | 882 | config_io + orchestrator |
| `training/models/flops.py` | 875 | op_cost 公式可考虑拆表 |
| `cli.py` | 869 | 两条入口分离 |

#### 第 4 层 · 收尾

| 类型 | 具体内容 |
|---|---|
| **测试盲区** | 6 子包零 tests：executor / simulator / memory / policy_model / layers / hardware；training 内 anchor / trace 也无 tests |
| **半成品** | 4 处 Phase 3 TODO：`training/trace/exporter.py:34`、`training/search/space.py:22`、`training/search/estimator.py:196`、`training/anchor/validate.py:6` |
| **脏文件** | 根目录 `tmp_baseline.yaml` + `tmp_compute_dtype_ab.py` + `tmp_fp4_ab.py` + `_debug_fusion.py` + `python/zrt/deepseek_v4_training.xlsx`（532KB，不该入代码） |

---

### 11.2 阻塞决策：必须先选 A / B / C

```
两 IR 共存 vs 收敛到 OpGraph vs 承认两个产品
        A              B  ← 已选择并实施        C
```

| 选项 | 工作量 | 风险 | 适合场景 |
|---|---|---|---|
| **A** 正式桥接（adapter 一等公民） | 高 | 长期维护 adapter | 两条路径都保留且要更多复用 |
| **B** 收敛到 OpGraph（**已选择**） | 极高 | spec 路径变慢，失去"不抓图就估算"优势 | 长期单一产品 |
| **C** 承认两个产品（共享层只下沉到 hw/dtype/comm 内核） | 中 | 短期工作量集中在拆桥 | 当前客户群已分化 |

**选择 B（收敛到 OpGraph）的理由**：

1. 长期维护两套 IR 的成本高于一次收敛
2. 去重后才能做跨路径全局优化（如统一调度、统一成本模型）
3. 可通过 OpGraph 子集化（只构建用到的子图）缓解"不抓图就估算"的场景

**实施进展**（详见 § 12）：

- ✅ Phase 1-4：基础设施搭建（OpGraph builder、Transform Pipeline、adapter、导入约定）
- ✅ Phase 5A：补齐 `_report_from_transformed()` 30 个缺失字段
- ✅ Phase 5B：Pipeline 精度对齐（`_original_graph` metadata 触发精确计算）
- ✅ Phase 6：`estimate()` 委托 `estimate_via_pipeline()`，删除 ~120 行独立逻辑
- ✅ Phase B1：消费者迁移（25 个 flops 函数 + stage + schedules + comm + quant）
- ✅ Phase B2：生产者迁移（`build_opgraph_direct()` + OpGraph-native shard/cast）
- ⏳ Phase B3：删除旧 IR（198 处测试引用待迁移）

---

### 11.3 五阶段路线图（5-7 周）

可视化见 [§ 8 阶段路线图](diagrams/09_stage_roadmap.svg)。文字版：

```
阶段 0 止血     1-2 天    零风险，立即做
   ↓
阶段 1 统一     1 周      single source of truth（DType/comm/FLOPs）
   ↓
阶段 2 拆解     1-2 周    7 个超大文件
   ↓
阶段 3 固化     2 周      补 9 子包 tests + anchor 当回归契约
   ↓
阶段 4 收口     1 周      public API 清单 + CI 阈值守门
```

#### 阶段 0 · 止血（1-2 天，零风险）

每项独立可 PR：

- [ ] 删 3 个 `tmp_*.{yaml,py}` + `_debug_fusion.py`
- [ ] 把 `python/zrt/deepseek_v4_training.xlsx` 532KB 移出代码目录
- [ ] **修 html_writer.py:1304**（删 1 行 `hub.simulate_graph()`，改读 annotations）— **见 § 5.2**
- [ ] 盘点 4 处 Phase 3 TODO：要么实现要么删，不要让"看似完成"的代码继续诱导新依赖
- [ ] 给 26 个 `*Pass` 类做归属清单（analysis / parallel / training / optim / debug）

#### 阶段 1 · 统一基础设施（1 周）

按依赖顺序，每项独立 PR：

- [ ] **DType 方向纠偏**（头号任务）：现状是 `ir/types.py` 顶层 import `training.spec.dtype`，方向反。需先定 canonical 落点（建议抽到中立的 `ir/` 或新建 `common/`），再让 `training.spec.dtype` 反过来 re-export，最后删 `layers/op_base.py:5` 简陋版 → DType 收成 1 份
- [ ] **Chrome Trace 三合一** → `report/chrome_trace.py` + `executor/chrome_trace.py` + `training/trace/exporter.py`，保留最完整的一份
- [ ] **量化桥收口**：`training/models/{quant,quant_dispatch,promotion}` 被 transform 反向调 8 处——抽成中立共享层，否则量化逻辑越写桥越多
- [ ] **通信成本 single source** → 保留 `training/models/comm.py`，让 `transform/analysis/comm_latency.py` 改 143 行薄 wrapper
- [ ] **per-op FLOPs 统一到 OpCost** → 让 `FlopsPass` 输出 `OpCost` 形态，删 `op_cost` 里的重复公式
- [ ] **拆桥**（如果选 C）：28 处里能立刻删的：`memory/model.py:11` 反向 + `transform/context.py` Muon 常量

#### 阶段 2 · 拆超大文件（1-2 周，可并行）

每文件独立 PR，按优先级：

| 优先级 | 文件 | 拆成 |
|---:|---|---|
| 1 | `transform/exporter.py` 2085 | chrome_trace / yaml_export / json_export |
| 2 | `simulator/backends/roofline.py` 1866 | op_catalog + engine + dtype_table |
| 3 | `training/io/html_exporter.py` 1933 | css / template / aggregator / writer |
| 4 | `training/compose/schedules.py` 1279 | 6 Composer 各一文件 |
| 5 | `report/html_writer.py` 1387 | 同 4 |
| 6 | `transform/analysis/training.py` 1082 | FlopsPass / MemoryPass / PipelinePass |
| 7 | `graph/patches.py` 928 | 按 patch 类型 |

#### 阶段 3 · 固化（2 周）

- [ ] 6 子包补基础单测：executor / simulator / memory / policy_model / layers / hardware
- [ ] training 内 3 子包补测：io / anchor / trace
- [ ] **anchor YAML 当回归契约**：每条路径要能跑过自己一组 anchor 才算固化
- [ ] CI 加文件大小阈值（>800 行 fail），防止再次膨胀

#### 阶段 4 · API 收口（1 周）

- [ ] `pipeline.py` 882 行拆 `config_io.py` + `orchestrator.py`
- [ ] `cli.py` 的两条入口 `_run_inference_pipeline` / `_run_estimate` 正式成为 public function
- [ ] 写 `python/zrt/__init__.py` 的稳定 API 清单（哪些 internal / 哪些可被外部依赖）
- [ ] 指定每个共享层的 owner（comm / flops / memory 公式有人 review，防止再分叉）

---

### 11.4 关键判断点（评审必须回答）

| # | 问题 | 影响后续阶段 |
|---|---|---|
| 1 | **选 A / B / C？** | 决定阶段 1-2 的实施范围（C 要删桥，工作量比 A 大） |
| 2 | **anchor 现在的覆盖够当回归基线吗？** | 不够的话阶段 3 要提前到阶段 0 后做，否则拆文件没法验证 |
| 3 | **谁是共享层 owner？** | 一旦 comm/flops 是 single source，必须有人 review 这些公式 |

---

### 11.5 推荐立即行动

1. **马上做阶段 0 的 5 项**（1-2 天）——零风险、纯收益、可独立 PR
2. **同时回答 § 11.4 的 3 个判断点**（评审会议上拍板）
3. 阶段 0 完成 + 判断点确定后，用 `writing-plans` 把阶段 1 拆成可执行的 implementation plan（含每个 PR 的 acceptance criteria）

---


## 附录 A — 命令复现

```bash
# LOC 排名
find python/zrt -name "*.py" -not -path "*__pycache__*" | xargs wc -l | sort -rn | head -50

# 跨包 import 计数
grep -rhE "^from (python\.)?zrt\.[a-z_]+" python/zrt --include="*.py" \
  | sed -E 's/^from (python\.)?zrt\.([a-z_]+).*/\2/' | sort | uniq -c | sort -rn

# 谁依赖 training/
grep -rlE "from (python\.)?zrt\.training" python/zrt --include="*.py" | grep -v "/training/"
```
