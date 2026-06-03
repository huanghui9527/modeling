# ZRT-Sim — LLM 算子捕获与性能建模

使用 `TorchDispatchMode` 在 **FakeTensor**（`FakeTensorMode`）上拦截 causal LM 的前向/反向，提取完整 aten 算子序列，应用并行变换后基于硬件规格估算性能。**无需下载模型权重**，只需 `config.json`。

```
HF config.json  →  AutoModelForCausalLM.from_config + FakeTensorMode
              →  TorchDispatchMode + ModuleTracker (无权重 forward/backward)
              →  raw OpGraph  →  transform pipeline  →  DAGScheduler  →  报告
```

671B 参数的 DeepSeek-V3 也能在几秒内完成抓图。

---

## 安装

```bash
pip install -r requirements.txt              # 核心：torch / transformers / openpyxl / onnx
pip install -r server/requirements.txt       # 可选：HTTP 服务（fastapi + uvicorn）
```

已验证组合：Python 3.14 + torch 2.11.0 + transformers 5.4.0（兼容 4.36+，详见 `graph/compat.py`）。

---

## 快速开始（CLI）

```bash
# 抓图（HF Hub，只下载 config.json）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3-0324 --layers 4
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4

# 本地目录（含 config.json 即可）
python -m python.zrt --model-id ./hf_models/deepseek_v3 --layers 4

# 推理性能建模（抓图 + 调度 + 报告）
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm --tp 8
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm --tp 8 --quant int8

# 训练建模（3D 并行 + ZeRO + 重计算）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 4 --ep 2 --dp 2 --zero-stage 1 --optimizer adam --recompute-policy selective --total-params 671e9 --num-layers-full 61

# 训练抓图（显式指定阶段）
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 2 --phases train_forward train_backward

# 仅抓训练前向
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 2 --phases train_forward

# 推理 + 训练混合（四阶段一次完成）
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --phases prefill decode train_forward train_backward

# 训练性能建模（抓图 + 1F1B 调度 + MFU 估算）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8

# Graph-capture + MegaMoE（融合 routed expert dispatch / compute / combine）
# --mega-moe-waves 0 表示自动选择合法 wave 数；也可以改成显式值，如 4。
python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --ep 8 --mega-moe --mega-moe-waves 0

# 完整 3D 并行 + ZeRO + 批次配置
python -m python.zrt --model-id hf_models/deepseek_v3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 4 --ep 2 --dp 2 --zero-stage 1 --optimizer adam --micro-batch 1 --global-batch 1024 --total-params 671e9 --num-layers-full 61

# 启用激活重计算
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --train --hw nvidia_h100_sxm --tp 8 --gradient-checkpointing

# 激活重计算策略（none / full / selective）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --recompute-policy full
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --recompute-policy selective

# Spec-based 训练估算（无需抓图，纯分析）→ 默认输出 Excel
python -m python.zrt --estimate-config python/zrt/training/configs/llama3_70b_3d.yaml

# Spec-based MegaMoE 示例（YAML 内 strategy.mega_moe: true）
python -m python.zrt --estimate-config python/zrt/training/configs/deepseek_v4_pro_3d_ascend_910c_mega_moe.yaml

# 网格搜索（Pareto 前沿）
python -m python.zrt --search-config python/zrt/training/configs/llama3_70b_3d.yaml

# torch.compile 图模式（替代 eager TorchDispatchMode 路径）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 2 --phases train_backward --graph-mode
```

**输出**默认写入 `output/<model_slug>/`。每个阶段（`prefill` / `decode` / `train_forward` / `train_backward`）一组文件：`_ops.xlsx`、`_raw_graph.json/.onnx`、`_fused_graph.json/.onnx`。指定 `--hw` 时额外生成 `reports/*_report.html` 和 `reports/*_trace.json`（`chrome://tracing`）。

完整参数：`python -m python.zrt --help`。

### `--layers` 选择建议

| 模型类型 | 推荐 |
|---------|------|
| 密集（Llama / Qwen2 / Mistral） | `2` |
| DeepSeek-V3（前 3 层密集，第 4 层起 MoE） | `4` |
| 第 1 层即 MoE（Mixtral / Qwen3-MoE） | `2` |

---

## Python API

```python
from python.zrt.pipeline import run_trace_phases

result = run_trace_phases(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    phases=("prefill", "decode"),
)
raw_graph, fused_graph = result.graphs["prefill"]
records = result.phase_records["prefill"]
print(result.output_dir)
```

复用已抓的图做训练建模：

```python
from python.zrt.transform.analysis.modeller import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry

hw = hw_registry.load("nvidia_h100_sxm")
report = estimate_training_from_graphs(
    forward_graph=result.graphs["train_forward"][0],
    backward_graph=result.graphs["train_backward"][0],
    hw_spec=hw,
    tp=8, pp=4, dp=2,
    total_params=671e9, num_layers_full=61,
)
print(report.summary())
```

应用并行 transform：

```python
from python.zrt.graph.transform_runner import run_transform
from python.zrt.transform import ParallelConfig, StreamConfig

output_dir, transformed_graph = run_transform(
    raw_graph=raw_graph,
    output_dir="output/qwen2_tp4ep2",
    parallel_config=ParallelConfig(tp=4, ep=2, dp=2),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw,
)
```

`ParallelConfig` 字段：`tp` / `pp` / `ep` / `dp` / `cp` / `sp`，例如 `ParallelConfig(tp=8, ep=2).describe()` → `"TP8-EP2"`。

---

## Prefill vs Decode

两阶段在**同一 FakeTensorMode 上下文**内运行，prefill 的 FakeTensor KV cache 直接传给 decode。

| 参数 | Prefill | Decode |
|------|---------|--------|
| `input_ids` | `(B, seq_len)` | `(B, 1)` |
| `position_ids` | `[[0..seq_len-1]]` | `[[seq_len]]` |
| `attention_mask` | `(1,1,seq_len,seq_len)` causal | `(1,1,1,seq_len+1)` 全零 |
| `past_key_values` | None | prefill 输出 |

---

## HTTP 服务

将 CLI 的三种模式封装为异步 REST 服务，提交后立即返回 `job_id`，通过轮询获取结果。

### 本机开发

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000/        → 启动页（launcher.html）
# 浏览器打开 http://localhost:8000/docs    → Swagger 交互文档
```

### 局域网部署（Linux 服务器）

`scripts/deploy_server.sh` 一键部署：装依赖 → 注册 systemd 服务（`Restart=always`，宕了自拉）→ 自动放行 ufw / firewalld / iptables → SELinux 端口标签 → **强制重启并清理端口占用**（杀掉残留的 uvicorn，杜绝 `Address already in use`）。

```bash
# 首次：装依赖 + 部署服务
sudo bash scripts/deploy_server.sh install

# 日常迭代：仅检查环境、重写 unit、强制重启（秒级）
sudo bash scripts/deploy_server.sh

# 自定义端口（默认 8001）
sudo bash scripts/deploy_server.sh install 8002
sudo bash scripts/deploy_server.sh 8002
```

| 环境变量 | 作用 |
|---------|------|
| `FORCE=1` | 强制重装依赖（即使 stamp 没过期），仅 `install` 模式生效 |
| `SKIP_FULL_DEPS=1` | 跳过 torch/transformers/networkx/openpyxl，仅装 fastapi+uvicorn。`/trace` 和 `/estimate` 将失效，仅保留 `/search` + `/health` |

部署完成后脚本会列出**所有非 loopback 网卡的 IPv4 + 完整 URL**，从同局域网 Windows 浏览器直接访问即可（前端 `launcher.html` 自动用 `location.origin` 作为 API base，无需手改）。常用运维命令：

```bash
systemctl status zrt-sim          # 状态
journalctl -u zrt-sim -f          # 实时日志（journald 自动轮转）
sudo systemctl disable --now zrt-sim && sudo rm /etc/systemd/system/zrt-sim.service  # 卸载
```

云主机额外步骤：在云厂商控制台的**安全组 / 网络 ACL** 也放行同一端口（脚本只能管 Linux 本机防火墙）。

### REST 接口

| 接口 | 等价 CLI |
|------|---------|
| `POST /trace` | `--model-id ... [--hw ...]`（抓图 / 推理 / 训练建模） |
| `POST /estimate` | `--estimate-config <yaml>` |
| `POST /search` | `--search-config <yaml>` |
| `GET /jobs` / `GET /jobs/{id}` | 列表 / 轮询 |
| `GET /hardware` / `GET /models` | 可用硬件 / 本地模型简写 |
| `GET /health` | 健康检查 |

任务状态：`pending` → `running` → `done` / `error`。结果体在 `done` 时填入 `result` 字段。详细 schema 见 `/docs`。

```python
import requests, time
resp = requests.post("http://localhost:8000/trace", json={
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "layers": 4, "hw": "nvidia_h100_sxm", "tp": 4,
})
job_id = resp.json()["id"]
while True:
    job = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
    if job["status"] in ("done", "error"): break
    time.sleep(5)
```

---

## 训练性能建模

`--train --hw` 同入口执行图捕获 + 调度（1F1B / VPP / DualPipe / ZB）+ MFU 估算。

**时延分解**（满足以下恒等式，单位 ms）：

| 字段 | 含义 | 关系 |
|------|------|------|
| `step_time_ms` | 完整单步 | `= pipeline_time + optimizer_time + optimizer_comm` |
| `pipeline_time_ms` | 流水线时间 | `= compute_time + exposed_comm` |
| `compute_time_ms` | 关键路径计算 | `= fwd_compute + bwd_compute + recompute_time` |
| `bubble_ms` | 流水线空泡 | `= warmup_ms + cooldown_ms` |
| `exposed_comm_ms` | 暴露通信（阻塞计算） | `= Σ *_exposed_ms` |
| `hidden_comm_ms` | 掩盖通信（重叠） | `= Σ *_hidden_ms` |
| `optimizer_time_ms` / `optimizer_comm_ms` | 优化器步 / 优化器通信 | — |

**MFU vs HFU**：MFU 不含重计算 FLOPs，HFU 含。开启重计算时 `hfu > mfu`。

**各并行组的暴露 / 掩盖**：

| 组 | 暴露字段 | 掩盖字段 | 掩盖机制 |
|----|---------|---------|---------|
| TP | `tp_exposed_ms` | `tp_hidden_ms` | CoC（10% 残留）/ MC2（全掩盖） |
| EP | `ep_exposed_ms` | `ep_hidden_ms` | wave-overlap（Expert GEMM ↔ A2A） |
| DP | `dp_exposed_ms` | `dp_hidden_ms` | pipeline bubble 期内 AR/RS |
| CP / PP | `*_exposed_ms` | — | — |

**重计算策略**：`--recompute-policy` 取 `none` / `full`（全部前向重算）/ `selective`（仅 Attention 算子）；`recompute_time_ms` 从 `bwd_compute` 独立拆出。

---

## 训练寻优

训练寻优有两类入口：

1. `python -m python.zrt --search-config <yaml>`：基于单个 YAML 配置生成并行策略 Pareto 前沿，适合快速验证某个训练配置空间。
2. `python/zrt/training/search/training_search_util.py`：批量扫描模型、硬件、seq、TP/CP/PP/EP/DP、重计算、量化策略等组合，输出全量结果和硬件对比报表，适合做大规模硬件/策略选型。

运行批量寻优脚本：

```bash
python python\zrt\training\search\training_search_util.py
```

脚本底部的 `training_param_grid` 是主要配置入口，常用字段包括：

| 字段 | 说明 |
|------|------|
| `model` / `hw` / `world_size` | 模型、硬件 YAML 名称和总卡数 |
| `seq_len` / `total_token` / `micro_batch` | 序列长度、总 token 数、micro batch；当前逻辑按 `total_token // seq_len` 推导 `global_batch`，适合使用规整 token |
| `tp` / `cp` / `pp` / `ep` / `dp` | 并行维度；`dp: "auto"` 时由 `world_size / (tp * cp * pp)` 自动推导 |
| `recompute` / `quant_preset` / `optimizer` | 重计算、混合精度/量化预设、优化器 |
| `pp_schedule` / `vpp_chunks` | PP 调度策略和虚拟流水段数 |

如果要把不同硬件按组对比，使用 `comparison_hw_groups`：

```python
comparison_hw_groups = [
    ["nvidia_b300", "nvidia_gb300_nvl576"],
    ["nvidia_b300", "ascend_910c"],
]
```

每个对比组的第一个硬件作为吞吐归一化基准。脚本会在每个 `对比组 × 硬件 × seq_len` 内按 `tokens_per_sec` 选出最优配置。

主要输出目录为：

```text
output/training_search/{model}_ws_{world_size}/
```

关键输出：

| 文件 | 内容 |
|------|------|
| `results_summary.csv` | 全量可行配置结果；包含 `step_time_ms`、`tokens_per_sec`、MFU/HFU、计算/通信/空泡/显存等字段 |
| `{model}_best_config_analysis.xlsx` | 按硬件和 seq 选出的最优配置分析表；包含 `raw_data` 和 `analysis` 两个 sheet |
| `{model}_{hw}_seq{seq}_ws{world}_best.xlsx` | 每个硬件和 seq 的最优配置明细（开启 `export_best_excel=True` 时生成） |

`analysis` sheet 中的占比口径：

| 指标 | 公式 |
|------|------|
| 计算时间 | `fwd_compute_ms + bwd_compute_ms + recompute_time_ms` |
| 计算占比 | `计算时间 / step_time_ms` |
| TP/EP/PP/DP/CP 通信占比 | 对应 `*_exposed_ms / step_time_ms` |
| 优化器占比 | `optimizer_compute_ms / step_time_ms` |
| 空泡占比 | `bubble_time_ms / step_time_ms` |
| 集群吞吐归一化 | 当前硬件 `tokens_per_sec / 同组同 seq 的基准硬件 tokens_per_sec` |

Excel 报表会自动合并并居中 `组号` 列；表头为蓝底，偶数对比组整组数据行为绿底。

---

## 多 tier 互联拓扑

`estimate` 路线由 `zrt.training.topology.CommDomain` 统一分派通信成本。硬件 YAML 支持任意层数：

```yaml
# Legacy 2-tier（H100/A100/910B/910C/B300）
interconnect:
  intra_node:
    type: NVLink4
    bandwidth_gbps: 900
    num_devices: 8
  inter_node:
    type: InfiniBand_NDR
    bandwidth_gbps: 400
    num_devices: 0          # 0 = 无界

# N-tier（NVL576 / Ascend SuperPod）
interconnect:
  tiers:
    - { name: nvlink_tray,   bandwidth_gbps: 1800, num_devices: 4 }
    - { name: nvswitch_rack, bandwidth_gbps: 900,  num_devices: 72 }
    - { name: ib_rail,       bandwidth_gbps: 800,  num_devices: 576 }
    - { name: ib_spine,      bandwidth_gbps: 400,  num_devices: 0 }
```

- `num_devices` 单调递增，最外层 `0` 表示无界。
- Search 路线的资源分配单元由**最内层 bounded tier** 容量推导，不再接受 `gpus_per_node`。

`CommDomain(system, strategy)` 一次构建、贯穿整个 `estimate()`：`domain.time(coll)` / `domain.ranks("TP")` / `domain.tier("EP")` / `domain.pp_p2p_link()` / `domain.summary()`。

---

## 并行策略约束（`--estimate-config`）

```
TP × CP × PP × DP = world_size
```

EP 在 DP group 内部，不占独立 rank。常见硬约束：

| 维度 | 规则 |
|------|------|
| TP | `num_heads % TP == 0`、`num_kv_heads % TP == 0`、`ffn % TP == 0` |
| EP | `num_experts % EP == 0`、`DP % EP == 0` |
| PP | `PP ≤ num_layers`、`global_batch / (micro_batch × DP) ≥ PP` |
| Batch | `global_batch % (micro_batch × DP) == 0` |
| ZeRO | `zero_stage ≥ 1` 时 `DP > 1` |
| VPP | `num_layers % (PP × vpp_chunks) == 0` |
| CP-Ring | `seq_len % (CP × 128) == 0` |

跨硬件 tier 的 TP / EP / CP 会触发性能警告（A2A / AR 落到带宽较低的外层）。

---

## 可用硬件

| 名称 | 互联 |
|------|------|
| `nvidia_h100_sxm` / `nvidia_h800` / `nvidia_a100_80g` / `nvidia_b300` | 2-tier |
| `nvidia_gb300_nvl576` | 4-tier (tray → rack → IB rail → IB spine) |
| `ascend_910b` / `ascend_910c` | 2-tier (HCCS / RoCE) |
| `ascend_910c_superpod` | 3-tier (HCCS supernode → RoCE pod → RoCE spine) |

YAML 在 `python/zrt/hardware/configs/`。

---

## 支持的模型

**开箱即用**（在 transformers 注册表中）：LLaMA-3、Qwen2 / Qwen2.5、Qwen3（dense + MoE）、Mistral、Mixtral、Gemma2、Phi-4。

**需要 `trust_remote_code`**（自动处理）：

| 模型 | 特殊处理 |
|------|---------|
| DeepSeek-V3 | MoE meta patch |
| DeepSeek-V3.2 | MoE + Indexer patch |
| DeepSeek-V4 | MoE patch + V4 推理 stubs；CUDA 融合 `v4_q_norm` / `v4_kv_quant` / `v4_sparse_attn`；NPU 融合 `npu_sas`（SWA/CSA/HCA 三变体） |

`hf_models/` 受**只读**约束：所有运行时兼容性修复必须写在 `python/zrt/graph/patches.py`，通过 monkey-patch 注入。唯一例外：`config.json` 可加 `auto_map` 字段。

新增自定义架构：(1) `hf_models/<name>/` 放入含 `auto_map` 的 `config.json` + modeling 文件；(2) 在 `compat._LOCAL_REGISTRY` 添加 `model_type` 与 Hub ID 映射；(3) 如遇 transformers API 兼容问题，在 `apply_version_shims()` 增补 shim。

---

## 项目结构

```
python/zrt/
├── pipeline.py             # 顶层抓图入口：run_trace / run_trace_phases
├── cli.py / __main__.py
├── graph/                  # 抓图：FakeTensor + TorchDispatchMode
│   ├── model_loader.py     # load_model() → (model, config, fake_mode)
│   ├── dispatch.py         # RecordingDispatch + TensorTracker（aten 拦截）
│   ├── tracker.py          # ModuleTracker（forward hooks）
│   ├── graph_builder.py    # build_op_graph
│   ├── transform_runner.py # 在抓到的图上跑 transform pipeline
│   ├── patches.py          # MoE / Indexer / V4 stubs
│   ├── compat.py           # transformers 版本 shim + 本地模型注册表
│   └── v4_fake_kernels.py  # DeepSeek-V4 自定义算子 FakeTensor stub
│
├── transform/              # 算子图变换
│   ├── context.py          # TransformContext / ParallelConfig / StreamConfig
│   ├── pipeline.py         # build_default_pipeline
│   ├── fusion/             # 融合包（rules / core / matching / building / ...）
│   ├── parallel/           # TP / EP / PP / DP / CP passes
│   ├── analysis/           # FLOPs / Roofline / 通信 / 训练建模
│   ├── optim/              # Quant / EPLB / SharedExpert / MTP
│   └── training/           # 训练专属 passes（Recompute / Offload / Optimizer / ZeroFSDP）
│
├── fusion/discover/        # 为新模型生成 fusion-rule YAML 草稿
├── executor/scheduler.py   # 拓扑排序 + 多流分配 → Timeline
├── simulator/              # SimulatorHub（Roofline / Regression / ProfileDB / Tiling）+ backends
├── hardware/               # HardwareSpec + YAML 配置
├── ir/                     # OpGraph / OpNode / OpEdge / TensorMeta
├── report/                 # summary / html_writer / chrome_trace / onnx_exporter / dot_exporter
└── training/               # 训练性能建模（spec / compose / search / topology / trace / ...）

hf_models/                  # 本地模型副本（只读！）
server/                     # FastAPI HTTP 服务
validation/                 # 端到端验证（python -m validation.cli）
tests/                      # pytest 套件，含 tests/training/anchors/ MFU 锚点回归
```

---

## 实现要点

**MoE Meta Patch**：MoE 模块的 `.cpu().numpy()` 路由在 FakeTensor 上失败。工具按 duck typing 检测（有 `experts: nn.ModuleList`）并替换为简化 forward：gate（捕获路由）→ `experts[0]`（捕获专家 MLP）→ shared expert（如有）。返回类型自动适配：DeepSeek/Qwen-MoE 返回单 tensor，Mixtral 返回 `(hidden, router_logits)`。

**两阶段融合**：(1) 同一 leaf module 触发的连续算子归为一组；(2) 若 parent module 内 ≤30 子算子，进一步合并到 parent 级别。融合规则集中在 `transform/fusion/`，按平台（cuda / ascend_npu / cpu / generic）分派。

**零开销算子过滤**：`view` / `reshape` / `expand` / `permute` / `contiguous` / `slice` / `unsqueeze` / `squeeze` / `arange` / `zeros` / `ones` / `full` / `tril` / `triu` / ... 默认跳过。

**版本 shim**（`graph/compat.py`）：在模型 import 前向 `transformers.*` 注入被移除的符号——`is_flash_attn_greater_or_equal_2_10`、`is_torch_fx_available`、`DynamicCache.from_legacy_cache` / `to_legacy_cache` / `get_usable_length` / `seen_tokens` / `get_max_length`——实现 transformers 4.x → 5.x 透明工作。

**Autocast FakeTensor 兼容**：`apply_compat_patches()` 将未知 device type 重定向到 `cpu`（对 FakeTensor 是 no-op），规避 transformers 4.50+ RoPE 在 FakeTensor 上调用 `torch.autocast` 失败的问题。

**输出 Excel**（抓图阶段，6 sheet）：Model Config / Fused Operators / Raw Operator Sequence / Summary / By Layer / Fusion Rules。同时生成 `*_fusion_rules.json`。指定 `--hw` 时额外生成 HTML 报告和 Chrome Trace。

---

## 测试

```bash
pytest tests/ -v 2>&1 | tail -n 50

# 单组 / 单函数
pytest tests/test_transform.py::test_tp_shape_modification -v
pytest tests/training/test_captured_graph_modelling.py -v
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v   # MFU 锚点回归

# 跳过网络用例
pytest -m "not network"
```

`tests/training/anchors/*.yaml` 用 GPT-3 175B / LLaMA-3 70B / DeepSeek-V3·V3.2·V4 的 published reference 数据锚定 `mfu` 和 `step_time_ms`，作为回归门；`strict_mfu_check: false` 标记的 anchor 处于**校准模式**（参考值已知，模拟器尚未对齐）。

---

## 注意事项

- 训练模块命令需要 `PYTHONPATH=python`（训练子包用 `zrt.*` 导入而非 `python.zrt.*`）。
- DeepSeek-V3-0324 是 V3 的 3 月 24 日更新版，**不是** V3.2。
- 不要把模型权重、生成的报告、本地虚拟环境提交进仓库；大体积产物放 `output/`。

设计文档分散在 `docs/`（部分可能与当前实现不完全同步，以代码为准）。
