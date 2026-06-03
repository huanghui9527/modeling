# Graph Capture MegaMoE Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add graph-capture MegaMoE fusion using the existing `mega_moe` and `mega_moe_waves` configuration names.

**Architecture:** Extend `ExpertGroupedMMPass` so `ctx.training.mega_moe=True` emits `mega_moe` nodes instead of the existing `GroupedMatMul -> silu -> GroupedMatMul` routed expert block. Reuse spec-path MegaMoE cost semantics for FLOPs and internal EP dispatch/combine overlap, and keep the current GroupedMM path unchanged when the flag is off.

**Tech Stack:** Python dataclasses, graph IR (`OpGraph`, `OpNode`, `Edge`, `TensorMeta`), transform passes, pytest, OpenPyXL report smoke tests.

---

## File Structure

- Modify `python/zrt/transform/context.py`
  - Add `TrainingConfig.mega_moe` and `TrainingConfig.mega_moe_waves`.

- Modify `python/zrt/transform/analysis/modeller.py`
  - Add `mega_moe` and `mega_moe_waves` parameters to `estimate_training_from_graphs()`.
  - Pass them into `TrainingConfig`.

- Modify `python/zrt/training/models/mega_moe.py`
  - Add a metadata-based helper so both spec `Op` and graph-capture `OpNode` can reuse the same cost terms.

- Modify `python/zrt/transform/parallel/expert_grouped_mm.py`
  - Add MegaMoE branch inside the existing forward and backward grouped expert fusion logic.
  - Build `mega_moe` nodes with spec-compatible metadata and without `ep_needs_a2a`.

- Modify `python/zrt/transform/analysis/passes.py`
  - Teach `FlopsPass` to compute graph-capture `mega_moe` FLOPs/read/write bytes and avoid MoE double scaling.
  - Teach `_calculate_grad_flops()` that `mega_moe` has dx and dw FLOPs equal to forward MegaMoE FLOPs.

- Modify `python/zrt/simulator/backends/roofline.py`
  - Add a graph-native `mega_moe` formula path that reads the metadata and returns nonzero FLOPs/bytes.

- Modify `python/zrt/transform/analysis/training.py`
  - Add internal MegaMoE EP comm exposed/hidden aggregation in `TrainingPipelinePass`.
  - Include `mega_moe_waves` in the strategy proxy when available.

- Modify `tests/test_transform.py`
  - Add fast synthetic unit tests for off/on behavior, metadata, no external A2A, and FLOPs scaling.

- Modify `tests/IT/test_ep_e2e.py`
  - Add a small DeepSeek-V4 graph-capture smoke test for `mega_moe=True`.

---

### Task 1: Add Graph-Capture Configuration Fields

**Files:**
- Modify: `python/zrt/transform/context.py`
- Modify: `python/zrt/transform/analysis/modeller.py`
- Test: `tests/training/test_captured_graph_modelling.py` or `tests/test_transform.py`

- [ ] **Step 1: Write the failing config plumbing test**

Add this test near other `estimate_training_from_graphs()` configuration tests. If there is no better local section, append it to `tests/training/test_captured_graph_modelling.py`:

```python
def test_estimate_training_from_graphs_passes_mega_moe_fields_to_context():
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.ir.types import DType, TensorMeta
    from python.zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec
    from python.zrt.transform.analysis import estimate_training_from_graphs

    hw = HardwareSpec(
        name="test",
        compute=ComputeSpec(bf16_tflops=1000, fp16_tflops=1000),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3000),
        interconnect=InterconnectSpec(),
    )
    x = TensorMeta.from_shape_dtype("x", (2, 4), DType.BF16)
    y = TensorMeta.from_shape_dtype("y", (2, 4), DType.BF16)
    node = OpNode(id="n0", op_type="aten.add.Tensor", inputs=[x], outputs=[y])
    graph = OpGraph("cfg", "train_forward", nodes={"n0": node})

    _report, ctx, _graphs = estimate_training_from_graphs(
        forward_graph=graph,
        hw_spec=hw,
        return_transformed=True,
        mega_moe=True,
        mega_moe_waves=4,
    )

    assert ctx.training.mega_moe is True
    assert ctx.training.mega_moe_waves == 4
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\training\test_captured_graph_modelling.py::test_estimate_training_from_graphs_passes_mega_moe_fields_to_context -q
```

Expected: fail with `TypeError: estimate_training_from_graphs() got an unexpected keyword argument 'mega_moe'`.

- [ ] **Step 3: Add fields to `TrainingConfig`**

In `python/zrt/transform/context.py`, add after `dp_overlap_in_bubble`:

```python
    # Graph-capture MegaMoE fusion. Uses the same user-facing names as
    # training.spec.strategy.Strategy.
    mega_moe: bool = False
    mega_moe_waves: int = 0
```

- [ ] **Step 4: Add parameters to `estimate_training_from_graphs()`**

In `python/zrt/transform/analysis/modeller.py`, add parameters after `recompute_policy`:

```python
    mega_moe: bool = False,
    mega_moe_waves: int = 0,
```

Then pass them into `TrainingConfig(...)`:

```python
            recompute_policy=recompute_policy,
            mega_moe=mega_moe,
            mega_moe_waves=mega_moe_waves,
            pp_schedule=pp_schedule,
```

- [ ] **Step 5: Run the config test and verify it passes**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\training\test_captured_graph_modelling.py::test_estimate_training_from_graphs_passes_mega_moe_fields_to_context -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit config plumbing**

```powershell
git add python/zrt/transform/context.py python/zrt/transform/analysis/modeller.py tests/training/test_captured_graph_modelling.py
git commit -m "feat: plumb graph capture mega moe config"
```

---

### Task 2: Add Shared MegaMoE Metadata Cost Helper

**Files:**
- Modify: `python/zrt/training/models/mega_moe.py`
- Modify: `tests/training/test_mega_moe_cost.py`

- [ ] **Step 1: Write the failing helper test**

Append to `tests/training/test_mega_moe_cost.py`:

```python
def test_mega_moe_cost_terms_from_meta_matches_op_wrapper():
    meta = {
        "m": 16,
        "micro_batch": 2,
        "n": 32,
        "n_local": 8,
        "k": 64,
        "k_local": 16,
        "top_k": 4,
        "num_experts": 8,
        "experts_per_rank": 2,
        "act_bytes": 2,
        "out_bytes": 2,
        "moe_act_bytes": 1,
        "weight_stored_bytes": 0.5,
        "fwd_multiplier": 12,
        "quant_variant": "w4a8",
    }

    from zrt.training.models.mega_moe import mega_moe_cost_terms_from_meta

    via_meta = mega_moe_cost_terms_from_meta(meta)
    via_op = mega_moe_cost_terms(_mega_moe_op(**meta))

    assert via_meta == via_op
    assert via_meta.fwd_multiplier == 3
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\training\test_mega_moe_cost.py::test_mega_moe_cost_terms_from_meta_matches_op_wrapper -q
```

Expected: fail with `ImportError` for `mega_moe_cost_terms_from_meta`.

- [ ] **Step 3: Implement `mega_moe_cost_terms_from_meta()`**

In `python/zrt/training/models/mega_moe.py`, replace the body of `mega_moe_cost_terms()` with a wrapper and move current logic into a new helper:

```python
from collections.abc import Mapping
from typing import Any


def mega_moe_cost_terms(op: Op) -> MegaMoECostTerms:
    return mega_moe_cost_terms_from_meta(op.meta)


def mega_moe_cost_terms_from_meta(meta: Mapping[str, Any]) -> MegaMoECostTerms:
    m = int(meta["m"])
    micro_batch = int(meta.get("micro_batch", 1))
    tokens = micro_batch * m
    n_logical = int(meta["n"])
    activation_n = int(meta.get("n_local", n_logical))
    k_eff = int(meta.get("k_local", meta["k"]))
    n = n_logical if "k_local" in meta else activation_n
    top_k = int(meta["top_k"])
    local_experts = int(meta.get("experts_per_rank", meta.get("num_experts", top_k)))
    raw_fwd_multiplier = float(meta.get("fwd_multiplier", 3))
    fwd_multiplier = _mega_moe_fwd_multiplier(raw_fwd_multiplier, top_k)
    act_bytes = float(meta.get("act_bytes", 2))
    moe_act_bytes = float(meta.get("moe_act_bytes", act_bytes))
    out_bytes = float(meta.get("out_bytes", act_bytes))
    weight_stored_bytes = float(meta.get("weight_stored_bytes", meta.get("weight_bytes", 2)))

    activation_input_bytes = float(tokens * activation_n * act_bytes)
    activation_output_bytes = float(tokens * activation_n * out_bytes)
    moe_activation_input_bytes = float(tokens * activation_n * moe_act_bytes)
    weight_bytes = float(local_experts * k_eff * n * fwd_multiplier * weight_stored_bytes)
    fwd_bytes = activation_input_bytes + activation_output_bytes + weight_bytes
    fwd_flops = float(2 * tokens * top_k * k_eff * n * fwd_multiplier)

    return MegaMoECostTerms(
        tokens=tokens,
        n=n,
        activation_n=activation_n,
        k_eff=k_eff,
        top_k=top_k,
        local_experts=local_experts,
        fwd_multiplier=fwd_multiplier,
        quant_variant=str(meta.get("quant_variant", "standard")),
        activation_input_bytes=activation_input_bytes,
        activation_output_bytes=activation_output_bytes,
        moe_activation_input_bytes=moe_activation_input_bytes,
        weight_bytes=weight_bytes,
        fwd_bytes=fwd_bytes,
        fwd_flops=fwd_flops,
    )
```

- [ ] **Step 4: Run MegaMoE cost tests**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\training\test_mega_moe_cost.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit shared helper**

```powershell
git add python/zrt/training/models/mega_moe.py tests/training/test_mega_moe_cost.py
git commit -m "refactor: share mega moe cost terms from metadata"
```

---

### Task 3: Add MegaMoE Fusion Branch In ExpertGroupedMMPass

**Files:**
- Modify: `python/zrt/transform/parallel/expert_grouped_mm.py`
- Modify: `tests/test_transform.py`

- [ ] **Step 1: Add failing forward fusion test**

Append to `tests/test_transform.py` near existing `ExpertGroupedMMPass` tests:

```python
def test_expert_grouped_mm_emits_mega_moe_when_enabled():
    src = _linear_node("src", "input", (2, 8), (2, 8))
    gate = _linear_node("gate", "transformer.layers.0.ffn.experts.0.w1", (2, 8), (2, 4))
    up = _linear_node("up", "transformer.layers.0.ffn.experts.0.w3", (2, 8), (2, 4))
    down = _linear_node("down", "transformer.layers.0.ffn.experts.0.w2", (2, 4), (2, 8))
    sink = _linear_node("sink", "post", (2, 8), (2, 8))
    for n in (gate, up, down):
        n.annotations.update({"phase": "fwd", "ep_needs_a2a": True, "ep_experts_local": 2})
    graph = OpGraph(
        name="mega_moe_fwd",
        phase="train",
        nodes={n.id: n for n in (src, gate, up, down, sink)},
        edges=[
            Edge("src", 0, "gate", 0, src.outputs[0]),
            Edge("src", 0, "up", 0, src.outputs[0]),
            Edge("gate", 0, "down", 0, gate.outputs[0]),
            Edge("up", 0, "down", 0, up.outputs[0]),
            Edge("down", 0, "sink", 0, down.outputs[0]),
        ],
        metadata={"seq_len": 4, "hidden": 8},
    )
    ctx = _ctx(ep=2)
    ctx.training.mega_moe = True
    ctx.training.mega_moe_waves = 4
    ctx.profile = SimpleNamespace(num_experts=4, moe_active=2)

    out = ExpertGroupedMMPass().run(graph, ctx)

    mega_nodes = [n for n in out.nodes.values() if n.op_type == "mega_moe"]
    assert len(mega_nodes) == 1
    mega = mega_nodes[0]
    assert mega.annotations["fused_by"] == "mega_moe_graph_capture"
    assert mega.annotations["fused_dispatch_compute_combine"] is True
    assert "ep_needs_a2a" not in mega.annotations
    assert "src" in out.predecessors(mega.id)
    assert "sink" in out.successors(mega.id)
    assert not any(n.op_type == "GroupedMatMul" for n in out.nodes.values())
    meta = mega.annotations["mega_moe_meta"]
    assert meta["m"] == 4
    assert meta["n"] == 8
    assert meta["n_local"] == 8
    assert meta["k"] == 4
    assert meta["k_local"] == 4
    assert meta["micro_batch"] == 1
    assert meta["num_experts"] == 4
    assert meta["top_k"] == 2
    assert meta["requested_waves"] == 4
    assert meta["ep"] == 2
    assert meta["experts_per_rank"] == 2
    assert meta["fwd_multiplier"] == 6
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\test_transform.py::test_expert_grouped_mm_emits_mega_moe_when_enabled -q
```

Expected: fail because no `mega_moe` node exists.

- [ ] **Step 3: Add a helper to detect the flag**

In `python/zrt/transform/parallel/expert_grouped_mm.py`, add:

```python
def _mega_moe_enabled(ctx: "TransformContext") -> bool:
    training = getattr(ctx, "training", None)
    return bool(getattr(training, "mega_moe", False))
```

- [ ] **Step 4: Add metadata builder helper**

In the same file, add:

```python
def _dtype_bytes(tensors: list[TensorMeta], default: float = 2.0) -> float:
    if tensors:
        return float(tensors[0].dtype.itemsize)
    return default


def _make_mega_moe_meta(
    *,
    ctx: "TransformContext",
    graph: "OpGraph",
    num_experts: int,
    ep: int,
    experts_per_rank: int,
    seq: int,
    hidden: int,
    ffn: int,
    n_local: int,
    k_local: int,
    input_tensors: list[TensorMeta],
    output_tensors: list[TensorMeta],
) -> dict:
    topk = ctx.profile.moe_active if ctx.profile else 1
    training = getattr(ctx, "training", None)
    micro_batch = training.micro_batch if training else 1
    return {
        "m": seq,
        "n": hidden,
        "k": ffn,
        "micro_batch": micro_batch,
        "num_experts": num_experts,
        "top_k": topk,
        "requested_waves": int(getattr(training, "mega_moe_waves", 0) or 0),
        "act_bytes": _dtype_bytes(input_tensors),
        "out_bytes": _dtype_bytes(output_tensors),
        "moe_act_bytes": _dtype_bytes(input_tensors),
        "weight_bytes": 2,
        "weight_stored_bytes": 2,
        "quant_variant": "standard",
        "fwd_multiplier": 3 * topk,
        "swiglu_clamp": None,
        "fused_dispatch_compute_combine": True,
        "ep": ep,
        "experts_per_rank": experts_per_rank,
        "n_local": n_local,
        "k_local": k_local,
    }
```

- [ ] **Step 5: Add `mega_moe` node constructor**

In the same file, add:

```python
def _make_mega_moe_node(
    node_id: str,
    scope: str,
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    src_node: OpNode,
    meta: dict,
    *,
    phase: str,
) -> OpNode:
    node = OpNode(
        id=node_id,
        op_type="mega_moe",
        inputs=inputs,
        outputs=outputs,
        attrs=dict(meta),
        scope=scope,
        layer=src_node.layer,
        category="compute",
        component="moe.mega_moe",
        fused_from=["GroupedMatMul", "aten.silu", "GroupedMatMul"],
        num_sub_ops=3,
        fusion_level="parent",
    )
    for key in ("recompute",):
        val = src_node.annotations.get(key)
        if val is not None and phase != "bwd":
            node.annotations[key] = val
    node.annotations["phase"] = phase
    node.annotations["fused_by"] = "mega_moe_graph_capture"
    node.annotations["fused_dispatch_compute_combine"] = True
    node.annotations["mega_moe_meta"] = dict(meta)
    node.annotations["ep_a2a_inserted"] = True
    return node
```

- [ ] **Step 6: Add forward branch in `_fuse_experts()`**

In the forward section after dimension validation and before current GroupedMM node construction, add:

```python
            if _mega_moe_enabled(ctx):
                self._fuse_forward_mega_moe_layer(
                    g=g,
                    layer_key=layer_key,
                    nodes=nodes,
                    gates=gates,
                    ups=ups,
                    downs=downs,
                    old_ids=old_ids,
                    in_edges=in_edges,
                    out_edges=out_edges,
                    num_experts=num_experts,
                    ep=ep,
                    experts_per_rank=G,
                    seq=seq,
                    hidden=hidden,
                    ffn=ffn,
                    H_in=H_in,
                    H_out=H_out,
                    ctx=ctx,
                )
                continue
```

Implement `_fuse_forward_mega_moe_layer()` below `_fuse_experts()` using the same remove/add/rewire pattern as the GroupedMM path:

```python
    def _fuse_forward_mega_moe_layer(
        self, *, g, layer_key, nodes, gates, ups, downs, old_ids,
        in_edges, out_edges, num_experts, ep, experts_per_rank,
        seq, hidden, ffn, H_in, H_out, ctx,
    ) -> None:
        input_tensor = gates[0].inputs[0] if gates[0].inputs else TensorMeta.from_shape_dtype("mega_moe_in", (seq, H_in), DType.BF16)
        output_tensor = downs[0].outputs[0] if downs[0].outputs else TensorMeta.from_shape_dtype("mega_moe_out", (seq, H_out), DType.BF16)
        meta = _make_mega_moe_meta(
            ctx=ctx,
            graph=g,
            num_experts=num_experts,
            ep=ep,
            experts_per_rank=experts_per_rank,
            seq=seq,
            hidden=hidden,
            ffn=ffn,
            n_local=H_in,
            k_local=ffn,
            input_tensors=[input_tensor],
            output_tensors=[output_tensor],
        )
        mega_id = f"{layer_key.replace('.','_')}_mega_moe"
        mega = _make_mega_moe_node(
            mega_id,
            f"{layer_key}.moe",
            [input_tensor],
            [output_tensor],
            gates[0],
            meta,
            phase="fwd",
        )
        g.edges = [e for e in g.edges if e.src not in old_ids and e.dst not in old_ids]
        for nid in old_ids:
            g.nodes.pop(nid, None)
        g.nodes[mega_id] = mega
        seen_src = set()
        for e in in_edges:
            key = (e.src, e.src_idx)
            if key not in seen_src:
                seen_src.add(key)
                g.edges.append(Edge(e.src, e.src_idx, mega_id, 0, mega.inputs[0]))
        for e in out_edges:
            g.edges.append(Edge(mega_id, 0, e.dst, e.dst_idx, mega.outputs[0]))
        g._rebuild_adjacency()
```

- [ ] **Step 7: Run the new forward test**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\test_transform.py::test_expert_grouped_mm_emits_mega_moe_when_enabled -q
```

Expected: pass.

- [ ] **Step 8: Add no external A2A test**

Append:

```python
def test_comm_inserter_skips_mega_moe_internal_a2a():
    src = _linear_node("src", "input", (2, 8), (2, 8))
    gate = _linear_node("gate", "transformer.layers.0.ffn.experts.0.w1", (2, 8), (2, 4))
    up = _linear_node("up", "transformer.layers.0.ffn.experts.0.w3", (2, 8), (2, 4))
    down = _linear_node("down", "transformer.layers.0.ffn.experts.0.w2", (2, 4), (2, 8))
    sink = _linear_node("sink", "post", (2, 8), (2, 8))
    for n in (gate, up, down):
        n.annotations.update({"phase": "fwd", "ep_needs_a2a": True, "ep_experts_local": 2})
    graph = OpGraph(
        name="mega_moe_comm",
        phase="train",
        nodes={n.id: n for n in (src, gate, up, down, sink)},
        edges=[
            Edge("src", 0, "gate", 0, src.outputs[0]),
            Edge("src", 0, "up", 0, src.outputs[0]),
            Edge("gate", 0, "down", 0, gate.outputs[0]),
            Edge("up", 0, "down", 0, up.outputs[0]),
            Edge("down", 0, "sink", 0, down.outputs[0]),
        ],
        metadata={"seq_len": 4, "hidden": 8},
    )
    ctx = _ctx(ep=2)
    ctx.training.mega_moe = True
    ctx.profile = SimpleNamespace(num_experts=4, moe_active=2)

    grouped = ExpertGroupedMMPass().run(graph, ctx)
    out = CommInserterPass().run(grouped, ctx)

    assert any(n.op_type == "mega_moe" for n in out.nodes.values())
    assert not any(
        n.op_type == "comm.all_to_all"
        and n.annotations.get("ep_role") in {"dispatch", "combine"}
        for n in out.nodes.values()
    )
```

- [ ] **Step 9: Run focused transform tests**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\test_transform.py -q
```

Expected: all tests in `tests/test_transform.py` pass.

- [ ] **Step 10: Commit forward MegaMoE fusion branch**

```powershell
git add python/zrt/transform/parallel/expert_grouped_mm.py tests/test_transform.py
git commit -m "feat: fuse graph capture routed experts as mega moe"
```

---

### Task 4: Preserve Backward Phase MegaMoE Fusion

**Files:**
- Modify: `python/zrt/transform/parallel/expert_grouped_mm.py`
- Modify: `tests/test_transform.py`

- [ ] **Step 1: Add failing backward test**

Append:

```python
def test_expert_grouped_mm_emits_mega_moe_backward_when_enabled():
    src = _linear_node("src", "input", (2, 8), (2, 8))
    down = _linear_node("down_bwd", "transformer.layers.0.ffn.experts.0.w2", (2, 8), (2, 4))
    gate = _linear_node("gate_bwd", "transformer.layers.0.ffn.experts.0.w1", (2, 4), (2, 4))
    up = _linear_node("up_bwd", "transformer.layers.0.ffn.experts.0.w3", (2, 4), (2, 4))
    sink = _linear_node("sink", "post", (2, 4), (2, 4))
    for n in (down, gate, up):
        n.annotations.update({"phase": "bwd", "ep_needs_a2a": True, "recompute": True})
    graph = OpGraph(
        name="mega_moe_bwd",
        phase="train",
        nodes={n.id: n for n in (src, down, gate, up, sink)},
        edges=[
            Edge("src", 0, "down_bwd", 0, src.outputs[0]),
            Edge("down_bwd", 0, "gate_bwd", 0, down.outputs[0]),
            Edge("down_bwd", 0, "up_bwd", 0, down.outputs[0]),
            Edge("gate_bwd", 0, "sink", 0, gate.outputs[0]),
            Edge("up_bwd", 0, "sink", 0, up.outputs[0]),
        ],
        metadata={"seq_len": 4, "hidden": 8},
    )
    ctx = _ctx(ep=2)
    ctx.training.mega_moe = True
    ctx.profile = SimpleNamespace(num_experts=4, moe_active=2)

    out = ExpertGroupedMMPass().run(graph, ctx)

    mega_nodes = [n for n in out.nodes.values() if n.op_type == "mega_moe"]
    assert len(mega_nodes) == 1
    mega = mega_nodes[0]
    assert mega.annotations["phase"] == "bwd"
    assert "recompute" not in mega.annotations
    assert "src" in out.predecessors(mega.id)
    assert "sink" in out.successors(mega.id)
```

- [ ] **Step 2: Run the backward test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\test_transform.py::test_expert_grouped_mm_emits_mega_moe_backward_when_enabled -q
```

Expected: fail because backward still emits grouped nodes or no MegaMoE node.

- [ ] **Step 3: Add MegaMoE branch to `_fuse_backward_layer()`**

At the point after `in_edges/out_edges` validation and before constructing grouped backward nodes, add:

```python
        if _mega_moe_enabled(getattr(g, "_transform_context", None)):
            ...
```

Do not attach context to `OpGraph`. Instead, change the `_fuse_backward_layer()` signature to receive `ctx` and `num_experts`, and call it from `_fuse_experts()` as:

```python
                self._fuse_backward_layer(
                    g, layer_key, nodes, G, M, tokens_per_ep_rank, hidden,
                    ctx=ctx, num_experts=num_experts,
                )
```

Then implement:

```python
        if _mega_moe_enabled(ctx):
            input_tensor = downs[0].inputs[0] if downs[0].inputs else TensorMeta.from_shape_dtype("mega_moe_bwd_in", (M, H_out), DType.BF16)
            output_tensor = TensorMeta.from_shape_dtype("mega_moe_bwd_out", (G, M, gate_up_dim), DType.BF16)
            meta = _make_mega_moe_meta(
                ctx=ctx,
                graph=g,
                num_experts=num_experts,
                ep=ctx.parallel.ep,
                experts_per_rank=G,
                seq=g.metadata.get("seq_len", 128),
                hidden=hidden,
                ffn=ffn,
                n_local=H_out,
                k_local=ffn,
                input_tensors=[input_tensor],
                output_tensors=[output_tensor],
            )
            mega_id = f"{layer_key.replace('.','_')}_mega_moe_bwd"
            mega = _make_mega_moe_node(
                mega_id,
                f"{layer_key}.moe",
                [input_tensor],
                [output_tensor],
                downs[0],
                meta,
                phase="bwd",
            )
            g.edges = [e for e in g.edges if e.src not in old_ids and e.dst not in old_ids]
            for nid in old_ids:
                g.nodes.pop(nid, None)
            g.nodes[mega_id] = mega
            seen_src = set()
            for e in in_edges:
                key = (e.src, e.src_idx)
                if key not in seen_src:
                    seen_src.add(key)
                    g.edges.append(Edge(e.src, e.src_idx, mega_id, 0, mega.inputs[0]))
            for e in out_edges:
                g.edges.append(Edge(mega_id, 0, e.dst, e.dst_idx, mega.outputs[0]))
            g._rebuild_adjacency()
            return
```

- [ ] **Step 4: Run backward and full transform tests**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\test_transform.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit backward MegaMoE fusion**

```powershell
git add python/zrt/transform/parallel/expert_grouped_mm.py tests/test_transform.py
git commit -m "feat: fuse graph capture mega moe backward path"
```

---

### Task 5: Add Analysis Support For Graph-Capture MegaMoE

**Files:**
- Modify: `python/zrt/transform/analysis/passes.py`
- Modify: `python/zrt/simulator/backends/roofline.py`
- Modify: `python/zrt/transform/analysis/training.py`
- Modify: `tests/test_transform.py`
- Modify: `tests/training/test_phase1_bugfixes.py` if a recompute-specific assertion fits better there

- [ ] **Step 1: Add failing FLOPs test**

Append to `tests/test_transform.py`:

```python
def test_flops_pass_handles_graph_capture_mega_moe_without_double_scaling():
    from zrt.training.models.mega_moe import mega_moe_cost_terms_from_meta

    meta = {
        "m": 4,
        "n": 8,
        "k": 4,
        "micro_batch": 1,
        "num_experts": 4,
        "top_k": 2,
        "requested_waves": 4,
        "act_bytes": 2,
        "out_bytes": 2,
        "moe_act_bytes": 2,
        "weight_bytes": 2,
        "weight_stored_bytes": 2,
        "quant_variant": "standard",
        "fwd_multiplier": 6,
        "swiglu_clamp": None,
        "fused_dispatch_compute_combine": True,
        "ep": 2,
        "experts_per_rank": 2,
        "n_local": 8,
        "k_local": 4,
    }
    node = OpNode(
        id="mega",
        op_type="mega_moe",
        inputs=[_t("x", (4, 8))],
        outputs=[_t("y", (4, 8))],
        attrs=dict(meta),
        scope="model.layers.0.ffn.experts.mega",
        annotations={
            "phase": "fwd",
            "fused_by": "mega_moe_graph_capture",
            "mega_moe_meta": dict(meta),
        },
    )
    graph = OpGraph(
        name="mega_flops",
        phase="train",
        nodes={"mega": node},
        metadata={"moe_active_experts": 6, "moe_total_experts": 8},
    )

    out = FlopsPass().run(graph, _ctx(ep=2))
    terms = mega_moe_cost_terms_from_meta(meta)
    actual = out.nodes["mega"].annotations

    assert actual["flops"] == int(terms.fwd_flops)
    assert actual["flops_fwd"] == int(terms.fwd_flops)
    assert actual["flops_dx"] == int(terms.fwd_flops)
    assert actual["flops_dw"] == int(terms.fwd_flops)
    assert actual["read_bytes"] == int(terms.fwd_bytes - terms.activation_output_bytes)
    assert actual["write_bytes"] == int(terms.activation_output_bytes)
```

- [ ] **Step 2: Run the FLOPs test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\test_transform.py::test_flops_pass_handles_graph_capture_mega_moe_without_double_scaling -q
```

Expected: fail because `FlopsPass` does not special-case `mega_moe`.

- [ ] **Step 3: Update `FlopsPass` for MegaMoE**

In `python/zrt/transform/analysis/passes.py`, import:

```python
from zrt.training.models.mega_moe import mega_moe_cost_terms_from_meta
```

Add before the `sem_flops` branch in `run()`:

```python
            if node.op_type == "mega_moe" or node.annotations.get("fused_by") == "mega_moe_graph_capture":
                meta = node.annotations.get("mega_moe_meta") or node.attrs
                terms = mega_moe_cost_terms_from_meta(meta)
                flops = terms.fwd_flops
                read_b = terms.fwd_bytes - terms.activation_output_bytes
                write_b = terms.activation_output_bytes
                node.annotations["flops"] = int(flops)
                node.annotations["read_bytes"] = int(read_b)
                node.annotations["write_bytes"] = int(write_b)
                if is_train:
                    rec_mult = 2.0 if is_external_recompute_node(node) and node.annotations.get("phase") != "bwd" else 1.0
                    node.annotations["flops_fwd"] = int(flops * rec_mult)
                    node.annotations["flops_dx"] = int(flops)
                    node.annotations["flops_dw"] = int(flops)
                continue
```

- [ ] **Step 4: Add roofline formula for `mega_moe`**

In `python/zrt/simulator/backends/roofline.py`, import the helper near other imports:

```python
from zrt.training.models.mega_moe import mega_moe_cost_terms_from_meta
```

Add:

```python
def _mega_moe_graph(node: "OpNode") -> FMR:
    meta = node.annotations.get("mega_moe_meta") or node.attrs
    terms = mega_moe_cost_terms_from_meta(meta)
    read_b = terms.fwd_bytes - terms.activation_output_bytes
    write_b = terms.activation_output_bytes
    return terms.fwd_flops, read_b, write_b
```

Register:

```python
    "mega_moe":                         _mega_moe_graph,
```

- [ ] **Step 5: Run FLOPs and roofline focused tests**

Run:

```powershell
$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m pytest tests\test_transform.py::test_flops_pass_handles_graph_capture_mega_moe_without_double_scaling tests\IT\test_roofline_formulas.py -k "grouped_mm or moe" -q
```

Expected: tests pass.

- [ ] **Step 6: Add internal EP comm timing aggregation**

In `python/zrt/transform/analysis/training.py`, import:

```python
from zrt.training.models.mega_moe import (
    _mega_moe_combine_bytes,
    _mega_moe_dispatch_bytes,
    mega_moe_cost_terms_from_meta,
    resolve_mega_moe_waves,
    simulate_wave_pipeline,
)
from python.zrt.transform.analysis.comm_latency import _estimate_comm_latency
```

Inside `TrainingPipelinePass.run()`, after overlap node aggregation and before `exposed_comm_ms` is assigned, add:

```python
        mega_moe_comm_total_us = 0.0
        mega_moe_exposed_us = 0.0
        mega_moe_hidden_us = 0.0
        for node in g.nodes.values():
            if node.op_type != "mega_moe":
                continue
            meta = node.annotations.get("mega_moe_meta") or node.attrs
            ep = int(meta.get("ep", ctx.parallel.ep if ctx.parallel else 1))
            if ep <= 1:
                continue
            terms = mega_moe_cost_terms_from_meta(meta)
            compute_s = float(node.annotations.get("base_latency_us", node.annotations.get("latency_us", 0.0))) / 1e6
            if compute_s <= 0:
                continue
            experts_per_rank = int(meta.get("experts_per_rank", max(1, terms.local_experts)))
            hardware_waves = int(getattr(getattr(hw, "compute", None), "ep_overlap_waves", 0) or 0)
            requested = int(getattr(ctx.training, "mega_moe_waves", 0) or meta.get("requested_waves", 0) or 0)
            waves = resolve_mega_moe_waves(
                requested=requested,
                hardware_waves=hardware_waves,
                experts_per_rank=experts_per_rank,
            )
            intra_node_devices = hw.interconnect.intra_node.num_devices
            cross_node = ep > intra_node_devices
            link = hw.interconnect.inter_node if cross_node else hw.interconnect.intra_node
            bandwidth_bps = link.effective_bw_bps(ep)
            dispatch_bytes = int(_mega_moe_dispatch_bytes(terms, ep))
            combine_bytes = int(_mega_moe_combine_bytes(terms, ep))
            dispatch_us = _estimate_comm_latency(
                "all_to_all", ep, dispatch_bytes, bandwidth_bps, link.latency_us
            )
            combine_us = _estimate_comm_latency(
                "all_to_all", ep, combine_bytes, bandwidth_bps, link.latency_us
            )
            node.annotations["mega_moe_dispatch_us"] = dispatch_us
            node.annotations["mega_moe_combine_us"] = combine_us
            pipeline = simulate_wave_pipeline(
                waves=waves,
                dispatch_s=dispatch_us / 1e6,
                compute_s=compute_s,
                combine_s=combine_us / 1e6,
            )
            mega_moe_comm_total_us += pipeline.comm_total_s * 1e6
            mega_moe_exposed_us += pipeline.exposed_comm_s * 1e6
            mega_moe_hidden_us += pipeline.hidden_comm_s * 1e6
        total_comm_us += mega_moe_comm_total_us
        total_exposed_us += mega_moe_exposed_us
        hidden_us += mega_moe_hidden_us
        step_time_us -= mega_moe_hidden_us
```

- [ ] **Step 7: Commit analysis support**

```powershell
git add python/zrt/transform/analysis/passes.py python/zrt/simulator/backends/roofline.py python/zrt/transform/analysis/training.py tests/test_transform.py
git commit -m "feat: analyze graph capture mega moe nodes"
```

---

### Task 6: Add End-to-End Graph-Capture Smoke Coverage

**Files:**
- Modify: `tests/IT/test_ep_e2e.py`
- Modify: `tests/IT/test_recompute_e2e.py` only if recompute smoke is not covered by synthetic tests

- [ ] **Step 1: Add E2E helper support**

In `tests/IT/test_ep_e2e.py`, update `_run_estimate()` to accept:

```python
    mega_moe: bool = False,
    mega_moe_waves: int = 0,
```

Pass through:

```python
        mega_moe=mega_moe,
        mega_moe_waves=mega_moe_waves,
```

- [ ] **Step 2: Add DeepSeek-V4 smoke test**

Append to the EP E2E class:

```python
    def test_mega_moe_graph_capture_smoke(self, captured):
        report, _ctx, transformed = _run_estimate(
            captured,
            ep=8,
            mega_moe=True,
            mega_moe_waves=4,
            return_transformed=True,
            model_id="hf_models/deepseek_v4",
        )
        graph = transformed["unified"]
        mega = [n for n in graph.nodes.values() if n.op_type == "mega_moe"]
        assert mega, "Expected MegaMoE nodes when mega_moe=True"
        assert report.step_time_ms > 0
        assert report.training_flops > 0
        assert "mega_moe" in report.fused_ops_summary
        assert not any(
            n.op_type == "comm.all_to_all"
            and n.annotations.get("ep_role") in {"dispatch", "combine"}
            and ".moe" in n.scope.lower()
            for n in graph.nodes.values()
        )
```

- [ ] **Step 3: Run the E2E smoke**

Run:

```powershell
$env:PYTHONPATH='python'; $env:TMP='D:\modeling\.pytest-tmp-mega-moe-e2e'; $env:TEMP='D:\modeling\.pytest-tmp-mega-moe-e2e'; .\.venv\Scripts\python.exe -m pytest --basetemp=.pytest-tmp-mega-moe-e2e tests\IT\test_ep_e2e.py::TestEPE2E::test_mega_moe_graph_capture_smoke -q
```

Expected: pass or skip only for the existing fake-mode skip reason.

- [ ] **Step 4: Run recompute regression**

Run:

```powershell
$env:PYTHONPATH='python'; $env:TMP='D:\modeling\.pytest-tmp-mega-moe-recompute'; $env:TEMP='D:\modeling\.pytest-tmp-mega-moe-recompute'; .\.venv\Scripts\python.exe -m pytest --basetemp=.pytest-tmp-mega-moe-recompute tests\IT\test_recompute_e2e.py -q
```

Expected: existing recompute E2E remains green.

- [ ] **Step 5: Commit E2E smoke**

```powershell
git add tests/IT/test_ep_e2e.py tests/IT/test_recompute_e2e.py
git commit -m "test: cover graph capture mega moe e2e"
```

---

### Task 7: Final Regression And Cleanup

**Files:**
- No planned code files.

- [ ] **Step 1: Run targeted regression suite**

Run:

```powershell
$env:PYTHONPATH='python'; $env:TMP='D:\modeling\.pytest-tmp-mega-moe-final'; $env:TEMP='D:\modeling\.pytest-tmp-mega-moe-final'; .\.venv\Scripts\python.exe -m pytest --basetemp=.pytest-tmp-mega-moe-final tests\test_transform.py tests\training\test_phase1_bugfixes.py tests\IT\test_ep_e2e.py tests\IT\test_recompute_e2e.py tests\transform\fusion\test_dsv4_rules.py tests\training\test_mega_moe_config.py tests\training\test_mega_moe_ir.py tests\training\test_mega_moe_cost.py tests\training\test_mega_moe_sharding.py tests\training\test_mega_moe_stage.py tests\training\test_mega_moe_integration.py tests\training\test_mega_moe_reports.py -q
```

Expected: all pass, except existing documented skips.

- [ ] **Step 2: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 3: Inspect git status**

Run:

```powershell
git status --short --branch
```

Expected: only intentional tracked changes are committed; existing untracked temp/report files may remain untouched.

---

## Execution Notes For Subagents

- Do not delete or modify unrelated untracked files in the workspace.
- Do not re-enable the DeepSeek-V4 YAML class-level `MoE` fusion rule.
- Preserve existing behavior when `ctx.training.mega_moe` is false.
- Keep `tests/training/test_phase1_bugfixes.py` changes if needed; despite the path name, existing graph-capture recompute tests live there.
- Prefer synthetic graph tests before slower DSV4 E2E tests.
- Keep commits small and task-aligned.
