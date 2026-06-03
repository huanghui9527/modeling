# Mega MoE Spec Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in spec-path `mega_moe` fused operator that replaces `routed_expert_ffn + EP A2A collectives` and supports `standard` and `w4a8` quant variants.

**Architecture:** Keep the existing path as the default. When `Strategy.mega_moe` is enabled, builders emit `Op(kind="mega_moe")`, sharding writes EP/TP/CP metadata without inserting routed EP collectives, and stage timing uses a fused dispatch/compute/combine wave model. Quant variant is inferred from existing model dtypes rather than a second user-facing switch.

**Tech Stack:** Python dataclasses, pytest, existing ZRT training IR (`zrt.training.ir`), stage composer (`zrt.training.compose.stage`), dtype and config loader modules.

---

## File Structure

Modify these existing files:

- `python/zrt/training/spec/strategy.py`: add `mega_moe` and `mega_moe_waves` strategy fields.
- `python/zrt/training/io/config_loader.py`: parse YAML `mega_moe` and `mega_moe_waves`.
- `python/zrt/training/search/training_search_util.py`: propagate the new strategy fields through search configs and preserve `ep_overlap_waves` in generated `SystemSpec`.
- `python/zrt/training/ir/builders.py`: pass `Strategy` into MoE FFN construction and emit `mega_moe` when enabled.
- `python/zrt/training/ir/shard.py`: skip EP A2A collectives for `mega_moe`, apply TP/CP/EP sharding metadata, and keep router shape adjustment.
- `python/zrt/training/models/flops.py`: add `op.kind == "mega_moe"` cost branch.
- `python/zrt/training/compose/stage.py`: special-case `mega_moe` timing and report EP exposed/hidden communication from the fused wave model.
- `python/zrt/training/io/html_exporter.py`: show meaningful formulas for `mega_moe`.
- `python/zrt/training/io/excel_exporter.py`: surface `mega_moe` strategy fields and let the Ops sheet display `mega_moe` formula details.

Create these new files:

- `python/zrt/training/models/mega_moe.py`: focused helpers for quant variant inference, wave count resolution, per-phase bytes/FLOPs, and wave-pipeline timing.
- `tests/training/test_mega_moe_config.py`: strategy/config/search parsing coverage.
- `tests/training/test_mega_moe_ir.py`: IR shape, sharding, and quant-variant metadata coverage.
- `tests/training/test_mega_moe_cost.py`: cost model, wave model, stage timing, and no-double-counting coverage.

Do not modify graph-capture code in this plan.

---

### Task 1: Strategy And Config Switch

**Files:**
- Modify: `python/zrt/training/spec/strategy.py`
- Modify: `python/zrt/training/io/config_loader.py`
- Modify: `python/zrt/training/search/training_search_util.py`
- Test: `tests/training/test_mega_moe_config.py`
- Test: `tests/training/test_training_search_util.py`

- [ ] **Step 1: Write failing strategy/config tests**

Create `tests/training/test_mega_moe_config.py`:

```python
from __future__ import annotations

from zrt.training.io.config_loader import _parse_strategy
from zrt.training.search.training_search_util import _make_strategy_from_config
from zrt.training.spec.strategy import Strategy


def test_strategy_defaults_keep_mega_moe_disabled():
    strategy = Strategy()

    assert strategy.mega_moe is False
    assert strategy.mega_moe_waves == 0


def test_parse_strategy_accepts_mega_moe_switch():
    strategy = _parse_strategy({
        "ep": 8,
        "mega_moe": True,
        "mega_moe_waves": 4,
    })

    assert strategy.ep == 8
    assert strategy.mega_moe is True
    assert strategy.mega_moe_waves == 4


def test_search_strategy_accepts_mega_moe_switch():
    strategy = _make_strategy_from_config({
        "ep": 8,
        "mega_moe": True,
        "mega_moe_waves": 2,
    })

    assert strategy.ep == 8
    assert strategy.mega_moe is True
    assert strategy.mega_moe_waves == 2
```

Append this test to `tests/training/test_training_search_util.py` inside `TestMakeStrategyFromConfig`:

```python
    def test_mega_moe_fields(self):
        config = {"mega_moe": True, "mega_moe_waves": 4}
        strategy = _make_strategy_from_config(config)

        assert strategy.mega_moe is True
        assert strategy.mega_moe_waves == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_config.py tests/training/test_training_search_util.py::TestMakeStrategyFromConfig::test_mega_moe_fields -v
```

Expected: FAIL because `Strategy` has no `mega_moe` or `mega_moe_waves` fields.

- [ ] **Step 3: Add strategy fields**

In `python/zrt/training/spec/strategy.py`, add these fields after `ep_overlap`:

```python
    ep_overlap: bool = False
    # Spec-path fused MoE operator. When enabled, MoE routed experts are built
    # as one mega_moe op that includes dispatch, expert FFN, and combine.
    mega_moe: bool = False
    # 0 means resolve from hardware ep_overlap_waves or a conservative default.
    mega_moe_waves: int = 0
    dualbatch: bool = False
```

- [ ] **Step 4: Parse YAML strategy fields**

In `python/zrt/training/io/config_loader.py`, add these keyword arguments in `_parse_strategy()` immediately after `ep_overlap`:

```python
        ep_overlap=d.get("ep_overlap", False),
        mega_moe=d.get("mega_moe", False),
        mega_moe_waves=int(d.get("mega_moe_waves", 0)),
        dualbatch=d.get("dualbatch", False),
```

- [ ] **Step 5: Parse search strategy fields**

In `python/zrt/training/search/training_search_util.py`, update every `Strategy(...)` construction that already passes `ep_overlap` to also pass the new fields.

For `_make_strategy_from_config()`:

```python
        ep_overlap=config.get("ep_overlap", False),
        mega_moe=config.get("mega_moe", False),
        mega_moe_waves=int(config.get("mega_moe_waves", 0)),
        cp_kind=CPKind(config.get("cp_kind", "none")),
```

For `_build_strategy_for_validation()`:

```python
            ep_overlap=other_config.get("ep_overlap", False),
            mega_moe=other_config.get("mega_moe", False),
            mega_moe_waves=int(other_config.get("mega_moe_waves", 0)),
            cp_kind=CPKind(other_config.get("cp_kind", "none")),
```

Also add `ep_overlap_waves=hw.compute.ep_overlap_waves` to each `GPU(...)` construction in this file:

```python
            ep_overlap_waves=hw.compute.ep_overlap_waves,
            compute_efficiency=hw.compute.compute_efficiency,
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_config.py tests/training/test_training_search_util.py::TestMakeStrategyFromConfig -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/zrt/training/spec/strategy.py python/zrt/training/io/config_loader.py python/zrt/training/search/training_search_util.py tests/training/test_mega_moe_config.py tests/training/test_training_search_util.py
git commit -m "feat: add mega moe strategy switch"
```

---

### Task 2: Mega MoE Helper Module

**Files:**
- Create: `python/zrt/training/models/mega_moe.py`
- Test: `tests/training/test_mega_moe_cost.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/training/test_mega_moe_cost.py` with these initial tests:

```python
from __future__ import annotations

import pytest

from zrt.training.models.mega_moe import (
    infer_quant_variant,
    resolve_mega_moe_waves,
    simulate_wave_pipeline,
)
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


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


def test_infer_quant_variant_standard_for_bf16():
    model = _moe_model()

    assert infer_quant_variant(model) == "standard"


def test_infer_quant_variant_w4a8_for_fp4_weights_and_fp8_moe_acts():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )

    assert infer_quant_variant(model) == "w4a8"


def test_resolve_waves_prefers_valid_divisor_not_above_target():
    assert resolve_mega_moe_waves(requested=6, hardware_waves=0, experts_per_rank=8) == 4
    assert resolve_mega_moe_waves(requested=0, hardware_waves=4, experts_per_rank=8) == 4
    assert resolve_mega_moe_waves(requested=0, hardware_waves=0, experts_per_rank=3) == 3


def test_wave_pipeline_one_wave_is_serial():
    result = simulate_wave_pipeline(waves=1, dispatch_s=1.0, compute_s=3.0, combine_s=2.0)

    assert result.total_s == pytest.approx(6.0)
    assert result.exposed_comm_s == pytest.approx(3.0)
    assert result.hidden_comm_s == pytest.approx(0.0)


def test_wave_pipeline_more_waves_hide_comm_when_compute_dominates():
    one = simulate_wave_pipeline(waves=1, dispatch_s=1.0, compute_s=6.0, combine_s=1.0)
    four = simulate_wave_pipeline(waves=4, dispatch_s=0.25, compute_s=1.5, combine_s=0.25)

    assert four.total_s < one.total_s
    assert four.hidden_comm_s > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'zrt.training.models.mega_moe'`.

- [ ] **Step 3: Create helper module**

Create `python/zrt/training/models/mega_moe.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec


FP8_DTYPES = {Dtype.FP8_E4M3, Dtype.FP8_E5M2}


@dataclass(frozen=True)
class WavePipelineResult:
    total_s: float
    exposed_comm_s: float
    hidden_comm_s: float
    comm_total_s: float


def infer_quant_variant(model: ModelSpec) -> str:
    if (
        model.routed_expert_weight_dtype is Dtype.FP4
        and model.effective_moe_act_dtype() in FP8_DTYPES
    ):
        return "w4a8"
    return "standard"


def resolve_mega_moe_waves(
    *,
    requested: int,
    hardware_waves: int,
    experts_per_rank: int,
) -> int:
    if experts_per_rank <= 1:
        return 1
    target = requested if requested > 0 else (hardware_waves if hardware_waves > 0 else 4)
    target = max(1, min(int(target), experts_per_rank))
    divisors = [d for d in range(1, experts_per_rank + 1) if experts_per_rank % d == 0]
    valid = [d for d in divisors if d <= target]
    return max(valid) if valid else 1


def simulate_wave_pipeline(
    *,
    waves: int,
    dispatch_s: float,
    compute_s: float,
    combine_s: float,
) -> WavePipelineResult:
    if waves <= 0:
        return WavePipelineResult(0.0, 0.0, 0.0, 0.0)

    dispatch_done: list[float | None] = [None] * waves
    compute_done: list[float | None] = [None] * waves
    combine_done: list[bool] = [False] * waves

    comm_cursor = 0.0
    compute_cursor = 0.0
    next_dispatch = 0
    next_compute = 0
    completed_combines = 0

    def schedule_ready_compute() -> None:
        nonlocal next_compute, compute_cursor
        while next_compute < waves and dispatch_done[next_compute] is not None:
            start = max(compute_cursor, dispatch_done[next_compute] or 0.0)
            compute_cursor = start + compute_s
            compute_done[next_compute] = compute_cursor
            next_compute += 1

    while completed_combines < waves:
        schedule_ready_compute()

        ready_combine = None
        for wave_id, done_at in enumerate(compute_done):
            if done_at is not None and not combine_done[wave_id] and done_at <= comm_cursor:
                ready_combine = wave_id
                break

        if ready_combine is not None:
            comm_cursor += combine_s
            combine_done[ready_combine] = True
            completed_combines += 1
            continue

        if next_dispatch < waves:
            comm_cursor += dispatch_s
            dispatch_done[next_dispatch] = comm_cursor
            next_dispatch += 1
            continue

        future_combines = [
            done_at for wave_id, done_at in enumerate(compute_done)
            if done_at is not None and not combine_done[wave_id] and done_at > comm_cursor
        ]
        if future_combines:
            comm_cursor = min(future_combines)
            continue

        schedule_ready_compute()

    total = max(comm_cursor, compute_cursor)
    comm_total = waves * (dispatch_s + combine_s)
    exposed = max(0.0, total - waves * compute_s)
    exposed = min(exposed, comm_total)
    hidden = max(0.0, comm_total - exposed)
    return WavePipelineResult(total, exposed, hidden, comm_total)
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/zrt/training/models/mega_moe.py tests/training/test_mega_moe_cost.py
git commit -m "feat: add mega moe modeling helpers"
```

---

### Task 3: IR Builder Emits Mega MoE

**Files:**
- Modify: `python/zrt/training/ir/builders.py`
- Test: `tests/training/test_mega_moe_ir.py`

- [ ] **Step 1: Write failing IR tests**

Create `tests/training/test_mega_moe_ir.py`:

```python
from __future__ import annotations

from zrt.training.ir.builders import build_graph
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy


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


def test_mega_moe_off_keeps_legacy_routed_expert_and_a2a():
    model = _moe_model()
    graph = build_graph(model, Strategy(ep=4, mega_moe=False))

    assert any(op.name == "L0.routed_expert_ffn" for op in graph.ops)
    assert not any(op.kind == "mega_moe" for op in graph.ops)
    assert sum(1 for c in graph.collectives if c.group == "EP" and c.kind == "A2A") == 2


def test_mega_moe_on_replaces_routed_expert_op():
    model = _moe_model()
    graph = build_graph(model, Strategy(ep=4, mega_moe=True, mega_moe_waves=2))

    mega_ops = [op for op in graph.ops if op.kind == "mega_moe"]
    assert len(mega_ops) == 1
    assert mega_ops[0].name == "L0.mega_moe"
    assert not any(op.name == "L0.routed_expert_ffn" for op in graph.ops)
    assert any(op.name == "L0.shared_up_proj" for op in graph.ops)
    assert any(op.name == "L0.expert_agg" for op in graph.ops)
    assert sum(1 for c in graph.collectives if c.group == "EP" and c.kind == "A2A") == 0


def test_mega_moe_standard_meta():
    model = _moe_model()
    graph = build_graph(model, Strategy(ep=4, mega_moe=True, mega_moe_waves=2))
    op = next(op for op in graph.ops if op.kind == "mega_moe")

    assert op.meta["m"] == model.seq_len
    assert op.meta["n"] == model.hidden
    assert op.meta["k"] == model.moe_ffn
    assert op.meta["num_experts"] == model.num_experts
    assert op.meta["top_k"] == model.top_k
    assert op.meta["requested_waves"] == 2
    assert op.meta["quant_variant"] == "standard"
    assert op.meta["weight_stored_bytes"] == Dtype.BF16.stored_bytes


def test_mega_moe_w4a8_meta_from_existing_quant_fields():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )
    graph = build_graph(model, Strategy(ep=4, mega_moe=True, mega_moe_waves=2))
    op = next(op for op in graph.ops if op.kind == "mega_moe")

    assert op.meta["quant_variant"] == "w4a8"
    assert op.meta["act_bytes"] == Dtype.FP8_E4M3.bytes
    assert op.meta["weight_bytes"] == Dtype.FP4.bytes
    assert op.meta["weight_stored_bytes"] == Dtype.FP4.stored_bytes
```

- [ ] **Step 2: Run IR tests to verify they fail**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_ir.py -v
```

Expected: FAIL because `build_graph()` still emits `L0.routed_expert_ffn` when `mega_moe=True`.

- [ ] **Step 3: Pass strategy through MoE builders**

In `python/zrt/training/ir/builders.py`, add this import:

```python
from zrt.training.models.mega_moe import infer_quant_variant
```

Change `_build_moe_ffn_ops` signature:

```python
def _build_moe_ffn_ops(model: ModelSpec, layer_id: int, seq: int,
                       prefix: str, layer_kind: LayerKind,
                       act_dtype: Dtype, strategy: Strategy | None = None) -> list[Op]:
```

Change `_moe_block` signature:

```python
    model: ModelSpec | None = None,
    strategy: Strategy | None = None,
) -> list[Op]:
```

When calling `_build_moe_ffn_ops`, pass `strategy=strategy`:

```python
        ops.extend(_build_moe_ffn_ops(model, layer_id, seq, prefix,
                                       LayerKind.MOE, act_dtype,
                                       strategy=strategy))
```

Change `_mtp_block` signature to accept `strategy: Strategy | None = None`, and pass it into `_moe_block`.

In `build_graph()`, pass `strategy=strategy` into both `_moe_block(...)` and `_mtp_block(...)`.

- [ ] **Step 4: Emit mega_moe Op when the switch is on**

Replace the current routed expert append in `_build_moe_ffn_ops` with this conditional:

```python
    if strategy is not None and strategy.mega_moe:
        routed_kind = "mega_moe"
        routed_name = f"{prefix}.mega_moe"
        routed_meta = {
            "m": seq,
            "n": h,
            "k": model.moe_ffn,
            "micro_batch": strategy.micro_batch,
            "num_experts": model.num_experts,
            "top_k": model.top_k,
            "requested_waves": strategy.mega_moe_waves,
            "act_bytes": model.effective_moe_act_dtype().bytes,
            "out_bytes": model.act_dtype.bytes,
            "weight_bytes": model.routed_expert_weight_dtype.bytes,
            "weight_stored_bytes": model.routed_expert_weight_dtype.stored_bytes,
            "quant_variant": infer_quant_variant(model),
            "fwd_multiplier": 3 * model.top_k,
            "swiglu_clamp": model.swiglu_clamp,
            "fused_dispatch_compute_combine": True,
        }
    else:
        routed_kind = "matmul"
        routed_name = f"{prefix}.routed_expert_ffn"
        routed_meta = {
            "m": seq,
            "n": h,
            "k": model.moe_ffn,
            "fwd_multiplier": 3 * model.top_k,
            "swiglu_clamp": model.swiglu_clamp,
            "fused_weight_dims": True,
        }

    ops.append(Op(name=routed_name, kind=routed_kind,
        inputs=[_tensor("x_ln2", (seq, h), model.effective_moe_act_dtype())],
        outputs=[_tensor("routed_ffn_out", (seq, h), model.act_dtype)],
        meta=routed_meta,
        layer_id=layer_id, layer_kind=layer_kind, component="routed_expert"))
```

- [ ] **Step 5: Run IR tests**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_ir.py -v
```

Expected: some tests may still fail because EP A2A insertion and sharding do not yet understand `mega_moe`.

- [ ] **Step 6: Commit the builder change only after Task 4 passes**

Do not commit at this step if tests still fail due to sharding. Continue to Task 4 and commit both builder and sharding changes together.

---

### Task 4: Sharding And Collective Insertion For Mega MoE

**Files:**
- Modify: `python/zrt/training/ir/shard.py`
- Test: `tests/training/test_mega_moe_ir.py`

- [ ] **Step 1: Extend failing sharding tests**

Append these tests to `tests/training/test_mega_moe_ir.py`:

```python
def test_mega_moe_ep_meta_and_router_shape():
    model = _moe_model(num_experts=8)
    graph = build_graph(model, Strategy(ep=4, mega_moe=True, mega_moe_waves=2))
    mega = next(op for op in graph.ops if op.kind == "mega_moe")
    router = next(op for op in graph.ops if op.name == "L0.router")

    assert mega.meta["ep"] == 4
    assert mega.meta["experts_per_rank"] == 2
    assert router.outputs[0].shape_local == (model.seq_len, 2)


def test_mega_moe_tp_and_cp_sharding_metadata():
    from zrt.training.spec.strategy import CPKind

    model = _moe_model(hidden=1024, moe_ffn=2048, num_experts=8)
    strategy = Strategy(tp=2, cp=4, cp_kind=CPKind.ULYSSES, ep=4, mega_moe=True)
    graph = build_graph(model, strategy)
    mega = next(op for op in graph.ops if op.kind == "mega_moe")

    assert mega.meta["m"] == model.seq_len // 4
    assert mega.meta["k_local"] == model.moe_ffn // 2
    assert mega.inputs[0].shape_local == (model.seq_len // 4, model.hidden)
    assert mega.outputs[0].shape_local == (model.seq_len // 4, model.hidden // 2)
```

- [ ] **Step 2: Run sharding tests to verify they fail**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_ir.py -v
```

Expected: FAIL because `mega_moe` lacks EP metadata and TP/CP sharding metadata.

- [ ] **Step 3: Skip EP A2A insertion for mega_moe**

In `_insert_ep_collectives()` in `python/zrt/training/ir/shard.py`, keep the current `routed_expert` handling but add an explicit `mega_moe` branch before it:

```python
            if op.kind == "mega_moe":
                continue

            if op.kind == "matmul" and "routed_expert" in op.name:
                collectives.append(Collective(
                    name=f"a2a_before_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="both",
                ))
                collectives.append(Collective(
                    name=f"a2a_after_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))
```

- [ ] **Step 4: Apply TP sharding to mega_moe**

In `_apply_tp_sharding()`, inside the `if op.kind == "matmul": ...` block sibling section, add:

```python
        elif op.kind == "mega_moe":
            k = op.meta.get("k", 0)
            n = op.meta.get("n", 0)
            if k > 0:
                op.meta["k_local"] = k // shard.tp
            if n > 0:
                n_local = n // shard.tp
                for t in op.outputs:
                    if t.shape_logical and t.shape_logical[-1] == n:
                        t.shape_local = (t.shape_logical[0], n_local)
```

Use this exact assertion after editing: `op.meta["k_local"]` should become `moe_ffn/tp` for fused expert GEMM accounting, input shape should remain the hidden activation shape, and output shape should become `(seq, hidden/tp)`.

- [ ] **Step 5: Apply CP sharding to mega_moe**

In `_apply_cp_sharding()`, after the generic tensor shape update loop and before attention handling, add:

```python
        if op.kind == "mega_moe":
            if "m" in op.meta:
                op.meta["m"] = op.meta["m"] // shard.cp
            continue
```

- [ ] **Step 6: Apply EP metadata to mega_moe**

In `_apply_ep_sharding()`, add:

```python
        if op.kind == "mega_moe":
            op.meta["ep"] = shard.ep
            op.meta["experts_per_rank"] = experts_per_rank
            continue
```

Keep the existing router output shape logic below this branch unchanged.

- [ ] **Step 7: Run IR tests**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_ir.py tests/training/test_ir_dense.py::test_ep_a2a_phase_both tests/training/test_ir_dense.py::test_ep_a2a_bytes_with_tp -v
```

Expected: PASS.

- [ ] **Step 8: Commit builder and sharding changes**

```bash
git add python/zrt/training/ir/builders.py python/zrt/training/ir/shard.py tests/training/test_mega_moe_ir.py
git commit -m "feat: emit mega moe op in spec graph"
```

---

### Task 5: FLOPs And Byte Cost For Mega MoE

**Files:**
- Modify: `python/zrt/training/models/flops.py`
- Modify: `python/zrt/training/models/mega_moe.py`
- Test: `tests/training/test_mega_moe_cost.py`

- [ ] **Step 1: Add failing op_cost tests**

Append to `tests/training/test_mega_moe_cost.py`:

```python
from zrt.training.ir.builders import build_graph
from zrt.training.models.flops import op_cost
from zrt.training.spec.strategy import Strategy


def test_mega_moe_op_cost_has_forward_and_backward_compute():
    model = _moe_model()
    graph = build_graph(model, Strategy(ep=4, mega_moe=True))
    op = next(op for op in graph.ops if op.kind == "mega_moe")

    cost = op_cost(op, model)

    assert cost.fwd_cube_flops > 0
    assert cost.dx_cube_flops > 0
    assert cost.dw_cube_flops > 0
    assert cost.fwd_bytes > 0


def test_mega_moe_w4a8_uses_lower_weight_bytes_than_bf16():
    bf16_model = _moe_model()
    w4a8_model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )

    bf16_op = next(op for op in build_graph(bf16_model, Strategy(ep=4, mega_moe=True)).ops if op.kind == "mega_moe")
    w4a8_op = next(op for op in build_graph(w4a8_model, Strategy(ep=4, mega_moe=True)).ops if op.kind == "mega_moe")

    assert op_cost(w4a8_op, w4a8_model).fwd_bytes < op_cost(bf16_op, bf16_model).fwd_bytes
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py::test_mega_moe_op_cost_has_forward_and_backward_compute tests/training/test_mega_moe_cost.py::test_mega_moe_w4a8_uses_lower_weight_bytes_than_bf16 -v
```

Expected: FAIL because `op_cost()` returns zero for unknown `mega_moe`.

- [ ] **Step 3: Add per-op cost helper**

In `python/zrt/training/models/mega_moe.py`, add:

```python
@dataclass(frozen=True)
class MegaMoECostTerms:
    fwd_flops: float
    fwd_bytes: float
    dispatch_bytes_effective: int
    combine_bytes_effective: int
    dispatch_bytes_per_rank: int
    combine_bytes_per_rank: int


def mega_moe_cost_terms(op, model: ModelSpec) -> MegaMoECostTerms:
    m = int(op.meta.get("m", 0))
    micro_batch = int(op.meta.get("micro_batch", 1))
    n = int(op.meta.get("n", 0))
    k = int(op.meta.get("k_local", op.meta.get("k", 0)))
    top_k = int(op.meta.get("top_k", model.top_k))
    ep = int(op.meta.get("ep", 1))
    act_bytes = float(op.meta.get("act_bytes", model.effective_moe_act_dtype().bytes))
    out_bytes = float(op.meta.get("out_bytes", model.act_dtype.bytes))
    weight_bytes = float(op.meta.get("weight_stored_bytes", model.routed_expert_weight_dtype.stored_bytes))

    tokens = micro_batch * m
    l1 = 4.0 * tokens * top_k * n * k
    l2 = 2.0 * tokens * top_k * n * k
    swiglu = 5.0 * tokens * top_k * k
    flops = l1 + swiglu + l2

    experts_per_rank = max(1, int(op.meta.get("experts_per_rank", max(1, model.num_experts // max(ep, 1)))))
    weight_elems = experts_per_rank * 3 * n * k
    act_in = tokens * n * act_bytes
    act_out = tokens * n * out_bytes
    fwd_bytes = act_in + act_out + weight_elems * weight_bytes

    effective = int(tokens * n * top_k * act_bytes // max(ep, 1))
    remote_fraction = (ep - 1) / ep if ep > 1 else 0.0
    dispatch_rank = int(tokens * n * top_k * act_bytes * remote_fraction)
    combine_rank = int(tokens * n * top_k * out_bytes * remote_fraction)

    return MegaMoECostTerms(
        fwd_flops=flops,
        fwd_bytes=fwd_bytes,
        dispatch_bytes_effective=effective,
        combine_bytes_effective=int(tokens * n * top_k * out_bytes // max(ep, 1)),
        dispatch_bytes_per_rank=dispatch_rank,
        combine_bytes_per_rank=combine_rank,
    )
```

- [ ] **Step 4: Add flops branch**

In `python/zrt/training/models/flops.py`, import:

```python
from zrt.training.models.mega_moe import mega_moe_cost_terms
```

Add to `op_cost()`:

```python
    if op.kind == "mega_moe":
        return _mega_moe_cost(op, model)
```

Add this function near `_matmul_cost()`:

```python
def _mega_moe_cost(op: Op, model: ModelSpec) -> OpCost:
    terms = mega_moe_cost_terms(op, model)
    return OpCost(
        fwd_bytes=terms.fwd_bytes,
        dx_bytes=terms.fwd_bytes,
        dw_bytes=terms.fwd_bytes,
        fwd_cube_flops=terms.fwd_flops,
        dx_cube_flops=terms.fwd_flops,
        dw_cube_flops=terms.fwd_flops,
    )
```

- [ ] **Step 5: Run cost tests**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/zrt/training/models/mega_moe.py python/zrt/training/models/flops.py tests/training/test_mega_moe_cost.py
git commit -m "feat: model mega moe op cost"
```

---

### Task 6: Stage Timing With Fused Wave Pipeline

**Files:**
- Modify: `python/zrt/training/models/mega_moe.py`
- Modify: `python/zrt/training/compose/stage.py`
- Test: `tests/training/test_mega_moe_cost.py`

- [ ] **Step 1: Add failing stage timing tests**

Append to `tests/training/test_mega_moe_cost.py`:

```python
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.compose.stage import stage_time
from zrt.training.spec.system import GPU, SystemSpec


def _system(ep_overlap_waves: int = 4) -> SystemSpec:
    return SystemSpec(
        gpu=GPU(
            name="test",
            flops_bf16=1000,
            flops_fp8=2000,
            flops_fp4=4000,
            hbm_gb=80,
            hbm_bw_gbps=3000,
            ep_overlap_waves=ep_overlap_waves,
            compute_efficiency=1.0,
            mem_bw_efficiency=1.0,
        ),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0, topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=2.0, topology="fat_tree"),
        ),
        nodes=1,
        gpus_per_node=8,
    )


def test_stage_time_counts_mega_moe_ep_exposed_and_hidden_without_collectives():
    model = _moe_model(seq_len=512, hidden=1024, moe_ffn=2048, num_experts=8, top_k=2)
    strategy = Strategy(ep=4, mega_moe=True, mega_moe_waves=4)
    graph = build_graph(model, strategy)

    st = stage_time(graph.ops_for_layer(0), graph.collectives, model, _system(), strategy)

    assert not graph.collectives
    assert st.fwd > 0
    assert st.ep_exposed > 0
    assert st.ep_hidden > 0
    assert st.comm_fwd > 0


def test_mega_moe_waves_reduce_stage_time_when_compute_covers_comm():
    model = _moe_model(seq_len=512, hidden=1024, moe_ffn=2048, num_experts=8, top_k=2)
    system = _system()
    one_wave = Strategy(ep=4, mega_moe=True, mega_moe_waves=1)
    four_waves = Strategy(ep=4, mega_moe=True, mega_moe_waves=4)

    graph_one = build_graph(model, one_wave)
    graph_four = build_graph(model, four_waves)
    st_one = stage_time(graph_one.ops_for_layer(0), graph_one.collectives, model, system, one_wave)
    st_four = stage_time(graph_four.ops_for_layer(0), graph_four.collectives, model, system, four_waves)

    assert st_four.fwd < st_one.fwd
```

- [ ] **Step 2: Run stage tests to verify they fail**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py::test_stage_time_counts_mega_moe_ep_exposed_and_hidden_without_collectives tests/training/test_mega_moe_cost.py::test_mega_moe_waves_reduce_stage_time_when_compute_covers_comm -v
```

Expected: FAIL because `stage_time()` treats `mega_moe` as a normal op and does not populate EP exposed/hidden communication.

- [ ] **Step 3: Add stage timing helper**

In `python/zrt/training/models/mega_moe.py`, add:

```python
@dataclass(frozen=True)
class MegaMoEPhaseTime:
    total_s: float
    compute_effective_s: float
    exposed_comm_s: float
    hidden_comm_s: float
    raw_comm_s: float


@dataclass(frozen=True)
class MegaMoEStageTime:
    fwd: MegaMoEPhaseTime
    bwd: MegaMoEPhaseTime


def mega_moe_stage_time(op, model: ModelSpec, system, strategy, compute_time_fn) -> MegaMoEStageTime:
    from zrt.training.ir.training_graph import Collective
    from zrt.training.models.comm import collective_time, tier_for_group
    from zrt.training.models.flops import op_cost

    terms = mega_moe_cost_terms(op, model)
    ep = max(1, int(op.meta.get("ep", strategy.ep)))
    experts_per_rank = max(1, int(op.meta.get("experts_per_rank", model.num_experts // ep)))
    waves = resolve_mega_moe_waves(
        requested=int(op.meta.get("requested_waves", strategy.mega_moe_waves)),
        hardware_waves=getattr(system.gpu, "ep_overlap_waves", 0),
        experts_per_rank=experts_per_rank,
    )
    op.meta["waves"] = waves
    op.meta["dispatch_bytes_effective"] = terms.dispatch_bytes_effective
    op.meta["combine_bytes_effective"] = terms.combine_bytes_effective
    op.meta["dispatch_bytes_per_rank"] = terms.dispatch_bytes_per_rank
    op.meta["combine_bytes_per_rank"] = terms.combine_bytes_per_rank

    tier = tier_for_group("EP", ep, system)
    dispatch_t = collective_time(
        Collective(name=f"{op.name}.dispatch", kind="A2A", group="EP", bytes_=terms.dispatch_bytes_effective),
        ep,
        tier,
    )
    combine_t = collective_time(
        Collective(name=f"{op.name}.combine", kind="A2A", group="EP", bytes_=terms.combine_bytes_effective),
        ep,
        tier,
    )

    cost = op_cost(op, model, system)
    fwd_compute = compute_time_fn(cost, "fwd")
    bwd_compute = compute_time_fn(cost, "dx") + compute_time_fn(cost, "dw")

    def phase(total_compute: float) -> MegaMoEPhaseTime:
        result = simulate_wave_pipeline(
            waves=waves,
            dispatch_s=dispatch_t / waves,
            compute_s=total_compute / waves,
            combine_s=combine_t / waves,
        )
        return MegaMoEPhaseTime(
            total_s=result.total_s,
            compute_effective_s=max(0.0, result.total_s - result.exposed_comm_s),
            exposed_comm_s=result.exposed_comm_s,
            hidden_comm_s=result.hidden_comm_s,
            raw_comm_s=result.comm_total_s,
        )

    return MegaMoEStageTime(fwd=phase(fwd_compute), bwd=phase(bwd_compute))
```

- [ ] **Step 4: Special-case mega_moe in stage_time**

In `python/zrt/training/compose/stage.py`, import:

```python
from zrt.training.models.mega_moe import mega_moe_stage_time
```

At the start of `stage_time()`, initialize:

```python
    t_mega_ep_exposed_fwd = 0.0
    t_mega_ep_exposed_bwd = 0.0
    t_mega_ep_hidden = 0.0
```

In the main op loop, replace the body with this shape:

```python
    for op in stage_ops:
        if op.kind == "mega_moe":
            op_dtype = _resolve_compute_dtype(op, model)

            def _phase(cost: OpCost, phase: str) -> float:
                return _cost_phase_time(
                    cost,
                    phase,
                    system,
                    gpu_name,
                    gpu.overlap_ratio.get(op.kind, 0.0),
                    op_dtype,
                )

            mt = mega_moe_stage_time(op, model, system, strategy, _phase)
            t_fwd += mt.fwd.compute_effective_s
            t_bwd_dx += mt.bwd.compute_effective_s
            t_mega_ep_exposed_fwd += mt.fwd.exposed_comm_s
            t_mega_ep_exposed_bwd += mt.bwd.exposed_comm_s
            t_mega_ep_hidden += mt.fwd.hidden_comm_s + mt.bwd.hidden_comm_s
            continue

        cost = op_cost(op, model, system)
        overlap = gpu.overlap_ratio.get(op.kind, 0.0)
        op_dtype = _resolve_compute_dtype(op, model)
        fwd_t = _cost_phase_time(cost, "fwd", system, gpu_name, overlap, op_dtype)
        dx_t  = _cost_phase_time(cost, "dx",  system, gpu_name, overlap, op_dtype)
        dw_t  = _cost_phase_time(cost, "dw",  system, gpu_name, overlap, op_dtype)
        t_fwd    += fwd_t
        t_bwd_dx += dx_t
        t_bwd_dw += dw_t
```

After TP overlap, when building combined comm totals, include fused exposed EP:

```python
    t_comm_fwd = t_other_comm_fwd + t_tp_exposed_fwd + t_cp_comm_fwd + t_ep_raw_comm_fwd + t_mega_ep_exposed_fwd
    t_comm_bwd = t_other_comm_bwd + t_tp_exposed_bwd + t_cp_comm_bwd + t_ep_raw_comm_bwd + t_mega_ep_exposed_bwd
```

At the end, include fused hidden/exposed EP in the returned fields:

```python
        ep_hidden=t_ep_hidden + t_mega_ep_hidden,
        ep_exposed=t_ep_exposed_fwd + t_ep_exposed_bwd + t_mega_ep_exposed_fwd + t_mega_ep_exposed_bwd,
```

Do not add `t_mega_ep_exposed_*` into `t_ep_raw_comm_*`; that would let `ep_overlap` hide the same communication twice.

- [ ] **Step 5: Update EP parallel fraction and GEMM helpers**

In `_ep_parallel_fraction()`, treat `mega_moe` as EP-parallel:

```python
        if (op.kind == "matmul" and "routed_expert" in op.name) or op.kind == "mega_moe":
            t_ep += t
```

In `_ep_gemm_time()`, include `mega_moe` compute for any legacy callers:

```python
        if (op.kind == "matmul" and "routed_expert" in op.name) or op.kind == "mega_moe":
```

- [ ] **Step 6: Run stage tests**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py tests/training/test_ep_overlap.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/zrt/training/models/mega_moe.py python/zrt/training/compose/stage.py tests/training/test_mega_moe_cost.py
git commit -m "feat: time mega moe wave pipeline"
```

---

### Task 7: Report And Formula Visibility

**Files:**
- Modify: `python/zrt/training/io/html_exporter.py`
- Modify: `python/zrt/training/io/excel_exporter.py`
- Test: `tests/training/test_mega_moe_cost.py`

- [ ] **Step 1: Add failing formula/report tests**

Append to `tests/training/test_mega_moe_cost.py`:

```python
from zrt.training.io.html_exporter import _op_detail


def test_mega_moe_op_detail_mentions_quant_and_waves():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )
    graph = build_graph(model, Strategy(ep=4, mega_moe=True, mega_moe_waves=2))
    op = next(op for op in graph.ops if op.kind == "mega_moe")
    cost = op_cost(op, model)

    detail = _op_detail(op, cost)

    assert "mega_moe" in detail["fwd_formula"]
    assert "w4a8" in detail["fwd_formula"]
    assert "waves" in detail["fwd_bytes_formula"]
```

- [ ] **Step 2: Run formula test to verify it fails**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py::test_mega_moe_op_detail_mentions_quant_and_waves -v
```

Expected: FAIL because `_op_formula()` falls back to generic text for `mega_moe`.

- [ ] **Step 3: Add mega_moe formula branch**

In `python/zrt/training/io/html_exporter.py`, inside `_op_formula()` before the matmul branch, add:

```python
    if op.kind == "mega_moe":
        mm = m.get("m", 0)
        nn = m.get("n", 0)
        kk = m.get("k_local", m.get("k", 0))
        topk = m.get("top_k", 0)
        waves = m.get("waves", m.get("requested_waves", 0))
        quant = m.get("quant_variant", "standard")
        fwd_str = (
            f"mega_moe[{quant}]: L1+SwiGLU+L2 = "
            f"4*m*topk*n*k + 5*m*topk*k + 2*m*topk*n*k = "
            f"m={mm}, topk={topk}, n={nn}, k={kk} => {_fmt_e(ff)}"
        )
        bwd_str = f"dx+dw = 2*fwd = {_fmt_e(df + wf)}"
        bytes_str = (
            f"fused HBM + dispatch/combine meta, waves={waves}, "
            f"dispatch={m.get('dispatch_bytes_per_rank', 0)}B, "
            f"combine={m.get('combine_bytes_per_rank', 0)}B, "
            f"weight_stored_bpe={m.get('weight_stored_bytes', '-')}"
        )
        return fwd_str, bwd_str, bytes_str, bytes_str
```

- [ ] **Step 4: Show strategy fields in Excel**

In `python/zrt/training/io/excel_exporter.py`, in `strat_rows`, add after `EP Overlap`:

```python
        ["Mega MoE", str(strategy.mega_moe), ""],
        ["Mega MoE Waves", strategy.mega_moe_waves, ""],
```

- [ ] **Step 5: Run formula test**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_cost.py::test_mega_moe_op_detail_mentions_quant_and_waves -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/zrt/training/io/html_exporter.py python/zrt/training/io/excel_exporter.py tests/training/test_mega_moe_cost.py
git commit -m "feat: report mega moe variant details"
```

---

### Task 8: Integration And Backward Compatibility

**Files:**
- Modify only if tests reveal a defect in files touched by earlier tasks.
- Test: existing EP, mixed quant, and config tests.

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_config.py tests/training/test_mega_moe_ir.py tests/training/test_mega_moe_cost.py tests/training/test_ir_dense.py tests/training/test_ep_overlap.py tests/training/test_mixed_quant_preset.py tests/training/test_mixed_quant_dtype.py tests/training/test_mixed_quant_peak_selection.py -v
```

Expected: PASS.

- [ ] **Step 2: Run broader training smoke suite**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_opgraph_from_spec.py tests/training/test_flops.py tests/training/test_training_search_util.py tests/training/test_tp_overlap.py -v
```

Expected: PASS.

- [ ] **Step 3: Manually inspect default-off graph behavior**

Run:

```bash
PYTHONPATH=python python - <<'PY'
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy

model = ModelSpec(
    hidden=1024, ffn=4096, num_heads=16, num_kv_heads=16, head_dim=64,
    vocab=32000, seq_len=128, layers=[LayerKind.MOE],
    num_experts=8, moe_ffn=2048, top_k=2,
)
graph = build_graph(model, Strategy(ep=4))
print([op.name for op in graph.ops if "routed_expert" in op.name or op.kind == "mega_moe"])
print([(c.name, c.kind, c.group) for c in graph.collectives if c.group == "EP"])
PY
```

Expected output contains `L0.routed_expert_ffn`, no `mega_moe`, and two EP A2A collectives.

- [ ] **Step 4: Manually inspect switch-on graph behavior**

Run:

```bash
PYTHONPATH=python python - <<'PY'
from zrt.training.ir.builders import build_graph
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy

model = ModelSpec(
    hidden=1024, ffn=4096, num_heads=16, num_kv_heads=16, head_dim=64,
    vocab=32000, seq_len=128, layers=[LayerKind.MOE],
    num_experts=8, moe_ffn=2048, top_k=2,
    routed_expert_compute_dtype=Dtype.FP8_E4M3,
    routed_expert_weight_dtype=Dtype.FP4,
    moe_act_dtype=Dtype.FP8_E4M3,
)
graph = build_graph(model, Strategy(ep=4, mega_moe=True, mega_moe_waves=2))
mega = next(op for op in graph.ops if op.kind == "mega_moe")
print(mega.name, mega.meta["quant_variant"], mega.meta["requested_waves"])
print([(c.name, c.kind, c.group) for c in graph.collectives if c.group == "EP"])
PY
```

Expected output contains `L0.mega_moe w4a8 2` and an empty EP collective list.

- [ ] **Step 5: Commit any regression fixes**

If Step 1 or Step 2 required fixes, commit them:

```bash
git add python/zrt tests/training
git commit -m "fix: stabilize mega moe integration"
```

If no fixes were needed, do not create an empty commit.

---

### Task 9: Final Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run all new tests**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_mega_moe_config.py tests/training/test_mega_moe_ir.py tests/training/test_mega_moe_cost.py -v
```

Expected: PASS.

- [ ] **Step 2: Run core regression tests**

Run:

```bash
PYTHONPATH=python pytest tests/training/test_ir_dense.py tests/training/test_ep_overlap.py tests/training/test_mixed_quant_preset.py tests/training/test_mixed_quant_memory.py tests/training/test_flops.py -v
```

Expected: PASS.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional files are modified or the worktree is clean except for pre-existing unrelated untracked files.

- [ ] **Step 4: Summarize implementation**

Prepare a short completion note with:

```text
Implemented:
- Strategy-gated spec-path mega_moe op.
- Default-off legacy EP behavior.
- W4A8 quant variant inference from existing dtype fields.
- Wave-pipeline timing with EP exposed/hidden reporting.

Verified:
- <paste exact pytest commands and PASS summaries>
```
