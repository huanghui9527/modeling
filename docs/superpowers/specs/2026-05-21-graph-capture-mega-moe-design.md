# Graph Capture MegaMoE Fusion Design

## Goal

Implement graph-capture MegaMoE fusion with the same user-facing configuration
as the spec path: `mega_moe` and `mega_moe_waves`.

The graph-capture path should model the routed MoE expert region as one fused
`mega_moe` operator when enabled. The fused operator represents internal EP
dispatch, grouped expert compute, activation, and EP combine. It does not
include the router or shared expert path.

## Existing Context

The spec path already supports MegaMoE through `Strategy.mega_moe` and
`Strategy.mega_moe_waves`. When enabled, the spec IR builder emits one
`kind="mega_moe"` op for each routed expert region and avoids inserting
external EP A2A collectives around that op. Cost and stage timing are modeled
in `python/zrt/training/models/mega_moe.py`.

The graph-capture path currently uses the transform pipeline:

```text
ExpertParallelPass
ExpertGroupedMMPass
ContextParallelPass
RecomputePass
CommInserterPass
FusionPass
Analysis passes
```

`ExpertGroupedMMPass` already identifies routed expert matmuls and lowers them
to:

```text
GroupedMatMul(gate_up) -> aten.silu -> GroupedMatMul(down)
```

`CommInserterPass` then wraps that block with explicit EP dispatch and combine
A2A nodes.

## Chosen Approach

Extend `ExpertGroupedMMPass` with a MegaMoE branch instead of adding a separate
pass.

This keeps the implementation close to the existing GroupedMM lowering because
MegaMoE is built on top of the same routed expert grouping and shape
normalization. The pass already has the layer grouping, phase handling, expert
classification, local expert count, token partitioning, and external edge
rewiring that MegaMoE needs.

When `ctx.training.mega_moe` is false, existing GroupedMM behavior remains
unchanged. When it is true, the pass directly emits a single `mega_moe` node for
the routed expert region instead of emitting `GroupedMatMul -> silu ->
GroupedMatMul`.

## Configuration

Add graph-capture fields to `TrainingConfig`:

```python
mega_moe: bool = False
mega_moe_waves: int = 0
```

Add matching parameters to `estimate_training_from_graphs()` and pass them into
`TrainingConfig`.

The user-facing config names stay identical to the spec path. Internally, the
spec path continues to use `Strategy.mega_moe`, while graph capture uses
`TransformContext.training.mega_moe`.

## Fused Node Contract

The graph-capture `mega_moe` node represents only the routed expert path. It
inherits the original routed block's external inputs and outputs:

```text
external input -> mega_moe -> external output
```

The node must not retain `ep_needs_a2a`, because dispatch and combine are
internal to the MegaMoE operator. This prevents `CommInserterPass` from adding
external EP A2A nodes around it.

The fused node should carry:

```text
op_type = "mega_moe"
category = "compute"
annotations["fused_by"] = "mega_moe_graph_capture"
annotations["fused_dispatch_compute_combine"] = True
```

The node should also carry a spec-compatible metadata dictionary in
`annotations["mega_moe_meta"]` and mirror the important scalar values in
`attrs` where analysis/export code already looks for node-level fields.

Required metadata:

```text
m
n
k
micro_batch
num_experts
top_k
requested_waves
act_bytes
out_bytes
moe_act_bytes
weight_bytes
weight_stored_bytes
quant_variant
fwd_multiplier
swiglu_clamp
fused_dispatch_compute_combine
ep
experts_per_rank
n_local
k_local
```

`m` is the local token count after CP when CP is active. `n` is the logical
hidden dimension. `n_local` is the local activation hidden dimension after TP.
`k` is the logical MoE FFN dimension. `k_local` is the local MoE FFN dimension
after TP.

## Forward And Backward Handling

Forward fusion replaces the current forward routed expert block with one
`mega_moe` node.

Backward fusion must also preserve the stitched graph phase contract. The
implementation will emit phase-specific `mega_moe` nodes with
`annotations["phase"]` set to `fwd` or `bwd`, matching the current
`ExpertGroupedMMPass` separation between forward and backward grouped nodes.

Backward nodes must not be marked as external recompute replay. Recompute
annotations should follow the same rule as existing graph-capture recompute:
forward replay nodes may be marked, backward base nodes should not.

## Analysis

Graph-capture analysis must recognize `mega_moe`.

`FlopsPass` should avoid active-expert double scaling for nodes fused by
MegaMoE. It should compute forward, dx, and dw FLOPs using the same MegaMoE
cost semantics as the spec path. The preferred implementation is to refactor
`python/zrt/training/models/mega_moe.py` so the core cost terms can be computed
from a metadata mapping, then adapt both spec `Op` and graph-capture `OpNode` to
that mapping.

`RooflinePass` and latency modeling should produce nonzero compute latency for
`mega_moe`. Internal dispatch/combine exposed and hidden communication should
reuse the spec path wave pipeline semantics. The important behavior is that
external EP A2A nodes disappear when MegaMoE is enabled, while EP communication
still contributes through MegaMoE's internal timing fields.

## Error Handling

If a routed expert layer cannot be safely recognized, MegaMoE fusion should
skip that layer and leave existing GroupedMM behavior intact. Safe skip cases
include missing gate/up/down matmuls, inconsistent expert dimensions, missing
external inputs or outputs, or invalid `num_experts % ep`.

This keeps the graph-capture path conservative: enabling MegaMoE should not
break capture results for unsupported MoE shapes.

## Testing

Add fast synthetic graph tests first:

1. `mega_moe=False` keeps current behavior: GroupedMM nodes are emitted and
   `CommInserterPass` adds external EP A2A nodes.
2. `mega_moe=True` emits `mega_moe` nodes and no external routed EP A2A nodes.
3. The fused node has complete spec-compatible metadata.
4. FLOPs and latency are nonzero and are not scaled twice by `top_k`,
   `moe_active_experts`, or EP local expert count.
5. Forward and backward phases are preserved on stitched graphs.
6. Recompute full/selective behavior remains compatible with the existing
   recompute E2E expectations.

Add one DeepSeek-V4 graph-capture smoke test after the synthetic tests pass. It
should assert stable invariants, not exact node counts:

```text
mega_moe nodes exist
external routed EP A2A nodes are absent
step time and training FLOPs are positive
report fused_ops_summary includes mega_moe
```

Run existing regression suites for:

```text
tests/test_transform.py
tests/training/test_phase1_bugfixes.py
tests/IT/test_ep_e2e.py
tests/IT/test_recompute_e2e.py
tests/transform/fusion/test_dsv4_rules.py
tests/training/test_mega_moe_*.py
```

## Out Of Scope

This design does not re-enable the DSV4 YAML class-level `MoE` fusion rule.
That path conflicts with the graph-capture EP/GroupedMM lowering and should
remain separate.

This design also does not require graph-capture to support every spec-path
quantization variant immediately. The node metadata should be shaped so W4A8
and stored-weight bytes can be modeled when those fields are available.
