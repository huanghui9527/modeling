# Mega MoE H800 Calibration Design

## Goal

Calibrate the spec-path `mega_moe` timing model against DeepGEMM Mega MoE
benchmark data, prioritizing time accuracy on the target hardware profile:
NVIDIA H800 SXM-class hardware.

The calibration must remain explainable. If the modeled time differs too much
from the benchmark, the tooling should report the gap and likely bottleneck
instead of silently forcing a black-box multiplier.

## References

- DeepGEMM PR #316: https://github.com/deepseek-ai/DeepGEMM/pull/316
- DeepGEMM Mega MoE test/statistics code:
  https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/tests/test_mega_moe.py

The DeepGEMM test code is the source of truth for benchmark accounting:
reported TFLOPS, HBM bandwidth, interconnect bandwidth, and time are derived
from per-rank token counts, EP=8 routing, FP4 expert weights, FP8 intermediate
activations, and BF16 outputs.

## Hardware Baseline

Use the repo's H800 profile as the calibration hardware baseline:
`python/zrt/hardware/configs/nvidia_h800.yaml`.

Assumptions:

- Compute is the same class as H100 SXM:
  - BF16/FP16 peak: 989 TFLOP/s
  - FP8 peak: 3958 TFLOP/s
  - FP4 peak: 7916 TFLOP/s, defined as `2 * FP8` for this calibration
- HBM:
  - Capacity: 80 GB
  - Bandwidth: 3350 GB/s
- Intra-node interconnect:
  - H800 export-restricted NVLink effective peak baseline: 400 GB/s
  - Use existing interconnect efficiency machinery for effective bandwidth
- EP wave overlap:
  - Start with existing `ep_overlap_waves: 4`

The first implementation should update `nvidia_h800.yaml` so `fp4_tops` is
explicitly `7916` rather than falling back to FP8. This is a calibration
assumption, not a claim that H800 has native FP4 Tensor Cores.

## Calibration Data Shape

Add a small structured dataset for DeepGEMM PR #316 Mega MoE rows. Each row
should preserve raw benchmark fields instead of baking them into code:

- `source`: PR URL or label
- `model_variant`: `deepseek_v4_flash` or `deepseek_v4_pro`
- `ep`: expected to be 8 for the PR data
- `tokens_per_rank`
- `hidden`
- `moe_ffn`
- `num_experts`
- `top_k`
- `time_us`
- `reported_tflops`
- `reported_hbm_gbps`
- `reported_interconnect_gbps`

The dataset can live as Python data in a test fixture or as a small YAML/JSON
file. Prefer a data file if more benchmark rows are copied from the PR, because
it makes future benchmark refreshes easier to review.

## Modeling Strategy

Use a two-step calibration flow:

1. **Comparison first**
   - Build the matching spec-path `mega_moe` graph for each benchmark row.
   - Run sharding, `op_cost`, and `stage_time` using H800 hardware.
   - Compare modeled `mega_moe` forward time against benchmark `time_us`.
   - Emit per-row diagnostics for modeled compute time, fused EP exposed comm,
     fused EP hidden comm, and total modeled forward time.

2. **Explainable correction**
   - If error is moderate, add calibrated effective factors by component:
     compute, HBM, and interconnect.
   - Do not use one global magic multiplier as the default implementation.
   - Keep calibrated factors optional and scoped to `mega_moe` on H800.

The initial implementation may stop after comparison if the measured mismatch
is large. The design explicitly allows a discussion checkpoint before changing
the model formula.

## Error Policy

Use time as the primary target:

- `abs(model_time - benchmark_time) / benchmark_time <= 25%`
  - Acceptable for first calibration pass.
  - Add optional calibrated component factors only if they improve consistency
    across token sizes and variants.
- `25% < error <= 40%`
  - Mark as degraded.
  - Produce diagnostics and require review before applying a correction.
- `error > 40%`
  - Stop automatic calibration.
  - Report likely cause:
    - compute-bound mismatch
    - HBM bandwidth mismatch
    - NVLink/interconnect mismatch
    - wave-pipeline mismatch
    - benchmark hardware mismatch
  - Discuss the fix before implementation proceeds.

This follows the user's requirement: if the gap is large, discuss the solution
instead of silently forcing the estimate to match.

## Implementation Boundaries

In scope:

- H800 hardware profile adjustment for FP4 peak assumption.
- DeepGEMM PR #316 benchmark fixture.
- Comparison helper/report for benchmark rows.
- Tests that assert the comparison path runs and produces meaningful error
  metrics.
- Optional component calibration only if the first comparison shows a stable,
  explainable correction.

Out of scope for the first calibration pass:

- Recalibrating non-`mega_moe` MoE paths.
- Changing the semantics of visible graph tensor dtypes.
- Replacing existing generic HBM/interconnect efficiency curves globally.
- Fitting a black-box model across all GPUs.

## Data Flow

For each benchmark row:

1. Convert row metadata into a minimal `ModelSpec`.
2. Use `Strategy(ep=8, mega_moe=True, mega_moe_waves=...)`.
3. Use H800 `SystemSpec`.
4. Build and shard the graph through the existing spec path.
5. Locate `Op(kind="mega_moe")`.
6. Compute:
   - `op_cost(op, model, system)`
   - `mega_moe_cost_terms(op)`
   - `mega_moe_stage_time(...)` or `stage_time(...)`
7. Compare modeled forward time to benchmark `time_us`.
8. Produce a row-level diagnostic object.

## Reporting

The comparison report should include:

- benchmark row identity
- benchmark time in microseconds
- modeled forward time in microseconds
- absolute and percent error
- modeled compute time
- modeled fused EP exposed comm
- modeled fused EP hidden comm
- effective compute TFLOPS implied by the benchmark
- effective HBM and interconnect bandwidth implied by the benchmark

This report should be readable in test output or exported through a small helper.
It does not need full HTML/Excel integration in the first pass.

## Testing

Add focused tests for:

- H800 FP4 peak is `2 * FP8` in the calibration hardware path.
- DeepGEMM benchmark rows can be converted into spec-path inputs.
- Comparison output includes benchmark time, modeled time, error ratio, and
  component diagnostics.
- Large mismatch handling returns a discuss-required/degraded status rather
  than applying a silent multiplier.

Existing mega_moe tests should remain green:

- `tests/training/test_mega_moe_cost.py`
- `tests/training/test_mega_moe_stage.py`
- `tests/training/test_mega_moe_integration.py`

## Open Decisions

The first implementation should answer these with data before changing the core
model:

- Is the H800 mismatch dominated by compute, HBM, or NVLink?
- Does one component correction fit both V4-Flash and V4-Pro?
- Does the same correction fit all `tokens_per_rank` rows?
- Should calibrated factors live in the H800 hardware YAML, a model calibration
  file, or a `mega_moe`-specific calibration fixture?

If these answers are not stable, stop and discuss before modifying production
timing formulas.
