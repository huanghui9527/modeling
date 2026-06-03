from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from zrt.training.ir.training_graph import Collective, Op
from zrt.training.models.comm import collective_time, tier_for_group
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec


FP8_DTYPES = {Dtype.FP8_E4M3, Dtype.FP8_E5M2}


@dataclass(frozen=True)
class WavePipelineResult:
    total_s: float
    exposed_comm_s: float
    hidden_comm_s: float
    comm_total_s: float


@dataclass(frozen=True)
class MegaMoECostTerms:
    tokens: int
    compute_tokens: int
    n: int
    activation_n: int
    k_eff: int
    top_k: int
    local_experts: int
    fwd_multiplier: float
    quant_variant: str
    activation_input_bytes: float
    activation_output_bytes: float
    moe_activation_input_bytes: float
    weight_bytes: float
    fwd_bytes: float
    fwd_flops: float


@dataclass(frozen=True)
class MegaMoEPhaseTime:
    compute_s: float
    dispatch_s: float
    combine_s: float
    exposed_comm_s: float
    hidden_comm_s: float
    total_s: float


@dataclass(frozen=True)
class MegaMoEStageTime:
    fwd: MegaMoEPhaseTime
    bwd: MegaMoEPhaseTime

    @property
    def comm_fwd_s(self) -> float:
        return self.fwd.exposed_comm_s

    @property
    def comm_bwd_s(self) -> float:
        return self.bwd.exposed_comm_s

    @property
    def ep_exposed_s(self) -> float:
        return self.fwd.exposed_comm_s + self.bwd.exposed_comm_s

    @property
    def ep_hidden_s(self) -> float:
        return self.fwd.hidden_comm_s + self.bwd.hidden_comm_s


def infer_quant_variant(model: ModelSpec) -> str:
    if (
        model.routed_expert_weight_dtype is Dtype.FP4
        and model.effective_moe_act_dtype() in FP8_DTYPES
    ):
        return "w4a8"
    return "standard"


def mega_moe_cost_terms(op: Op) -> MegaMoECostTerms:
    return mega_moe_cost_terms_from_meta(op.meta)


def mega_moe_cost_terms_from_meta(meta: Mapping[str, Any]) -> MegaMoECostTerms:
    m = int(meta["m"])
    micro_batch = int(meta.get("micro_batch", 1))
    tokens = micro_batch * m
    n_logical = int(meta["n"])
    activation_n = int(meta.get("n_local", n_logical))
    k_eff = int(meta.get("k_local", meta["k"]))
    # A fused routed FFN has both column-parallel and row-parallel pieces.
    # Once k_local is present, using n_local for compute as well would apply
    # TP twice. Keep n_local for boundary activations/A2A, but use the logical
    # hidden dimension for expert GEMM and stored-weight traffic.
    n = n_logical if "k_local" in meta else activation_n
    top_k = int(meta["top_k"])
    compute_tokens = int(meta.get("compute_tokens_per_rank", tokens * top_k))
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
    fwd_flops = float(2 * compute_tokens * k_eff * n * fwd_multiplier)

    return MegaMoECostTerms(
        tokens=tokens,
        compute_tokens=compute_tokens,
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


def mega_moe_stage_time(
    op: Op,
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
    gpu_name_or_gpu: str | GPU,
    *,
    fwd_compute_s: float,
    dx_compute_s: float,
    dw_compute_s: float,
) -> MegaMoEStageTime:
    """Internal dispatch/compute/combine timing for one fused mega_moe op."""
    terms = mega_moe_cost_terms(op)
    ep = int(op.meta.get("ep", strategy.ep))
    experts_per_rank = int(
        op.meta.get("experts_per_rank", max(1, terms.local_experts))
    )
    if ep <= 1:
        empty_fwd = MegaMoEPhaseTime(fwd_compute_s, 0.0, 0.0, 0.0, 0.0, fwd_compute_s)
        bwd_compute_s = dx_compute_s + dw_compute_s
        empty_bwd = MegaMoEPhaseTime(bwd_compute_s, 0.0, 0.0, 0.0, 0.0, bwd_compute_s)
        return MegaMoEStageTime(fwd=empty_fwd, bwd=empty_bwd)

    gpu = gpu_name_or_gpu if isinstance(gpu_name_or_gpu, GPU) else system.gpu
    dispatch_s = _mega_moe_a2a_time(
        name=f"{op.name}.dispatch",
        bytes_=_mega_moe_dispatch_bytes(terms, ep),
        ep=ep,
        system=system,
    )
    combine_s = _mega_moe_a2a_time(
        name=f"{op.name}.combine",
        bytes_=_mega_moe_combine_bytes(terms, ep),
        ep=ep,
        system=system,
    )

    requested = int(strategy.mega_moe_waves or op.meta.get("requested_waves", 0))
    bwd_compute_s = dx_compute_s + dw_compute_s
    if requested > 0:
        waves = resolve_mega_moe_waves(
            requested=requested,
            hardware_waves=gpu.ep_overlap_waves,
            experts_per_rank=experts_per_rank,
        )
    else:
        waves = select_mega_moe_waves_by_pipeline(
            experts_per_rank=experts_per_rank,
            fwd_compute_s=fwd_compute_s,
            bwd_compute_s=bwd_compute_s,
            dispatch_s=dispatch_s,
            combine_s=combine_s,
        )

    fwd = _phase_time(
        waves=waves,
        compute_s=fwd_compute_s,
        dispatch_s=dispatch_s,
        combine_s=combine_s,
    )
    bwd = _phase_time(
        waves=waves,
        compute_s=bwd_compute_s,
        dispatch_s=dispatch_s,
        combine_s=combine_s,
    )
    return MegaMoEStageTime(fwd=fwd, bwd=bwd)


def _mega_moe_dispatch_bytes(terms: MegaMoECostTerms, ep: int) -> float:
    return terms.moe_activation_input_bytes * terms.top_k


def _mega_moe_combine_bytes(terms: MegaMoECostTerms, ep: int) -> float:
    return terms.activation_output_bytes * terms.top_k


def _mega_moe_a2a_time(
    *, name: str, bytes_: float, ep: int, system: SystemSpec,
) -> float:
    collective = Collective(name=name, kind="A2A", group="EP", bytes_=int(bytes_))
    tier = tier_for_group("EP", ep, system)
    return collective_time(collective, ep, tier)


def _phase_time(
    *, waves: int, compute_s: float, dispatch_s: float, combine_s: float,
) -> MegaMoEPhaseTime:
    if waves <= 1:
        pipeline = simulate_wave_pipeline(
            waves=1,
            dispatch_s=dispatch_s,
            compute_s=compute_s,
            combine_s=combine_s,
        )
    else:
        pipeline = simulate_wave_pipeline(
            waves=waves,
            dispatch_s=dispatch_s / waves,
            compute_s=compute_s / waves,
            combine_s=combine_s / waves,
        )
    return MegaMoEPhaseTime(
        compute_s=compute_s,
        dispatch_s=dispatch_s,
        combine_s=combine_s,
        exposed_comm_s=pipeline.exposed_comm_s,
        hidden_comm_s=pipeline.hidden_comm_s,
        total_s=pipeline.total_s,
    )


def _mega_moe_fwd_multiplier(raw_fwd_multiplier: float, top_k: int) -> float:
    legacy_multiplier = 3 * top_k
    if top_k > 1 and raw_fwd_multiplier == legacy_multiplier:
        return raw_fwd_multiplier / top_k
    return raw_fwd_multiplier


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


def select_mega_moe_waves_by_pipeline(
    *,
    experts_per_rank: int,
    fwd_compute_s: float,
    bwd_compute_s: float,
    dispatch_s: float,
    combine_s: float,
) -> int:
    """Pick the wave count with the best modeled dispatch/compute/combine pipeline.

    This auto mode deliberately uses only quantities already known to the
    performance model. It does not require NVIDIA SM counts or Ascend AI/Cube
    core counts, which are not always available for the target hardware.
    """
    if experts_per_rank <= 1:
        return 1

    best_waves = 1
    best_key: tuple[float, float, int] | None = None
    for waves in _wave_divisors(experts_per_rank):
        if waves <= 1:
            fwd = simulate_wave_pipeline(
                waves=1,
                dispatch_s=dispatch_s,
                compute_s=fwd_compute_s,
                combine_s=combine_s,
            )
            bwd = simulate_wave_pipeline(
                waves=1,
                dispatch_s=dispatch_s,
                compute_s=bwd_compute_s,
                combine_s=combine_s,
            )
        else:
            fwd = simulate_wave_pipeline(
                waves=waves,
                dispatch_s=dispatch_s / waves,
                compute_s=fwd_compute_s / waves,
                combine_s=combine_s / waves,
            )
            bwd = simulate_wave_pipeline(
                waves=waves,
                dispatch_s=dispatch_s / waves,
                compute_s=bwd_compute_s / waves,
                combine_s=combine_s / waves,
            )
        key = (
            fwd.total_s + bwd.total_s,
            fwd.exposed_comm_s + bwd.exposed_comm_s,
            waves,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_waves = waves
    return best_waves


def _wave_divisors(experts_per_rank: int) -> list[int]:
    return [
        d
        for d in range(1, experts_per_rank + 1)
        if experts_per_rank % d == 0
    ]


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
            done_at
            for wave_id, done_at in enumerate(compute_done)
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
