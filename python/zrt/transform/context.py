"""TransformContext and configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, TYPE_CHECKING

from zrt.training.spec.dtype import Dtype


def _str_to_bool(v: str) -> bool:
    """Parse a YAML-style boolean string."""
    return v.lower().strip() in ("true", "1", "yes")


# Inference-only quantization formats that don't support training.
_INFERENCE_ONLY = frozenset({
    "int8", "int4", "w8a8", "w4a16", "w4a8",
    "w8a8kv8", "w4a8kv4", "w4a16kv8", "w8a4",
})


if TYPE_CHECKING:
    from python.zrt.hardware.spec import HardwareSpec


@dataclass
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1
    cp: int = 1
    sp: bool = False

    @property
    def total_devices(self) -> int:
        return self.tp * self.pp * self.ep * self.dp * self.cp

    def describe(self) -> str:
        parts = []
        if self.tp > 1: parts.append(f"TP{self.tp}")
        if self.ep > 1: parts.append(f"EP{self.ep}")
        if self.pp > 1: parts.append(f"PP{self.pp}")
        if self.dp > 1: parts.append(f"DP{self.dp}")
        if self.cp > 1: parts.append(f"CP{self.cp}")
        if self.sp:     parts.append("SP")
        return "-".join(parts) or "single"


@dataclass
class StreamConfig:
    """Multi-stream execution configuration."""
    num_compute_streams: int = 1
    num_comm_streams: int = 1

    @property
    def total(self) -> int:
        return self.num_compute_streams + self.num_comm_streams

    def compute_stream_id(self, idx: int = 0) -> int:
        return idx % self.num_compute_streams

    def comm_stream_id(self, idx: int = 0) -> int:
        return self.num_compute_streams + (idx % self.num_comm_streams)


@dataclass
class QuantConfig:
    weight:     str = "bf16"   # "int8", "int4", "w8a8", "w4a16", ...
    activation: str = "bf16"
    kv_cache:   str = "bf16"
    # Per-component overrides (empty = inherit from activation)
    attn_activation:    str = ""   # attention Q/K/V/O projections
    expert_activation:  str = ""   # routed MoE expert GEMMs
    shared_activation:  str = ""   # shared expert GEMMs

    @property
    def weight_bytes(self) -> float:
        _map = {"int4": 0.5, "int8": 1.0, "fp8": 1.0, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}
        return _map.get(self.weight.lower(), 2.0)

    def activation_for_component(self, component: str) -> str:
        """Return per-component activation dtype, falling back to global activation."""
        # Check per-component overrides first
        if component.startswith("attn.") and self.attn_activation:
            return self.attn_activation
        if component.startswith("moe.experts.") and self.expert_activation:
            return self.expert_activation
        if component.startswith("moe.shared.") and self.shared_activation:
            return self.shared_activation
        # Fallback to global activation
        return self.activation


@dataclass
class GraphQuantProfile:
    """Per-component dtype profile for graph-based training quantization.

    Mirrors the dtype fields of ``ModelSpec`` so the graph path can reuse
    spec dtype resolution logic without constructing a full ``ModelSpec``.
    All fields use the spec ``Dtype`` enum; defaults match BF16 baseline.
    """

    # Global dtypes
    param_dtype: Dtype = Dtype.BF16
    grad_dtype: Dtype = Dtype.FP32
    master_dtype: Dtype = Dtype.FP32
    act_dtype: Dtype = Dtype.BF16

    # Per-component compute dtype
    attn_compute_dtype: Dtype = Dtype.BF16
    shared_expert_compute_dtype: Dtype = Dtype.BF16
    routed_expert_compute_dtype: Dtype = Dtype.BF16

    # Per-component weight dtype (None → fallback to param_dtype in __post_init__)
    attn_weight_dtype: Dtype | None = None
    shared_expert_weight_dtype: Dtype | None = None
    routed_expert_weight_dtype: Dtype | None = None

    # Per-region activation dtype (None → fallback to act_dtype)
    attn_act_dtype: Dtype | None = None
    moe_act_dtype: Dtype | None = None
    residual_dtype: Dtype | None = None

    # Per-component grad dtype
    attn_grad_dtype: Dtype = Dtype.FP32
    shared_expert_grad_dtype: Dtype = Dtype.FP32
    routed_expert_grad_dtype: Dtype = Dtype.FP32

    # Hardware-realization policy flags (mirrors spec QuantPolicy)
    ln_softmax_promote_fp32: bool = True
    assume_all_casts_fused: bool = True

    # KV cache dtype (inference path annotation; not used in training memory)
    kv_cache_dtype: Dtype = Dtype.BF16

    def __post_init__(self) -> None:
        # Per-component weight dtype defaults to param_dtype when unset.
        # Mirrors ModelSpec.__post_init__ so manual GraphQuantProfile()
        # construction inherits the same fallback semantics.
        if self.attn_weight_dtype is None:
            self.attn_weight_dtype = self.param_dtype
        if self.shared_expert_weight_dtype is None:
            self.shared_expert_weight_dtype = self.param_dtype
        if self.routed_expert_weight_dtype is None:
            self.routed_expert_weight_dtype = self.param_dtype

    def effective_attn_act_dtype(self) -> Dtype:
        return self.attn_act_dtype if self.attn_act_dtype is not None else self.act_dtype

    def effective_moe_act_dtype(self) -> Dtype:
        return self.moe_act_dtype if self.moe_act_dtype is not None else self.act_dtype

    def effective_residual_dtype(self) -> Dtype:
        return self.residual_dtype if self.residual_dtype is not None else self.act_dtype

    @classmethod
    def from_scalar(cls, quant: str | None) -> "GraphQuantProfile | None":
        """Normalize a legacy scalar ``quant`` string into a profile.

        Compatibility table from plan §9a.  Returns ``None`` for no-quant
        baseline (BF16).  Raises ``ValueError`` for inference-only dtypes.
        """
        if quant is None:
            return None

        q = quant.lower().strip()
        if q in ("bf16", "bfloat16"):
            return cls()  # explicit BF16 baseline

        if q in ("fp8", "fp8_e4m3"):
            # fp8_mixed preset: FP8 compute+activation for MoE, BF16 weights
            return cls(
                routed_expert_compute_dtype=Dtype.FP8_E4M3,
                moe_act_dtype=Dtype.FP8_E4M3,
            )

        if q == "fp8_e5m2":
            return cls(
                routed_expert_compute_dtype=Dtype.FP8_E5M2,
                moe_act_dtype=Dtype.FP8_E5M2,
            )

        if q in ("fp4", "mxfp4", "nvfp4"):
            return cls.from_preset("deepseek_v4_paper_fp4")

        if q in _INFERENCE_ONLY:
            raise ValueError(
                f"{q!r} is an inference-only quantization format; "
                f"use fp8_e4m3 or fp4 for training"
            )

        raise ValueError(f"unknown quant dtype {quant!r}")

    @classmethod
    def from_preset(cls, preset_name: str) -> "GraphQuantProfile":
        """Load from the shared ``_QUANT_PRESETS`` table in config_loader."""
        from zrt.training.io.config_loader import _QUANT_PRESETS

        if preset_name not in _QUANT_PRESETS:
            raise KeyError(
                f"unknown quant_preset {preset_name!r}; "
                f"valid: {sorted(_QUANT_PRESETS)}"
            )
        preset = _QUANT_PRESETS[preset_name]
        return cls(
            param_dtype=Dtype.parse(preset.get("param_dtype", "bf16")),
            act_dtype=Dtype.parse(preset.get("act_dtype", "bf16")),
            attn_compute_dtype=Dtype.parse(preset.get("attn_compute_dtype", "bf16")),
            shared_expert_compute_dtype=Dtype.parse(preset.get("shared_expert_compute_dtype", "bf16")),
            routed_expert_compute_dtype=Dtype.parse(preset.get("routed_expert_compute_dtype", "bf16")),
            attn_weight_dtype=Dtype.parse(preset.get("attn_weight_dtype", "bf16")),
            shared_expert_weight_dtype=Dtype.parse(preset.get("shared_expert_weight_dtype", "bf16")),
            routed_expert_weight_dtype=Dtype.parse(preset.get("routed_expert_weight_dtype", "bf16")),
            attn_act_dtype=Dtype.parse(preset["attn_act_dtype"]) if "attn_act_dtype" in preset else None,
            moe_act_dtype=Dtype.parse(preset["moe_act_dtype"]) if "moe_act_dtype" in preset else None,
            residual_dtype=Dtype.parse(preset["residual_dtype"]) if "residual_dtype" in preset else None,
            attn_grad_dtype=Dtype.parse(preset.get("attn_grad_dtype", "fp32")),
            shared_expert_grad_dtype=Dtype.parse(preset.get("shared_expert_grad_dtype", "fp32")),
            routed_expert_grad_dtype=Dtype.parse(preset.get("routed_expert_grad_dtype", "fp32")),
        )

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "GraphQuantProfile":
        """Build from a structured dict with dtype field names matching spec YAML."""
        from zrt.training.io.config_loader import _expand_quant_preset
        expanded = _expand_quant_preset(d)
        kwargs: dict[str, Any] = {}
        _flag_fields = {"ln_softmax_promote_fp32", "assume_all_casts_fused"}
        for f in fields(cls):
            if f.name in _flag_fields:
                if f.name in expanded:
                    v = expanded[f.name]
                    kwargs[f.name] = _str_to_bool(v) if isinstance(v, str) else bool(v)
            elif f.name in expanded:
                kwargs[f.name] = Dtype.parse(expanded[f.name])
        return cls(**kwargs)


@dataclass
class OffloadConfig:
    """Host-device memory offloading configuration."""
    pct: float = 0.0          # Fraction of each component to offload [0.0, 1.0]
    opt_state: bool = False   # Offload optimizer state (Adam momentum/variance)
    grads: bool = False       # Offload gradients after reduction
    params: bool = False      # Offload parameters (requires CPU-GPU sync)


@dataclass
class TrainingConfig:
    """Training-specific configuration for performance modelling."""

    # Optimizer settings
    optimizer: str = "adam"  # "adam", "adamw", "muon"
    zero_stage: int = 1  # 0=none, 1=opt_state, 2=grads+opt, 3=weights+grads+opt
    muon_ns_steps: int | None = None  # Newton-Schulz iterations for Muon
    muon_param_fraction: float | None = None  # Fraction of params using Muon
    muon_rotation: bool = True  # Moonshot rotation optimization for Muon

    # Batch size
    micro_batch: int = 1
    global_batch: int = 32

    # Recompute policy
    recompute_policy: str = "none"  # "none", "full", "selective"

    # Graph-capture MegaMoE controls
    mega_moe: bool = False
    mega_moe_waves: int = 0

    # Pipeline schedule
    pp_schedule: str = "1f1b"  # "1f1b", "interleaved", "dualpipe", "dualpipev", "zb"
    vpp_chunks: int = 1
    pp_mode: str = "trace"     # "trace" = grid-based PPStitcher, "formula" = PipelineComposer

    # Optional explicit layer→stage assignment for PP; length must equal
    # the number of traced transformer layers.  None → greedy bin-packing.
    pp_layer_assignment: list[int] | None = None

    # Context parallel strategy (auto-set based on model type)
    # - cp > 1: DeepSeek-V4 → "compressed", others → "ulysses"
    # - cp <= 1: "none" (no CP)
    # User can override by setting cp_kind explicitly
    cp_kind: str = "none"  # Will be auto-set to "ulysses" or "compressed" when cp > 1

    # Overlap DP allreduce with PP bubble window
    dp_overlap_in_bubble: bool = True

    # CoC (Communication-over-Computation) for TP all_reduce
    # When True, TP comm nodes start after 1/K of the predecessor compute
    # (K-wave overlap), reducing exposed comm time in trace mode.
    tp_coc: bool = False
    tp_coc_tile_k: int = 4

    # Memory offloading (optional, disabled by default)
    offload: OffloadConfig | None = None

    # Model geometry for CP shape split (used by ContextParallelPass)
    seq_len: int = 2048  # Sequence length (required for seq/cp split)
    hidden: int = 7168   # Hidden dimension (required for CP communication sizing)

    def resolve_cp_kind(self, model_id: str = "", cp: int = 1) -> str:
        """Resolve cp_kind based on model type and CP configuration.
        
        Rules:
        - cp <= 1 → "none" (no context parallel)
        - cp > 1 and user specified cp_kind != "none" → keep user choice
        - cp > 1 and model is DeepSeek-V4 → "compressed" (two-stage CP)
        - cp > 1 and other models → "ulysses" (A2A-based CP)
        
        Args:
            model_id: HF model ID (e.g., "deepseek-ai/DeepSeek-V3")
            cp: Context parallel factor
            
        Returns:
            Resolved cp_kind string
        """
        # If user explicitly set cp_kind (not "none"), respect it
        if self.cp_kind != "none":
            return self.cp_kind
        
        # cp <= 1: no CP needed
        if cp <= 1:
            return "none"
        
        # cp > 1: auto-select based on model type
        model_lower = model_id.lower()
        
        # DeepSeek-V4 uses compressed (two-stage) CP
        if "deepseek" in model_lower and ("v4" in model_lower or "v3.2" in model_lower):
            return "compressed"
        
        # All other models use Ulysses (A2A-based) CP
        return "ulysses"

    def effective_ns_steps(self, model_type: str | None = None) -> int:
        """Return effective NS steps, handling None fallback logic.

        Priority:
          1. self.muon_ns_steps (explicit config)
          2. _MUON_NS_STEPS_DEFAULTS[model_type] (model type lookup)
          3. Default 5
        """
        if self.muon_ns_steps is not None:
            return self.muon_ns_steps
        from zrt.training.spec.strategy import _MUON_NS_STEPS_DEFAULTS
        if model_type is not None:
            return _MUON_NS_STEPS_DEFAULTS.get(model_type, 5)
        return 5

    @property
    def num_microbatches(self) -> int:
        return self.global_batch // self.micro_batch


@dataclass
class FusionConfig:
    """User-facing operator-fusion controls.

    Loaded from YAML (default search path or ``--fusion-config``).  Filters
    which fusion rules fire during a single ``FusionPass`` run.

    enabled_rules
        ``None`` → use each rule's ``default_phases`` for the active phase.
        Non-empty set → only rules whose ``name`` is in this set are active.
    disabled_rules
        Always subtracted after ``enabled_rules`` resolution.  Useful for
        starting from defaults and removing a few specific rules.
    allow_structural_collapse
        Re-enables the legacy ``op_type = module_class`` fallback for
        unmatched multi-op buckets.  Off by default — unmatched ops stay
        as raw aten nodes.
    merge_sibling_classes
        Module class names for which ``_merge_parent_groups`` is allowed
        to fuse runs of identical sibling instances (e.g. ``Expert`` lists
        in MoE).  Off by default to keep buckets aligned with single
        forward calls.
    """

    enabled_rules: set[str] | None = None
    disabled_rules: set[str] = field(default_factory=set)
    allow_structural_collapse: bool = False
    merge_sibling_classes: set[str] = field(default_factory=set)

    @classmethod
    def merged(cls, base: "FusionConfig | None",
               override: "FusionConfig | None") -> "FusionConfig":
        """Layer ``override`` on top of ``base`` (override wins on every field)."""
        b = base or cls()
        if override is None:
            return cls(
                enabled_rules=set(b.enabled_rules) if b.enabled_rules is not None else None,
                disabled_rules=set(b.disabled_rules),
                allow_structural_collapse=b.allow_structural_collapse,
                merge_sibling_classes=set(b.merge_sibling_classes),
            )
        return cls(
            enabled_rules=(set(override.enabled_rules)
                           if override.enabled_rules is not None
                           else (set(b.enabled_rules) if b.enabled_rules is not None else None)),
            disabled_rules=set(b.disabled_rules) | set(override.disabled_rules),
            allow_structural_collapse=override.allow_structural_collapse,
            merge_sibling_classes=set(b.merge_sibling_classes) | set(override.merge_sibling_classes),
        )


@dataclass
class TransformContext:
    hw_spec:      "HardwareSpec"
    parallel:     ParallelConfig  = field(default_factory=ParallelConfig)
    stream_config: StreamConfig   = field(default_factory=StreamConfig)
    quant:        QuantConfig | None = None
    quant_profile: GraphQuantProfile | None = None  # Structured per-component dtype profile
    training:     TrainingConfig | None = None  # Training-specific config
    fusion:       FusionConfig    = field(default_factory=FusionConfig)
    optim_flags:  set[str]        = field(default_factory=set)
    phase:        str             = "prefill"
    profile:      Any             = None   # ModelProfile (optional)
    stack:        Any             = None   # SoftwareStack (optional)
    model_id:     str             = ""     # HF model id, used by FusionPass to load platform rules

    @property
    def is_training(self) -> bool:
        return self.training is not None

    def phase_for_fusion(self) -> str:
        """Return ``"training"`` when training, else ``"inference"``."""
        return "training" if self.is_training else "inference"
