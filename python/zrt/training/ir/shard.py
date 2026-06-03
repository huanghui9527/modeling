"""Sharding pass — apply TP/CP/EP sharding to IR and insert collectives.

Entry points:
  - :func:`insert_collectives` — legacy Graph path
  - :func:`insert_collectives_opgraph` — OpGraph-native path (Phase B2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zrt.ir.graph import OpGraph
    from zrt.ir.node import OpNode

from zrt.training.ir.training_graph import Collective, Graph
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import CPKind, Strategy


def _is_moe_layer(model: ModelSpec, layer_id: int) -> bool:
    """True when the layer at ``layer_id`` is a MoE block.

    Used to pick activation dtype for FFN-side TP collectives — MoE FFN
    runs in ``effective_moe_act_dtype`` while dense FFN stays at
    ``act_dtype``.
    """
    if 0 <= layer_id < len(model.layers):
        return model.layers[layer_id] == LayerKind.MOE
    return False


class ShardPlan:
    """Holds sharding parameters derived from Strategy."""

    def __init__(self, strategy: Strategy):
        self.tp = strategy.tp
        self.cp = strategy.cp
        self.ep = strategy.ep
        self.dp = strategy.dp
        self.pp = strategy.pp
        self.cp_kind = strategy.cp_kind
        self.cp_ulysses, self.cp_ring = strategy.hybrid_cp_factors()
        self.sp = strategy.tp > 1  # Megatron SP on when TP>1

    def shard_col_parallel(self, n: int) -> int:
        """Shard column dimension by TP."""
        return n // self.tp

    def shard_row_parallel(self, n: int) -> int:
        """Shard row dimension by TP (output dimension stays full via RS)."""
        return n // self.tp

    def shard_seq_parallel(self, s: int) -> int:
        """Shard sequence dimension by CP (Ulysses)."""
        return s // self.cp if self.cp > 1 else s

    def shard_expert_parallel(self, n: int) -> int:
        """Shard experts by EP (each rank gets n/experts)."""
        return n // self.ep if self.ep > 1 else n

    @property
    def has_cp(self) -> bool:
        """Whether context parallelism is enabled."""
        return self.cp > 1

    @property
    def has_ep(self) -> bool:
        """Whether expert parallelism is enabled."""
        return self.ep > 1

    @property
    def has_tp(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp > 1


def insert_collectives(graph: Graph, model: ModelSpec, strategy: Strategy) -> None:
    """Insert TP/CP/EP collectives into the graph IN-PLACE.

    Megatron-TP pattern per dense layer:
      - AG before QKV (if input was RS'd by previous layer)
      - RS after O_proj
      - AG before FFN up/gate
      - RS after FFN down

    Ulysses-CP pattern:
      - A2A before attention (splits sequence across ranks)
      - A2A after attention (gathers sequence)

    EP pattern (MoE only):
      - All-to-All before routed expert FFN
      - All-to-All after routed expert FFN
    """
    shard = ShardPlan(strategy)
    collectives: list[Collective] = []

    # Insert TP collectives
    if shard.has_tp:
        _insert_tp_collectives(graph, shard, model, collectives)

    # Insert CP collectives (Ulysses sequence parallel)
    if shard.has_cp:
        _insert_cp_collectives(graph, shard, model, collectives)

    # Insert EP collectives (expert parallel for MoE)
    if shard.has_ep:
        _insert_ep_collectives(graph, shard, model, strategy, collectives)

    graph.collectives.extend(collectives)

    # Apply TP/CP sharding to global ops (layer_id < 0: lm_head, final_ln, etc.)
    # These are outside the per-layer ranges and would otherwise be skipped.
    _apply_global_hc_sharding(graph, strategy, model.seq_len)
    _apply_global_token_op_sharding(graph, strategy)
    _apply_global_lm_head_sharding(graph, strategy)


def _shard_hc_sequence(op, factor: int, seq: int) -> None:
    """Shard Hyper-Connection token dimension by an additional factor."""
    if factor <= 1:
        return
    if "s" in op.meta and op.meta["s"] > 0:
        op.meta["s"] = max(1, op.meta["s"] // factor)
    if "bytes_fwd" in op.meta:
        op.meta["bytes_fwd"] = max(1, int(op.meta["bytes_fwd"]) // factor)
    for t in op.inputs + op.outputs:
        if t.shape_local:
            t.shape_local = (max(1, t.shape_local[0] // factor),) + t.shape_local[1:]


def _apply_global_hc_sharding(graph: Graph, strategy: Strategy, seq: int) -> None:
    """Apply TP/CP token sharding to HC ops outside layer_index."""
    factor = max(1, strategy.tp) * max(1, strategy.cp)
    if factor <= 1:
        return
    for op in graph.ops:
        if op.layer_id < 0 and op.kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            _shard_hc_sequence(op, factor, seq)


def _shard_global_token_sequence(op, factor: int) -> None:
    """Shard the leading token dimension for global sequence-local ops."""
    if factor <= 1:
        return
    if "m" in op.meta and op.meta["m"] > 0:
        op.meta["m"] = max(1, op.meta["m"] // factor)
    if "s" in op.meta and op.meta["s"] > 0:
        op.meta["s"] = max(1, op.meta["s"] // factor)
    if "bytes_fwd" in op.meta:
        op.meta["bytes_fwd"] = max(1, int(op.meta["bytes_fwd"]) // factor)
    for t in op.inputs + op.outputs:
        if t.shape_local:
            t.shape_local = (max(1, t.shape_local[0] // factor),) + t.shape_local[1:]


def _apply_global_token_op_sharding(graph: Graph, strategy: Strategy) -> None:
    """Apply CP token sharding to non-lm_head global token ops."""
    factor = max(1, strategy.cp)
    if factor <= 1:
        return
    for op in graph.ops:
        if op.layer_id < 0 and op.kind in ("embed", "ln", "rmsnorm"):
            _shard_global_token_sequence(op, factor)


def _apply_global_lm_head_sharding(graph: Graph, strategy: Strategy) -> None:
    """Apply TP vocab and CP sequence sharding to global lm_head."""
    tp = max(1, strategy.tp)
    cp = max(1, strategy.cp)
    if tp <= 1 and cp <= 1:
        return
    for op in graph.ops:
        if op.layer_id >= 0 or op.kind != "lm_head":
            continue
        m = op.meta.get("m", 0)
        n = op.meta.get("n", 0)
        if cp > 1 and m > 0:
            op.meta["m"] = max(1, m // cp)
            for t in op.inputs + op.outputs:
                if t.shape_logical and t.shape_logical[0] == m:
                    t.shape_local = (max(1, t.shape_local[0] // cp),) + t.shape_local[1:]
        if tp > 1 and n > 0:
            n_local = max(1, n // tp)
            op.meta["n_local"] = n_local
            for t in op.outputs:
                if t.shape_logical and t.shape_logical[-1] == n:
                    t.shape_local = t.shape_local[:-1] + (n_local,)


def _insert_tp_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert TP collectives (AG/RS pairs) into the graph."""
    seq = model.seq_len
    h = model.hidden
    h_attn = model.num_heads * model.head_dim
    h_kv = model.num_kv_heads * model.head_dim
    ffn = model.ffn
    # v2 region-level dtype: AG/RS around attention use attn region's
    # activation dtype; FFN-side use the dense/MoE region. For MoE layers
    # the FFN-side TP collectives wrap shared/routed experts → use moe_act.
    act_bytes = model.act_dtype.bytes               # legacy / fallback / TP-shape metadata
    attn_act_bytes = model.effective_attn_act_dtype().bytes
    moe_act_bytes = model.effective_moe_act_dtype().bytes

    # When CP is enabled, sequence is already sharded by CP
    # TP AG/RS should operate on the CP-sharded sequence
    if shard.cp > 1:
        seq = seq // shard.cp

    for layer_id, (start, end) in graph.layer_index.items():
        # AG before QKV gathers an x_ln1 tile (attention region dtype after
        # the fused LN epilog in v2 builders). RS after O_proj scatters
        # attn_proj which is in the same dtype.
        ag_attn_bytes = seq * h * attn_act_bytes  # AG before QKV
        rs_attn_bytes = seq * h * attn_act_bytes  # RS after O_proj
        # FFN AG/RS: in a dense block this wraps the dense FFN (act_dtype);
        # in MoE this wraps shared expert ups/downs (moe_act). Pick the
        # right dtype using the layer's kind from layer_index.
        ffn_bytes_per = moe_act_bytes if _is_moe_layer(model, layer_id) else act_bytes
        ag_ffn_bytes = seq * h * ffn_bytes_per   # AG before FFN up
        rs_ffn_bytes = seq * h * ffn_bytes_per   # RS after FFN down

        for i in range(start, end):
            op = graph.ops[i]

            # AG before QKV projection (gathers seq-sharded input for col-parallel)
            if op.kind == "matmul" and "qkv" in op.name:
                collectives.append(Collective(
                    name=f"ag_{op.name}",
                    kind="AG", group="TP",
                    bytes_=ag_attn_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

            # RS after O projection
            if op.kind == "matmul" and "o_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_attn_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

            # AG before FFN up projection
            if op.kind == "matmul" and "up_proj" in op.name:
                collectives.append(Collective(
                    name=f"ag_{op.name}",
                    kind="AG", group="TP",
                    bytes_=ag_ffn_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

            # RS after FFN down projection
            if op.kind == "matmul" and "down_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_ffn_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

        # Adjust tensor shapes for TP sharding
        _apply_tp_sharding(graph, start, end, shard, h, h_attn, h_kv, ffn, seq, act_bytes)


def _insert_cp_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert CP collectives for different CP strategies.

    Ulysses-CP pattern:
      - A2A before attention core (scatter-seq / gather-heads)
      - A2A after attention core (gather-seq / scatter-heads)
      - Each A2A transfers seq/cp × hidden_tp bytes

    Ring-CP pattern:
      - P2P send/recv pairs, cp rounds per attention
      - Each P2P round transfers seq/cp × hidden_tp bytes
      - P2P overlaps with FA tile computation

    Hybrid-CP pattern:
      - Combines Ulysses A2A with Ring P2P overlap
      - Explicit factors use cp_ulysses for head sharding and cp_ring for P2P
        rounds. Legacy configs without factors preserve the previous total-cp
        approximation.

    Note: When TP is enabled, hidden dimension is already sharded by TP.
    CP communication should use hidden_tp = hidden / tp, not full hidden.
    """
    h = model.hidden
    if shard.tp > 1:
        h = h // shard.tp
    # CP shards the sequence INSIDE the attention region, so activation
    # bytes use the attention region dtype (FP8 in mixed-quant V4).
    act_bytes = model.effective_attn_act_dtype().bytes
    cp = shard.cp

    # If cp_kind is NONE, skip collective insertion but still apply
    # CP sharding metadata for FLOPs calculation (cp > 1 but unspecified kind).
    if shard.cp_kind == CPKind.NONE:
        for layer_id, (start, end) in graph.layer_index.items():
            _apply_cp_sharding(graph, start, end, shard, model.seq_len, model.hidden)
        return

    for layer_id, (start, end) in graph.layer_index.items():
        # Note: Insert collectives BEFORE sharding metadata.
        # Collectives use model.seq_len directly (not op.meta["s"]),
        # so sharding order doesn't affect communication calculation.
        # Sharding metadata is for subsequent FLOPs calculation.
        for i in range(start, end):
            op = graph.ops[i]

            if op.kind not in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
                continue

            if shard.cp_kind == CPKind.ULYSSES:
                a2a_bytes = (model.seq_len // cp) * h * act_bytes
                collectives.append(Collective(
                    name=f"a2a_fwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_fwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="bwd",
                ))

            elif shard.cp_kind == CPKind.RING:
                p2p_bytes = (model.seq_len // cp) * h * act_bytes
                # Forward: P2P overlaps with FA tile computation
                collectives.append(Collective(
                    name=f"p2p_ring_fwd_{op.name}",
                    kind="P2P", group="CP",
                    bytes_=p2p_bytes,
                    inserted_before=op.name,
                    rounds=cp,
                    overlap=True,
                    phase="fwd",
                ))
                # Backward: P2P for gradient propagation (similar pattern)
                collectives.append(Collective(
                    name=f"p2p_ring_bwd_{op.name}",
                    kind="P2P", group="CP",
                    bytes_=p2p_bytes,
                    inserted_before=op.name,
                    rounds=cp,
                    overlap=True,
                    phase="bwd",
                ))

            elif shard.cp_kind == CPKind.HYBRID:
                a2a_bytes = (model.seq_len // cp) * h * act_bytes
                collectives.append(Collective(
                    name=f"a2a_fwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                p2p_bytes = (model.seq_len // cp) * h * act_bytes
                collectives.append(Collective(
                    name=f"p2p_ring_{op.name}",
                    kind="P2P", group="CP",
                    bytes_=p2p_bytes,
                    inserted_before=op.name,
                    rounds=shard.cp_ring,
                    overlap=True,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_fwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="bwd",
                ))
            
            elif shard.cp_kind == CPKind.COMPRESSED:
                # DeepSeek-V4 two-stage compressed CP
                # Determine layer type (CSA/HCA/SWA) from model spec
                cp_type = model.get_layer_cp_type(layer_id)
                
                # SWA-only layers don't participate in CP communication
                if cp_type == 'swa':
                    continue
                
                # Import compressed CP analyzer
                from zrt.training.models.compressed_cp import (
                    CompressedCPConfig, 
                    CompressedCPCommAnalyzer,
                )
                
                # Configure compressed CP based on layer type
                if cp_type == 'csa':
                    compression_ratio = 4  # CSA: m=4
                elif cp_type == 'hca':
                    compression_ratio = 128  # HCA: m'=128
                else:
                    continue  # Unknown type, skip
                
                # Create analyzer for this layer type
                cp_config = CompressedCPConfig(
                    cp_size=cp,
                    compression_ratio_csa=4 if cp_type == 'csa' else 4,
                    compression_ratio_hca=128 if cp_type == 'hca' else 128,
                    kv_head_dim=model.head_dim,
                )
                analyzer = CompressedCPCommAnalyzer(cp_config)
                
                # Stage 1: P2P boundary exchange (forward + backward)
                if cp_type == 'csa':
                    stage1_bytes = analyzer.stage1_comm_bytes_csa()
                else:  # hca
                    stage1_bytes = analyzer.stage1_comm_bytes_hca()
                
                collectives.append(Collective(
                    name=f"p2p_boundary_fwd_{op.name}_stage1_{cp_type}",
                    kind="P2P", group="CP",
                    bytes_=stage1_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"p2p_boundary_bwd_{op.name}_stage1_{cp_type}",
                    kind="P2P", group="CP",
                    bytes_=stage1_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))
                
                # Stage 2: AllGather compressed KV (forward + backward)
                if cp_type == 'csa':
                    stage2_bytes = analyzer.stage2_comm_bytes_csa(model.seq_len)
                else:  # hca
                    stage2_bytes = analyzer.stage2_comm_bytes_hca(model.seq_len)
                
                collectives.append(Collective(
                    name=f"allgather_kv_fwd_{op.name}_stage2_{cp_type}",
                    kind="AG", group="CP",
                    bytes_=stage2_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"allgather_kv_bwd_{op.name}_stage2_{cp_type}",
                    kind="AG", group="CP",
                    bytes_=stage2_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))

        _apply_cp_sharding(graph, start, end, shard, model.seq_len, model.hidden)


def _insert_ep_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec, strategy: Strategy,
    collectives: list[Collective],
) -> None:
    """Insert EP collectives (A2A pairs for expert parallel).

    EP pattern (MoE only):
      - A2A before routed expert FFN (routes tokens to expert ranks)
      - A2A after routed expert FFN (gathers expert outputs)
    """
    h = model.hidden
    if shard.tp > 1:
        h = h // shard.tp
    seq = model.seq_len
    if shard.cp > 1:
        seq = seq // shard.cp
    moe_ffn = model.moe_ffn if model.moe_ffn > 0 else model.ffn
    # v2: EP A2A dispatches tokens INTO the MoE region — payload should
    # use the MoE region's activation dtype (FP8 in mixed-quant V4 →
    # halves the A2A volume vs BF16 baseline).
    act_bytes = model.effective_moe_act_dtype().bytes

    # EP A2A payload — per-rank participating buffer (bus-bw convention).
    #
    # Each rank holds ``(micro_batch, seq/cp, hidden/tp)`` activation and
    # top-k routes each token, so its dispatch input buffer is
    # ``micro_batch * seq * h * topk * act_bytes`` (seq, h already CP/TP
    # sharded). ``collective_time()`` applies ``× (N-1)/N`` to extract the
    # cross-rank portion, so we MUST NOT pre-divide by EP here — doing so
    # under-counts EP A2A by exactly ``ep`` (history: a /ep divisor sat
    # here for years; calibration absorbed the error until ep > 16 grids
    # made it visible — see verify_ep_a2a_bug.py for the 94× delta on
    # ep=128, top_k=6).
    micro_batch = strategy.micro_batch
    topk = model.top_k
    a2a_bytes = micro_batch * seq * h * topk * act_bytes

    for layer_id, (start, end) in graph.layer_index.items():
        # Check if this is an MoE layer
        if layer_id >= len(model.layers):
            continue
        if model.layers[layer_id].value != "moe":
            continue

        for i in range(start, end):
            op = graph.ops[i]

            if op.kind == "mega_moe":
                continue

            # A2A before routed expert FFN
            if op.kind == "matmul" and "routed_expert" in op.name:
                collectives.append(Collective(
                    name=f"a2a_before_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="both",
                ))

            # A2A after routed expert FFN
            if op.kind == "matmul" and "routed_expert" in op.name:
                collectives.append(Collective(
                    name=f"a2a_after_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

        # Adjust tensor shapes for EP sharding (experts sharded across ranks)
        _apply_ep_sharding(graph, start, end, shard, model.num_experts, model.top_k)


def _apply_tp_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan,
    h: int, h_attn: int, h_kv: int, ffn: int, seq: int, act_bytes: int,
) -> None:
    """Adjust tensor shape_local for TP sharding on ops in [start, end)."""
    if shard.tp <= 1:
        return

    for i in range(start, end):
        op = graph.ops[i]

        if op.kind == "mega_moe":
            k = op.meta.get("k", 0)
            n = op.meta.get("n", 0)
            if k > 0:
                op.meta["k_local"] = k // shard.tp
            if n > 0:
                n_local = n // shard.tp
                op.meta["n_local"] = n_local
                for t in op.outputs:
                    if t.shape_logical and t.shape_logical[-1] == n:
                        t.shape_local = (t.shape_logical[0], n_local)
        elif op.kind == "matmul":
            m = op.meta.get("m", 0)
            n = op.meta.get("n", 0)
            k = op.meta.get("k", 0)

            # Column-parallel ops: output dimension (n) sharded by TP
            col_parallel = any(p in op.name for p in (
                "qkv", "q_a_proj", "q_b_proj", "kv_a_proj",
                "wq_a", "wq_b", "wkv",
                "up_proj", "gate_proj", "shared_up_proj", "shared_gate_proj",
                "comp_wkv", "comp_wgate",
                "idx_wq_b", "idx_weights", "idx_comp_wkv", "idx_comp_wgate",
            ))
            # Row-parallel ops: input dimension (k) sharded by TP
            row_parallel = any(p in op.name for p in (
                "o_proj", "down_proj", "shared_down_proj",
                "wo_a", "wo_b", "kv_b_proj",
            ))

            if col_parallel:
                n_local = n // shard.tp
                updated = False
                for t in op.outputs:
                    if t.shape_logical and t.shape_logical[-1] == n:
                        t.shape_local = (t.shape_logical[0], n_local)
                        updated = True
                if not updated:
                    op.meta["n_local"] = n_local
            elif row_parallel:
                k_local = k // shard.tp
                updated = False
                for t in op.inputs:
                    if t.shape_logical and t.shape_logical[-1] == k:
                        t.shape_local = (t.shape_logical[0], k_local)
                        updated = True
                if not updated:
                    op.meta["k_local"] = k_local
            elif "router" in op.name:
                n_local = n // shard.tp
                for t in op.outputs:
                    if t.shape_logical and t.shape_logical[-1] == n:
                        t.shape_local = (t.shape_logical[0], n_local)
            elif "routed_expert" in op.name:
                # routed_expert_ffn fuses up+gate+down projections into one op.
                # up/gate are column-parallel (n sharded), down is row-parallel
                # (k sharded). Total FLOPs = 6*m*n*k*top_k / tp (single tp factor).
                # If we set both n_local AND k_local, _matmul_cost computes
                # 2*m*n_local*k_local*multiplier = 6*m*n*k*top_k / tp² — WRONG.
                # Fix: only set k_local in meta (so cost uses full n * k_local/tp),
                # but still update both tensor shapes for memory tracking.
                k_local = k // shard.tp
                n_local = n // shard.tp
                op.meta["k_local"] = k_local
                # Deliberately do NOT set n_local — prevents tp² division.
                for t in op.inputs:
                    if t.shape_logical and t.shape_logical[-1] == k:
                        t.shape_local = (t.shape_logical[0], k_local)
                for t in op.outputs:
                    if t.shape_logical and t.shape_logical[-1] == n:
                        t.shape_local = (t.shape_logical[0], n_local)
        elif op.kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
            if "heads" in op.meta:
                heads_before_tp = op.meta["heads"]
                op.meta["heads"] = max(1, op.meta["heads"] // shard.tp)
                op.meta["heads_tp"] = op.meta["heads"]
                op.meta["heads_before_tp"] = heads_before_tp
            if "kv_heads" in op.meta:
                kv_heads_before_tp = op.meta["kv_heads"]
                op.meta["kv_heads"] = max(1, op.meta["kv_heads"] // shard.tp)
                op.meta["kv_heads_tp"] = op.meta["kv_heads"]
                op.meta["kv_heads_before_tp"] = kv_heads_before_tp
            if "h_kv" in op.meta:
                op.meta["h_kv"] = max(1, op.meta["h_kv"] // shard.tp)
            for t in op.inputs + op.outputs:
                if t.shape_logical and t.shape_logical[-1] == h_attn:
                    t.shape_local = (t.shape_logical[0], max(1, h_attn // shard.tp))
                elif t.shape_logical and t.shape_logical[-1] == h_kv:
                    if "kv_heads" in op.meta:
                        kv_heads_local = op.meta["kv_heads"]
                        kv_heads_before_tp = op.meta.get("kv_heads_before_tp", kv_heads_local)
                        kv_head_dim = max(1, h_kv // kv_heads_before_tp)
                        t.shape_local = (t.shape_logical[0], kv_heads_local * kv_head_dim)
                    else:
                        t.shape_local = (t.shape_logical[0], max(1, h_kv // shard.tp))
        elif op.kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            # Hyper-Connections are token-local: TP does not shard hc or h.
            # The mixes-Linear operates on (hc·h) features that semantically
            # belong to *one* residual stream replicated across hc copies; the
            # learned hc_*_fn matrix is small (hc·h × mix_hc) and replicated on
            # every TP rank, so introducing TP here would only add comm without
            # reducing compute.  CP (sequence-parallel) handling — when added —
            # would scale the seq dim of meta["s"] independently.
            _shard_hc_sequence(op, shard.tp, seq)
        elif op.kind == "indexer_topk":
            # Shard indexer heads by TP: ih → ih_local
            ih = op.meta.get("ih", 0)
            if ih > 0:
                ih_local = max(1, ih // shard.tp)
                op.meta["ih_local"] = ih_local
                op.meta["world_factor"] = shard.tp
                id_ = op.meta.get("id", 0)
                for t in op.inputs:
                    if "idx_q" in t.name and t.shape_logical:
                        t.shape_local = (t.shape_logical[0], ih_local * id_)
                    elif "idx_w" in t.name and t.shape_logical:
                        t.shape_local = (t.shape_logical[0], ih_local)
                # Scale bytes_fwd: idx_q and idx_w are sharded by ih/TP,
                # idx_kv uses compressed kv_len (for CSA), output is replicated.
                kv_len = op.meta.get("kv_len", op.meta.get("s", 0))
                idx_q_bytes = op.meta.get("s", 0) * ih_local * id_ * 2
                idx_kv_bytes = kv_len * id_ * 2
                idx_w_bytes = op.meta.get("s", 0) * ih_local * 2
                idx_out_bytes = op.meta.get("s", 0) * op.meta.get("topk", 0) * 2
                op.meta["bytes_fwd"] = idx_q_bytes + idx_kv_bytes + idx_w_bytes + idx_out_bytes
        elif op.kind == "lm_head":
            # Column-parallel: vocab (n) sharded by TP
            n = op.meta.get("n", 0)
            if n > 0:
                n_local = n // shard.tp
                op.meta["n_local"] = n_local
                for t in op.outputs:
                    if t.shape_logical and t.shape_logical[-1] == n:
                        t.shape_local = (t.shape_logical[0], n_local)
            if "bytes_fwd" in op.meta:
                op.meta["bytes_fwd"] = int(op.meta["bytes_fwd"]) // shard.tp
        elif op.kind == "compressor_pool":
            # Shard compressor dim by TP: d → d_local
            d = op.meta.get("d", 0)
            if d > 0:
                d_local = max(1, d // shard.tp)
                op.meta["d_local"] = d_local
                op.meta["world_factor"] = shard.tp
                if "bytes_fwd" in op.meta:
                    op.meta["bytes_fwd"] = op.meta["bytes_fwd"] // shard.tp
        elif op.kind in ("ln", "rmsnorm", "rope", "swiglu", "add"):
            for t in op.inputs + op.outputs:
                if t.shape_logical and t.shape_logical[-1] in (h, h_attn, h_kv, ffn):
                    t.shape_local = (t.shape_logical[0], max(1, t.shape_logical[-1] // shard.tp))


def _apply_cp_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan, seq: int, hidden: int,
) -> None:
    """Adjust tensor shape_local for CP sharding.

    Ulysses-CP: A2A scatters heads, gathers seq.
      Per-rank attn work: s_local = full_seq, heads_local = heads_tp // cp.
    Ring-CP: seq split across cp ranks, KV chunks streamed via P2P.
      s_local = seq / cp, heads_local = heads_tp.
    Hybrid-CP: Ulysses head sharding plus Ring sequence tiling.
      s_local = seq / cp_ring, heads_local = heads_tp // cp_ulysses,
      cp_tiles = cp_ring. Legacy configs use total cp for both factors.
    """
    if not shard.has_cp:
        return

    for i in range(start, end):
        op = graph.ops[i]

        for t in op.inputs + op.outputs:
            if t.shape_logical and len(t.shape_logical) > 0:
                if t.shape_logical[0] == seq:
                    t.shape_local = (max(1, t.shape_local[0] // shard.cp),) + t.shape_local[1:]

        if op.kind == "mega_moe":
            if "m" in op.meta:
                op.meta["m"] = op.meta["m"] // shard.cp
        elif op.kind == "matmul":
            # For meta-authoritative matmuls (grouped/fused where meta k != input shape),
            # _matmul_cost reads m from meta — must divide it here.
            meta_k = op.meta.get("k", 0)
            if "m" in op.meta and meta_k > 0 and op.inputs and op.inputs[0].shape_logical[-1] != meta_k:
                op.meta["m"] = op.meta["m"] // shard.cp
        elif op.kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
            heads_tp = op.meta.get("heads_tp", op.meta.get("heads", 0))
            kv_heads_tp = op.meta.get("kv_heads_tp", op.meta.get("kv_heads"))

            if shard.cp_kind == CPKind.ULYSSES:
                # Ulysses A2A1: scatter heads across CP ranks, gather seq.
                # Per-rank attn work: full seq, heads_tp // cp heads.
                op.meta["heads"] = max(1, heads_tp // shard.cp)
                if kv_heads_tp is not None:
                    op.meta["kv_heads"] = max(1, kv_heads_tp // shard.cp)
                op.meta["heads_gathered_by_cp"] = False
                # NOTE: do NOT divide op.meta["s"] — full seq is on this
                # rank during attention.
            elif shard.cp_kind == CPKind.HYBRID:
                if "s" in op.meta:
                    op.meta["s"] = op.meta["s"] // shard.cp_ring
                op.meta["heads"] = max(1, heads_tp // shard.cp_ulysses)
                if kv_heads_tp is not None:
                    op.meta["kv_heads"] = max(1, kv_heads_tp // shard.cp_ulysses)
                op.meta["heads_gathered_by_cp"] = False
                op.meta["cp_tiles"] = shard.cp_ring
            elif shard.cp_kind == CPKind.RING:
                if "s" in op.meta:
                    op.meta["s"] = op.meta["s"] // shard.cp
                # heads unchanged; ring tiles via cp rounds
                op.meta["heads_gathered_by_cp"] = False
                op.meta["cp_tiles"] = shard.cp
            else:
                # CPKind.NONE / COMPRESSED — preserve existing behavior
                if "s" in op.meta:
                    op.meta["s"] = op.meta["s"] // shard.cp
        elif op.kind in ("ln", "rope", "swiglu", "add"):
            pass  # shape_local already updated by the generic loop above
        elif op.kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            if "s" in op.meta and op.meta["s"] > 0:
                op.meta["s"] = max(1, op.meta["s"] // shard.cp)
            if "bytes_fwd" in op.meta:
                op.meta["bytes_fwd"] = max(1, int(op.meta["bytes_fwd"]) // shard.cp)
        elif op.kind == "compressor_pool":
            # Sequence dimension is sharded by CP. Each rank pools its local
            # chunk independently (m=4 pooling is local, no cross-rank comm).
            s = op.meta.get("s", 0)
            if s > 0:
                op.meta["s"] = s // shard.cp
                op.meta["world_factor"] = shard.cp
            if "bytes_fwd" in op.meta:
                op.meta["bytes_fwd"] = int(op.meta["bytes_fwd"]) // shard.cp
        elif op.kind == "indexer_topk":
            # In Ulysses CP, query sequence is split by CP. The compressed KV
            # needs to be gathered via A2A before scoring, so kv_len stays
            # global. FLOPs scale as 1/CP (s → s/cp, kv_len unchanged).
            s = op.meta.get("s", 0)
            if s > 0:
                op.meta["s"] = s // shard.cp
                op.meta["world_factor"] = shard.cp
            # Scale bytes_fwd: idx_q, idx_w, and output are sharded by CP,
            # but idx_kv stays global (gathered from all ranks).
            kv_len = op.meta.get("kv_len", 0)
            id_ = op.meta.get("id", 0)
            topk = op.meta.get("topk", 0)
            ih = op.meta.get("ih_local", op.meta.get("ih", 0))
            s_local = s // shard.cp if s > 0 else 0
            idx_q_bytes = s_local * ih * id_ * 2
            idx_kv_bytes = kv_len * id_ * 2  # global
            idx_w_bytes = s_local * ih * 2
            idx_out_bytes = s_local * topk * 2
            op.meta["bytes_fwd"] = idx_q_bytes + idx_kv_bytes + idx_w_bytes + idx_out_bytes


def _apply_ep_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan,
    num_experts: int, top_k: int,
) -> None:
    """Adjust tensor shape_local for EP sharding (expert parallel).

    For EP, each rank handles num_experts/ep experts.
    The routed expert FFN only processes a subset of experts.
    """
    if not shard.has_ep:
        return

    experts_per_rank = num_experts // shard.ep

    for i in range(start, end):
        op = graph.ops[i]

        # Routed expert FFN: under uniform routing the per-rank work is
        # invariant to EP. Each rank has experts_per_rank = num_experts/ep
        # local experts. Tokens flow into them from `ep` ranks (the EP group),
        # each contributing seq*top_k/num_experts tokens per expert. The two
        # factors cancel: per-rank work = seq * top_k * moe_ffn * hidden * 6,
        # which is what fwd_multiplier=3*top_k already encodes. So no scaling
        # here. Load imbalance (expert_imbalance) is applied later in
        # compose/stage.py via _ep_parallel_fraction.
        if op.kind == "matmul" and "routed_expert" in op.name:
            pass  # intentionally left unscaled — see comment above

        if op.kind == "mega_moe":
            num_experts = op.meta.get("num_experts", 0)
            if num_experts % shard.ep != 0:
                raise ValueError(
                    f"num_experts({num_experts}) not divisible by EP({shard.ep})"
                )
            op.meta["ep"] = shard.ep
            op.meta["experts_per_rank"] = num_experts // shard.ep

        # Router output: num_experts -> experts_per_rank
        if op.kind == "matmul" and "router" in op.name:
            for t in op.outputs:
                if len(t.shape_logical) > 1:
                    # Router output: (seq, num_experts) -> (seq, experts_per_rank)
                    if t.shape_logical[1] == num_experts:
                        t.shape_local = (t.shape_logical[0], experts_per_rank)


# ── OpGraph-native sharding pass (Phase B2) ─────────────────────────────────

def insert_collectives_opgraph(
    graph: "OpGraph", model: ModelSpec, strategy: Strategy,
) -> None:
    """Insert TP/CP/EP collectives into the OpGraph IN-PLACE.

    This is the OpGraph-native version of :func:`insert_collectives`.
    It creates comm.* OpNodes and inserts them into the graph at the
    appropriate positions, then applies TP/CP/EP sharding to tensor shapes.

    Parameters
    ----------
    graph : OpGraph
        The computation graph to modify in-place.
    model : ModelSpec
        Model architecture specification.
    strategy : Strategy
        Parallel strategy configuration.
    """
    from zrt.ir.graph import OpGraph
    from zrt.ir.node import OpNode
    from zrt.ir.types import TensorMeta
    from zrt.ir.edge import Edge

    shard = ShardPlan(strategy)
    comm_nodes: list[tuple[str, OpNode, str]] = []
    comm_counter = [0]

    if shard.has_tp:
        _insert_tp_collectives_opgraph(graph, shard, model, comm_nodes, comm_counter)

    if shard.has_cp:
        _insert_cp_collectives_opgraph(graph, shard, model, comm_nodes, comm_counter)

    if shard.has_ep:
        _insert_ep_collectives_opgraph(graph, shard, model, strategy, comm_nodes, comm_counter)

    for ref_id, comm_node, position in comm_nodes:
        ref_node = graph.nodes.get(ref_id)
        if ref_node:
            comm_node.layer = ref_node.layer
        graph.add_node(comm_node)
        if position == "after":
            graph.add_edge(Edge(src=ref_id, src_idx=0, dst=comm_node.id, dst_idx=0))
        else:
            graph.add_edge(Edge(src=comm_node.id, src_idx=0, dst=ref_id, dst_idx=0))

    _apply_global_hc_sharding_opgraph(graph, strategy, model.seq_len)
    _apply_global_token_op_sharding_opgraph(graph, strategy)
    _apply_global_lm_head_sharding_opgraph(graph, strategy)


def _insert_tp_collectives_opgraph(
    graph: "OpGraph", shard: ShardPlan, model: ModelSpec,
    comm_nodes: list[tuple[str, "OpNode", str]], comm_counter: list[int],
) -> None:
    """Insert TP collectives (AG/RS pairs) into the OpGraph."""
    from zrt.ir.node import OpNode

    seq = model.seq_len
    h = model.hidden
    h_attn = model.num_heads * model.head_dim
    h_kv = model.num_kv_heads * model.head_dim
    ffn = model.ffn
    act_bytes = model.act_dtype.bytes
    attn_act_bytes = model.effective_attn_act_dtype().bytes
    moe_act_bytes = model.effective_moe_act_dtype().bytes

    if shard.cp > 1:
        seq = seq // shard.cp

    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue

        layer_id = int(node.layer) if node.layer and node.layer.isdigit() else -1
        if layer_id < 0:
            continue

        ffn_bytes_per = moe_act_bytes if _is_moe_layer(model, layer_id) else act_bytes
        ag_attn_bytes = seq * h * attn_act_bytes
        rs_attn_bytes = seq * h * attn_act_bytes
        ag_ffn_bytes = seq * h * ffn_bytes_per
        rs_ffn_bytes = seq * h * ffn_bytes_per

        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if spec_kind == "matmul":
            if "qkv" in node.id:
                comm_counter[0] += 1
                comm_nodes.append((node.id, _make_comm_node_opgraph(
                    f"ag_{node.id}", "AG", "TP", ag_attn_bytes, comm_counter[0], "both",
                    inserted_after=node.id), "after"))

            if "o_proj" in node.id:
                comm_counter[0] += 1
                comm_nodes.append((node.id, _make_comm_node_opgraph(
                    f"rs_{node.id}", "RS", "TP", rs_attn_bytes, comm_counter[0], "both",
                    inserted_after=node.id), "after"))

            if "up_proj" in node.id or "shared_up_proj" in node.id:
                comm_counter[0] += 1
                comm_nodes.append((node.id, _make_comm_node_opgraph(
                    f"ag_{node.id}", "AG", "TP", ag_ffn_bytes, comm_counter[0], "both",
                    inserted_after=node.id), "after"))

            if "down_proj" in node.id or "shared_down_proj" in node.id:
                comm_counter[0] += 1
                comm_nodes.append((node.id, _make_comm_node_opgraph(
                    f"rs_{node.id}", "RS", "TP", rs_ffn_bytes, comm_counter[0], "both",
                    inserted_after=node.id), "after"))

    _apply_tp_sharding_opgraph(graph, shard, h, h_attn, h_kv, ffn, seq, act_bytes)


def _insert_cp_collectives_opgraph(
    graph: "OpGraph", shard: ShardPlan, model: ModelSpec,
    comm_nodes: list[tuple[str, "OpNode", str]], comm_counter: list[int],
) -> None:
    """Insert CP collectives into the OpGraph."""
    from zrt.ir.node import OpNode

    h = model.hidden
    if shard.tp > 1:
        h = h // shard.tp
    act_bytes = model.effective_attn_act_dtype().bytes
    cp = shard.cp

    if shard.cp_kind == CPKind.NONE:
        _apply_cp_sharding_opgraph(graph, shard, model.seq_len, model.hidden)
        return

    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue

        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if spec_kind not in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
            continue

        if shard.cp_kind == CPKind.ULYSSES:
            a2a_bytes = (model.seq_len // cp) * h * act_bytes
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"a2a_fwd_before_{node.id}", "A2A", "CP", a2a_bytes, comm_counter[0], "fwd",
                inserted_before=node.id), "before"))
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"a2a_fwd_after_{node.id}", "A2A", "CP", a2a_bytes, comm_counter[0], "fwd",
                inserted_after=node.id), "after"))
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"a2a_bwd_before_{node.id}", "A2A", "CP", a2a_bytes, comm_counter[0], "bwd",
                inserted_before=node.id), "before"))
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"a2a_bwd_after_{node.id}", "A2A", "CP", a2a_bytes, comm_counter[0], "bwd",
                inserted_after=node.id), "after"))

        elif shard.cp_kind == CPKind.RING:
            p2p_bytes = (model.seq_len // cp) * h * act_bytes
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"p2p_ring_fwd_{node.id}", "P2P", "CP", p2p_bytes, comm_counter[0], "fwd",
                rounds=cp, overlap=True, inserted_before=node.id), "before"))
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"p2p_ring_bwd_{node.id}", "P2P", "CP", p2p_bytes, comm_counter[0], "bwd",
                rounds=cp, overlap=True, inserted_before=node.id), "before"))

    _apply_cp_sharding_opgraph(graph, shard, model.seq_len, model.hidden)


def _insert_ep_collectives_opgraph(
    graph: "OpGraph", shard: ShardPlan, model: ModelSpec, strategy: Strategy,
    comm_nodes: list[tuple[str, "OpNode", str]], comm_counter: list[int],
) -> None:
    """Insert EP collectives (A2A pairs for expert parallel) into the OpGraph."""
    from zrt.ir.node import OpNode

    h = model.hidden
    if shard.tp > 1:
        h = h // shard.tp
    seq = model.seq_len
    if shard.cp > 1:
        seq = seq // shard.cp
    moe_ffn = model.moe_ffn if model.moe_ffn > 0 else model.ffn
    act_bytes = model.effective_moe_act_dtype().bytes

    micro_batch = strategy.micro_batch
    topk = model.top_k
    a2a_bytes = micro_batch * seq * h * topk * act_bytes

    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue

        layer_id = int(node.layer) if node.layer and node.layer.isdigit() else -1
        if layer_id < 0 or layer_id >= len(model.layers):
            continue
        if model.layers[layer_id].value != "moe":
            continue

        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if spec_kind == "matmul" and "routed_expert" in node.id:
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"a2a_before_{node.id}", "A2A", "EP", a2a_bytes, comm_counter[0], "both",
                inserted_before=node.id), "before"))
            comm_counter[0] += 1
            comm_nodes.append((node.id, _make_comm_node_opgraph(
                f"a2a_after_{node.id}", "A2A", "EP", a2a_bytes, comm_counter[0], "both",
                inserted_after=node.id), "after"))

    _apply_ep_sharding_opgraph(graph, shard, model.num_experts, model.top_k)


def _make_comm_node_opgraph(
    name: str, kind: str, group: str, bytes_: int, counter: int, phase: str,
    rounds: int = 1, overlap: bool = False, inserted_after: str = None,
    inserted_before: str = None,
) -> "OpNode":
    """Create a comm OpNode."""
    from zrt.ir.node import OpNode

    _KIND_TO_COMM_OP = {
        "AG": "comm.all_gather",
        "RS": "comm.reduce_scatter",
        "AR": "comm.all_reduce",
        "A2A": "comm.all_to_all",
        "P2P": "comm.send_recv",
    }
    op_type = _KIND_TO_COMM_OP.get(kind, f"comm.{kind.lower()}")

    attrs = {
        "comm_kind": kind,
        "comm_group": group,
        "comm_bytes": bytes_,
        "comm_rounds": rounds,
        "comm_overlap": overlap,
        "comm_phase": phase,
        "spec_kind": "comm",
    }
    if inserted_after:
        attrs["inserted_after"] = inserted_after
    if inserted_before:
        attrs["inserted_before"] = inserted_before

    return OpNode(
        id=f"comm_{counter}",
        op_type=op_type,
        attrs=attrs,
        scope="",
        category="communication",
        component="comm",
        layer="",
        name=name,
    )


def _apply_tp_sharding_opgraph(
    graph: "OpGraph", shard: ShardPlan,
    h: int, h_attn: int, h_kv: int, ffn: int, seq: int, act_bytes: int,
) -> None:
    """Adjust tensor shape_local for TP sharding on OpGraph nodes."""
    if shard.tp <= 1:
        return

    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue

        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if spec_kind == "matmul":
            m = node.attrs.get("m", 0)
            n = node.attrs.get("n", 0)
            k = node.attrs.get("k", 0)

            col_parallel = any(p in node.id for p in (
                "qkv", "q_a_proj", "q_b_proj", "kv_a_proj",
                "wq_a", "wq_b", "wkv",
                "up_proj", "gate_proj", "shared_up_proj", "shared_gate_proj",
                "comp_wkv", "comp_wgate",
                "idx_wq_b", "idx_weights", "idx_comp_wkv", "idx_comp_wgate",
            ))
            row_parallel = any(p in node.id for p in (
                "o_proj", "down_proj", "shared_down_proj",
                "wo_a", "wo_b", "kv_b_proj",
            ))

            if col_parallel:
                n_local = n // shard.tp
                updated = False
                for t in node.outputs:
                    if t.shape and t.shape[-1] == n:
                        t.shape_local = (t.shape[0], n_local)
                        updated = True
                if not updated:
                    node.attrs["n_local"] = n_local
            elif row_parallel:
                k_local = k // shard.tp
                updated = False
                for t in node.inputs:
                    if t.shape and t.shape[-1] == k:
                        t.shape_local = (t.shape[0], k_local)
                        updated = True
                if not updated:
                    node.attrs["k_local"] = k_local
            elif "router" in node.id:
                n_local = n // shard.tp
                for t in node.outputs:
                    if t.shape and t.shape[-1] == n:
                        t.shape_local = (t.shape[0], n_local)
            elif "routed_expert" in node.id:
                k_local = k // shard.tp
                n_local = n // shard.tp
                node.attrs["k_local"] = k_local
                for t in node.inputs:
                    if t.shape and t.shape[-1] == k:
                        t.shape_local = (t.shape[0], k_local)
                for t in node.outputs:
                    if t.shape and t.shape[-1] == n:
                        t.shape_local = (t.shape[0], n_local)
        elif spec_kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
            if "heads" in node.attrs:
                heads_before_tp = node.attrs["heads"]
                node.attrs["heads"] = max(1, node.attrs["heads"] // shard.tp)
                node.attrs["heads_tp"] = node.attrs["heads"]
                node.attrs["heads_before_tp"] = heads_before_tp
            if "h_kv" in node.attrs:
                node.attrs["h_kv"] = max(1, node.attrs["h_kv"] // shard.tp)
            for t in node.inputs + node.outputs:
                if t.shape and t.shape[-1] in (h_attn, h_kv):
                    t.shape_local = (t.shape[0], max(1, t.shape[-1] // shard.tp))
        elif spec_kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            _shard_hc_sequence_opgraph(node, shard.tp, seq)
        elif spec_kind == "indexer_topk":
            ih = node.attrs.get("ih", 0)
            if ih > 0:
                ih_local = max(1, ih // shard.tp)
                node.attrs["ih_local"] = ih_local
                node.attrs["world_factor"] = shard.tp
                id_ = node.attrs.get("id", 0)
                for t in node.inputs:
                    if "idx_q" in t.id and t.shape:
                        t.shape_local = (t.shape[0], ih_local * id_)
                    elif "idx_w" in t.id and t.shape:
                        t.shape_local = (t.shape[0], ih_local)
                kv_len = node.attrs.get("kv_len", node.attrs.get("s", 0))
                idx_q_bytes = node.attrs.get("s", 0) * ih_local * id_ * 2
                idx_kv_bytes = kv_len * id_ * 2
                idx_w_bytes = node.attrs.get("s", 0) * ih_local * 2
                idx_out_bytes = node.attrs.get("s", 0) * node.attrs.get("topk", 0) * 2
                node.attrs["bytes_fwd"] = idx_q_bytes + idx_kv_bytes + idx_w_bytes + idx_out_bytes
        elif spec_kind == "lm_head":
            n = node.attrs.get("n", 0)
            if n > 0:
                n_local = n // shard.tp
                node.attrs["n_local"] = n_local
                for t in node.outputs:
                    if t.shape and t.shape[-1] == n:
                        t.shape_local = (t.shape[0], n_local)
            if "bytes_fwd" in node.attrs:
                node.attrs["bytes_fwd"] = int(node.attrs["bytes_fwd"]) // shard.tp
        elif spec_kind == "compressor_pool":
            d = node.attrs.get("d", 0)
            if d > 0:
                d_local = max(1, d // shard.tp)
                node.attrs["d_local"] = d_local
                node.attrs["world_factor"] = shard.tp
                if "bytes_fwd" in node.attrs:
                    node.attrs["bytes_fwd"] = node.attrs["bytes_fwd"] // shard.tp
        elif spec_kind in ("ln", "rmsnorm", "rope", "swiglu", "add"):
            for t in node.inputs + node.outputs:
                if t.shape and t.shape[-1] in (h, h_attn, h_kv, ffn):
                    t.shape_local = (t.shape[0], max(1, t.shape[-1] // shard.tp))


def _apply_cp_sharding_opgraph(
    graph: "OpGraph", shard: ShardPlan, seq: int, hidden: int,
) -> None:
    """Adjust tensor shape_local for CP sharding on OpGraph nodes."""
    if not shard.has_cp:
        return

    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue

        for t in node.inputs + node.outputs:
            if t.shape and len(t.shape) > 0:
                if t.shape[0] == seq:
                    t.shape_local = (max(1, (t.shape_local[0] if t.shape_local else t.shape[0]) // shard.cp),) + (t.shape_local[1:] if t.shape_local and len(t.shape_local) > 1 else t.shape[1:])

        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if spec_kind == "matmul":
            meta_k = node.attrs.get("k", 0)
            if "m" in node.attrs and meta_k > 0 and node.inputs and node.inputs[0].shape and node.inputs[0].shape[-1] != meta_k:
                node.attrs["m"] = node.attrs["m"] // shard.cp
        elif spec_kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
            heads_tp = node.attrs.get("heads_tp", node.attrs.get("heads", 0))

            if shard.cp_kind == CPKind.ULYSSES:
                node.attrs["heads"] = max(1, heads_tp // shard.cp)
                node.attrs["heads_gathered_by_cp"] = False
            elif shard.cp_kind == CPKind.HYBRID:
                if "s" in node.attrs:
                    node.attrs["s"] = node.attrs["s"] // shard.cp
                node.attrs["heads"] = max(1, heads_tp // shard.cp)
                node.attrs["heads_gathered_by_cp"] = False
                node.attrs["cp_tiles"] = shard.cp
            elif shard.cp_kind == CPKind.RING:
                if "s" in node.attrs:
                    node.attrs["s"] = node.attrs["s"] // shard.cp
                node.attrs["heads_gathered_by_cp"] = False
                node.attrs["cp_tiles"] = shard.cp
            else:
                if "s" in node.attrs:
                    node.attrs["s"] = node.attrs["s"] // shard.cp
        elif spec_kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            if "s" in node.attrs and node.attrs["s"] > 0:
                node.attrs["s"] = max(1, node.attrs["s"] // shard.cp)
            if "bytes_fwd" in node.attrs:
                node.attrs["bytes_fwd"] = max(1, int(node.attrs["bytes_fwd"]) // shard.cp)
        elif spec_kind == "compressor_pool":
            s = node.attrs.get("s", 0)
            if s > 0:
                node.attrs["s"] = s // shard.cp
                node.attrs["world_factor"] = shard.cp
            if "bytes_fwd" in node.attrs:
                node.attrs["bytes_fwd"] = int(node.attrs["bytes_fwd"]) // shard.cp
        elif spec_kind == "indexer_topk":
            s = node.attrs.get("s", 0)
            if s > 0:
                node.attrs["s"] = s // shard.cp
                node.attrs["world_factor"] = shard.cp
            kv_len = node.attrs.get("kv_len", 0)
            id_ = node.attrs.get("id", 0)
            topk = node.attrs.get("topk", 0)
            ih = node.attrs.get("ih_local", node.attrs.get("ih", 0))
            s_local = s // shard.cp if s > 0 else 0
            idx_q_bytes = s_local * ih * id_ * 2
            idx_kv_bytes = kv_len * id_ * 2
            idx_w_bytes = s_local * ih * 2
            idx_out_bytes = s_local * topk * 2
            node.attrs["bytes_fwd"] = idx_q_bytes + idx_kv_bytes + idx_w_bytes + idx_out_bytes


def _apply_ep_sharding_opgraph(
    graph: "OpGraph", shard: ShardPlan,
    num_experts: int, top_k: int,
) -> None:
    """Adjust tensor shape_local for EP sharding on OpGraph nodes."""
    if not shard.has_ep:
        return

    experts_per_rank = num_experts // shard.ep

    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue

        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if spec_kind == "matmul" and "router" in node.id:
            for t in node.outputs:
                if len(t.shape) > 1:
                    if t.shape[1] == num_experts:
                        t.shape_local = (t.shape[0], experts_per_rank)


def _shard_hc_sequence_opgraph(node: "OpNode", factor: int, seq: int) -> None:
    """Shard Hyper-Connection token dimension by an additional factor (OpGraph version)."""
    if factor <= 1:
        return
    if "s" in node.attrs and node.attrs["s"] > 0:
        node.attrs["s"] = max(1, node.attrs["s"] // factor)
    if "bytes_fwd" in node.attrs:
        node.attrs["bytes_fwd"] = max(1, int(node.attrs["bytes_fwd"]) // factor)
    for t in node.inputs + node.outputs:
        if t.shape_local:
            t.shape_local = (max(1, t.shape_local[0] // factor),) + t.shape_local[1:]


def _apply_global_hc_sharding_opgraph(graph: "OpGraph", strategy: Strategy, seq: int) -> None:
    """Apply TP/CP token sharding to HC ops outside layer_index (OpGraph version)."""
    factor = max(1, strategy.tp) * max(1, strategy.cp)
    if factor <= 1:
        return
    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue
        layer_id = int(node.layer) if node.layer and node.layer.isdigit() else -1
        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if layer_id < 0 and spec_kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            _shard_hc_sequence_opgraph(node, factor, seq)


def _apply_global_token_op_sharding_opgraph(graph: "OpGraph", strategy: Strategy) -> None:
    """Apply CP token sharding to non-lm_head global token ops (OpGraph version)."""
    factor = max(1, strategy.cp)
    if factor <= 1:
        return
    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue
        layer_id = int(node.layer) if node.layer and node.layer.isdigit() else -1
        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if layer_id < 0 and spec_kind in ("embed", "ln", "rmsnorm"):
            _shard_global_token_sequence_opgraph(node, factor)


def _shard_global_token_sequence_opgraph(node: "OpNode", factor: int) -> None:
    """Shard the leading token dimension for global sequence-local ops (OpGraph version)."""
    if factor <= 1:
        return
    if "m" in node.attrs and node.attrs["m"] > 0:
        node.attrs["m"] = max(1, node.attrs["m"] // factor)
    if "s" in node.attrs and node.attrs["s"] > 0:
        node.attrs["s"] = max(1, node.attrs["s"] // factor)
    if "bytes_fwd" in node.attrs:
        node.attrs["bytes_fwd"] = max(1, int(node.attrs["bytes_fwd"]) // factor)
    for t in node.inputs + node.outputs:
        if t.shape_local:
            t.shape_local = (max(1, t.shape_local[0] // factor),) + t.shape_local[1:]


def _apply_global_lm_head_sharding_opgraph(graph: "OpGraph", strategy: Strategy) -> None:
    """Apply TP vocab and CP sequence sharding to global lm_head (OpGraph version)."""
    tp = max(1, strategy.tp)
    cp = max(1, strategy.cp)
    if tp <= 1 and cp <= 1:
        return
    for node in graph.nodes.values():
        if node.op_type.startswith("comm."):
            continue
        layer_id = int(node.layer) if node.layer and node.layer.isdigit() else -1
        spec_kind = node.attrs.get("spec_kind", node.op_type)
        if layer_id >= 0 or spec_kind != "lm_head":
            continue
        m = node.attrs.get("m", 0)
        n = node.attrs.get("n", 0)
        if cp > 1 and m > 0:
            node.attrs["m"] = max(1, m // cp)
            for t in node.inputs + node.outputs:
                if t.shape and t.shape[0] == m:
                    t.shape_local = (max(1, (t.shape_local[0] if t.shape_local else t.shape[0]) // cp),) + (t.shape_local[1:] if t.shape_local and len(t.shape_local) > 1 else t.shape[1:])
        if tp > 1 and n > 0:
            n_local = max(1, n // tp)
            node.attrs["n_local"] = n_local
            for t in node.outputs:
                if t.shape and t.shape[-1] == n:
                    t.shape_local = (t.shape_local[:-1] if t.shape_local else t.shape[:-1]) + (n_local,)
