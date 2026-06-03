"""ZRT-Sim FastAPI service.

Wraps the three CLI modes (graph capture, spec estimate, grid search) as
async background jobs with a simple in-memory job store.

Launch (from project root):
    uvicorn server.main:app --host 0.0.0.0 --port 8000

    # with auto-reload during development:
    uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

Poll a job:
    GET /jobs/{job_id}

Interactive docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import EstimateRequest, JobResponse, JobStatus, SearchRequest, TraceRequest
from . import stats

# Ensure 'python/' is on sys.path so that zrt.* imports inside the training
# module (which uses 'from zrt.*') resolve correctly.
_python_dir = str(Path(__file__).parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

# Matches CLI output directory: output/estimate/{config_slug}_{timestamp}.html
_estimates_dir = Path(__file__).parent.parent / "output" / "estimate"
_estimates_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ZRT-Sim API",
    description="LLM performance modelling and simulation service.",
    version="1.0.0",
)

# Allow requests from file:// (origin=null) and any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_origin_regex=r"https?://localhost(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated HTML reports (mirrors CLI output/estimate/ directory)
app.mount("/estimate", StaticFiles(directory=str(_estimates_dir)), name="estimate")

_launcher = Path(__file__).parent / "launcher.html"


@app.get("/", include_in_schema=False)
def serve_launcher():
    return FileResponse(_launcher, media_type="text/html")

# ── In-memory job store ───────────────────────────────────────────────────────
# Each entry: {id, status, result, error, created_at, finished_at}
_jobs: dict[str, dict[str, Any]] = {}
# Per-search live state (model/system/strategies) for lazy detail rendering.
# Kept out of _jobs because Strategy objects aren't JSON-serializable and
# would break /jobs listing.
_search_states: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
    return job_id


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        _jobs[job_id].update(kwargs)


def _snapshot(job_id: str) -> dict:
    with _lock:
        return dict(_jobs[job_id])


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/health", tags=["utility"])
def health():
    return {"status": "ok"}



@app.get("/hardware", tags=["utility"], summary="List available hardware specs")
def list_hardware():
    from python.zrt.hardware.registry import list_available
    return {"hardware": list_available()}


@app.get("/hardware/{name}", tags=["utility"], summary="Get hardware spec details")
def get_hardware_spec(name: str):
    import yaml as _yaml
    cfg = Path(__file__).parent.parent / "python" / "zrt" / "hardware" / "configs" / f"{name}.yaml"
    if not cfg.exists():
        raise HTTPException(404, detail=f"Hardware '{name}' not found")
    data = _yaml.safe_load(cfg.read_text(encoding="utf-8"))
    compute = data.get("compute", {})
    memory  = data.get("memory",  {})
    intra   = data.get("interconnect", {}).get("intra_node", {})
    inter   = data.get("interconnect", {}).get("inter_node", {})
    return {
        "name":       data.get("name", name),
        "vendor":     data.get("vendor", ""),
        "bf16_tflops": compute.get("bf16_tflops"),
        "fp8_tops":    compute.get("fp8_tops"),
        "memory_gb":   memory.get("capacity_gb"),
        "hbm_gbps":    memory.get("hbm_bandwidth_gbps"),
        "intra_type":  intra.get("type", ""),
        "intra_bw":    intra.get("bandwidth_gbps"),
        "inter_type":  inter.get("type", ""),
        "inter_bw":    inter.get("bandwidth_gbps"),
    }


@app.get("/hardware/{name}/raw", tags=["utility"], summary="Get raw hardware YAML content")
def get_hardware_raw(name: str):
    cfg = Path(__file__).parent.parent / "python" / "zrt" / "hardware" / "configs" / f"{name}.yaml"
    if not cfg.exists():
        raise HTTPException(404, detail=f"Hardware '{name}' not found")
    return {"content": cfg.read_text(encoding="utf-8")}


@app.get("/models", tags=["utility"], summary="List training model configs")
def list_models():
    import re
    import yaml as _yaml
    models_dir = Path(__file__).parent.parent / "python" / "zrt" / "training" / "configs" / "models"
    result = []
    for f in sorted(models_dir.glob("*.yaml")):
        try:
            d = _yaml.safe_load(f.read_text(encoding="utf-8"))
            # Parse layers string e.g. "[dense]*3+[moe]*57+[mtp]*1"
            layers_raw = d.get("layers", "")
            layer_counts: dict = {}
            if isinstance(layers_raw, str):
                for m in re.finditer(r'\[(\w+)\]\*(\d+)', layers_raw):
                    layer_counts[m.group(1)] = int(m.group(2))
            # Attention type
            kv_lora = d.get("kv_lora_rank")
            num_kv  = d.get("num_kv_heads", d.get("num_heads", 1))
            num_h   = d.get("num_heads", 1)
            if kv_lora:
                attn = "MLA"
            elif num_kv == 1:
                attn = "MQA"
            elif num_kv < num_h:
                attn = "GQA"
            else:
                attn = "MHA"
            result.append({
                "key":          f.stem,
                "name":         d.get("name", f.stem),
                "hidden":       d.get("hidden"),
                "ffn":          d.get("ffn") if not d.get("num_experts") else d.get("moe_ffn"),
                "num_heads":    num_h,
                "num_kv_heads": num_kv,
                "head_dim":     d.get("head_dim"),
                "attn":         attn,
                "vocab":        d.get("vocab"),
                "seq_len":      d.get("seq_len"),
                "layer_counts": layer_counts,
                "num_experts":  d.get("num_experts"),
                "top_k":        d.get("top_k"),
                "param_dtype":  d.get("param_dtype"),
                "is_moe":       bool(d.get("num_experts")),
            })
        except Exception:
            result.append({"key": f.stem, "name": f.stem})
    return {"models": result}


@app.get("/models/{key}/raw", tags=["utility"], summary="Get raw model spec YAML content")
def get_model_raw(key: str):
    cfg = Path(__file__).parent.parent / "python" / "zrt" / "training" / "configs" / "models" / f"{key}.yaml"
    if not cfg.exists():
        raise HTTPException(404, detail=f"Model '{key}' not found")
    return {"content": cfg.read_text(encoding="utf-8")}


@app.get("/stats", tags=["utility"], summary="Aggregate task submission counts")
def get_stats():
    """Totals per task kind across all users (per-user detail lives in stats.json)."""
    return stats.read_totals()


# ── Job polling ───────────────────────────────────────────────────────────────

@app.get("/jobs", tags=["jobs"], summary="List all submitted jobs")
def list_jobs() -> List[dict]:
    with _lock:
        return list(_jobs.values())


@app.get(
    "/jobs/{job_id}",
    tags=["jobs"],
    response_model=JobResponse,
    summary="Poll job status and result",
)
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/jobs/{job_id}/artifacts/{filename}", tags=["jobs"], include_in_schema=False)
def get_job_artifact(job_id: str, filename: str):
    artifact = _resolve_job_artifact(job_id, filename)
    if artifact.suffix == ".xlsx":
        return FileResponse(
            artifact,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=artifact.name,
        )
    return FileResponse(artifact, media_type="text/html")


def _resolve_job_artifact(job_id: str, filename: str) -> Path:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail=f"Job '{job_id}' not found")
    result = job.get("result") or {}
    candidates: dict[str | None, str | None] = {
        result.get("html_filename"): result.get("html_path"),
        result.get("excel_filename"): result.get("excel_path"),
    }
    # Search jobs publish a per-config detail HTML for each Pareto entry; the
    # top-1 best is rendered eagerly, others on first click via /search/{id}/render/{idx}.
    for entry in (result.get("pareto_frontier") or []):
        fn = entry.get("html_filename")
        if fn:
            candidates[fn] = entry.get("html_path")
    path = candidates.get(filename)
    if not path:
        raise HTTPException(404, detail=f"Artifact '{filename}' not found")
    artifact = Path(path)
    if not artifact.exists() or artifact.name != filename:
        raise HTTPException(404, detail=f"Artifact '{filename}' not found")
    return artifact


# ── POST /trace ───────────────────────────────────────────────────────────────

@app.post(
    "/trace",
    tags=["jobs"],
    response_model=JobResponse,
    status_code=202,
    summary="Submit a graph-capture (+ optional perf modelling) job",
    description=(
        "Traces the operator sequence of an HF causal LM and optionally runs the "
        "inference or training performance modelling pipeline. "
        "Returns a job_id immediately; poll GET /jobs/{job_id} for completion."
    ),
)
def submit_trace(req: TraceRequest, bg: BackgroundTasks):
    job_id = _new_job()
    try:
        stats.record_submission(req.username, "trace")
    except Exception:
        pass
    bg.add_task(_trace_worker, job_id, req)
    return _snapshot(job_id)


def _trace_worker(job_id: str, req: TraceRequest) -> None:
    _update_job(job_id, status=JobStatus.RUNNING)
    try:
        result = _do_trace(req)
        _update_job(
            job_id,
            status=JobStatus.DONE,
            result=result,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def _do_trace(req: TraceRequest) -> dict:
    from python.zrt.pipeline import run_trace_phases, _MODEL_DIRS

    # Resolve model_id: 'local:<shorthand>' → absolute hf_models/ path
    if req.model_id.startswith("local:"):
        shorthand = req.model_id[len("local:"):]
        if shorthand not in _MODEL_DIRS:
            raise ValueError(
                f"Unknown local model '{shorthand}'. "
                f"Available: {list(_MODEL_DIRS.keys())}"
            )
        model_id = str(
            Path(__file__).parent.parent / "hf_models" / _MODEL_DIRS[shorthand]
        )
    else:
        model_id = req.model_id

    phases = (
        ("train_forward", "train_backward")
        if req.train
        else tuple(req.phases or ["prefill", "decode"])
    )

    target_layers: Optional[List[int]] = None
    if req.target_layers:
        target_layers = [int(x.strip()) for x in req.target_layers.split(",")]

    # When target_layers is explicit, disable auto_layers
    auto_layers = req.auto_layers if target_layers is None else False

    out_dir = Path(req.output_dir) if req.output_dir else None

    trace_result = run_trace_phases(
        model_id=model_id,
        num_layers=req.layers,
        batch_size=req.batch_size,
        seq_len=req.seq_len,
        output_dir=out_dir,
        phases=phases,
        target_layers=target_layers,
        auto_layers=auto_layers,
        platform=req.platform,
        graph_mode=req.graph_mode,
        gradient_checkpointing=req.gradient_checkpointing,
    )

    result: dict[str, Any] = {
        "output_dir": str(trace_result.output_dir),
        "phases": list(trace_result.graphs.keys()),
        "summary": None,
    }

    if not req.hw:
        return result

    # Run perf modelling pipeline
    import python.zrt.hardware.registry as hw_registry
    from python.zrt.cli import _run_inference_pipeline, _run_training_modelling

    hw = hw_registry.load(req.hw)
    fake_args = SimpleNamespace(
        hw=req.hw,
        tp=req.tp,
        pp=req.pp,
        ep=req.ep,
        dp=req.dp,
        cp=req.cp,
        quant=req.quant,
        batch_size=req.batch_size,
        seq_len=req.seq_len,
        # Training extras (only used when req.train is True)
        total_params=req.total_params,
        hidden=req.hidden,
        layers=req.layers,
        num_layers_full=req.num_layers_full,
        zero_stage=req.zero_stage,
        optimizer=req.optimizer,
        muon_rotation=req.muon_rotation,
        muon_ns_steps=req.muon_ns_steps,
        micro_batch=req.micro_batch,
        global_batch=req.global_batch,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        if req.train:
            _run_training_modelling(fake_args, model_id, hw, trace_result)
        else:
            _run_inference_pipeline(fake_args, model_id, hw, trace_result)

    summary = buf.getvalue().strip()
    if summary:
        result["summary"] = summary

    return result


# ── POST /estimate ────────────────────────────────────────────────────────────

@app.post(
    "/estimate",
    tags=["jobs"],
    response_model=JobResponse,
    status_code=202,
    summary="Submit a spec-based training estimate job",
    description=(
        "Runs spec-based training estimation from a YAML config — no model weights "
        "or graph capture required. "
        "Provide either config_path (server-side file) or config_content (raw YAML)."
    ),
)
def submit_estimate(req: EstimateRequest, bg: BackgroundTasks):
    if not req.config_path and not req.config_content:
        raise HTTPException(422, detail="Provide either config_path or config_content")
    job_id = _new_job()
    try:
        stats.record_submission(req.username, "estimate")
    except Exception:
        pass
    bg.add_task(_estimate_worker, job_id, req)
    return _snapshot(job_id)


def _estimate_worker(job_id: str, req: EstimateRequest) -> None:
    _update_job(job_id, status=JobStatus.RUNNING)
    try:
        result = _do_estimate(req, job_id)
        _update_job(
            job_id,
            status=JobStatus.DONE,
            result=result,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def _do_estimate(req: EstimateRequest, job_id: str) -> dict:
    from python.zrt.training.io.config_loader import load_specs
    from python.zrt.training.ir.builders import build_graph
    from python.zrt.training.models.flops import op_cost
    from python.zrt.training.search.estimator import estimate
    from python.zrt.training.search.report import report_summary, report_to_dict
    from python.zrt.training.io.excel_exporter import export_estimate_excel
    from python.zrt.training.io.html_exporter import export_estimate_html

    config_path, tmp = _resolve_yaml(req.config_path, req.config_content)
    try:
        model, system, strategy, _capture = load_specs(config_path)

        # Build graph once so we can reuse it for op_costs and estimate
        graph = build_graph(model, strategy)

        op_costs: dict = {}
        for op in graph.ops:
            op_costs[op.name] = op_cost(op, model, system)

        report = estimate(model, system, strategy, graph=graph)

        # Mirror CLI naming: {config_slug}_{YYYYMMDD_HHMMSS}.html
        _slug = Path(req.config_path).stem if req.config_path else "estimate"
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(req.output_dir) if req.output_dir else _estimates_dir
        _base_name = f"{_slug}_{_ts}"
        _html_filename = f"{_base_name}.html"
        _excel_filename = f"{_base_name}.xlsx"
        excel_path = export_estimate_excel(
            report=report,
            graph=graph,
            model=model,
            system=system,
            strategy=strategy,
            op_costs=op_costs,
            output_path=out_dir / _excel_filename,
        )
        html_path = export_estimate_html(
            report=report,
            graph=graph,
            model=model,
            system=system,
            strategy=strategy,
            op_costs=op_costs,
            output_path=out_dir / _html_filename,
        )

        # ── Structured model detail ──────────────────────────────────────
        from collections import Counter
        layer_counts = dict(Counter(str(l).split(".")[-1].lower() for l in model.layers))
        is_mla = model.kv_lora_rank > 0
        is_mqa = model.num_kv_heads == 1 and not is_mla
        is_gqa = model.num_kv_heads < model.num_heads and not is_mla
        attn_type = "MLA" if is_mla else ("MQA" if is_mqa else ("GQA" if is_gqa else "MHA"))
        model_detail = {
            "hidden":       model.hidden,
            "ffn":          model.moe_ffn if model.num_experts else model.ffn,
            "num_heads":    model.num_heads,
            "num_kv_heads": model.num_kv_heads,
            "head_dim":     model.head_dim,
            "attn_type":    attn_type,
            "vocab":        model.vocab,
            "seq_len":      model.seq_len,
            "num_layers":   len(model.layers),
            "layer_counts": layer_counts,
            "num_experts":  model.num_experts,
            "top_k":        model.top_k,
            "is_moe":       bool(model.num_experts),
            "param_dtype":  str(model.param_dtype).split(".")[-1],
            "total_params": model.total_params(),
        }
        # ── Structured hardware detail ────────────────────────────────────
        gpu = system.gpu
        try:
            intra = system.interconnect.intra_node
            inter = system.interconnect.inter_node
            intra_info = f"{intra.type}  {intra.bandwidth_gbps} GB/s"
            inter_info = f"{inter.type}  {inter.bandwidth_gbps} GB/s"
        except Exception:
            intra_info, inter_info = "", ""
        hw_detail = {
            "name":          gpu.name,
            "bf16_tflops":   gpu.flops_bf16,
            "fp8_tops":      gpu.flops_fp8,
            "hbm_gb":        gpu.hbm_gb,
            "hbm_bw_gbps":   gpu.hbm_bw_gbps,
            "nodes":         system.nodes,
            "gpus_per_node": system.gpus_per_node,
            "world_size":    system.world_size,
            "intra":         intra_info,
            "inter":         inter_info,
        }
        # ── Structured strategy detail ────────────────────────────────────
        strategy_detail = {
            "tp": strategy.tp, "cp": strategy.cp,
            "pp": strategy.pp, "ep": strategy.ep, "dp": strategy.dp,
            "zero_stage":      strategy.zero_stage,
            "micro_batch":     strategy.micro_batch,
            "global_batch":    strategy.global_batch,
            "num_microbatches": strategy.num_microbatches(),
            "optimizer":       str(strategy.optimizer).split(".")[-1],
            "tp_overlap":      str(strategy.tp_overlap).split(".")[-1],
            "pp_schedule":     str(strategy.pp_schedule).split(".")[-1],
        }

        return {
            "summary":         report_summary(report),
            "data":            report_to_dict(report),
            "html_filename":   _html_filename,
            "excel_filename":  _excel_filename,
            "html_url":        f"/jobs/{job_id}/artifacts/{_html_filename}",
            "excel_url":       f"/jobs/{job_id}/artifacts/{_excel_filename}",
            "html_path":       str(html_path),
            "excel_path":      str(excel_path),
            "model_detail":    model_detail,
            "hw_detail":       hw_detail,
            "strategy_detail": strategy_detail,
        }
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)


# ── POST /search ──────────────────────────────────────────────────────────────

@app.post(
    "/search",
    tags=["jobs"],
    response_model=JobResponse,
    status_code=202,
    summary="Submit a parallel strategy grid-search job",
    description=(
        "Grid-searches parallel strategies (TP/CP/PP/EP/DP/ZeRO/PPSched) for a "
        "training config and returns the Pareto-optimal frontier. "
        "Provide either config_path (server-side file) or config_content (raw YAML)."
    ),
)
def submit_search(req: SearchRequest, bg: BackgroundTasks):
    if not req.config_path and not req.config_content:
        raise HTTPException(422, detail="Provide either config_path or config_content")
    job_id = _new_job()
    try:
        stats.record_submission(req.username, "search")
    except Exception:
        pass
    bg.add_task(_search_worker, job_id, req)
    return _snapshot(job_id)


def _search_worker(job_id: str, req: SearchRequest) -> None:
    _update_job(job_id, status=JobStatus.RUNNING)
    try:
        result = _do_search(req, job_id)
        _update_job(
            job_id,
            status=JobStatus.DONE,
            result=result,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def _do_search(req: SearchRequest, job_id: str) -> dict:
    from python.zrt.training.io.config_loader import load_specs
    from python.zrt.training.search.report import report_to_dict

    config_path, tmp = _resolve_yaml(req.config_path, req.config_content)
    try:
        model, system, strategy, _capture = load_specs(config_path)
        space = _build_search_space(req, strategy)
        # Pair each surviving (strategy, report) so we can render any Pareto
        # entry later. The stock grid_search() drops the originating strategy.
        all_pairs = _grid_search_with_strategies(model, system, space)
        if req.search_mode == "pareto":
            frontier_pairs = _pareto_frontier_with_strategies(all_pairs)
        else:
            frontier_pairs = _filter_sort_with_strategies(
                all_pairs, req.filters, req.sort_by, req.sort_desc, req.top_n,
            )

        _slug = Path(req.config_path).stem if req.config_path else "search"
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(req.output_dir) if req.output_dir else _estimates_dir
        base_name = f"{_slug}_search_{_ts}"

        pareto_data: list[dict] = []
        for idx, (strat, report) in enumerate(frontier_pairs):
            d = report_to_dict(report)
            d["idx"] = idx
            d["strategy_params"] = _strategy_to_params(strat)
            if idx == 0:
                # Render the best config synchronously so the user can click
                # straight through. Others render on first click.
                try:
                    html_path = _render_strategy_detail(
                        model=model, system=system, strategy=strat,
                        slug=f"{base_name}_top1",
                        output_dir=out_dir,
                    )
                    d["html_filename"] = html_path.name
                    d["html_path"] = str(html_path)
                    d["html_url"] = f"/jobs/{job_id}/artifacts/{html_path.name}"
                except Exception as exc:
                    d["render_error"] = str(exc)
            pareto_data.append(d)

        if req.output and pareto_data:
            out = Path(req.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(pareto_data, indent=2))

        # Stash search state in a dedicated dict for lazy detail rendering.
        with _lock:
            _search_states[job_id] = {
                "model": model,
                "system": system,
                "frontier_strategies": [p[0] for p in frontier_pairs],
                "output_dir": out_dir,
                "base_name": base_name,
            }

        return {
            "total_configs": len(all_pairs),
            "pareto_count": len(frontier_pairs),
            "pareto_frontier": pareto_data,
            "mode": req.search_mode,
            "sort_by": req.sort_by,
            "sort_desc": req.sort_desc,
            "filters": [f.model_dump() for f in (req.filters or [])],
        }
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)


# ── /search helpers ───────────────────────────────────────────────────────────

def _build_search_space(req: SearchRequest, strategy: Any) -> Any:
    """Translate SearchRequest overrides into a SearchSpace.

    Anything left unset on the request keeps the SearchSpace default. Numeric
    lists are forwarded verbatim; enum-valued lists (pp_schedules, optimizer
    kinds) are mapped through their constructors.
    """
    from python.zrt.training.search.space import SearchSpace
    from python.zrt.training.spec.strategy import OptKind, PPSched

    kwargs: dict[str, Any] = {
        "micro_batch": req.micro_batch if req.micro_batch is not None else strategy.micro_batch,
        "global_batch": req.global_batch if req.global_batch is not None else strategy.global_batch,
    }
    # Pure-int list overrides — same key on both sides.
    for key in ("tp_values", "cp_values", "pp_values", "ep_values", "dp_values",
                "zero_stages", "vpp_chunks_values"):
        val = getattr(req, key, None)
        if val:
            kwargs[key] = val
    if req.max_memory_gb is not None:
        kwargs["max_memory_gb"] = req.max_memory_gb
    if req.pp_schedules:
        kwargs["pp_schedules"] = [PPSched(s) for s in req.pp_schedules]
    if req.recompute_policies:
        kwargs["recompute_policies"] = req.recompute_policies
    if req.optimizer_values:
        kwargs["optimizer_values"] = [OptKind(s) for s in req.optimizer_values]
    return SearchSpace(**kwargs)


def _grid_search_with_strategies(model: Any, system: Any, space: Any) -> list[tuple[Any, Any]]:
    """grid_search variant that keeps each report paired with its strategy.

    Mirrors search.estimator.grid_search but returns (strategy, report) tuples
    so the launcher can re-render any frontier entry later.
    """
    from python.zrt.training.search.estimator import estimate

    pairs: list[tuple[Any, Any]] = []
    for strat in space.strategies(system.world_size):
        try:
            strat.validate(model, system)
        except ValueError:
            continue
        try:
            report = estimate(model, system, strat)
            if report.memory is not None:
                peak_gb = report.memory.peak_overall / 1e9
                if peak_gb > space.max_memory_gb:
                    continue
            pairs.append((strat, report))
        except Exception:
            continue
    pairs.sort(key=lambda p: p[1].step_time_ms)
    return pairs


def _filter_sort_with_strategies(
    pairs: list[tuple[Any, Any]],
    filters: Any,
    sort_by: str,
    descending: bool,
    top_n: int,
) -> list[tuple[Any, Any]]:
    """Constraint-filter + single-metric ranking (filter_sort mode).

    Reuses the shared predicate/sort from `search.metric_filters` so the HTTP
    path and the offline `training_search_util` script agree on semantics.
    Memory feasibility (`max_memory_gb`) is already applied upstream in
    `_grid_search_with_strategies`.
    """
    from python.zrt.training.search.metric_filters import (
        report_passes_filters, sort_reports_with_strategies,
    )
    flt = [f.model_dump() if hasattr(f, "model_dump") else dict(f) for f in (filters or [])]
    kept = [p for p in pairs if report_passes_filters(p[1], flt)]
    kept = sort_reports_with_strategies(
        kept, sort_by=sort_by or "tokens_per_sec", ascending=not descending,
    )
    if top_n and top_n > 0:
        kept = kept[:top_n]
    return kept


def _pareto_frontier_with_strategies(pairs: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    if not pairs:
        return []
    sorted_pairs = sorted(
        pairs,
        key=lambda p: (
            p[1].step_time_ms,
            p[1].memory.total / 1e9 if p[1].memory else float("inf"),
        ),
    )
    out: list[tuple[Any, Any]] = []
    min_mem = float("inf")
    for strat, rep in sorted_pairs:
        mem_gb = rep.memory.total / 1e9 if rep.memory else None
        if not out:
            out.append((strat, rep))
            min_mem = mem_gb if mem_gb is not None else float("inf")
        elif mem_gb is not None and mem_gb < min_mem:
            out.append((strat, rep))
            min_mem = mem_gb
    return out


def _strategy_to_params(strat: Any) -> dict:
    """Surface the few fields the UI shows alongside the strategy string."""
    def _enum_name(v: Any) -> Any:
        return getattr(v, "name", v) if v is not None else None
    return {
        "tp": strat.tp,
        "cp": strat.cp,
        "pp": strat.pp,
        "ep": strat.ep,
        "dp": strat.dp,
        "zero_stage": strat.zero_stage,
        "vpp_chunks": getattr(strat, "vpp_chunks", 1),
        "pp_schedule": _enum_name(getattr(strat, "pp_schedule", None)),
        "optimizer": _enum_name(getattr(strat, "optimizer", None)),
        "recompute": (
            "selective" if getattr(strat.recompute, "per_layer", None) and
            any("attn" in v for v in strat.recompute.per_layer.values())
            else "full" if getattr(strat.recompute, "per_layer", None) and
            any("full" in v for v in strat.recompute.per_layer.values())
            else "none"
        ),
    }


def _render_strategy_detail(*, model: Any, system: Any, strategy: Any,
                            slug: str, output_dir: Path) -> Path:
    """Reuse the /estimate HTML exporter to render a single (model, system, strategy)
    triple. Returns the path to the produced HTML file."""
    from python.zrt.training.ir.builders import build_graph
    from python.zrt.training.models.flops import op_cost
    from python.zrt.training.search.estimator import estimate
    from python.zrt.training.io.html_exporter import export_estimate_html

    graph = build_graph(model, strategy)
    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    report = estimate(model, system, strategy, graph=graph)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = export_estimate_html(
        report=report,
        graph=graph,
        model=model,
        system=system,
        strategy=strategy,
        op_costs=op_costs,
        output_path=output_dir / f"{slug}.html",
    )
    return Path(html_path)


@app.post(
    "/search/{job_id}/render/{idx}",
    tags=["jobs"],
    summary="Render a Pareto-frontier entry's detail HTML on demand",
    description=(
        "Generates the same HTML page as /estimate for the idx-th Pareto entry of "
        "a completed /search job. Idempotent: returns the cached URL if already rendered. "
        "Top-1 (idx=0) is auto-rendered at search completion, so this is normally "
        "used for the rest of the frontier."
    ),
)
def render_search_entry(job_id: str, idx: int):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail=f"Job '{job_id}' not found")
    if job.get("status") != JobStatus.DONE:
        raise HTTPException(409, detail=f"Job is {job.get('status')}, not done")
    result = job.get("result") or {}
    frontier = result.get("pareto_frontier") or []
    if not (0 <= idx < len(frontier)):
        raise HTTPException(404, detail=f"idx {idx} out of range (0..{len(frontier)-1})")
    entry = frontier[idx]
    if entry.get("html_url"):
        return {
            "html_filename": entry["html_filename"],
            "html_url": entry["html_url"],
            "cached": True,
        }

    with _lock:
        state = _search_states.get(job_id)
    if not state:
        raise HTTPException(410, detail="Search state expired (server restarted?). Re-run /search.")

    strategies = state["frontier_strategies"]
    if idx >= len(strategies):
        raise HTTPException(404, detail=f"idx {idx} has no stored strategy")
    try:
        html_path = _render_strategy_detail(
            model=state["model"],
            system=state["system"],
            strategy=strategies[idx],
            slug=f"{state['base_name']}_idx{idx:03d}",
            output_dir=state["output_dir"],
        )
    except Exception as exc:
        raise HTTPException(500, detail=f"Render failed: {exc}")

    html_filename = html_path.name
    html_url = f"/jobs/{job_id}/artifacts/{html_filename}"
    with _lock:
        # Re-read inside the lock in case the job was updated concurrently.
        current_result = _jobs[job_id].get("result") or {}
        current_frontier = current_result.get("pareto_frontier") or []
        if 0 <= idx < len(current_frontier):
            current_frontier[idx]["html_filename"] = html_filename
            current_frontier[idx]["html_path"] = str(html_path)
            current_frontier[idx]["html_url"] = html_url
            current_result["pareto_frontier"] = current_frontier
            _jobs[job_id]["result"] = current_result
    return {"html_filename": html_filename, "html_url": html_url, "cached": False}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve_yaml(
    config_path: Optional[str],
    config_content: Optional[str],
) -> tuple[str, Optional[str]]:
    """Return (path_to_use, tmp_path_to_delete).

    If config_path is given, use it directly (tmp_path is None).
    Otherwise write config_content to a temp file.
    """
    if config_path:
        return config_path, None
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(config_content)
    tmp.close()
    return tmp.name, tmp.name
