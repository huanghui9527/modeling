"""CaptureConfig: parameters for real HF model graph capture.

Used by Path A (抓图建模) to configure ``build_captured_graph()`` when
a real HuggingFace model should be traced instead of using the spec-driven
fallback.

Usage::

    from zrt.training.spec.capture_config import CaptureConfig

    capture = CaptureConfig(model_id="meta-llama/Llama-3-70B")
    report = estimate_via_pipeline(model, system, strategy, capture=capture)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CaptureConfig:
    """Configuration for real HF model graph capture.

    Parameters
    ----------
    model_id : str
        HuggingFace Hub ID or local path to the model.
    num_layers : int
        Number of transformer layers to instantiate (default: 4).
    seq_len : int
        Input sequence length for tracing (default: 128).
    batch_size : int
        Input batch size for tracing (default: 1).
    target_layers : list[int] | None
        Specific layer indices to capture. None = auto-infer first dense + first sparse.
    gradient_checkpointing : bool
        Whether to enable gradient checkpointing during capture.
    graph_mode : bool
        Whether to use torch.compile-based graph capture instead of TorchDispatchMode.
    """
    model_id: str
    num_layers: int = 4
    seq_len: int = 128
    batch_size: int = 1
    target_layers: Optional[list[int]] = None
    gradient_checkpointing: bool = False
    graph_mode: bool = False

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
