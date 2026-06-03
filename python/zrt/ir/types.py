"""IR primitive types: DType (unified with spec Dtype) and TensorMeta."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# DType is now a re-export of the unified spec Dtype enum.
# All members (FP32, BF16, FP8_*, FP4, INT*, BOOL, UNKNOWN) and
# properties (bytes, itemsize, bits) live in spec/dtype.py.
from python.zrt.training.spec.dtype import Dtype as DType


# ── torch dtype string → DType ────────────────────────────────────────────────

_TORCH_TO_DTYPE: dict[str, DType] = {
    "torch.float32":  DType.FP32,
    "torch.float":    DType.FP32,
    "torch.float16":  DType.FP16,
    "torch.half":     DType.FP16,
    "torch.bfloat16": DType.BF16,
    "torch.float8_e4m3fn":  DType.FP8_E4M3,
    "torch.float8_e5m2":    DType.FP8_E5M2,
    "torch.int8":   DType.INT8,
    "torch.int32":  DType.INT32,
    "torch.int":    DType.INT32,
    "torch.int64":  DType.INT64,
    "torch.long":   DType.INT64,
    "torch.uint8":  DType.UINT8,
    "torch.bool":   DType.BOOL,
}


def dtype_from_torch(s: str) -> DType:
    """Map a torch dtype string (e.g. 'torch.bfloat16') to DType."""
    return _TORCH_TO_DTYPE.get(s.strip(), DType.UNKNOWN)


def dtype_from_str(s: str) -> DType:
    """Map a DType value string (e.g. 'bf16') or torch dtype string to DType.

    Tries the DType enum value first, then falls back to the torch dtype map.
    """
    s = s.strip()
    try:
        return DType(s)
    except ValueError:
        return dtype_from_torch(s)


# ─────────────────────────────────────────────────────────────────────────────
# Shape parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_shape(s: str) -> tuple[int, ...]:
    """Parse a single shape string such as '[1, 128, 7168]' → (1, 128, 7168).

    Returns an empty tuple for empty/scalar inputs.
    """
    s = s.strip().strip("[]")
    if not s:
        return ()
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        return ()


def split_shape_list(s: str) -> list[str]:
    """Split '[1, 128], [7168]' → ['[1, 128]', '[7168]'].

    Handles nested brackets correctly.
    """
    if not s:
        return []
    result: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                result.append(token)
            current = []
        else:
            current.append(ch)
    token = "".join(current).strip()
    if token:
        result.append(token)
    return result


def memory_bytes(shape: Sequence[int], dtype: DType) -> int:
    """Total memory in bytes for a tensor of the given shape and dtype."""
    if not shape:
        return int(math.ceil(dtype.itemsize))
    n = 1
    for d in shape:
        n *= d
    return int(math.ceil(n * dtype.itemsize))


# ─────────────────────────────────────────────────────────────────────────────
# TensorMeta
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TensorMeta:
    """Lightweight descriptor for a single tensor.

    ``shape`` holds the logical (pre-sharding) dimensions; ``shape_local``
    holds the per-rank dimensions after TP/CP/EP sharding.  When
    ``shape_local`` is ``None`` the tensor has not been sharded and
    ``effective_shape`` returns ``shape``.
    """
    id: str
    shape: tuple[int, ...]
    dtype: DType
    mem_bytes: int
    shape_local: tuple[int, ...] | None = None

    @property
    def effective_shape(self) -> tuple[int, ...]:
        return self.shape_local if self.shape_local is not None else self.shape

    def num_elements(self) -> int:
        n = 1
        for d in self.effective_shape:
            n *= d
        return n

    def nbytes(self) -> int:
        return self.num_elements() * self.dtype.bytes

    @classmethod
    def from_shape_dtype(cls, tensor_id: str,
                         shape: tuple[int, ...], dtype: DType,
                         shape_local: tuple[int, ...] | None = None) -> "TensorMeta":
        local = shape_local if shape_local is not None else shape
        return cls(
            id=tensor_id,
            shape=shape,
            dtype=dtype,
            mem_bytes=memory_bytes(local, dtype),
            shape_local=shape_local,
        )

    @classmethod
    def from_strings(cls, tensor_id: str,
                     shape_str: str, dtype_str: str) -> "TensorMeta":
        shape = parse_shape(shape_str)
        dtype = dtype_from_torch(dtype_str) if dtype_str else DType.UNKNOWN
        return cls.from_shape_dtype(tensor_id, shape, dtype)

    def with_shape(self, new_shape: tuple[int, ...]) -> "TensorMeta":
        return TensorMeta.from_shape_dtype(self.id, new_shape, self.dtype)

    def with_dtype(self, new_dtype: DType) -> "TensorMeta":
        return TensorMeta.from_shape_dtype(self.id, self.shape, new_dtype)

    def __repr__(self) -> str:
        shape_str = "×".join(str(d) for d in self.shape) if self.shape else "scalar"
        return f"TensorMeta({self.id}, {shape_str}, {self.dtype.value})"
