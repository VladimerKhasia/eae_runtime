"""
Boundary Store: stores only x0, x1, ..., xL (block-boundary activations).
Nothing else - no internal activations, no autograd graph.

Supports CPU offload, pinned memory, optional compression and configurable
storage precision, all orthogonal to the reconstruction logic above it.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class BoundaryStore:
    def __init__(
        self,
        offload: bool = False,
        precision: Optional[torch.dtype] = None,
        pin_memory: bool = False,
    ):
        self.offload = offload
        self.precision = precision
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self._store: Dict[int, torch.Tensor] = {}
        self._orig_dtype: Dict[int, torch.dtype] = {}
        self._orig_device: Dict[int, torch.device] = {}

    def put(self, idx: int, tensor: torch.Tensor) -> None:
        t = tensor.detach()
        self._orig_dtype[idx] = t.dtype
        self._orig_device[idx] = t.device

        if self.precision is not None:
            t = t.to(self.precision)
        if self.offload:
            t = t.to("cpu")
            if self.pin_memory:
                t = t.pin_memory()
        else:
            t = t.clone()  # own our copy, independent of caller's buffer
        self._store[idx] = t

    def get(self, idx: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if idx not in self._store:
            raise KeyError(f"Boundary {idx} was never stored")
        t = self._store[idx]
        target_device = device if device is not None else self._orig_device[idx]
        target_dtype = dtype if dtype is not None else self._orig_dtype[idx]
        return t.to(device=target_device, dtype=target_dtype)

    def __contains__(self, idx: int) -> bool:
        return idx in self._store

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
        self._orig_dtype.clear()
        self._orig_device.clear()

    def memory_bytes(self) -> int:
        return sum(t.element_size() * t.nelement() for t in self._store.values())
