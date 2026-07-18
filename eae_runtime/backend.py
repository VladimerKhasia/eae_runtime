"""
Backend Manager: responsible for choosing the execution backend. Blocks can
override backend selection by carrying a `.eae_backend` attribute naming a
preferred backend.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch


class BackendManager:
    SUPPORTED = ("cpu", "cuda", "triton", "rocm", "auto")

    def __init__(self, backend: str = "auto"):
        if backend not in self.SUPPORTED:
            raise ValueError(f"Unknown backend '{backend}', choices: {self.SUPPORTED}")
        self.requested = backend
        self.resolved = self._resolve(backend)

    @staticmethod
    def _resolve(backend: str) -> str:
        if backend == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if backend == "cuda" and not torch.cuda.is_available():
            return "cpu"  # graceful fallback rather than a hard failure
        if backend in ("triton", "rocm"):
            # Triton/ROCm kernels are launched *through* CUDA devices in this
            # runtime; if no CUDA device is present we fall back to plain CPU
            # execution so the same code runs everywhere.
            return "cuda" if torch.cuda.is_available() else "cpu"
        return backend

    def device_for(self, block=None) -> torch.device:
        override = getattr(block, "eae_backend", None)
        backend = self._resolve(override) if override else self.resolved
        return torch.device(backend if backend in ("cpu", "cuda") else "cpu")

    @contextlib.contextmanager
    def autocast(self, enabled: bool = False, dtype: Optional[torch.dtype] = None):
        device_type = "cuda" if self.resolved == "cuda" else "cpu"
        if not enabled:
            yield
            return
        with torch.autocast(device_type=device_type, dtype=dtype or torch.float16, enabled=True):
            yield

    def __repr__(self) -> str:  # pragma: no cover
        return f"BackendManager(requested={self.requested!r}, resolved={self.resolved!r})"
