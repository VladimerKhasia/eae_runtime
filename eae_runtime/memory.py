"""
Memory Manager: owns temporary tensors, scratch buffers, reconstruction
workspace. Provides request()/release()/reuse() and exposes runtime
statistics. The allocator should minimize allocations, fragmentation and
synchronization.

Two policies are provided out of the box:
  * PoolMemoryPolicy  - buckets free tensors by (shape, dtype, device) and
                         reuses them, avoiding repeated cudaMalloc/alloc churn.
  * NullMemoryPolicy   - always allocates fresh / frees immediately; useful
                         as a correctness baseline and for leak tests.

Users can implement their own MemoryPolicy (Memory Policy API) without
touching reconstruction logic.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch

from .events import EventBus, EventType

PoolKey = Tuple[Tuple[int, ...], torch.dtype, str]


class MemoryPolicy:
    """Base class for pluggable memory policies."""

    def allocate(self, shape, dtype, device) -> torch.Tensor:
        raise NotImplementedError

    def free(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def stats(self) -> Dict[str, int]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class NullMemoryPolicy(MemoryPolicy):
    """No pooling: every request is a fresh allocation, every release is a
    real deallocation (subject to Python/CUDA garbage collection)."""

    def __init__(self):
        self._active = 0
        self._peak = 0
        self._allocations = 0

    def allocate(self, shape, dtype, device) -> torch.Tensor:
        t = torch.empty(shape, dtype=dtype, device=device)
        self._active += 1
        self._allocations += 1
        self._peak = max(self._peak, self._active)
        return t

    def free(self, tensor: torch.Tensor) -> None:
        self._active = max(0, self._active - 1)

    def stats(self) -> Dict[str, int]:
        return {"active": self._active, "peak": self._peak, "allocations": self._allocations, "pooled": 0}

    def reset(self) -> None:
        self._active = 0
        self._peak = 0
        self._allocations = 0


class PoolMemoryPolicy(MemoryPolicy):
    """Reuses freed tensors of matching (shape, dtype, device) instead of
    reallocating. Thread-safe (relevant for the AsyncScheduler)."""

    def __init__(self, max_pool_per_key: int = 8):
        self._pool: Dict[PoolKey, list] = {}
        self._lock = threading.Lock()
        self._active = 0
        self._peak = 0
        self._allocations = 0
        self._reuses = 0
        self._max_pool_per_key = max_pool_per_key

    @staticmethod
    def _key(shape, dtype, device) -> PoolKey:
        return (tuple(shape), dtype, str(device))

    def allocate(self, shape, dtype, device) -> torch.Tensor:
        key = self._key(shape, dtype, device)
        with self._lock:
            bucket = self._pool.get(key)
            if bucket:
                t = bucket.pop()
                self._reuses += 1
                self._active += 1
                self._peak = max(self._peak, self._active)
                return t
            self._allocations += 1
            self._active += 1
            self._peak = max(self._peak, self._active)
        return torch.empty(shape, dtype=dtype, device=device)

    def free(self, tensor: torch.Tensor) -> None:
        key = self._key(tensor.shape, tensor.dtype, tensor.device)
        with self._lock:
            self._active = max(0, self._active - 1)
            bucket = self._pool.setdefault(key, [])
            if len(bucket) < self._max_pool_per_key:
                bucket.append(tensor.detach())

    def stats(self) -> Dict[str, int]:
        pooled = sum(len(v) for v in self._pool.values())
        return {
            "active": self._active,
            "peak": self._peak,
            "allocations": self._allocations,
            "reuses": self._reuses,
            "pooled": pooled,
        }

    def reset(self) -> None:
        with self._lock:
            self._pool.clear()
            self._active = 0
            self._peak = 0
            self._allocations = 0
            self._reuses = 0


_POLICIES = {
    "pool": PoolMemoryPolicy,
    "none": NullMemoryPolicy,
}


class MemoryManager:
    """Facade used by the rest of the runtime. Wraps a MemoryPolicy and adds
    event emission + a scratch-buffer convenience API."""

    def __init__(self, policy="pool", event_bus: Optional[EventBus] = None):
        if isinstance(policy, str):
            if policy not in _POLICIES:
                raise ValueError(f"Unknown memory policy '{policy}', choices: {list(_POLICIES)}")
            policy = _POLICIES[policy]()
        elif isinstance(policy, type) and issubclass(policy, MemoryPolicy):
            policy = policy()
        self.policy: MemoryPolicy = policy
        self.event_bus = event_bus or EventBus()

    def request(self, shape, dtype=torch.float32, device="cpu") -> torch.Tensor:
        t = self.policy.allocate(shape, dtype, device)
        self.event_bus.emit(EventType.MEMORY_ALLOCATED, shape=tuple(shape), dtype=str(dtype), device=str(device))
        return t

    def release(self, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        shape = tuple(tensor.shape)
        self.policy.free(tensor)
        self.event_bus.emit(EventType.MEMORY_RELEASED, shape=shape)

    def reuse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Zero out and hand back a tensor for reuse without a full
        free+allocate round trip."""
        with torch.no_grad():
            tensor.zero_()
        return tensor

    def stats(self) -> Dict[str, int]:
        return self.policy.stats()

    def reset(self) -> None:
        self.policy.reset()
