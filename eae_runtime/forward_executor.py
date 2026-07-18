"""
Forward Executor: detached forward, boundary extraction, boundary storage.
No autograd graph is kept alive - this is what makes the runtime
memory-cheap relative to a naive PyTorch backward.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .boundary_store import BoundaryStore
from .events import EventBus, EventType


class ForwardExecutor:
    def __init__(self, event_bus: Optional[EventBus] = None, compute_dtype: Optional[torch.dtype] = None):
        self.event_bus = event_bus or EventBus()
        self.compute_dtype = compute_dtype

    @torch.no_grad()
    def run(self, blocks: List[nn.Module], x0: torch.Tensor, boundary_store: BoundaryStore) -> torch.Tensor:
        self.event_bus.emit(EventType.FORWARD_STARTED, num_blocks=len(blocks))

        x = x0
        boundary_store.put(0, x)
        for i, block in enumerate(blocks):
            compute_x = x.to(self.compute_dtype) if self.compute_dtype is not None else x
            x = block(compute_x)
            boundary_store.put(i + 1, x)

        self.event_bus.emit(EventType.FORWARD_FINISHED, num_boundaries=len(boundary_store))
        return x
