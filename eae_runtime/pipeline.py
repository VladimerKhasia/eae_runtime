"""
AdjointPipeline: the central abstraction that turns EAE from a memory
optimization into a research platform.

Every reverse step exchanges an AdjointState through this programmable
pipeline instead of the scheduler/reconstruction engine passing tensors
directly between steps:

    AdjointState -> Pass1 -> Pass2 -> ... -> PassN -> next local VJP

A researcher working on gradient compression, synthetic gradients,
error-feedback, distributed communication or adaptive optimization only
ever implements a new Pass. They never need to understand or modify the
reconstruction engine, scheduler, or memory manager.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .adjoint import AdjointState
from .events import EventBus, EventType
from .passes.base import EAEPass


class AdjointPipeline:
    def __init__(self, passes: Optional[List[EAEPass]] = None, event_bus: Optional[EventBus] = None,
                 profiler=None):
        self.passes: List[EAEPass] = list(passes) if passes else []
        self.event_bus = event_bus or EventBus()
        self.profiler = profiler

    def add_pass(self, p: EAEPass) -> None:
        self.passes.append(p)

    def run(self, adjoint: AdjointState, context: Optional[dict] = None) -> AdjointState:
        context = context or {}
        for p in self.passes:
            start = time.perf_counter()
            adjoint = p(adjoint, context)
            elapsed = time.perf_counter() - start
            if self.profiler is not None:
                self.profiler.record("pass:" + p.name, elapsed)
            self.event_bus.emit(EventType.PASS_APPLIED, pass_name=p.name, layer_id=adjoint.layer_id,
                                 elapsed_seconds=elapsed)
        return adjoint

    def __call__(self, adjoint: AdjointState, context: Optional[dict] = None) -> AdjointState:
        return self.run(adjoint, context)

    def __len__(self) -> int:
        return len(self.passes)


# Backwards-compatible alias matching the spec's "Pass Manager" component
# name; AdjointPipeline is the concrete implementation of that role.
PassManager = AdjointPipeline
