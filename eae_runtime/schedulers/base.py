"""
Scheduler API. This is the heart of the runtime's reverse execution.

Instead of a hardcoded `for block in reversed(blocks): ...` loop, the
runtime delegates reverse execution to a swappable BaseScheduler. Every
scheduler performs, per block, the same logical sequence:

    reconstruct -> launch local VJP -> run adjoint pipeline -> free memory -> continue

but *how* that sequence is scheduled (in order, with prefetching, in
pipeline stages across microbatches, or across distributed workers) is
entirely up to the scheduler implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..adjoint import AdjointState
from ..boundary_store import BoundaryStore
from ..events import EventBus, EventType
from ..memory import MemoryManager
from ..pipeline import AdjointPipeline
from ..profiler import Profiler
from ..reconstruction import ReconstructionEngine


@dataclass
class ReverseContext:
    """Everything a scheduler needs to run the reverse pass. Kept as a
    plain dataclass so custom schedulers can be written without importing
    half the runtime."""

    blocks: List[nn.Module]
    block_names: List[str]
    boundary_store: BoundaryStore
    reconstruction_engine: ReconstructionEngine
    pipeline: AdjointPipeline
    memory_manager: MemoryManager
    event_bus: EventBus
    profiler: Profiler
    initial_adjoint: AdjointState


class BaseScheduler:
    """Subclass and implement `run(context)`.

    Must return a dict mapping each `nn.Parameter` to its accumulated
    gradient tensor for this step.
    """

    name: str = "BaseScheduler"

    def run(self, context: ReverseContext) -> Dict[nn.Parameter, torch.Tensor]:
        raise NotImplementedError

    # -- shared helpers available to all schedulers --------------------- #
    @staticmethod
    def _accumulate(target: Dict[nn.Parameter, torch.Tensor], updates: Dict[nn.Parameter, torch.Tensor]) -> None:
        for p, g in updates.items():
            if p in target:
                target[p] = target[p] + g
            else:
                target[p] = g.clone()

    @staticmethod
    def _reconstruct_one(context: ReverseContext, i: int, adjoint: AdjointState):
        """Reconstruct block `i` (0-indexed) given the adjoint at its output
        boundary (i+1). Returns (new_adjoint_at_input_boundary, param_grads)."""
        block = context.blocks[i]
        name = context.block_names[i]
        input_activation = context.boundary_store.get(i)

        with context.profiler.track(f"reconstruct:{name}"):
            new_adjoint, param_grads = context.reconstruction_engine.reconstruct(
                block, input_activation, adjoint, block_name=name
            )

        new_adjoint = context.pipeline.run(new_adjoint, context={"block_index": i, "block_name": name})

        context.memory_manager.release(input_activation)
        context.event_bus.emit(EventType.SCHEDULER_STEP, block_index=i, block_name=name)
        return new_adjoint, param_grads
