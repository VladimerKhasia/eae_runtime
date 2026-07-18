from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..adjoint import AdjointState
from ..events import EventType
from .base import BaseScheduler, ReverseContext


class PipelineScheduler(BaseScheduler):
    """Splits the batch dimension into `num_microbatches` chunks and runs a
    full reverse pass per chunk, accumulating parameter gradients across
    chunks. This is a single-process emulation of pipeline-parallel
    microbatching: for any block whose forward is applied independently
    per-sample (the common case - Linear/activation/attention blocks
    without cross-batch statistics), the accumulated result is numerically
    identical to running the whole batch through SequentialScheduler.
    """

    name = "PipelineScheduler"

    def __init__(self, num_microbatches: int = 1):
        self.num_microbatches = max(1, num_microbatches)

    @staticmethod
    def _slice_adjoint(adjoint: AdjointState, start: int, end: int) -> AdjointState:
        new = adjoint.clone()
        new.tensor = adjoint.tensor[start:end]
        return new

    def run(self, context: ReverseContext) -> Dict[nn.Parameter, torch.Tensor]:
        num_blocks = len(context.blocks)
        param_grads: Dict[nn.Parameter, torch.Tensor] = {}

        batch_size = context.initial_adjoint.tensor.shape[0]
        num_mb = max(1, min(self.num_microbatches, batch_size))
        chunk = (batch_size + num_mb - 1) // num_mb

        for mb in range(num_mb):
            start, end = mb * chunk, min(batch_size, (mb + 1) * chunk)
            if start >= end:
                continue
            adjoint = self._slice_adjoint(context.initial_adjoint, start, end)

            for i in reversed(range(num_blocks)):
                block = context.blocks[i]
                name = context.block_names[i]
                full_input = context.boundary_store.get(i)
                input_activation = full_input[start:end]

                with context.profiler.track(f"reconstruct:{name}:mb{mb}"):
                    new_adjoint, updates = context.reconstruction_engine.reconstruct(
                        block, input_activation, adjoint, block_name=name
                    )
                new_adjoint = context.pipeline.run(
                    new_adjoint, context={"block_index": i, "block_name": name, "microbatch": mb}
                )
                context.event_bus.emit(
                    EventType.SCHEDULER_STEP, block_index=i, block_name=name, microbatch=mb
                )
                adjoint = new_adjoint
                self._accumulate(param_grads, updates)

        return param_grads
