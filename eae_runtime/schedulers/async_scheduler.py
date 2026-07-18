from __future__ import annotations

import concurrent.futures
from typing import Dict

import torch
import torch.nn as nn

from ..events import EventType
from .base import BaseScheduler, ReverseContext


class AsyncScheduler(BaseScheduler):
    """Same numerical result as SequentialScheduler, but overlaps the (I/O
    bound) boundary-activation fetch for block i-1 with the (compute bound)
    reconstruction of block i using a background thread. This matters most
    when the BoundaryStore is CPU-offloaded and the device is CUDA.

    The reverse dependency chain (adjoint_{i-1} depends on adjoint_i) is
    inherently sequential, so only the *fetch* is prefetched - the VJP
    compute itself still runs in strict order, guaranteeing identical
    results to SequentialScheduler.
    """

    name = "AsyncScheduler"

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers

    def run(self, context: ReverseContext) -> Dict[nn.Parameter, torch.Tensor]:
        num_blocks = len(context.blocks)
        order = list(reversed(range(num_blocks)))
        adjoint = context.initial_adjoint
        param_grads: Dict[nn.Parameter, torch.Tensor] = {}

        if not order:
            return param_grads

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            def fetch(i):
                return context.boundary_store.get(i)

            pending = {order[0]: executor.submit(fetch, order[0])}

            for pos, i in enumerate(order):
                input_activation = pending.pop(i).result()

                if pos + 1 < len(order):
                    nxt = order[pos + 1]
                    pending[nxt] = executor.submit(fetch, nxt)

                block = context.blocks[i]
                name = context.block_names[i]
                with context.profiler.track(f"reconstruct:{name}"):
                    new_adjoint, updates = context.reconstruction_engine.reconstruct(
                        block, input_activation, adjoint, block_name=name
                    )
                new_adjoint = context.pipeline.run(new_adjoint, context={"block_index": i, "block_name": name})
                context.memory_manager.release(input_activation)
                context.event_bus.emit(EventType.SCHEDULER_STEP, block_index=i, block_name=name, async_prefetch=True)

                adjoint = new_adjoint
                self._accumulate(param_grads, updates)

        return param_grads
