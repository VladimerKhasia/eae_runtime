from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseScheduler, ReverseContext
from .sequential import SequentialScheduler

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None


class DistributedScheduler(BaseScheduler):
    """Assigns contiguous block ranges to ranks and pipelines the adjoint
    across process boundaries via a Communication Pass (compress -> send ->
    receive -> decompress) at each rank boundary.

    Requires `torch.distributed` to be initialized (e.g. via
    `torch.distributed.init_process_group`) with world_size > 1. When that
    is not the case - e.g. running single-process, as in unit tests and
    most local development - it transparently falls back to
    SequentialScheduler so the same code path is exercised and produces
    identical, verifiable gradients.
    """

    name = "DistributedScheduler"

    def __init__(self, block_ranks: Optional[List[int]] = None):
        """`block_ranks[i]` = rank that owns block i. If omitted, blocks are
        striped evenly across the current world size."""
        self.block_ranks = block_ranks
        self._fallback = SequentialScheduler()

    def _is_distributed_active(self) -> bool:
        return dist is not None and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def run(self, context: ReverseContext) -> Dict[nn.Parameter, torch.Tensor]:
        if not self._is_distributed_active():
            warnings.warn(
                "DistributedScheduler: torch.distributed is not initialized "
                "with world_size > 1; falling back to SequentialScheduler "
                "(numerically identical, single-process).",
                RuntimeWarning,
            )
            return self._fallback.run(context)

        # -- real multi-process path -------------------------------------
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        num_blocks = len(context.blocks)
        ranks = self.block_ranks or [i * world_size // num_blocks for i in range(num_blocks)]

        adjoint = context.initial_adjoint
        param_grads: Dict[nn.Parameter, torch.Tensor] = {}

        for i in reversed(range(num_blocks)):
            owner = ranks[i]
            if owner == rank:
                adjoint, updates = self._reconstruct_one(context, i, adjoint)
                self._accumulate(param_grads, updates)
                # Communication pass: hand the adjoint to the neighboring
                # rank that owns block i-1, if different.
                if i > 0 and ranks[i - 1] != rank:
                    dist.send(adjoint.tensor.contiguous(), dst=ranks[i - 1])
            elif i > 0 and ranks[i - 1] == rank and owner != rank:
                buf = torch.empty_like(adjoint.tensor)
                dist.recv(buf, src=owner)
                adjoint = adjoint.clone()
                adjoint.tensor = buf

        return param_grads
