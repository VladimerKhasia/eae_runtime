from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BaseScheduler, ReverseContext


class SequentialScheduler(BaseScheduler):
    """The reference scheduler: reconstruct -> VJP -> pipeline -> free,
    strictly in order from the last block to the first. All other
    schedulers must reproduce this scheduler's numerical result exactly."""

    name = "SequentialScheduler"

    def run(self, context: ReverseContext) -> Dict[nn.Parameter, torch.Tensor]:
        num_blocks = len(context.blocks)
        adjoint = context.initial_adjoint
        param_grads: Dict[nn.Parameter, torch.Tensor] = {}

        for i in reversed(range(num_blocks)):
            adjoint, updates = self._reconstruct_one(context, i, adjoint)
            self._accumulate(param_grads, updates)

        return param_grads
