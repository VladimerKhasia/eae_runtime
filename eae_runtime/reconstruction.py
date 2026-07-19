"""
Reconstruction Engine: given a block, a saved boundary activation, and an
incoming AdjointState, it rebuilds the local computation graph, computes the
VJP using PyTorch's *local* autograd (torch.autograd.grad), and returns a
new AdjointState plus parameter gradients.

No global graph ever exists. PyTorch owns local autograd / VJP / kernels;
the runtime owns everything around it (see spec, Primary Design Principle 2).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .adjoint import AdjointState
from .events import EventBus, EventType


class ReconstructionEngine:
    def __init__(self, event_bus: Optional[EventBus] = None, compute_dtype: Optional[torch.dtype] = None):
        self.event_bus = event_bus or EventBus()
        self.compute_dtype = compute_dtype

    def reconstruct(
        self,
        block: nn.Module,
        input_activation: torch.Tensor,
        adjoint: AdjointState,
        block_name: str = "",
    ) -> Tuple[AdjointState, Dict[nn.Parameter, torch.Tensor]]:
        """Rebuild `block`'s local graph from `input_activation`, run the
        forward again with autograd enabled, and back-propagate `adjoint`
        through it using local VJP only.

        Returns:
            (new_adjoint_for_block_input, {param: grad_tensor})
        """
        compute_dtype = self.compute_dtype or input_activation.dtype

        x = input_activation.detach().to(compute_dtype).clone().requires_grad_(True)

        with torch.enable_grad():
            out = block(x)

        params = [p for p in block.parameters() if p.requires_grad]

        grad_outputs = adjoint.tensor.to(device=out.device, dtype=out.dtype)
        if grad_outputs.shape != out.shape:
            raise RuntimeError(
                f"Adjoint shape {tuple(grad_outputs.shape)} does not match "
                f"block output shape {tuple(out.shape)} for block '{block_name}'"
            )

        inputs: List[torch.Tensor] = [x, *params]
        grads = torch.autograd.grad(
            outputs=out,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=False,
            allow_unused=True,
        )
        grad_x, param_grads_list = grads[0], grads[1:]

        if grad_x is None:
            grad_x = torch.zeros_like(x)

        param_grads = {p: g for p, g in zip(params, param_grads_list) if g is not None}

        new_adjoint = AdjointState(
            tensor=grad_x.detach(),
            layer_id=adjoint.layer_id - 1,
            block=block_name,
            metadata=dict(adjoint.metadata),
        )
        new_adjoint.record(f"reconstruct:{block_name}")

        self.event_bus.emit(
            EventType.BLOCK_RECONSTRUCTED,
            block=block_name,
            layer_id=adjoint.layer_id,
            grad_norm=float(torch.linalg.vector_norm(grad_x.float()).item()),
        )

        # free the local graph explicitly
        del out
        return new_adjoint, param_grads