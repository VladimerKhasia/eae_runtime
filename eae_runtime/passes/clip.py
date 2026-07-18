from __future__ import annotations

import torch

from ..adjoint import AdjointState
from .base import EAEPass


class ClipPass(EAEPass):
    """Clips the adjoint tensor's norm to `max_norm`, mirroring
    torch.nn.utils.clip_grad_norm_ but operating on a single adjoint in the
    pipeline rather than a full parameter list."""

    name = "ClipPass"

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        new = adjoint.clone()
        norm = torch.linalg.vector_norm(new.tensor.float())
        if norm > self.max_norm:
            scale = self.max_norm / (norm + 1e-6)
            new.tensor = new.tensor * scale.to(new.tensor.dtype)
        return new
