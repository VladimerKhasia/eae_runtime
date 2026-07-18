from __future__ import annotations

import torch

from ..adjoint import AdjointState
from .base import EAEPass


class RegularizationPass(EAEPass):
    """Adds a decay term proportional to the adjoint itself, i.e. a simple
    stand-in for gradient-space regularization (e.g. an L2 penalty on the
    *gradient*, distinct from weight decay which acts on parameters)."""

    name = "RegularizationPass"

    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        if self.strength == 0.0:
            return adjoint.clone()
        new = adjoint.clone()
        new.tensor = new.tensor * (1.0 - self.strength)
        return new


class GaussianNoisePass(EAEPass):
    """Injects zero-mean Gaussian noise into the adjoint, useful for studying
    differential-privacy style gradient noising or robustness."""

    name = "GaussianNoisePass"

    def __init__(self, std: float = 0.0, generator: torch.Generator = None):
        self.std = std
        self.generator = generator

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        if self.std == 0.0:
            return adjoint.clone()
        new = adjoint.clone()
        noise = torch.randn(
            new.tensor.shape, dtype=new.tensor.dtype, device=new.tensor.device, generator=self.generator
        ) * self.std
        new.tensor = new.tensor + noise
        return new
