from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..adjoint import AdjointState
from .base import EAEPass


class SyntheticGradientPass(EAEPass):
    """Decoupled Neural Interfaces style synthetic gradient pass.

    Wraps a small learned predictor `synthesizer(x) -> approx_grad` that can
    be substituted for the true adjoint once it has been warmed up, letting
    a researcher study update-unlocking / async credit assignment schemes.
    During `warmup_steps`, the true adjoint passes through unchanged and is
    used purely as a regression target to train the synthesizer; after
    warmup, the pass optionally *replaces* the adjoint with the synthetic
    estimate when `use_synthetic=True`.
    """

    name = "SyntheticGradientPass"

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        warmup_steps: int = 0,
        use_synthetic: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.synthesizer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        if device is not None:
            self.synthesizer = self.synthesizer.to(device)
        self.optimizer = torch.optim.Adam(self.synthesizer.parameters(), lr=lr)
        self.warmup_steps = warmup_steps
        self.use_synthetic = use_synthetic
        self._step = 0
        self.last_loss: Optional[float] = None

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        new = adjoint.clone()
        flat = new.tensor.reshape(new.tensor.shape[0], -1) if new.tensor.dim() > 1 else new.tensor.unsqueeze(0)

        with torch.enable_grad():
            pred = self.synthesizer(flat.detach().float())
            target = flat.detach().float()
            loss = torch.nn.functional.mse_loss(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.last_loss = loss.item()
        self._step += 1

        if self.use_synthetic and self._step > self.warmup_steps:
            with torch.no_grad():
                synthetic = self.synthesizer(flat.float()).to(new.tensor.dtype)
            new.tensor = synthetic.reshape(new.tensor.shape)
            new.metadata["synthetic_gradient_loss"] = self.last_loss
        return new
