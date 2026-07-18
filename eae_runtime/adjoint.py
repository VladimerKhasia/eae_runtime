"""
AdjointState: the first-class runtime object that flows through the EAE runtime.

Design principle (see spec, "Explicit Adjoint States"):
    The runtime never passes raw tensors internally. Everything flows through
    AdjointState. Passes, schedulers and the reconstruction engine all read
    and write AdjointState objects, never bare torch.Tensors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class AdjointState:
    """A gradient ("adjoint") flowing backward through the runtime, plus metadata.

    Attributes:
        tensor: the actual gradient tensor (d Loss / d activation).
        layer_id: index of the block boundary this adjoint sits at (L, L-1, ..., 0).
        block: human readable name of the block that produced/consumes this adjoint.
        dtype: logical dtype this adjoint should be materialized at.
        device: logical device this adjoint should be materialized at.
        metadata: free-form dict for passes to stash information.
        history: ordered list of pass names that have touched this adjoint,
                 for debugging / provenance / profiling.
    """

    tensor: torch.Tensor
    layer_id: int
    block: str = ""
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = self.tensor.dtype
        if self.device is None:
            self.device = self.tensor.device

    # ------------------------------------------------------------------ #
    # Rich adjoint API - users should almost never touch .tensor directly
    # ------------------------------------------------------------------ #
    def norm(self, p: float = 2.0) -> torch.Tensor:
        """L-p norm of the underlying gradient tensor."""
        return torch.linalg.vector_norm(self.tensor.float(), ord=p)

    def statistics(self) -> Dict[str, float]:
        """Cheap summary statistics, useful for logging/profiling passes."""
        t = self.tensor.detach().float()
        return {
            "mean": t.mean().item(),
            "std": t.std(unbiased=False).item() if t.numel() > 1 else 0.0,
            "min": t.min().item(),
            "max": t.max().item(),
            "norm": torch.linalg.vector_norm(t).item(),
            "numel": t.numel(),
            "has_nan": bool(torch.isnan(t).any().item()),
            "has_inf": bool(torch.isinf(t).any().item()),
        }

    def quantize(self, dtype: torch.dtype = torch.float16) -> "AdjointState":
        """Return a new AdjointState with the tensor cast down to `dtype`."""
        new = self.clone()
        new.tensor = new.tensor.to(dtype)
        new.dtype = dtype
        return new

    def dequantize(self, dtype: torch.dtype = torch.float32) -> "AdjointState":
        new = self.clone()
        new.tensor = new.tensor.to(dtype)
        new.dtype = dtype
        return new

    def compress(self, ratio: float = 0.5) -> "AdjointState":
        """Simple magnitude-pruning compression pass, kept out of the runtime
        core on purpose - this is exactly the kind of thing a real Pass would
        do; it lives here only as a convenience primitive on the adjoint."""
        new = self.clone()
        flat = new.tensor.flatten()
        k = max(1, int(flat.numel() * (1.0 - ratio)))
        if k < flat.numel():
            threshold = flat.abs().kthvalue(flat.numel() - k + 1).values
            mask = flat.abs() >= threshold
            flat = flat * mask
            new.tensor = flat.view_as(new.tensor)
        return new

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "AdjointState":
        new = self.clone()
        new.tensor = new.tensor.to(device=device or new.device, dtype=dtype or new.dtype)
        new.device = new.tensor.device
        new.dtype = new.tensor.dtype
        return new

    def clone(self) -> "AdjointState":
        return AdjointState(
            tensor=self.tensor,
            layer_id=self.layer_id,
            block=self.block,
            dtype=self.dtype,
            device=self.device,
            metadata=dict(self.metadata),
            history=list(self.history),
        )

    def detach(self) -> "AdjointState":
        new = self.clone()
        new.tensor = new.tensor.detach()
        return new

    def record(self, pass_name: str) -> None:
        """Append a pass name to this adjoint's provenance history."""
        self.history.append(pass_name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AdjointState(layer_id={self.layer_id}, block={self.block!r}, "
            f"shape={tuple(self.tensor.shape)}, dtype={self.dtype}, "
            f"device={self.device}, history={self.history})"
        )
