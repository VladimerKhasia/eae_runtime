from __future__ import annotations

import torch

from ..adjoint import AdjointState
from .base import EAEPass


class FP16Pass(EAEPass):
    """Downcasts the adjoint to fp16 and back, simulating the numerical
    effect of a reduced-precision communication/compute path."""

    name = "FP16Pass"

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        new = adjoint.clone()
        orig_dtype = new.tensor.dtype
        new.tensor = new.tensor.to(torch.float16).to(orig_dtype)
        return new


class Int8QuantizationPass(EAEPass):
    """Fake-quantizes the adjoint to int8 with a per-tensor symmetric scale,
    then immediately dequantizes. This is the standard 'fake quant' trick
    used to study the effect of low-precision gradient compression without
    needing real int8 kernels. A production FP8Pass would follow the same
    shape but call into real fp8 kernels via the Backend Manager."""

    name = "Int8QuantizationPass"

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        new = adjoint.clone()
        t = new.tensor.float()
        max_abs = t.abs().max()
        if max_abs == 0:
            return new
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(t / scale), -127, 127)
        dq = (q * scale).to(adjoint.tensor.dtype)
        new.tensor = dq
        new.metadata["quantization_scale"] = scale.item()
        return new


# Alias matching the spec's example name; same fake-quant strategy, kept
# distinct so a future PR can swap in a real fp8 kernel behind this class
# without touching Int8QuantizationPass or any user code that references it.
class FP8Pass(EAEPass):
    name = "FP8Pass"

    def __init__(self):
        self._impl = Int8QuantizationPass()

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        return self._impl.process(adjoint, context)
