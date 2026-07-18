"""
Pass API: every adjoint passes through Pass1 -> Pass2 -> ... -> Passn before
reaching the next block. Passes are plugins; the runtime never knows what
they do.
"""

from __future__ import annotations

from typing import Optional

from ..adjoint import AdjointState


class EAEPass:
    """Base class for a single stage in the Adjoint Pipeline.

    Subclasses implement `process(adjoint, context) -> AdjointState`.
    `context` is a plain dict with useful ambient info (current block name,
    layer_id, training step, event_bus, etc.) that a pass may read but
    should not mutate destructively.
    """

    name: str = "EAEPass"

    def process(self, adjoint: AdjointState, context: Optional[dict] = None) -> AdjointState:
        raise NotImplementedError

    def __call__(self, adjoint: AdjointState, context: Optional[dict] = None) -> AdjointState:
        result = self.process(adjoint, context or {})
        result.record(self.name)
        return result
