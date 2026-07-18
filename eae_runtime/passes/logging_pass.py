from __future__ import annotations

from typing import Optional

from ..adjoint import AdjointState
from ..events import EventBus, EventType
from .base import EAEPass


class LoggingPass(EAEPass):
    """Emits an ADJOINT_MODIFIED event carrying adjoint.statistics(); does
    not otherwise touch the adjoint. Demonstrates that observation-only
    passes are just as first-class as mutating ones."""

    name = "LoggingPass"

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or EventBus()

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        stats = adjoint.statistics()
        self.event_bus.emit(
            EventType.ADJOINT_MODIFIED,
            layer_id=adjoint.layer_id,
            block=adjoint.block,
            pass_name=self.name,
            **stats,
        )
        return adjoint.clone()
