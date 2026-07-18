"""
Event system: everything important that happens inside the runtime emits an
Event. Researchers should never need to edit runtime code just to observe
execution - they subscribe to the EventBus instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class EventType:
    FORWARD_STARTED = "ForwardStarted"
    FORWARD_FINISHED = "ForwardFinished"
    BLOCK_RECONSTRUCTED = "BlockReconstructed"
    ADJOINT_CREATED = "AdjointCreated"
    ADJOINT_MODIFIED = "AdjointModified"
    MEMORY_ALLOCATED = "MemoryAllocated"
    MEMORY_RELEASED = "MemoryReleased"
    SCHEDULER_STEP = "SchedulerStep"
    KERNEL_LAUNCH = "KernelLaunch"
    COMMUNICATION_START = "CommunicationStart"
    COMMUNICATION_FINISH = "CommunicationFinish"
    TRAIN_STEP_STARTED = "TrainStepStarted"
    TRAIN_STEP_FINISHED = "TrainStepFinished"
    PASS_APPLIED = "PassApplied"


@dataclass
class Event:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Event({self.type}, {self.payload})"


class EventBus:
    """A minimal, synchronous pub/sub bus.

    Subscribers are plain callables `fn(event: Event) -> None`. Subscribing
    to `"*"` receives every event, regardless of type.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._log: List[Event] = []
        self._recording = False

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def emit(self, event_type: str, **payload: Any) -> Event:
        event = Event(type=event_type, payload=payload)
        if self._recording:
            self._log.append(event)
        for callback in self._subscribers.get(event_type, []):
            callback(event)
        for callback in self._subscribers.get("*", []):
            callback(event)
        return event

    # -- recording / introspection, handy for tests -------------------- #
    def start_recording(self) -> None:
        self._recording = True
        self._log = []

    def stop_recording(self) -> List[Event]:
        self._recording = False
        return self._log

    @property
    def log(self) -> List[Event]:
        return list(self._log)

    def events_of_type(self, event_type: str) -> List[Event]:
        return [e for e in self._log if e.type == event_type]
