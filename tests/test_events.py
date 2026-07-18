from eae_runtime import Event, EventBus, EventType


def test_subscribe_and_emit():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.FORWARD_STARTED, lambda e: received.append(e))
    bus.emit(EventType.FORWARD_STARTED, num_blocks=3)
    assert len(received) == 1
    assert received[0].type == EventType.FORWARD_STARTED
    assert received[0].payload["num_blocks"] == 3


def test_wildcard_subscriber_receives_everything():
    bus = EventBus()
    received = []
    bus.subscribe("*", lambda e: received.append(e.type))
    bus.emit(EventType.FORWARD_STARTED)
    bus.emit(EventType.MEMORY_ALLOCATED)
    assert received == [EventType.FORWARD_STARTED, EventType.MEMORY_ALLOCATED]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []

    def cb(e):
        received.append(e)

    bus.subscribe(EventType.SCHEDULER_STEP, cb)
    bus.emit(EventType.SCHEDULER_STEP)
    bus.unsubscribe(EventType.SCHEDULER_STEP, cb)
    bus.emit(EventType.SCHEDULER_STEP)
    assert len(received) == 1


def test_recording_and_filtering():
    bus = EventBus()
    bus.start_recording()
    bus.emit(EventType.MEMORY_ALLOCATED, shape=(2, 2))
    bus.emit(EventType.MEMORY_RELEASED, shape=(2, 2))
    bus.emit(EventType.MEMORY_ALLOCATED, shape=(3, 3))
    log = bus.stop_recording()
    assert len(log) == 3
    allocated = bus.events_of_type(EventType.MEMORY_ALLOCATED)
    assert len(allocated) == 2


def test_event_repr_contains_type():
    e = Event(type=EventType.ADJOINT_CREATED, payload={"a": 1})
    assert "AdjointCreated" in repr(e)
