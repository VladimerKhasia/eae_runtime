from .base import BaseScheduler, ReverseContext
from .sequential import SequentialScheduler
from .async_scheduler import AsyncScheduler
from .pipeline_scheduler import PipelineScheduler
from .distributed import DistributedScheduler

_REGISTRY = {
    "sequential": SequentialScheduler,
    "async": AsyncScheduler,
    "pipeline": PipelineScheduler,
    "distributed": DistributedScheduler,
}


def build_scheduler(spec, **kwargs) -> BaseScheduler:
    if isinstance(spec, BaseScheduler):
        return spec
    if isinstance(spec, str):
        if spec not in _REGISTRY:
            raise ValueError(f"Unknown scheduler '{spec}', choices: {list(_REGISTRY)}")
        cls = _REGISTRY[spec]
        if spec == "pipeline":
            return cls(num_microbatches=kwargs.get("num_microbatches", 1))
        return cls()
    if isinstance(spec, type) and issubclass(spec, BaseScheduler):
        return spec()
    raise TypeError(f"Cannot build scheduler from {spec!r}")


__all__ = [
    "BaseScheduler",
    "ReverseContext",
    "SequentialScheduler",
    "AsyncScheduler",
    "PipelineScheduler",
    "DistributedScheduler",
    "build_scheduler",
]
