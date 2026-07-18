"""
EAE Runtime: a standalone, programmable, block-local reverse-mode execution
engine that sits *above* PyTorch and orchestrates many local autograd
computations, instead of replacing PyTorch's autodiff engine.

Quick start:

    import torch.nn as nn
    from eae_runtime import EAERuntime, RuntimeConfig
    from eae_runtime.passes import ClipPass, GaussianNoisePass

    model = nn.Sequential(
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    config = RuntimeConfig(
        scheduler="sequential",
        memory="pool",
        backend="auto",
        passes=[ClipPass(max_norm=1.0)],
    )
    runtime = EAERuntime(model, optimizer, config)
    loss = runtime.train_step(x, lambda out: nn.functional.mse_loss(out, target))
"""

from .adjoint import AdjointState
from .backend import BackendManager
from .blocks import BlockDecomposer, EAEBlock
from .boundary_store import BoundaryStore
from .config import RuntimeConfig
from .events import Event, EventBus, EventType
from .forward_executor import ForwardExecutor
from .memory import MemoryManager, MemoryPolicy, NullMemoryPolicy, PoolMemoryPolicy
from .pipeline import AdjointPipeline, PassManager
from .profiler import Profiler
from .reconstruction import ReconstructionEngine
from .runtime import EAERuntime
from .schedulers import (
    AsyncScheduler,
    BaseScheduler,
    DistributedScheduler,
    PipelineScheduler,
    SequentialScheduler,
    build_scheduler,
)

__version__ = "2.0.0"

__all__ = [
    "AdjointState",
    "BackendManager",
    "BlockDecomposer",
    "EAEBlock",
    "BoundaryStore",
    "RuntimeConfig",
    "Event",
    "EventBus",
    "EventType",
    "ForwardExecutor",
    "MemoryManager",
    "MemoryPolicy",
    "NullMemoryPolicy",
    "PoolMemoryPolicy",
    "AdjointPipeline",
    "PassManager",
    "Profiler",
    "ReconstructionEngine",
    "EAERuntime",
    "AsyncScheduler",
    "BaseScheduler",
    "DistributedScheduler",
    "PipelineScheduler",
    "SequentialScheduler",
    "build_scheduler",
]
