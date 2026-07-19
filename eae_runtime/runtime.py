"""
EAERuntime: the top-level object users instantiate. Ties together the
Forward Executor, Boundary Store, Reverse Scheduler, Reconstruction Engine,
Memory Manager, Adjoint Pipeline, Backend Manager, Event Bus and Profiler.

    runtime = EAERuntime(model, optimizer, config)
    runtime.train_step(batch)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .adjoint import AdjointState
from .backend import BackendManager
from .blocks import BlockDecomposer
from .boundary_store import BoundaryStore
from .config import RuntimeConfig
from .events import EventBus, EventType
from .forward_executor import ForwardExecutor
from .logging_utils import attach_structured_logging
from .memory import MemoryManager
from .pipeline import AdjointPipeline
from .profiler import Profiler
from .reconstruction import ReconstructionEngine
from .schedulers import ReverseContext, build_scheduler


class EAERuntime:
    def __init__(
        self,
        model: Union[nn.Sequential, List[nn.Module], nn.Module],
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[RuntimeConfig] = None,
    ):
        self.config = config or RuntimeConfig()
        self.optimizer = optimizer

        self.blocks: List[nn.Module] = BlockDecomposer.decompose(model)
        self.block_names: List[str] = [
            getattr(b, "eae_name", None) or f"{type(b).__name__}_{i}" for i, b in enumerate(self.blocks)
        ]

        self.event_bus = EventBus()
        if self.config.log_level:
            attach_structured_logging(self.event_bus, level=self.config.log_level)

        self.profiler = Profiler(enabled=self.config.enable_profiler)

        self.backend_manager = (
            self.config.backend if isinstance(self.config.backend, BackendManager) else BackendManager(self.config.backend)
        )

        self.memory_manager = MemoryManager(policy=self.config.memory, event_bus=self.event_bus)

        self.boundary_store = BoundaryStore(
            offload=self.config.boundary_offload,
            precision=self.config.boundary_precision,
            pin_memory=self.config.pin_memory,
        )

        self.forward_executor = ForwardExecutor(event_bus=self.event_bus, compute_dtype=self.config.compute_dtype)
        self.reconstruction_engine = ReconstructionEngine(event_bus=self.event_bus, compute_dtype=self.config.compute_dtype)

        passes = list(self.config.passes)
        self.pipeline = AdjointPipeline(passes=passes, event_bus=self.event_bus, profiler=self.profiler)

        self.scheduler = build_scheduler(self.config.scheduler, num_microbatches=self.config.num_microbatches)

        self.last_loss: Optional[float] = None
        self.last_grad_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the detached forward pass, populating the boundary store, and
        return the (detached) model output."""
        self.boundary_store.clear()
        with self.profiler.track("forward"):
            out = self.forward_executor.run(self.blocks, x, self.boundary_store)
        return out

    def backward(self, output: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Given the detached forward output, compute the loss (with a fresh
        leaf tensor so we can extract dLoss/dOutput via ordinary autograd),
        then run the reverse scheduler through the whole block stack.

        Returns (loss_tensor, param_grads).
        """
        output_leaf = output.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            loss = loss_fn(output_leaf)
            loss.backward()

        if output_leaf.grad is None:
            raise RuntimeError(
                "loss_fn produced no gradient w.r.t. the model output; make "
                "sure it actually depends on its input tensor."
            )

        initial_adjoint = AdjointState(
            tensor=output_leaf.grad.detach(),
            layer_id=len(self.blocks),
            block="output",
        )
        initial_adjoint = self.pipeline.run(initial_adjoint, context={"block_index": len(self.blocks), "block_name": "output"})

        context = ReverseContext(
            blocks=self.blocks,
            block_names=self.block_names,
            boundary_store=self.boundary_store,
            reconstruction_engine=self.reconstruction_engine,
            pipeline=self.pipeline,
            memory_manager=self.memory_manager,
            event_bus=self.event_bus,
            profiler=self.profiler,
            initial_adjoint=initial_adjoint,
        )

        with self.profiler.track("reverse_pass"):
            param_grads = self.scheduler.run(context)

        if self.config.grad_clip_norm is not None:
            squared_norms = [(g.float() ** 2).sum() for g in param_grads.values()]
            total_norm = torch.stack(squared_norms).sum().sqrt() if squared_norms else torch.tensor(0.0)
            if total_norm > self.config.grad_clip_norm:
                scale = self.config.grad_clip_norm / (total_norm + 1e-6)
                param_grads = {p: g * scale for p, g in param_grads.items()}

        return loss.detach(), param_grads

    def apply_gradients(self, param_grads: Dict[nn.Parameter, torch.Tensor]) -> None:
        for p, g in param_grads.items():
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad = p.grad + g

    def zero_grad(self) -> None:
        for block in self.blocks:
            for p in block.parameters():
                p.grad = None

    def train_step(self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
        self.event_bus.emit(EventType.TRAIN_STEP_STARTED)
        self.zero_grad()

        out = self.forward(x)
        loss, param_grads = self.backward(out, loss_fn)
        self.apply_gradients(param_grads)

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.last_loss = loss.item()
        total_grad_norm = 0.0
        if param_grads:
            squared_norms = [(g.float() ** 2).sum() for g in param_grads.values()]
            total_grad_norm = float(torch.stack(squared_norms).sum().sqrt().item())
        self.last_grad_stats = {
            "num_params_with_grad": len(param_grads),
            "total_grad_norm": total_grad_norm,
        }
        self.event_bus.emit(EventType.TRAIN_STEP_FINISHED, loss=self.last_loss, **self.last_grad_stats)
        return self.last_loss

    # convenience for tests / advanced users needing raw grads without an
    # optimizer step:
    def compute_gradients(self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.zero_grad()
        out = self.forward(x)
        loss, param_grads = self.backward(out, loss_fn)
        return loss, param_grads

    def parameters(self):
        for block in self.blocks:
            yield from block.parameters()