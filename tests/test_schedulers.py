import pytest
import torch
import torch.nn as nn

from eae_runtime import (
    AdjointPipeline,
    AdjointState,
    BoundaryStore,
    EventBus,
    MemoryManager,
    Profiler,
    ReconstructionEngine,
    build_scheduler,
)
from eae_runtime.schedulers import (
    AsyncScheduler,
    BaseScheduler,
    DistributedScheduler,
    PipelineScheduler,
    ReverseContext,
    SequentialScheduler,
)


def _make_context(blocks, x, target, num_microbatches=1):
    boundary_store = BoundaryStore()
    xcur = x
    boundary_store.put(0, xcur)
    with torch.no_grad():
        for i, b in enumerate(blocks):
            xcur = b(xcur)
            boundary_store.put(i + 1, xcur)

    out_leaf = xcur.detach().clone().requires_grad_(True)
    loss = nn.functional.mse_loss(out_leaf, target)
    loss.backward()
    initial_adjoint = AdjointState(tensor=out_leaf.grad, layer_id=len(blocks))

    ctx = ReverseContext(
        blocks=blocks,
        block_names=[f"b{i}" for i in range(len(blocks))],
        boundary_store=boundary_store,
        reconstruction_engine=ReconstructionEngine(),
        pipeline=AdjointPipeline(),
        memory_manager=MemoryManager(policy="pool"),
        event_bus=EventBus(),
        profiler=Profiler(),
        initial_adjoint=initial_adjoint,
    )
    return ctx, loss


@pytest.fixture
def toy_blocks():
    torch.manual_seed(0)
    return [nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)]


def test_build_scheduler_from_string():
    s = build_scheduler("sequential")
    assert isinstance(s, SequentialScheduler)


def test_build_scheduler_from_instance_is_passthrough():
    inst = AsyncScheduler(max_workers=4)
    s = build_scheduler(inst)
    assert s is inst


def test_build_scheduler_unknown_string_raises():
    with pytest.raises(ValueError):
        build_scheduler("not-a-real-scheduler")


def test_sequential_scheduler_returns_grad_for_every_param(toy_blocks):
    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    ctx, loss = _make_context(toy_blocks, x, target)
    grads = SequentialScheduler().run(ctx)
    total_params = sum(1 for b in toy_blocks for _ in b.parameters())
    assert len(grads) == total_params


def test_async_scheduler_matches_sequential_scheduler(toy_blocks):
    x = torch.randn(3, 4)
    target = torch.randn(3, 2)

    ctx1, _ = _make_context(toy_blocks, x, target)
    seq_grads = SequentialScheduler().run(ctx1)

    ctx2, _ = _make_context(toy_blocks, x, target)
    async_grads = AsyncScheduler().run(ctx2)

    assert set(id(p) for p in seq_grads) == set(id(p) for p in async_grads)
    for p in seq_grads:
        match = next(q for q in async_grads if q is p)
        assert torch.allclose(seq_grads[p], async_grads[match], atol=1e-6)


def test_pipeline_scheduler_more_microbatches_than_batch_clamped(toy_blocks):
    x = torch.randn(2, 4)  # batch size 2
    target = torch.randn(2, 2)
    ctx, _ = _make_context(toy_blocks, x, target, num_microbatches=10)
    # should not crash even though num_microbatches > batch_size
    grads = PipelineScheduler(num_microbatches=10).run(ctx)
    assert len(grads) > 0


def test_pipeline_scheduler_single_microbatch_matches_sequential(toy_blocks):
    x = torch.randn(4, 4)
    target = torch.randn(4, 2)

    ctx1, _ = _make_context(toy_blocks, x, target)
    seq_grads = SequentialScheduler().run(ctx1)

    ctx2, _ = _make_context(toy_blocks, x, target)
    pipe_grads = PipelineScheduler(num_microbatches=1).run(ctx2)

    for p in seq_grads:
        match = next(q for q in pipe_grads if q is p)
        assert torch.allclose(seq_grads[p], pipe_grads[match], atol=1e-6)


def test_distributed_scheduler_warns_and_falls_back_single_process(toy_blocks):
    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    ctx, _ = _make_context(toy_blocks, x, target)
    with pytest.warns(RuntimeWarning):
        grads = DistributedScheduler().run(ctx)
    assert len(grads) > 0


def test_custom_scheduler_can_be_plugged_in(toy_blocks):
    """Demonstrates the Scheduler API: a user-defined scheduler subclassing
    BaseScheduler works without any runtime modification."""

    class ReverseButLoud(BaseScheduler):
        name = "ReverseButLoud"

        def run(self, context):
            calls = []
            adjoint = context.initial_adjoint
            grads = {}
            for i in reversed(range(len(context.blocks))):
                adjoint, updates = self._reconstruct_one(context, i, adjoint)
                calls.append(i)
                self._accumulate(grads, updates)
            assert calls == sorted(calls, reverse=True)
            return grads

    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    ctx, _ = _make_context(toy_blocks, x, target)
    grads = ReverseButLoud().run(ctx)
    assert len(grads) > 0
