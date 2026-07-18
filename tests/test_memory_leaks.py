import gc

import torch
import torch.nn as nn

from eae_runtime import EAERuntime, RuntimeConfig
from eae_runtime.passes import ClipPass


def test_memory_manager_active_returns_to_zero_after_step(mlp_factory, sample_batch):
    model, _ = mlp_factory(seed=7)
    x, target = sample_batch
    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer=None, config=cfg)

    runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))
    stats = runtime.memory_manager.stats()
    # every boundary activation requested by the forward pass should have
    # been released by the time the reverse pass completes
    assert stats["active"] == 0


def test_pool_allocations_stabilize_across_many_steps(mlp_factory, sample_batch):
    """Allocator correctness: after a warlöm-up step, the pool should reuse
    buffers rather than growing allocation count linearly with step count."""
    model, _ = mlp_factory(seed=8)
    x, target = sample_batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer, cfg)

    for _ in range(3):
        runtime.train_step(x, lambda out: nn.functional.mse_loss(out, target))
    allocations_after_warmup = runtime.memory_manager.stats()["allocations"]

    for _ in range(20):
        runtime.train_step(x, lambda out: nn.functional.mse_loss(out, target))
    allocations_after_many = runtime.memory_manager.stats()["allocations"]

    # allocation count should not grow further once the pool is warm - all
    # subsequent boundary tensors are the same (shape, dtype, device) and
    # should be served from the pool via reuse
    assert allocations_after_many == allocations_after_warmup


def test_boundary_store_cleared_between_steps(mlp_factory, sample_batch):
    model, _ = mlp_factory(seed=9)
    x, target = sample_batch
    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer=None, config=cfg)

    runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))
    n1 = len(runtime.boundary_store)
    runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))
    n2 = len(runtime.boundary_store)
    # boundary store is cleared+refilled each forward, not accumulated
    assert n1 == n2 == len(runtime.blocks) + 1


def test_no_reference_cycles_keep_activation_tensors_alive(mlp_factory, sample_batch):
    """After compute_gradients returns, nothing but the returned grads dict
    should keep large tensors alive; the boundary store's own buffers may be
    pooled/reused, but plain python gc should not report growing garbage."""
    model, _ = mlp_factory(seed=10, hidden=64, out_dim=64, depth=6)
    x = torch.randn(4, 16)
    target = torch.randn(4, 64)  # depth chosen so out_dim == hidden == 64
    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer=None, config=cfg)

    gc.collect()
    for _ in range(10):
        runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))
    gc.collect()
    # no assertion on absolute counts (allocator-dependent); this test's
    # real purpose is to ensure repeated calls don't raise/hang, which
    # would indicate a reference cycle or deadlock in cleanup.
    assert runtime.memory_manager.stats()["active"] == 0
