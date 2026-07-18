import pytest
import torch
import torch.nn as nn

from eae_runtime import EAERuntime, RuntimeConfig


def _reference_grads(model, x, target):
    model.zero_grad()
    out = model(x)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    return loss.detach(), {p: p.grad.clone() for p in model.parameters() if p.grad is not None}


def _param_map(model):
    """Stable mapping name -> parameter so we can compare across two
    independently-constructed but architecturally-identical models."""
    return dict(model.named_parameters())


@pytest.mark.parametrize("scheduler", ["sequential", "async"])
def test_gradients_match_plain_pytorch(mlp_factory, sample_batch, scheduler):
    model, twin = mlp_factory(seed=0)
    x, target = sample_batch

    ref_loss, ref_grads = _reference_grads(twin, x, target)

    cfg = RuntimeConfig(scheduler=scheduler, memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    loss, param_grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))

    assert torch.isclose(loss, ref_loss, atol=1e-6)

    ref_by_name = _param_map(twin)
    got_by_name = _param_map(model)
    name_of = {id(p): n for n, p in got_by_name.items()}

    assert len(param_grads) == len(ref_grads)
    for p, g in param_grads.items():
        n = name_of[id(p)]
        ref_g = ref_by_name[n].grad
        assert torch.allclose(g, ref_g, atol=1e-5, rtol=1e-4), f"mismatch for {n}"


def test_gradients_match_with_boundary_offload(mlp_factory, sample_batch):
    model, twin = mlp_factory(seed=1)
    x, target = sample_batch
    ref_loss, ref_grads = _reference_grads(twin, x, target)

    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu", boundary_offload=True)
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    loss, param_grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))

    ref_by_name = _param_map(twin)
    got_by_name = _param_map(model)
    name_of = {id(p): n for n, p in got_by_name.items()}
    for p, g in param_grads.items():
        n = name_of[id(p)]
        assert torch.allclose(g, ref_by_name[n].grad, atol=1e-5, rtol=1e-4), f"mismatch for {n}"


def test_gradients_match_with_pipeline_scheduler_microbatching(mlp_factory, sample_batch):
    model, twin = mlp_factory(seed=2)
    x, target = sample_batch  # batch size 6
    ref_loss, ref_grads = _reference_grads(twin, x, target)

    cfg = RuntimeConfig(scheduler="pipeline", num_microbatches=3, memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    loss, param_grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target, reduction="sum"))

    # reference must also use reduction="sum" so per-microbatch accumulation
    # is exactly comparable (mean would need re-weighting by chunk size)
    twin.zero_grad()
    out = twin(x)
    ref_loss_sum = nn.functional.mse_loss(out, target, reduction="sum")
    ref_loss_sum.backward()

    ref_by_name = _param_map(twin)
    got_by_name = _param_map(model)
    name_of = {id(p): n for n, p in got_by_name.items()}
    for p, g in param_grads.items():
        n = name_of[id(p)]
        assert torch.allclose(g, ref_by_name[n].grad, atol=1e-4, rtol=1e-3), f"mismatch for {n}"


def test_gradients_match_with_distributed_scheduler_fallback(mlp_factory, sample_batch):
    """Single-process: DistributedScheduler must fall back to Sequential and
    produce identical gradients (with a RuntimeWarning)."""
    model, twin = mlp_factory(seed=3)
    x, target = sample_batch
    ref_loss, ref_grads = _reference_grads(twin, x, target)

    cfg = RuntimeConfig(scheduler="distributed", memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    with pytest.warns(RuntimeWarning):
        loss, param_grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))

    ref_by_name = _param_map(twin)
    got_by_name = _param_map(model)
    name_of = {id(p): n for n, p in got_by_name.items()}
    for p, g in param_grads.items():
        n = name_of[id(p)]
        assert torch.allclose(g, ref_by_name[n].grad, atol=1e-5, rtol=1e-4)


def test_end_to_end_train_step_reduces_loss(mlp_factory, sample_batch):
    model, _ = mlp_factory(seed=4)
    x, target = sample_batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu")
    runtime = EAERuntime(model, optimizer, cfg)

    losses = [runtime.train_step(x, lambda out: nn.functional.mse_loss(out, target)) for _ in range(20)]
    assert losses[-1] < losses[0]
