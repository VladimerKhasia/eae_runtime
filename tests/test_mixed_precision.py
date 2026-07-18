import torch
import torch.nn as nn

from eae_runtime import EAERuntime, RuntimeConfig


def test_mixed_precision_reconstruction_dtype(mlp_factory, sample_batch):
    model, _ = mlp_factory(seed=5)
    x, target = sample_batch

    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu", compute_dtype=torch.float32)
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    loss, grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))
    assert torch.isfinite(loss)
    for g in grads.values():
        assert torch.isfinite(g).all()


def test_boundary_precision_fp16_storage_close_to_fp32_reference(mlp_factory, sample_batch):
    model, twin = mlp_factory(seed=6)
    x, target = sample_batch

    twin.zero_grad()
    ref_out = twin(x)
    ref_loss = nn.functional.mse_loss(ref_out, target)
    ref_loss.backward()

    cfg = RuntimeConfig(
        scheduler="sequential",
        memory="pool",
        backend="cpu",
        boundary_precision=torch.float16,  # store activations as fp16
    )
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    loss, grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))

    ref_by_name = dict(twin.named_parameters())
    got_by_name = dict(model.named_parameters())
    name_of = {id(p): n for n, p in got_by_name.items()}

    # fp16 storage introduces small error - use a loose but bounded tolerance
    for p, g in grads.items():
        n = name_of[id(p)]
        assert torch.allclose(g, ref_by_name[n].grad, atol=5e-2, rtol=5e-2)


def test_no_nans_with_fp16_boundary_and_compute():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(4, 16)
    target = torch.randn(4, 4)

    cfg = RuntimeConfig(
        scheduler="sequential",
        memory="pool",
        backend="cpu",
        boundary_precision=torch.float16,
        compute_dtype=torch.float32,
    )
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    loss, grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))
    assert torch.isfinite(loss)
    for g in grads.values():
        assert torch.isfinite(g).all()
