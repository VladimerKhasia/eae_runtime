import pytest
import torch
import torch.nn as nn

from eae_runtime import EAERuntime, RuntimeConfig


def _tiny_model():
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def test_backward_raises_clear_error_when_loss_fn_ignores_output():
    """If loss_fn doesn't actually depend on its input, output_leaf.grad
    would silently be None; the runtime should fail loudly and early
    instead of raising an opaque AttributeError deep inside AdjointState
    construction."""
    model = _tiny_model()
    runtime = EAERuntime(model, optimizer=None, config=RuntimeConfig(backend="cpu"))
    x = torch.randn(3, 4)

    def broken_loss_fn(_output):
        # constant loss: no gradient flows back to _output at all
        return torch.tensor(1.0, requires_grad=True)

    out = runtime.forward(x)
    with pytest.raises(RuntimeError, match="no gradient"):
        runtime.backward(out, broken_loss_fn)


def test_grad_clip_norm_scales_gradients_down():
    model = _tiny_model()
    cfg = RuntimeConfig(backend="cpu", grad_clip_norm=1e-6)
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    x = torch.randn(5, 4)
    target = torch.randn(5, 2)

    _, param_grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))

    total_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in param_grads.values()))
    assert total_norm <= cfg.grad_clip_norm + 1e-4


def test_grad_clip_norm_none_leaves_gradients_unscaled():
    model = _tiny_model()
    cfg = RuntimeConfig(backend="cpu", grad_clip_norm=None)
    runtime = EAERuntime(model, optimizer=None, config=cfg)
    x = torch.randn(5, 4)
    target = torch.randn(5, 2)
    _, param_grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))
    assert all(torch.isfinite(g).all() for g in param_grads.values())


def test_apply_gradients_accumulates_into_existing_grad():
    model = _tiny_model()
    runtime = EAERuntime(model, optimizer=None, config=RuntimeConfig(backend="cpu"))
    p = next(model.parameters())
    p.grad = torch.ones_like(p)
    runtime.apply_gradients({p: torch.ones_like(p)})
    assert torch.allclose(p.grad, torch.full_like(p, 2.0))


def test_zero_grad_clears_all_parameter_grads():
    model = _tiny_model()
    runtime = EAERuntime(model, optimizer=None, config=RuntimeConfig(backend="cpu"))
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    runtime.zero_grad()
    assert all(p.grad is None for p in model.parameters())