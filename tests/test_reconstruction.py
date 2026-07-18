import torch
import torch.nn as nn

from eae_runtime import AdjointState, ReconstructionEngine


def test_reconstruct_single_linear_matches_manual_vjp():
    torch.manual_seed(0)
    block = nn.Linear(4, 3)
    x = torch.randn(5, 4)

    # manual reference VJP
    x_ref = x.clone().requires_grad_(True)
    out_ref = block(x_ref)
    grad_out = torch.randn(5, 3)
    out_ref.backward(grad_out)
    ref_grad_x = x_ref.grad.clone()
    ref_grad_w = block.weight.grad.clone()
    ref_grad_b = block.bias.grad.clone()

    block.zero_grad()
    engine = ReconstructionEngine()
    adjoint = AdjointState(tensor=grad_out, layer_id=1)
    new_adjoint, param_grads = engine.reconstruct(block, x, adjoint, block_name="linear")

    assert torch.allclose(new_adjoint.tensor, ref_grad_x, atol=1e-6)
    assert torch.allclose(param_grads[block.weight], ref_grad_w, atol=1e-6)
    assert torch.allclose(param_grads[block.bias], ref_grad_b, atol=1e-6)
    assert new_adjoint.layer_id == 0


def test_reconstruct_relu_zeroes_negative_gradient_paths():
    block = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0, -3.0, 4.0]])
    adjoint = AdjointState(tensor=torch.ones(1, 4), layer_id=1)
    engine = ReconstructionEngine()
    new_adjoint, param_grads = engine.reconstruct(block, x, adjoint, block_name="relu")
    expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    assert torch.allclose(new_adjoint.tensor, expected)
    assert param_grads == {}  # ReLU has no parameters


def test_reconstruct_shape_mismatch_raises():
    block = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    adjoint = AdjointState(tensor=torch.randn(2, 5), layer_id=1)  # wrong shape
    engine = ReconstructionEngine()
    try:
        engine.reconstruct(block, x, adjoint, block_name="linear")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_reconstruct_does_not_require_grad_on_input_activation():
    """The stored boundary activation should never itself carry a graph -
    reconstruction must build its own leaf regardless."""
    block = nn.Linear(4, 4)
    x = torch.randn(3, 4)
    x.requires_grad_(False)
    adjoint = AdjointState(tensor=torch.randn(3, 4), layer_id=1)
    engine = ReconstructionEngine()
    new_adjoint, param_grads = engine.reconstruct(block, x, adjoint, block_name="linear")
    assert not new_adjoint.tensor.requires_grad
    assert new_adjoint.tensor.grad_fn is None
