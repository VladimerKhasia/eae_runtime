import torch

from eae_runtime import AdjointState


def test_adjoint_basic_construction():
    t = torch.randn(4, 8)
    a = AdjointState(tensor=t, layer_id=3, block="block3")
    assert a.layer_id == 3
    assert a.block == "block3"
    assert a.dtype == t.dtype
    assert a.device == t.device
    assert a.history == []


def test_adjoint_norm_matches_torch():
    t = torch.randn(4, 8)
    a = AdjointState(tensor=t, layer_id=0)
    assert torch.isclose(a.norm(), torch.linalg.vector_norm(t.float()))


def test_adjoint_statistics_keys():
    t = torch.randn(10)
    a = AdjointState(tensor=t, layer_id=0)
    stats = a.statistics()
    for k in ("mean", "std", "min", "max", "norm", "numel", "has_nan", "has_inf"):
        assert k in stats
    assert stats["numel"] == 10
    assert stats["has_nan"] is False
    assert stats["has_inf"] is False


def test_adjoint_quantize_dequantize_roundtrip_dtype():
    t = torch.randn(4, 4)
    a = AdjointState(tensor=t, layer_id=0)
    q = a.quantize(torch.float16)
    assert q.tensor.dtype == torch.float16
    assert q.dtype == torch.float16
    dq = q.dequantize(torch.float32)
    assert dq.tensor.dtype == torch.float32
    # original untouched
    assert a.tensor.dtype == torch.float32


def test_adjoint_clone_is_independent():
    t = torch.randn(3, 3)
    a = AdjointState(tensor=t, layer_id=1, metadata={"x": 1})
    b = a.clone()
    b.metadata["x"] = 2
    b.history.append("touched")
    assert a.metadata["x"] == 1
    assert a.history == []


def test_adjoint_detach_removes_grad_fn():
    t = torch.randn(3, 3, requires_grad=True)
    y = t * 2
    a = AdjointState(tensor=y, layer_id=0)
    assert a.tensor.requires_grad
    d = a.detach()
    assert not d.tensor.requires_grad


def test_adjoint_compress_zeroes_small_magnitude_entries():
    t = torch.tensor([0.01, 0.02, 5.0, -5.0, 0.03, 0.04])
    a = AdjointState(tensor=t, layer_id=0)
    c = a.compress(ratio=0.5)  # keep top 50% by magnitude
    nonzero = (c.tensor != 0).sum().item()
    assert nonzero <= 3


def test_adjoint_record_history():
    t = torch.randn(2, 2)
    a = AdjointState(tensor=t, layer_id=0)
    a.record("PassA")
    a.record("PassB")
    assert a.history == ["PassA", "PassB"]


def test_adjoint_to_device_dtype():
    t = torch.randn(2, 2)
    a = AdjointState(tensor=t, layer_id=0)
    moved = a.to(dtype=torch.float64)
    assert moved.tensor.dtype == torch.float64
    assert moved.dtype == torch.float64
