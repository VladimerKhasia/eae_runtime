import pytest
import torch

from eae_runtime import BackendManager


def test_auto_resolves_to_cpu_when_no_cuda():
    bm = BackendManager("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert bm.resolved == expected


def test_cpu_backend_stays_cpu():
    bm = BackendManager("cpu")
    assert bm.resolved == "cpu"
    assert bm.device_for() == torch.device("cpu")


def test_cuda_backend_falls_back_gracefully_without_gpu():
    bm = BackendManager("cuda")
    if torch.cuda.is_available():
        assert bm.resolved == "cuda"
    else:
        assert bm.resolved == "cpu"


def test_triton_backend_falls_back_to_cpu_without_gpu():
    bm = BackendManager("triton")
    if not torch.cuda.is_available():
        assert bm.resolved == "cpu"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        BackendManager("not-a-backend")


def test_block_can_override_backend():
    class Dummy:
        eae_backend = "cpu"

    bm = BackendManager("auto")
    assert bm.device_for(Dummy()) == torch.device("cpu")


def test_autocast_disabled_is_noop_context():
    bm = BackendManager("cpu")
    with bm.autocast(enabled=False):
        x = torch.randn(2, 2)
        y = x @ x
    assert y.dtype == torch.float32


def test_repr_contains_requested_and_resolved():
    bm = BackendManager("cpu")
    r = repr(bm)
    assert "cpu" in r
