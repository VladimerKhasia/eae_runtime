import torch

from eae_runtime import BoundaryStore


def test_put_get_roundtrip():
    store = BoundaryStore()
    t = torch.randn(3, 3)
    store.put(0, t)
    got = store.get(0)
    assert torch.allclose(got, t)


def test_store_owns_independent_copy():
    store = BoundaryStore()
    t = torch.randn(3, 3)
    store.put(0, t)
    t.add_(100.0)  # mutate the original after storing
    got = store.get(0)
    assert not torch.allclose(got, t)


def test_missing_key_raises():
    store = BoundaryStore()
    try:
        store.get(5)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_precision_downcast_and_restore_dtype():
    store = BoundaryStore(precision=torch.float16)
    t = torch.randn(4, 4, dtype=torch.float32)
    store.put(0, t)
    got = store.get(0)  # restores original dtype by default
    assert got.dtype == torch.float32
    # values should be close (fp16 roundtrip loses precision)
    assert torch.allclose(got, t, atol=1e-2)


def test_offload_to_cpu():
    store = BoundaryStore(offload=True)
    t = torch.randn(2, 2)
    store.put(0, t)
    assert store._store[0].device.type == "cpu"
    got = store.get(0)
    assert torch.allclose(got, t)


def test_clear_empties_store():
    store = BoundaryStore()
    store.put(0, torch.randn(2, 2))
    store.put(1, torch.randn(2, 2))
    assert len(store) == 2
    store.clear()
    assert len(store) == 0


def test_memory_bytes_reports_something_positive():
    store = BoundaryStore()
    store.put(0, torch.randn(100, 100))
    assert store.memory_bytes() > 0


def test_contains():
    store = BoundaryStore()
    store.put(3, torch.randn(1))
    assert 3 in store
    assert 4 not in store
