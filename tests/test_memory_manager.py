import torch

from eae_runtime import MemoryManager, NullMemoryPolicy, PoolMemoryPolicy


def test_null_policy_tracks_active_and_peak():
    mm = MemoryManager(policy="none")
    t1 = mm.request((4, 4))
    t2 = mm.request((4, 4))
    stats = mm.stats()
    assert stats["active"] == 2
    assert stats["peak"] == 2
    mm.release(t1)
    mm.release(t2)
    assert mm.stats()["active"] == 0


def test_pool_policy_reuses_freed_tensor():
    mm = MemoryManager(policy="pool")
    t1 = mm.request((8, 8), dtype=torch.float32, device="cpu")
    data_ptr_before = t1.data_ptr()
    mm.release(t1)
    t2 = mm.request((8, 8), dtype=torch.float32, device="cpu")
    assert mm.stats()["reuses"] >= 1
    assert t2.data_ptr() == data_ptr_before


def test_pool_policy_distinguishes_shape_dtype_device():
    mm = MemoryManager(policy="pool")
    a = mm.request((4, 4), dtype=torch.float32)
    mm.release(a)
    # different shape must not reuse a's buffer
    b = mm.request((4, 5), dtype=torch.float32)
    assert b.shape == (4, 5)
    stats = mm.stats()
    assert stats["allocations"] >= 2


def test_pool_max_size_limits_bucket_growth():
    policy = PoolMemoryPolicy(max_pool_per_key=2)
    mm = MemoryManager(policy=policy)
    tensors = [mm.request((2, 2)) for _ in range(5)]
    for t in tensors:
        mm.release(t)
    assert mm.stats()["pooled"] <= 2


def test_reset_clears_stats():
    mm = MemoryManager(policy="pool")
    t = mm.request((3, 3))
    mm.release(t)
    mm.reset()
    stats = mm.stats()
    assert stats["active"] == 0
    assert stats["peak"] == 0
    assert stats["pooled"] == 0


def test_events_emitted_on_allocate_and_release():
    mm = MemoryManager(policy="pool")
    mm.event_bus.start_recording()
    t = mm.request((2, 2))
    mm.release(t)
    log = mm.event_bus.log
    types = [e.type for e in log]
    assert "MemoryAllocated" in types
    assert "MemoryReleased" in types
