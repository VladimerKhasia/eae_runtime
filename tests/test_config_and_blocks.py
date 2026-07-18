import pytest
import torch
import torch.nn as nn

from eae_runtime import BlockDecomposer, EAEBlock, RuntimeConfig


def test_default_config_values():
    cfg = RuntimeConfig()
    assert cfg.scheduler == "sequential"
    assert cfg.memory == "pool"
    assert cfg.backend == "auto"
    assert cfg.passes == []
    assert cfg.compute_dtype is None
    assert cfg.boundary_offload is False


def test_config_seed_sets_torch_manual_seed():
    cfg = RuntimeConfig(seed=123)
    t1 = torch.randn(4)
    RuntimeConfig(seed=123)
    t2 = torch.randn(4)
    assert torch.allclose(t1, t2)


def test_config_accepts_pass_list():
    class FakePass:
        pass

    p = FakePass()
    cfg = RuntimeConfig(passes=[p])
    assert cfg.passes == [p]


def test_decompose_sequential():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    blocks = BlockDecomposer.decompose(model)
    assert len(blocks) == 3
    assert isinstance(blocks[0], nn.Linear)
    assert isinstance(blocks[1], nn.ReLU)


def test_decompose_list_of_modules():
    mods = [nn.Linear(2, 2), nn.Tanh()]
    blocks = BlockDecomposer.decompose(mods)
    assert blocks == mods


def test_decompose_single_module_wraps_in_list():
    m = nn.Linear(3, 3)
    blocks = BlockDecomposer.decompose(m)
    assert blocks == [m]


def test_decompose_invalid_type_raises():
    with pytest.raises(TypeError):
        BlockDecomposer.decompose(42)


def test_eae_block_forward_not_implemented_by_default():
    class MyBlock(EAEBlock):
        pass

    b = MyBlock()
    with pytest.raises(NotImplementedError):
        b.forward(torch.randn(1))


def test_eae_block_subclass_works_as_a_normal_module():
    class DoubleBlock(EAEBlock):
        def forward(self, x):
            return x * 2

    b = DoubleBlock()
    out = b(torch.ones(3))
    assert torch.allclose(out, torch.full((3,), 2.0))
