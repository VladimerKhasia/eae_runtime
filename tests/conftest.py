import copy

import pytest
import torch
import torch.nn as nn


def make_mlp(in_dim=16, hidden=16, out_dim=8, depth=4, seed=0):
    torch.manual_seed(seed)
    layers = []
    d = in_dim
    for i in range(depth):
        nd = hidden if i < depth - 1 else out_dim
        layers.append(nn.Linear(d, nd))
        if i < depth - 1:
            layers.append(nn.ReLU())
        d = nd
    return nn.Sequential(*layers)


@pytest.fixture
def mlp_factory():
    """Returns a callable that builds a fresh MLP with a given seed, plus a
    deep-copied twin so plain-PyTorch and EAE runtime paths start from
    identical weights."""

    def _factory(seed=0, **kwargs):
        model = make_mlp(seed=seed, **kwargs)
        twin = copy.deepcopy(model)
        return model, twin

    return _factory


@pytest.fixture
def sample_batch():
    torch.manual_seed(42)
    x = torch.randn(6, 16)
    target = torch.randn(6, 8)
    return x, target
