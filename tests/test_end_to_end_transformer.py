import copy

import torch
import torch.nn as nn

from eae_runtime import EAEBlock, EAERuntime, RuntimeConfig
from eae_runtime.passes import ClipPass, FP16Pass, LoggingPass, RegularizationPass


class ToyTransformerBlock(EAEBlock):
    """A minimal pre-norm Transformer encoder block: self-attention +
    feed-forward, each with a residual connection. Demonstrates that the
    runtime works with real, stateful, multi-parameter blocks - not just
    single nn.Linear layers - and that block granularity is exactly the
    level EAE naturally applies to (spec: "Focus first on Transformer-style
    models with repeated block structures")."""

    def __init__(self, dim, heads=2, ff_dim=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


def _build_stack(dim=16, num_blocks=3, seed=0):
    torch.manual_seed(seed)
    blocks = [ToyTransformerBlock(dim) for _ in range(num_blocks)]
    return blocks


def test_transformer_stack_gradient_equivalence():
    dim, seq, batch = 16, 5, 4
    blocks = _build_stack(dim=dim, num_blocks=3, seed=1)
    ref_blocks = copy.deepcopy(blocks)

    x = torch.randn(batch, seq, dim)
    target = torch.randn(batch, seq, dim)

    # plain PyTorch reference
    for b in ref_blocks:
        b.zero_grad()
    h = x
    for b in ref_blocks:
        h = b(h)
    ref_loss = nn.functional.mse_loss(h, target)
    ref_loss.backward()

    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu")
    runtime = EAERuntime(blocks, optimizer=None, config=cfg)
    loss, grads = runtime.compute_gradients(x, lambda out: nn.functional.mse_loss(out, target))

    assert torch.isclose(loss, ref_loss, atol=1e-5)

    ref_by_name = {}
    for i, b in enumerate(ref_blocks):
        for n, p in b.named_parameters():
            ref_by_name[f"{i}.{n}"] = p

    got_by_name = {}
    for i, b in enumerate(blocks):
        for n, p in b.named_parameters():
            got_by_name[f"{i}.{n}"] = p
    name_of = {id(p): n for n, p in got_by_name.items()}

    checked = 0
    for p, g in grads.items():
        n = name_of[id(p)]
        ref_g = ref_by_name[n].grad
        if ref_g is None:
            continue
        assert torch.allclose(g, ref_g, atol=1e-4, rtol=1e-3), f"mismatch at {n}"
        checked += 1
    assert checked > 0


def test_transformer_stack_trains_with_full_pass_pipeline_and_optimizer():
    dim, seq, batch = 16, 4, 4
    blocks = _build_stack(dim=dim, num_blocks=2, seed=2)
    x = torch.randn(batch, seq, dim)
    target = torch.randn(batch, seq, dim)

    all_params = [p for b in blocks for p in b.parameters()]
    optimizer = torch.optim.Adam(all_params, lr=1e-2)

    cfg = RuntimeConfig(
        scheduler="sequential",
        memory="pool",
        backend="cpu",
        passes=[RegularizationPass(strength=0.0), FP16Pass(), ClipPass(max_norm=5.0), LoggingPass()],
        grad_clip_norm=10.0,
    )
    runtime = EAERuntime(blocks, optimizer, cfg)

    losses = [runtime.train_step(x, lambda out: nn.functional.mse_loss(out, target)) for _ in range(15)]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0]

    report = runtime.profiler.report()
    assert any(k.startswith("reconstruct:") for k in report)
    assert any(k.startswith("pass:") for k in report)


def test_no_custom_tensor_type_plain_tensors_flow_through():
    """PyTorch compatibility principle: users write ordinary tensors and
    ordinary nn.Module; the runtime never requires a custom Tensor
    subclass anywhere on the user-facing surface."""
    dim = 8
    blocks = _build_stack(dim=dim, num_blocks=1, seed=3)
    x = torch.randn(2, 3, dim)
    assert type(x) is torch.Tensor

    cfg = RuntimeConfig(scheduler="sequential", memory="pool", backend="cpu")
    runtime = EAERuntime(blocks, optimizer=None, config=cfg)
    out = runtime.forward(x)
    assert type(out) is torch.Tensor
