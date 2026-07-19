import copy

import pytest
import torch
import torch.nn as nn

from eae_runtime import EAERuntime, RuntimeConfig
from eae_runtime.contrib import (
    CausalSelfAttention,
    PreNormTransformerBlock,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    apply_rotary,
)


def test_rmsnorm_output_shape_and_unit_scale_when_weight_is_one():
    dim = 16
    norm = RMSNorm(dim)
    x = torch.randn(2, 5, dim) * 3.0
    out = norm(x)
    assert out.shape == x.shape
    # RMS of the normalized output (before the learned weight, which starts
    # at 1) should be ~1 along the last dim.
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rotary_embedding_preserves_norm():
    """A rotation is norm-preserving: applying RoPE to a vector must not
    change its L2 norm, only its direction within each rotated pair."""
    head_dim = 8
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=32)
    x = torch.randn(2, 3, 10, head_dim)  # (batch, heads, seq, head_dim)
    cos, sin = rope(10)
    rotated = apply_rotary(x, cos, sin)
    assert rotated.shape == x.shape
    assert torch.allclose(
        torch.linalg.vector_norm(x, dim=-1),
        torch.linalg.vector_norm(rotated, dim=-1),
        atol=1e-4,
    )


def test_rotary_embedding_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(head_dim=7)


def test_causal_self_attention_output_shape():
    dim, heads, seq, batch = 32, 4, 6, 2
    attn = CausalSelfAttention(dim, heads)
    x = torch.randn(batch, seq, dim)
    out = attn(x)
    assert out.shape == (batch, seq, dim)


def test_causal_self_attention_is_causal():
    """Changing a future token must not change an earlier position's
    output - the defining property of causal (autoregressive) attention."""
    dim, heads, seq, batch = 16, 2, 5, 1
    torch.manual_seed(0)
    attn = CausalSelfAttention(dim, heads, rope=False)
    attn.eval()
    x = torch.randn(batch, seq, dim)
    out1 = attn(x)

    x2 = x.clone()
    x2[:, -1, :] = torch.randn(dim)  # perturb only the last token
    out2 = attn(x2)

    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6)
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])


def test_causal_self_attention_supports_grouped_query_attention():
    dim, heads, kv_heads, seq, batch = 32, 8, 2, 4, 2
    attn = CausalSelfAttention(dim, heads, num_kv_heads=kv_heads)
    x = torch.randn(batch, seq, dim)
    out = attn(x)
    assert out.shape == (batch, seq, dim)


def test_causal_self_attention_rejects_invalid_head_config():
    with pytest.raises(ValueError):
        CausalSelfAttention(dim=30, num_heads=4)  # 30 not divisible by 4
    with pytest.raises(ValueError):
        CausalSelfAttention(dim=32, num_heads=8, num_kv_heads=3)  # 8 % 3 != 0


def test_swiglu_output_shape_and_default_hidden_dim_rounding():
    dim = 64
    ffn = SwiGLU(dim, multiple_of=32)
    x = torch.randn(3, 7, dim)
    out = ffn(x)
    assert out.shape == x.shape
    assert ffn.gate_proj.out_features % 32 == 0


def test_prenorm_transformer_block_output_shape():
    dim, heads, seq, batch = 32, 4, 6, 3
    block = PreNormTransformerBlock(dim=dim, num_heads=heads)
    x = torch.randn(batch, seq, dim)
    out = block(x)
    assert out.shape == x.shape


def test_prenorm_transformer_stack_gradient_equivalence_with_eae_runtime():
    """The end-to-end proof: a stack of PreNormTransformerBlock (RMSNorm +
    GQA causal attention w/ RoPE + SwiGLU) trained through EAERuntime
    produces bit-for-bit-close gradients to plain PyTorch backward()."""
    dim, heads, kv_heads, seq, batch, depth = 24, 4, 2, 5, 3, 2
    torch.manual_seed(7)
    blocks = nn.ModuleList(
        [
            PreNormTransformerBlock(dim=dim, num_heads=heads, num_kv_heads=kv_heads, max_seq_len=32)
            for _ in range(depth)
        ]
    )
    ref_blocks = copy.deepcopy(blocks)

    x = torch.randn(batch, seq, dim)
    target = torch.randn(batch, seq, dim)

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
    name_of = {id(p): f"{i}.{n}" for i, b in enumerate(blocks) for n, p in b.named_parameters()}

    checked = 0
    for p, g in grads.items():
        n = name_of[id(p)]
        ref_g = ref_by_name[n].grad
        if ref_g is None:
            continue
        assert torch.allclose(g, ref_g, atol=1e-4, rtol=1e-3), f"mismatch at {n}"
        checked += 1
    assert checked > 0