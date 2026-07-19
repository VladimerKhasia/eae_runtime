"""
eae_runtime.contrib.transformer_blocks
=======================================

Reference, current-generation (2024-2026 era) Transformer building blocks
that are known-good EAE Runtime blocks: each `nn.Module` here has a
`forward(x) -> Tensor` that is a pure function of `x` and its own
parameters/buffers - exactly the contract `ReconstructionEngine` relies on
(see `eae_runtime.blocks.EAEBlock`).

These are deliberately **not** shipped as part of the runtime core. The
runtime is architecture-agnostic by design (see README "Non-goals"); this
module exists to:

  1. Prove, concretely, that the runtime handles modern Transformer
     internals: RMSNorm, rotary position embeddings (RoPE), grouped-query
     attention (GQA) via `F.scaled_dot_product_attention` (which
     transparently dispatches to a fused flash-attention / memory-
     efficient kernel on supported hardware), and SwiGLU feed-forward
     layers - the components of e.g. LLaMA / Mistral / Qwen-style models.
  2. Demonstrate the idiomatic pattern for handling information a block
     needs besides `x` (a causal mask, RoPE tables) *without* breaking the
     runtime's single-tensor forward contract: precompute it once in
     `__init__` / `register_buffer`, never as an extra forward() argument.
  3. Give you a working, forkable starting point. Copy this file into your
     own project and edit it; the runtime core never needs to change to
     support your architecture.

Usage::

    from eae_runtime.contrib import PreNormTransformerBlock

    blocks = nn.ModuleList([
        PreNormTransformerBlock(dim=512, num_heads=8, num_kv_heads=2)
        for _ in range(depth)
    ])
    runtime = EAERuntime(blocks, optimizer, config)

Deliberately not covered here (fork and extend instead):
  * Cross-attention / encoder-decoder blocks - attention over a second,
    externally supplied sequence isn't expressible as a pure function of a
    single `x`. Attach the memory tensor as an instance attribute the same
    way RoPE tables are attached below, or override `forward` to unpack a
    tensor you've concatenated/stacked upstream.
  * KV-cache incremental decoding - this reference targets full-sequence
    training; inference-time caching belongs in the caller's generation
    loop, not the runtime.
  * Mixture-of-Experts routing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import EAEBlock

__all__ = [
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "apply_rotary",
    "CausalSelfAttention",
    "SwiGLU",
    "PreNormTransformerBlock",
]


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (Zhang & Sennrich, 2019).

    The default normalization in LLaMA / Mistral / Gemma-style models:
    cheaper than LayerNorm (no mean-centering, no bias term) and matches
    or beats it empirically at scale.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).to(dtype) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embeddings (Su et al., 2021 / RoFormer) - the de
    facto standard positional scheme in modern decoder-only Transformers.

    Precomputes cos/sin tables as non-persistent buffers (so they move
    with `.to(device)` but are never saved to a checkpoint or trained) up
    to `max_seq_len`; `forward(seq_len)` slices out the prefix needed for
    the current sequence.
    """

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to `x` of shape (batch, heads, seq, head_dim)
    given per-position `cos`/`sin` tables of shape (seq, head_dim // 2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Causal (optionally grouped-query) self-attention built on
    `torch.nn.functional.scaled_dot_product_attention`, which dispatches
    to a fused flash-attention / memory-efficient kernel on supported
    hardware with no extra code here - the current recommended way to
    write attention in PyTorch (2.x+), rather than a hand-rolled
    `softmax(QK^T / sqrt(d)) @ V`.

    Set `num_kv_heads < num_heads` for grouped-query attention (GQA, as in
    LLaMA-2 70B / Mistral) to shrink the KV cache at inference time;
    `num_kv_heads=1` is multi-query attention (MQA). Leave it as `None`
    for ordinary multi-head attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        rope: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len) if rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(seq)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.num_heads * self.head_dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer, 2020) - the standard modern
    replacement for a plain Linear -> GELU -> Linear FFN, used in
    LLaMA / PaLM / Mistral. `hidden_dim` defaults to ~8/3 * dim rounded up
    to a multiple of `multiple_of` for hardware-friendly matmul shapes.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 256):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PreNormTransformerBlock(EAEBlock):
    """One decoder-only Transformer block in the current dominant
    architecture family (LLaMA/Mistral/Qwen-style): pre-norm residual
    stream, RMSNorm, causal (optionally grouped-query) attention with
    RoPE, RMSNorm, SwiGLU feed-forward.

    Stack `depth` of these in an `nn.ModuleList` and hand that straight to
    `EAERuntime` - `BlockDecomposer` treats each block as one
    reconstructable unit::

        blocks = nn.ModuleList([
            PreNormTransformerBlock(dim=512, num_heads=8, num_kv_heads=2)
            for _ in range(depth)
        ])
        runtime = EAERuntime(blocks, optimizer, config)

    Every sub-module here (`RMSNorm`, `CausalSelfAttention`, `SwiGLU`) is
    independently reusable if your architecture only needs one piece, and
    every EAE pass (`ClipPass`, `Int8QuantizationPass`, a custom
    compression/synthetic-gradient pass, ...) applies to the adjoint
    flowing out of this block exactly as it would for a plain `nn.Linear`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 4096,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLU(dim, hidden_dim=ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x