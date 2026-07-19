"""
End-to-end example: a small decoder-only Transformer language model,
trained through EAERuntime instead of `loss.backward()`.

This exercises the full stack in one script:
  * eae_runtime.contrib.PreNormTransformerBlock (RMSNorm + GQA causal
    attention w/ RoPE + SwiGLU) as the repeated block unit
  * a real gradient-pipeline: mixed precision compute, int8 fake-quant
    adjoint compression, gradient clipping, and structured logging, all
    as composable EAEPass instances
  * the PipelineScheduler (microbatched gradient accumulation)
  * the built-in profiler and event bus

Run:
    python examples/train_transformer_lm.py

This trains on a tiny synthetic copy-the-input task purely to demonstrate
the wiring and confirm the loss goes down - it is not a real language
modeling benchmark.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eae_runtime import EAERuntime, RuntimeConfig
from eae_runtime.contrib import PreNormTransformerBlock
from eae_runtime.passes import ClipPass, Int8QuantizationPass, LoggingPass


class TokenEmbedding(nn.Module):
    """Embedding lookup as its own EAE block. Kept separate from the
    Transformer blocks so it can be reconstructed/quantized independently
    (embeddings and attention layers often warrant different pass
    pipelines in practice, e.g. no int8 adjoint compression on the
    embedding gradient, which is typically already sparse)."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, token_ids_as_float: torch.Tensor) -> torch.Tensor:
        # The runtime passes a single Tensor through every block; token ids
        # are carried as a float tensor and cast back to long here so the
        # whole model still satisfies forward(Tensor) -> Tensor end to end.
        return self.embed(token_ids_as_float.long())


class LMHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


def make_model(vocab_size: int, dim: int, num_heads: int, depth: int, max_seq_len: int) -> nn.ModuleList:
    blocks: list[nn.Module] = [TokenEmbedding(vocab_size, dim)]
    blocks += [
        PreNormTransformerBlock(dim=dim, num_heads=num_heads, num_kv_heads=2, max_seq_len=max_seq_len)
        for _ in range(depth)
    ]
    blocks.append(LMHead(dim, vocab_size))
    return nn.ModuleList(blocks)


def synthetic_batch(vocab_size: int, seq_len: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    # toy task: predict the same sequence shifted by one position
    targets = torch.roll(tokens, shifts=-1, dims=1)
    return tokens.float(), targets


def main() -> None:
    torch.manual_seed(0)
    vocab_size, dim, num_heads, depth, seq_len, batch_size = 64, 64, 4, 3, 16, 8

    model = make_model(vocab_size, dim, num_heads, depth, max_seq_len=seq_len)
    optimizer = torch.optim.AdamW(
        [p for block in model for p in block.parameters()], lr=3e-3
    )

    config = RuntimeConfig(
        scheduler="pipeline",
        num_microbatches=2,
        memory="pool",
        backend="auto",
        passes=[
            Int8QuantizationPass(),  # simulate low-precision gradient compression
            ClipPass(max_norm=1.0),
            LoggingPass(),
        ],
        grad_clip_norm=5.0,
        log_level="WARNING",
    )
    runtime = EAERuntime(model, optimizer, config)

    print(f"Backend resolved to: {runtime.backend_manager.resolved}")
    print(f"Blocks: {runtime.block_names}")

    # Train on one fixed batch (the standard "can it overfit?" sanity
    # check) so the loss curve directly demonstrates that gradients are
    # flowing correctly end to end through the runtime, rather than
    # measuring language-modeling quality on resampled random tokens.
    tokens, targets = synthetic_batch(vocab_size, seq_len, batch_size)

    for step in range(50):
        loss = runtime.train_step(
            tokens, lambda logits: nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        )
        if step % 10 == 0 or step == 49:
            print(f"step {step:3d}  loss {loss:.4f}  grad_norm {runtime.last_grad_stats['total_grad_norm']:.4f}")

    print("\nProfiler report (top-level keys):")
    for name, stats in runtime.profiler.report().items():
        if name in ("forward", "reverse_pass"):
            print(f"  {name}: {stats}")


if __name__ == "__main__":
    main()