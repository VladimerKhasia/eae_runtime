"""
Block interface. The runtime reconstructs blocks independently - no block
should know anything about global training.

Per the spec's non-goals, we do NOT attempt automatic partitioning of
arbitrary nn.Module graphs. Manual / helper-based block specification is
sufficient: users hand us a `nn.Sequential` (or any list of `nn.Module`s
that compose linearly, e.g. repeated Transformer blocks) and we treat each
element as one reconstructable unit.
"""

from __future__ import annotations

from typing import List, Sequence

import torch.nn as nn


class EAEBlock(nn.Module):
    """Optional marker base class for a block usable by the EAE runtime.

    Subclassing this is *not* required - any `nn.Module` whose `forward`
    is a pure function of its input and its own parameters works fine.
    Subclass it when you want to be explicit about intent, or want the
    stricter contract this class documents:

      * forward(x) -> Tensor            (single tensor in, single tensor out)
      * parameters() -> Iterator[Tensor] (inherited from nn.Module)
      * no reliance on global/mutable state outside the block
    """

    def forward(self, x):  # pragma: no cover - documentation stub
        raise NotImplementedError


class BlockDecomposer:
    """Turns a user model into an ordered list of reconstructable blocks."""

    @staticmethod
    def from_sequential(seq: nn.Sequential) -> List[nn.Module]:
        return list(seq.children())

    @staticmethod
    def from_list(modules: Sequence[nn.Module]) -> List[nn.Module]:
        return list(modules)

    @staticmethod
    def decompose(model) -> List[nn.Module]:
        """Best-effort dispatcher: nn.Sequential -> children, nn.ModuleList ->
        children, list-like -> list, single nn.Module -> a one-block model.

        `nn.ModuleList` is handled explicitly (not just via generic
        `nn.Module` iteration) because it's the idiomatic way to hold a
        repeated Transformer block stack, e.g.
        `nn.ModuleList([TransformerBlock(...) for _ in range(depth)])`.
        """
        if isinstance(model, nn.Sequential):
            return BlockDecomposer.from_sequential(model)
        if isinstance(model, nn.ModuleList):
            return list(model)
        if isinstance(model, (list, tuple)):
            return BlockDecomposer.from_list(model)
        if isinstance(model, nn.Module):
            return [model]
        raise TypeError(
            f"Cannot decompose object of type {type(model)} into EAE blocks. "
            "Pass an nn.Sequential, an nn.ModuleList, a list of nn.Module, "
            "or a single nn.Module."
        )