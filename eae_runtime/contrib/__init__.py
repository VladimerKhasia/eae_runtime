"""
Optional, opt-in reference implementations built on top of the EAE Runtime
core. Nothing under `eae_runtime.contrib` is imported by
`eae_runtime.__init__`, and the runtime core has zero dependency on it -
this package exists purely to demonstrate and give you a starting point
for real Transformer architectures. Fork freely.
"""

from .transformer_blocks import (
    CausalSelfAttention,
    PreNormTransformerBlock,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    apply_rotary,
)

__all__ = [
    "CausalSelfAttention",
    "PreNormTransformerBlock",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "apply_rotary",
]