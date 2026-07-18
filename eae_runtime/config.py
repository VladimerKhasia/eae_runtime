"""
Single configuration object for the runtime. No scattered options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch


@dataclass
class RuntimeConfig:
    # scheduler: name string ("sequential" | "async" | "pipeline" | "distributed")
    # or a BaseScheduler instance for full user control.
    scheduler: Union[str, Any] = "sequential"

    # memory policy: name string ("pool" | "none") or a MemoryPolicy instance.
    memory: Union[str, Any] = "pool"

    # backend: name string ("cpu" | "cuda" | "triton" | "rocm") or a
    # BackendManager instance.
    backend: Union[str, Any] = "auto"

    # ordered list of EAEPass instances forming the adjoint pipeline.
    passes: List[Any] = field(default_factory=list)

    # mixed precision compute dtype used during reconstruction (None = fp32).
    compute_dtype: Optional[torch.dtype] = None

    # boundary store options
    boundary_offload: bool = False          # move boundary activations to CPU
    boundary_precision: Optional[torch.dtype] = None  # e.g. torch.float16 to save memory
    pin_memory: bool = False

    # pipeline scheduler option
    num_microbatches: int = 1

    # misc
    grad_clip_norm: Optional[float] = None
    seed: Optional[int] = None
    enable_profiler: bool = True
    log_level: str = "WARNING"

    def __post_init__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
