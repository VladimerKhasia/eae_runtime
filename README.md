# A programmable reverse-mode runtime for PyTorch

A production-quality runtime that replaces PyTorch's monolithic
`backward()` with a programmable, block-local reverse-mode execution engine
implementing **Explicit Adjoint Exposure (EAE)**. It builds on the the top of [EAE](https://github.com/VladimerKhasia/eae) and turnes reverse-mode execution itself into schedulable runtime. 

The runtime is **not** a new autodiff engine. It orchestrates many local
autograd computations using PyTorch's existing local autograd:

* **PyTorch owns:** local autograd, VJP computation, kernels.
* **The runtime owns:** forward scheduling, reverse scheduling,
  reconstruction, adjoint lifecycle, memory lifecycle, pass execution,
  backend selection, communication scheduling.

Users write ordinary `nn.Sequential` / `nn.Module` models. No custom tensor
type, no compiler, no FX, no PyTorch internals modified.

## Install

```bash
pip install -e .
```

## Quick start

```python
import torch
import torch.nn as nn
from eae_runtime import EAERuntime, RuntimeConfig
from eae_runtime.passes import ClipPass, Int8QuantizationPass

model = nn.Sequential(
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

config = RuntimeConfig(
    scheduler="sequential",          # "sequential" | "async" | "pipeline" | "distributed"
    memory="pool",                   # "pool" | "none"
    backend="auto",                  # "auto" | "cpu" | "cuda" | "triton" | "rocm"
    passes=[Int8QuantizationPass(), ClipPass(max_norm=1.0)],
    boundary_offload=False,          # move x0..xL to CPU between fwd/bwd
    boundary_precision=None,         # e.g. torch.float16 to shrink boundary storage
)

runtime = EAERuntime(model, optimizer, config)

x = torch.randn(32, 64)
target = torch.randint(0, 10, (32,))
loss = runtime.train_step(x, lambda out: nn.functional.cross_entropy(out, target))
```

## Architecture

```
User Model
   |
Block Decomposer            (blocks.py)
   |
Forward Executor            (forward_executor.py)   -> detached forward, no autograd graph
   |
Boundary State Store        (boundary_store.py)     -> stores only x0..xL
   |
Reverse Scheduler           (schedulers/)           -> pluggable: sequential/async/pipeline/distributed
   |            \
   |             \
Reconstruction    Memory Manager                     (reconstruction.py / memory.py)
Engine                 |            
   |                   |             
Adjoint Pipeline   Backend Manager                   (pipeline.py / backend.py)
   |
Local Autograd (PyTorch)
   |
CPU / CUDA
```

### The Adjoint Pipeline (the one addition beyond the base spec)

Every reverse step exchanges an `AdjointState` — never a raw tensor —
through a programmable pipeline:

```
AdjointState -> Pass1 -> Pass2 -> ... -> PassN -> next local VJP
```

A researcher working on gradient compression, synthetic gradients,
error-feedback, distributed communication, or adaptive optimization only
implements a new `EAEPass`. They never touch the reconstruction engine,
scheduler, or memory manager.

```python
from eae_runtime.passes import EAEPass

class MyPass(EAEPass):
    name = "MyPass"
    def process(self, adjoint, context=None):
        new = adjoint.clone()
        new.tensor = new.tensor * 0.9   # e.g. decay
        return new
```

### Extension points

| Want to add...          | Subclass                | Where                     |
|--------------------------|--------------------------|----------------------------|
| A gradient transform     | `EAEPass`                 | `eae_runtime/passes/`     |
| A reverse scheduling policy | `BaseScheduler`        | `eae_runtime/schedulers/` |
| A memory allocation strategy | `MemoryPolicy`        | `eae_runtime/memory.py`   |
| A new backend             | extend `BackendManager`  | `eae_runtime/backend.py`  |

None of these require modifying the runtime core.

### Built-in passes

`ClipPass`, `FP16Pass`, `Int8QuantizationPass` (aliased as `FP8Pass`),
`SyntheticGradientPass`, `RegularizationPass`, `GaussianNoisePass`,
`LoggingPass`.

### Built-in schedulers

* `SequentialScheduler` — the reference implementation; strict in-order
  reconstruct → VJP → pipeline → free.
* `AsyncScheduler` — overlaps boundary-activation prefetch (useful with
  `boundary_offload=True`) with reconstruction compute; numerically
  identical to `SequentialScheduler`.
* `PipelineScheduler` — splits the batch into microbatches and runs a full
  reverse pass per microbatch, accumulating gradients (single-process
  emulation of pipeline-parallel microbatching).
* `DistributedScheduler` — assigns block ranges to `torch.distributed`
  ranks and communicates adjoints across rank boundaries; falls back to
  `SequentialScheduler` (with a warning) when running single-process.

### Events

Every component emits structured events (`ForwardStarted`,
`BlockReconstructed`, `AdjointCreated`, `MemoryAllocated`,
`SchedulerStep`, `PassApplied`, ...) on an `EventBus`. Subscribe without
touching runtime code:

```python
runtime.event_bus.subscribe("BlockReconstructed", lambda e: print(e.payload))
```

Set `RuntimeConfig(log_level="DEBUG")` to also pipe every event through
Python's structured JSON logger.

### Profiling

```python
loss = runtime.train_step(x, loss_fn)
print(runtime.profiler.report())
# {'forward': {...}, 'reconstruct:Linear_0': {...}, 'pass:ClipPass': {...}, ...}
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite (`tests/`) covers:

* `AdjointState` API (norm, statistics, quantize/dequantize, compress, clone, detach)
* Event bus pub/sub, wildcard subscribers, recording
* Memory manager: pool reuse, null policy, allocator correctness, leak checks
* Boundary store: precision downcast, CPU offload, ownership semantics
* Reconstruction engine vs. manual VJP (`torch.autograd.grad`) reference
* **Gradient equivalence**: every scheduler vs. plain `model.backward()`,
  across float32, boundary-offloaded, and microbatched configurations
* All built-in passes, including a full multi-pass pipeline
* All built-in schedulers, plus a user-defined custom scheduler
* Backend manager device/dtype resolution and graceful CUDA→CPU fallback
* Profiler timing aggregation
* Mixed precision (compute dtype + boundary storage dtype) numerical bounds
* Memory-leak / allocator-stability checks over many training steps
* A full Transformer-style block stack (`nn.MultiheadAttention` + FFN,
  pre-norm, residual) trained end-to-end with a full pass pipeline
* Real two-process `torch.distributed` (gloo) correctness check for
  `DistributedScheduler` (skips gracefully if the sandbox blocks socket use)

## Non-goals (by design, matching the v2 spec)

* Automatic partitioning of arbitrary `nn.Module` graphs — blocks are
  specified manually or via `BlockDecomposer.from_sequential`/`from_list`.
* Replacing or modifying PyTorch's autograd internals.
* A new differentiation algorithm — local VJP is always PyTorch's own.
* A new compiler or graph IR.
* Day-one support for every architecture — the runtime targets
  Transformer-style models with repeated block structure first.
