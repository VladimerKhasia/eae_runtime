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
pip install eae-runtime  
```

<!-- # pip install -e . -->

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
git clone https://github.com/VladimerKhasia/eae_runtime
cd eae_runtime
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

## Non-goals by design

* Automatic partitioning of arbitrary `nn.Module` graphs — blocks are
  specified manually or via `BlockDecomposer.from_sequential`/`from_list`.
* Replacing or modifying PyTorch's autograd internals.
* A new differentiation algorithm — local VJP is always PyTorch's own.
* A new compiler or graph IR.
* Day-one support for every architecture — the runtime targets
  Transformer-style models with repeated block structure first.


## Implementation of [EAE paper](https://github.com/VladimerKhasia/eae) Alg. 2

```python
#@title Memory-Optimized EAE (EAE-GC) with `eae-runtime` & VRAM Profiling
# !pip install eae-runtime
import os, math, time, json, random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# --- CORRECTED IMPORTS ---
# AdjointState is at the root level, EAEPass is in .passes
from eae_runtime import EAERuntime, RuntimeConfig, AdjointState
from eae_runtime.passes import EAEPass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

assert torch.cuda.is_available(), "Enable a GPU runtime: Runtime > Change runtime type > T4 GPU"
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
print(f"GPU: {gpu_name}  (compute capability {cc_major}.{cc_minor})")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

BF16_OK = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16_OK else torch.float16
print(f"Mixed precision dtype selected: {AMP_DTYPE}  (GradScaler {'disabled' if BF16_OK else 'enabled'})")

SEED = 1337
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Colab form params
# ---------------------------------------------------------------------------
MODEL_SIZE = "small"  #@param ["tiny", "small", "base", "custom"]
MAX_STEPS = 800  #@param {type:"integer"}
SAMPLE_EVERY_STEPS = 250  #@param {type:"integer"}
EVAL_EVERY_STEPS = 200  #@param {type:"integer"}
SAVE_EVERY_STEPS = 500  #@param {type:"integer"}
EVAL_BATCHES = 20  #@param {type:"integer"}
GENERATION_PROMPTS = "The history of artificial intelligence,Once upon a time,In order to solve this problem,"  #@param {type:"string"}
MAX_NEW_TOKENS = 60  #@param {type:"integer"}

GENERATION_PROMPTS = [p.strip() for p in GENERATION_PROMPTS.split(",") if p.strip()]

@dataclass
class ModelConfig:
    vocab_size: int = 49152
    hidden_size: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4
    intermediate_size: int = 1408
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    use_qk_norm: bool = True

PRESETS = {
    "tiny":  dict(hidden_size=256,  n_layers=8,  n_heads=8,  n_kv_heads=2, intermediate_size=704,  max_seq_len=1024),
    "small": dict(hidden_size=512,  n_layers=12, n_heads=8,  n_kv_heads=4, intermediate_size=1408, max_seq_len=1024),
    "base":  dict(hidden_size=768,  n_layers=16, n_heads=12, n_kv_heads=4, intermediate_size=2048, max_seq_len=1024),
}

if MODEL_SIZE == "custom":
    custom_cfg = dict(hidden_size=640, n_layers=14, n_heads=10, n_kv_heads=2, intermediate_size=1728, max_seq_len=1024)
    model_cfg = ModelConfig(**custom_cfg)
else:
    model_cfg = ModelConfig(**PRESETS[MODEL_SIZE])

assert model_cfg.hidden_size % model_cfg.n_heads == 0
assert model_cfg.n_heads % model_cfg.n_kv_heads == 0
print(model_cfg)

@dataclass
class TrainConfig:
    dataset_name: str = "openbmb/Ultra-FineWeb"
    dataset_split: str = "en"
    shuffle_buffer_size: int = 10_000

    micro_batch_size: int = 8
    grad_accum_steps: int = 8
    max_steps: int = MAX_STEPS
    warmup_steps: int = 200
    max_lr: float = 4e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    eval_batches: int = EVAL_BATCHES
    eval_every: int = EVAL_EVERY_STEPS
    sample_every: int = SAMPLE_EVERY_STEPS
    sample_prompts: tuple = tuple(GENERATION_PROMPTS)
    max_new_tokens: int = MAX_NEW_TOKENS

    compile_model: bool = True
    log_every: int = 20
    save_every: int = SAVE_EVERY_STEPS
    checkpoint_dir: str = "/content/checkpoints"

train_cfg = TrainConfig()
os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

# ============================================================================
# Model Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight

def precompute_rope(head_dim, max_seq_len, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.hidden_size // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.use_qk_norm = cfg.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = q.transpose(1, 2), k.transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    """Refactored to be a pure function of `x` to fulfill EAEBlock requirements"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)
        
        head_dim = cfg.hidden_size // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_theta, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x), self.rope_cos, self.rope_sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class SOTALM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for blk in self.blocks: x = blk(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================================
# Tokenizer Setup
# ============================================================================
from transformers import AutoTokenizer

TOKENIZER_CANDIDATES = ["HuggingFaceTB/SmolLM2-135M", "gpt2"]
tokenizer = None
for name in TOKENIZER_CANDIDATES:
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        break
    except Exception: pass

if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
if tokenizer.eos_token_id is None: tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
model_cfg.vocab_size = len(tokenizer)

model = SOTALM(model_cfg).to(device)

if train_cfg.compile_model:
    try:
        # We can compile individual blocks natively now that eae-runtime manages the global graph
        for i in range(len(model.blocks)):
            model.blocks[i] = torch.compile(model.blocks[i])
        model.final_norm = torch.compile(model.final_norm)
        model.lm_head = torch.compile(model.lm_head)
    except Exception as e:
        pass

# ============================================================================
# Datasets
# ============================================================================
from datasets import load_dataset

class PackedTokenDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, seed=SEED):
        self.tokenizer = tokenizer; self.seq_len = seq_len; self.seed = seed

    def __iter__(self):
        buffer = []
        while True:
            stream = load_dataset(train_cfg.dataset_name, split=train_cfg.dataset_split, streaming=True)
            stream = stream.shuffle(seed=self.seed, buffer_size=train_cfg.shuffle_buffer_size)
            for example in stream:
                text = example.get("content", "")
                if not text: continue
                ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                ids.append(self.tokenizer.eos_token_id)
                buffer.extend(ids)
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

train_loader = DataLoader(PackedTokenDataset(tokenizer, model_cfg.max_seq_len, seed=SEED), 
                          batch_size=train_cfg.micro_batch_size, collate_fn=collate, num_workers=0)

# ============================================================================
# EAE RUNTIME INTEGRATION
# ============================================================================
class CaptureAdjointPass(EAEPass):
    """Pass to catch the adjoint flowing out of the 0-th block so we can 
    backpropagate it explicitly through the non-runtime embedding layers."""
    name = "CaptureAdjointPass"
    def __init__(self):
        self.grad = None

    def process(self, adjoint: AdjointState, context=None) -> AdjointState:
        if context and context.get("block_index") == 0:
            self.grad = adjoint.tensor.clone()
        return adjoint

capture_pass = CaptureAdjointPass()

# Using EAERuntime from pip install eae-runtime
runtime_config = RuntimeConfig(
    scheduler="async",     # eae-runtime's AsyncScheduler (overlaps VJP compute with CPU offload fetch)
    memory="pool",         # eae-runtime's PoolMemoryPolicy 
    backend="auto",
    passes=[capture_pass],
    compute_dtype=AMP_DTYPE
)

# Initialize the runtime ONLY on the sequential Transformer blocks
runtime = EAERuntime(model.blocks, optimizer=None, config=runtime_config)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=train_cfg.max_lr, weight_decay=train_cfg.weight_decay, betas=train_cfg.betas, fused=True
)
scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))

def get_lr(step, cfg):
    if step < cfg.warmup_steps: return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps: return cfg.max_lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.max_lr * cfg.min_lr_ratio + 0.5 * (1.0 + math.cos(math.pi * progress)) * cfg.max_lr * (1 - cfg.min_lr_ratio)

# ============================================================================
# TRAINING LOOP WITH VRAM PROFILING
# ============================================================================

data_iter = iter(train_loader)
model.train()
t0 = time.time()
running_loss = 0.0

start_step = 0
for step in range(start_step, train_cfg.max_steps):
    lr = get_lr(step, train_cfg)
    for pg in optimizer.param_groups: pg["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0
    
    for micro_step in range(train_cfg.grad_accum_steps):
        xb, yb = next(data_iter)
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        capture_pass.grad = None
        
        # 1. Normal PyTorch Embeddings
        with torch.enable_grad():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                x_emb = model.tok_emb(xb)
        
        # 2. EAE Runtime Forward (populates internal boundary store)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            out = runtime.forward(x_emb)

        # 3. Local Boundary Condition (LM Head)
        def loss_fn(out_leaf):
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model.lm_head(model.final_norm(out_leaf))
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), yb.view(-1), ignore_index=-100)
            
            loss_fn.item_val = loss.item() / train_cfg.grad_accum_steps
            scaled_loss = loss / train_cfg.grad_accum_steps
            if AMP_DTYPE == torch.float16:
                scaled_loss = scaler.scale(scaled_loss)
            return scaled_loss

        # 4. EAE Runtime Backward Engine
        _, param_grads = runtime.backward(out, loss_fn)
        runtime.apply_gradients(param_grads)
        
        # 5. Hand the adjoint gradient back to the PyTorch embedding graph
        if capture_pass.grad is not None:
            x_emb.backward(capture_pass.grad)

        step_loss += loss_fn.item_val

    # PyTorch Optimizer Step
    if AMP_DTYPE == torch.float16:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

    running_loss += step_loss

    # ========================== LOGGING & PROFILING ==========================
    if step % train_cfg.log_every == 0:
        elapsed = time.time() - t0
        avg_loss = running_loss / train_cfg.log_every if step > 0 else step_loss
        toks_per_sec = ((step + 1) * train_cfg.micro_batch_size * train_cfg.grad_accum_steps * model_cfg.max_seq_len) / elapsed
        
        # PyTorch Native VRAM Profile
        alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
        res_gb = torch.cuda.memory_reserved(device) / (1024**3)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        # EAE Runtime Pool Memory Policy Profile
        eae_stats = runtime.memory_manager.stats()
        pool_active = eae_stats.get("active", 0)
        pool_peak = eae_stats.get("peak", 0)
        pool_reuses = eae_stats.get("reuses", 0)
        
        print(f"step {step:5d} | loss {avg_loss:.4f} | ppl {math.exp(min(avg_loss, 20)):8.2f} | lr {lr:.2e} | tok/s {toks_per_sec:,.0f}")
        print(f"  --> PyTorch VRAM: Peak: {peak_gb:.2f}GB | Allocated: {alloc_gb:.2f}GB | Reserved: {res_gb:.2f}GB")
        print(f"  --> EAE Pool Mgr: Active: {pool_active} | Peak Tensors: {pool_peak} | Total Reuses: {pool_reuses}")
        
        running_loss = 0.0
        torch.cuda.reset_peak_memory_stats(device)
```
