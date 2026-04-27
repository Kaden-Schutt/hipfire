# Architecture

A 10,000-foot view of how a `hipfire run` becomes tokens. Read this
before contributing kernels or dispatch changes.

## Crates

```
crates/
├── engine/             model loaders, forward pass, KV cache, DeltaNet state
├── rdna-compute/       kernel dispatch, hipGraph capture, JIT loader
├── hip-bridge/         safe Rust FFI over libamdhip64.so
├── hipfire-quantize/   CPU-side safetensors / GGUF → .mq4 / .hf4 encoder
└── redline/            direct-KMD dispatch research (future, skips HIP)
```

`engine` depends on `rdna-compute`. `rdna-compute` depends on
`hip-bridge`. `hipfire-quantize` is standalone (no GPU deps) so a CI
node without ROCm can still build the quantizer.

## Request lifecycle

```
hipfire run qwen3.5:9b "..."

  CLI (cli/index.ts, Bun/TS)
    ├── resolve tag → ~/.hipfire/models/<file>
    ├── if running daemon detected → POST /v1/chat/completions
    └── else → spawn one-shot daemon binary

  Daemon (crates/engine/examples/daemon.rs)
    ├── HfqFile::open(path)               # mmap, read header + tensor index
    ├── config_from_hfq                   # rebuild LlamaConfig / Qwen35Config
    ├── Tokenizer::from_hfq_metadata      # vocab, merges, BOS/EOS
    ├── load_weights / load_weights_hfq   # tensor → GPU upload
    └── for each token: prompt + sample loop

  Forward pass (crates/engine/src/{llama,qwen35}.rs)
    ├── per layer:
    │   ├── rmsnorm
    │   ├── (rotate_x_for_mq if MQ-quantized)
    │   ├── QKV / attention / O proj
    │   │   └── DeltaNet linear-attn for Qwen3.5 LA layers
    │   ├── residual
    │   ├── ffn_norm
    │   └── gate / up / down (SwiGLU)
    ├── final norm + lm_head
    └── sample (greedy / top-p / repeat-penalty)

  Kernels (kernels/src/*.hip)
    ├── compiled at runtime via hipcc, cached at ~/.hipfire/bin/kernels/<arch>/
    └── invoked by rdna-compute::dispatch
```

## Two model paths

hipfire has two largely-independent model loaders:

| Path | Files | Targets |
|---|---|---|
| `llama.rs` | `crates/engine/src/llama.rs` | Llama / Qwen3 / Mistral / generic dense |
| `qwen35.rs` | `crates/engine/src/qwen35.rs` | Qwen 3.5 / 3.6 hybrid (DeltaNet + FullAttention) |

`config_from_hfq` (in `hfq.rs`) sniffs `architecture` in the model's
metadata blob and dispatches to the right loader. The qwen35 path adds
DeltaNet linear-attention layers, MoE expert routing (qwen3.5_moe), and
DFlash speculative decode hooks.

Tensor naming uses the HuggingFace safetensors convention:
`model.layers.{i}.self_attn.q_proj.weight`, etc. The GGUF input path
in `hipfire-quantize` translates llama.cpp's `blk.{i}.attn_q.weight`
naming to this convention at write time so both paths read the same
tensor names.

## Dispatch layering

`rdna-compute::dispatch` is the kernel-selection hot path. Every GEMM /
GEMV / norm / fused op routes through here:

```rust
pub fn gemm_qkv_hfq4g256(&self, ...) -> HipResult<()> {
    if has_wmma_f16(&self.arch) {
        return self.gemm_qkv_hfq4g256_wmma(...);
    }
    if has_dot2_f32_f16(&self.arch) {
        return self.gemm_qkv_hfq4g256_dot2(...);
    }
    self.gemm_qkv_hfq4g256_baseline(...)
}
```

Two principles:

1. Fast paths first; baseline last. Predicates are arch-feature checks
   (`has_wmma_f16`, `has_dot2_f32_f16`) defined at the top of
   `dispatch.rs`, not inline `arch.starts_with(...)` chains.
2. **No unreachable branches.** When a new arch absorbs a check that
   was matched by an older `|| starts_with("gfxN")` clause, drop the
   redundant clause in the same diff.

## Kernel build paths

```
kernels/src/<name>.hip                    # source
~/.hipfire/bin/kernels/<arch>/<name>.hsaco  # pre-compiled blob, hash-verified
~/.hipcc-cache/<hash>.hsaco                  # JIT fallback
```

On startup the runtime checks for a pre-compiled blob matching the
source hash. If present, mmap-load. If absent, JIT through hipcc and
cache. `hipfire diag` prints which kernels came from which path.

Per-arch variants follow the dot convention:

```
gemv_hfq4g256.hip                         # default
gemv_hfq4g256.gfx1100.hip                 # chip-specific override
gemv_hfq4g256.gfx1030.v4.hip              # chip-specific versioned
gemm_qkv_hfq4g256_wmma.gfx12.hip          # family-wide override
```

`scripts/compile-kernels.sh` resolves chip → family → default in that
order. Family tags (`.gfx12.hip`) cover gfx1200 + gfx1201 with a
single file.

## KV cache

Stored as `[seq_len][n_kv_heads][head_dim]` per layer. Layout depends
on `kv_cache` config:

| Mode | K format | V format |
|---|---|---|
| `q8` | Q8_0 | Q8_0 |
| `asym3` | Lloyd-Max rotated 3-bit | Q8_0 |
| `asym4` | Lloyd-Max rotated 4-bit | Q8_0 |
| `asym2` | rotated 2-bit | Q8_0 |

The "asym" name is because K and V get different bitwidths: K is the
multi-turn recall bottleneck (small bitwidth shifts → "Kendall"
instead of "Kaden") so it gets the rotation + careful quant; V is less
sensitive and stays Q8 for speed. See [QUANTIZATION.md](QUANTIZATION.md)
for the math.

## DeltaNet (Qwen 3.5 hybrid)

Qwen 3.5+ alternates FullAttention with DeltaNet linear-attention
layers. DeltaNet replaces softmax attention with a recurrent gated
linear update — O(1) per-token compute, fixed-size state, no KV cache
for those layers (the state IS the cache).

The Qwen 3.5 config carries a `layer_types` array that decides which
layers are linear vs full. A 1D causal conv across the time axis runs
before the linear-attention update for local mixing. Per-head
learnable decay + per-head per-token gate parameterize the state
update; see `crates/engine/src/qwen35.rs` for the exact form.

## DFlash (speculative decode)

`crates/engine/src/dflash.rs`. Target model + small same-family draft
model run in parallel; the draft proposes K tokens, the target
verifies in one batched forward pass and accepts the longest
correctly-predicted prefix. Average accepted-tokens-per-cycle (τ)
drives the speedup.

Draft pairing is registry-driven, not filename-derived. Each shipped
target tag has a sibling `:<size>-draft` registry entry pointing at a
distinct file (e.g. `qwen3.5:27b` ↔ `qwen3.5:27b-draft` with files
`qwen3.5-27b.mq4` and `qwen35-27b-dflash-mq4.hfq`). The CLI passes
the resolved draft path to the daemon, which loads both target and
draft together. Toggle with `hipfire config set dflash_mode
{auto,on,off}` — default is `off` as of v0.1.8 (opt-in until the
speedup is more universally a win).

## Observability

```
HIPFIRE_PROMPT_TOKEN_HEAT=1   # per-position BPE merge-rank heat
HIPFIRE_GRAPH=1               # enable hipGraph capture (debug; AR-only)
HIPFIRE_MEMSET_DUMP=1         # log every gpu memset call:line
```

Daemon log (`~/.hipfire/serve.log`) contains layer-load progress,
kernel JIT activity, and dispatch decisions. Tail it during first-load
and any first-time arch transition.

## Where to start contributing

- **A new arch port**: read `.skills/hipfire-arch-port/` first — it
  has the WMMA matrix, dispatch routing rules, validation gates, and
  the contributor onboarding workflow.
- **A new kernel variant**: `kernels/src/<existing>.<chip>.hip` and
  wire it in `kernels.rs` + `dispatch.rs`. Run the speed-gate
  (`scripts/speed-gate.sh --fast`) before committing.
- **A new GGUF dequant type** (Q5_K / IQ4_XS / etc.): port from
  llama.cpp's `ggml-quants.c` into
  `crates/hipfire-quantize/src/gguf_input.rs`.
- **A new model architecture** (Gemma, Mistral-NeMo, etc.): start
  with `llama.rs` as the template; add the architecture string to
  `from_gguf` / `from_hfq` and any tensor-shape divergences.
