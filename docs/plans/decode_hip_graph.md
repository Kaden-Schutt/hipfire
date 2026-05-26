# Decode HIP Graph Capture Design

**Status:** Design (not yet implemented)
**Date:** 2026-05-26
**Context:** dots.ocr decode profiling on gfx1151 (Strix Halo)

## Problem

Decode profiling on the dots.ocr Qwen2 1.5B model shows **76% of per-token
wall time is GPU idle** (host-side dispatch gap):

| Phase | Time/token | Share |
|---|---|---|
| GPU compute | 17.8 ms | 24% |
| Dispatch gap (host) | 57.4 ms | 76% |
| **Total wall** | **75.2 ms** | 100% |

567 kernel launches per decode token at ~101 µs average dispatch latency.
At 13 tok/s, decode accounts for 99% of end-to-end OCR runtime (347s of 352s).

**Theoretical with graph capture:** eliminating dispatch gap → ~18 ms/token
= ~56 tok/s (4.2× speedup). This is the single largest remaining lever.

## Existing Infrastructure

hipfire already has a full hipGraph capture stack:

- `Gpu.capture_mode: bool` — when true, `launch_maybe_blob` routes to the
  blob path (kernargs copied into `KernargBlob` heap allocations)
- `Gpu.capture_blobs: Vec<Vec<u8>>` — retained kernarg blobs (must outlive
  the graph)
- `Gpu.begin_graph_capture()` / `end_graph_capture()` / `graph_launch()` —
  generic capture lifecycle
- `Gpu.verify_graph_cache: HashMap<usize, (Graph, GraphExec, Vec<Vec<u8>>)>` —
  per-batch-size graph cache for DFlash verify
- `Gpu.replay_graph_cache: HashMap<usize, (Graph, GraphExec, Vec<Vec<u8>>)>` —
  per-step-count cache for DFlash replay
- `launch_maybe_blob()` — dispatches via blob during capture, normal
  `kernelParams` otherwise

All of this is in `crates/rdna-compute/src/dispatch.rs`.

## Dispatch Audit

The decode loop (`forward_step` in
`crates/hipfire-arch-qwen2/src/qwen2.rs:800-902`) calls these dispatch
functions per token. Functions marked ❌ use direct `hip.launch_kernel`
(stack pointers → dangling on replay) and must be converted to
`launch_maybe_blob` before graph capture works.

| Function | Launches/step | Status | Notes |
|---|---|---|---|
| `rmsnorm_f32` | 57 | ✅ | Already uses `launch_maybe_blob` |
| `gemv_q8_0` | 197 | ✅ | Already uses `launch_maybe_blob` (both wide + narrow) |
| `add_inplace_f32` | 56 | ✅ | Already uses `launch_maybe_blob` |
| `silu_mul_f32` | 28 | ✅ | Already uses `launch_maybe_blob` |
| `attention_flash_gqa` | 56 | ❌ | 2 direct launches (partial + reduce) |
| `rope_f32` | 28 | ❌ | 1 direct launch |
| `kv_cache_write` | 56 | ❌ | 1 direct launch |
| `bias_add_f32` | 84 | ❌ | 1 direct launch |
| `argmax_f32` | 1 | ❌ | Special: allocs temp buf + sync D2H copy |

**225 launches need conversion** (40% of total). The remaining 342 launches
(60%) are already graph-safe.

### argmax_f32 special case

`argmax_f32` (`dispatch.rs:20671`) does a synchronous `malloc → launch →
memcpy_dtoh → free` cycle. This is fundamentally incompatible with graph
capture (no malloc/memcpy during capture; async D2H would need a persistent
staging buffer + event sync). Options:

1. **Pre-allocate a result buffer** in `Qwen2State`, convert to async D2H
   with event sync outside the graph, launch via `launch_maybe_blob`.
2. **Leave argmax outside the graph** — only 1 launch/step, negligible cost.
   Graph replays through the FFN output, then a separate argmax launch
   reads `state.logits`. This is simpler and the 1-launch overhead is
   ~100 µs.

Recommendation: **option 2** (leave outside graph).

## Design

### Graph cache

Keyed by `n_chunks` bucket (attention grid size changes when seq_len crosses
a chunk_size boundary). Add to `Gpu`:

```rust
pub decode_graph_cache: HashMap<u32, (hip_bridge::Graph, hip_bridge::GraphExec, Vec<Vec<u8>>)>,
```

Where the key is `n_chunks` (u32). For dots.ocr with chunk_size=128:
- Positions 0–127: n_chunks=1
- Positions 128–255: n_chunks=2
- ...
- Positions 5095+: n_chunks=40

Over a 4633-token decode (positions 5095–9728), n_chunks goes from 40 to 77
= ~37 re-captures. Each capture takes ~5ms (warmup + capture + instantiate),
so ~185ms total overhead vs the ~260s saved. Net win: ~260s saved.

### Capture lifecycle

The decode loop currently looks like:

```
for each token:
    forward_step(gpu, ...)     // 567 kernel launches
```

With graph capture:

```
for each token:
    n_chunks = compute_n_chunks(pos + 1)
    if need_new_graph(n_chunks):
        warmup: forward_step(gpu, ...)     // 1 normal pass
        gpu.begin_decode_graph_capture(n_chunks)
        forward_step(gpu, ...)             // captured
        gpu.end_decode_graph_capture()
    gpu.replay_decode_graph(n_chunks)

    // Update mutable state between replays:
    gpu.memcpy_htod_auto(pos_buf, &[pos as i32])   // async on capture stream
    gpu.kv_cache_write(...)  // wait — this is IN the graph already

    // Argmax outside graph:
    gpu.argmax_f32(logits, vocab_size)
```

Wait — there's a subtlety. The graph captures the *entire* `forward_step`,
including kv_cache_write. On replay, the kv_cache_write positions are baked
into the graph. We need `pos_buf` (a device buffer) to be mutable between
replays — but the graph records the pointer, not the value, so updating
`pos_buf` contents before replay works.

Similarly, all device pointers (tensor buffers) in the graph are stable
addresses. Only scalar arguments (seq_len, pos) need to change. But under
the blob path, scalars are baked into the blob. This is the fundamental
challenge.

### Mutable arguments problem

The attention grid depends on `n_chunks = ceil(seq_len / chunk_size)`.
When `n_chunks` changes, we must re-capture. But within a single n_chunks
bucket, `seq_len` (a kernel argument) changes every token.

Under graph capture, `launch_maybe_blob` copies the current scalar values
into the blob. On replay, those same values are replayed. So even within
one n_chunks bucket, we'd need per-token graphs if seq_len is a kernel
argument.

**Option A: Re-capture every n_chunks boundary.** This works because
seq_len changes by 1 per token, and the only kernel that uses seq_len
as a grid dimension is attention (grid = `[n_kv_heads, n_chunks]`).
The partial kernel receives seq_len as a scalar arg for the S loop bound.
If we re-capture at every n_chunks boundary, the scalar seq_len only
changes by chunk_size within a capture — still not exact.

Actually, looking at `attention_flash_gqa_partial` more carefully: the
kernel uses `seq_len` as a loop bound (iterated up to seq_len), and the
grid uses `n_chunks`. So within one n_chunks bucket, seq_len changes
every token, but the kernel argument changes too. On replay, the old
seq_len value would be used — **wrong**.

This means we cannot simply replay the same graph for different seq_len
values. The options are:

**Option A: Per-n_chunks capture with device-side seq_len.** Pass seq_len
via a device buffer (like `pos_buf`), not a scalar kernarg. The kernel
reads seq_len from the device buffer at launch time. The graph records
the device buffer pointer; we update the contents before each replay.
This requires modifying `attention_flash_gqa_partial` to read seq_len
from a pointer instead of a kernarg.

**Option B: hipGraphExecKernelNodeSetParams.** ROCm supports updating
individual node parameters on an instantiated graph exec without
re-capture. This lets us update the seq_len scalar in the attention
node between replays. Requires node handle tracking.

**Option C: Re-capture every token.** Defeats the purpose — capture
overhead would exceed dispatch savings.

**Option D: Re-capture every n_chunks boundary + accept approximate
seq_len.** The kernel loops to seq_len, but if we capture with a
slightly larger seq_len (the bucket's maximum), the kernel reads a
few extra positions from the KV cache. The extra positions contain
zeros (unwritten cache entries), which contribute near-zero attention
weight. Quality impact: negligible (zeros → zero attention). This is
the simplest option.

### Recommended approach: Option D (over-seq capture)

Capture with `seq_len = bucket_end` (e.g., n_chunks × chunk_size) for
each bucket. The kernel scans a few extra positions (at most chunk_size-1)
which contain zeroed KV cache rows. Those contribute zero attention weight,
so output is numerically identical to exact-seq.

Advantages:
- No kernel changes needed
- Re-use existing capture infrastructure exactly
- Only ~37 re-captures over 4633 tokens

Disadvantages:
- ~3-6% attention compute overhead from scanning extra positions
  (up to 127 extra out of 5000+ = negligible)

### Additional mutable arguments

Other kernel arguments that change per token:
- `pos_buf` (position for kv_cache_write, rope) — already a device buffer,
  updated via `memcpy_htod` before replay. Graph records pointer → works.
- `embedding_lookup_q8` token id — outside graph (1 launch, negligible)
- KV cache pointers — stable addresses, no change
- Weight pointers — stable addresses, no change
- `n_chunks` grid dimension — changes only at bucket boundaries (re-capture)

## Implementation Plan

### Phase 1: Convert dispatch functions to launch_maybe_blob

Five functions need conversion. Each follows the same pattern as the
existing `rmsnorm_f32` conversion:

1. **`rope_f32`** (`dispatch.rs:21046`) — 1 launch, 7 kernargs
2. **`kv_cache_write`** (`dispatch.rs:24137`) — 1 launch, 4 kernargs
3. **`bias_add_f32`** (`dispatch.rs:26317`) — 1 launch, 4 kernargs
4. **`attention_flash_gqa`** (`dispatch.rs:21294`) — 2 launches (partial + reduce)
5. **`attention_flash`** (`dispatch.rs:21195`) — 2 launches (flash + reduce)

Pattern (example for `bias_add_f32`):

```rust
// Before (direct launch):
unsafe { self.hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }

// After (launch_maybe_blob):
self.launch_maybe_blob(
    "bias_add_f32", [blocks, 1, 1], [256, 1, 1], 0, &mut params,
    || {
        let mut b = hip_bridge::KernargBlob::new();
        b.push_ptr(xp); b.push_ptr(bp); b.push_i32(ni); b.push_i32(ti);
        b
    },
)
```

For `attention_flash_gqa`, both the partial and reduce launches need
conversion. The reduce can use the same blob builder pattern.

Estimated effort: ~150 lines changed across 5 functions. Mechanical.

### Phase 2: Add decode graph cache to Gpu

New fields on `Gpu`:

```rust
decode_graph_cache: HashMap<u32, (hip_bridge::Graph, hip_bridge::GraphExec, Vec<Vec<u8>>)>,
decode_graph_warmed_up: HashSet<u32>,
```

New methods (model after `verify_has_graph`, `begin_verify_graph_capture`,
etc.):

- `decode_has_graph(n_chunks) -> bool`
- `decode_needs_warmup(n_chunks) -> bool`
- `begin_decode_graph_capture(n_chunks)`
- `end_decode_graph_capture()`
- `replay_decode_graph(n_chunks)`

### Phase 3: Modify forward_step for capture/replay

In `crates/hipfire-arch-qwen2/src/qwen2.rs`, `forward_step`:

```rust
let n_chunks = ((pos + 1 + chunk_size - 1) / chunk_size) as u32;
if gpu.decode_has_graph(n_chunks) {
    // Update mutable device buffers
    gpu.memcpy_htod_auto(&state.pos_buf, &(pos as i32).to_ne_bytes())?;
    gpu.replay_decode_graph(n_chunks)?;
} else if gpu.decode_needs_warmup(n_chunks) {
    forward_step_inner(gpu, weights, cfg, state, token)?;  // normal warmup pass
    gpu.decode_mark_warmup_done(n_chunks);
} else {
    // Capture
    gpu.memcpy_htod_auto(&state.pos_buf, &(pos as i32).to_ne_bytes())?;
    gpu.begin_decode_graph_capture(n_chunks)?;
    forward_step_inner(gpu, weights, cfg, state, token)?;  // captured
    gpu.end_decode_graph_capture()?;
}
```

The argmax stays outside the graph (1 launch, ~100 µs, not worth the
complexity of pre-allocating a staging buffer).

### Phase 4: Handle seq_len in attention

For Option D (over-seq capture), the attention kernels receive
`seq_len = bucket_end` during capture. On replay, the extra positions
are zeroed KV cache → zero attention weight → numerically identical.

Need to check: is the KV cache zero-initialized? If not, we need to
`memset` new cache rows after allocation. Current code in
`Qwen2State::new` should already zero-init via `gpu.alloc_zeroed_tensor`
or similar — verify this.

### Phase 5: Correctness + perf validation

1. Run `ocr_e2e` with graph capture enabled, verify identical F1 score
2. Run coherence-gate on qwen3.5 (existing models) to ensure no regression
3. Measure decode tok/s: expect ~4× improvement
4. Benchmark re-capture overhead: should be <1% of total decode time

## Risks

1. **ROCm graph capture bugs.** hipGraph capture on ROCm has historically
   had issues with async memcpy, shared memory, and certain kernel launch
   configurations. The existing DFlash verify/replay paths have already
   shaken out most of these, but the decode path uses different kernels.

2. **KV cache zero-init assumption.** If the cache isn't zeroed, scanning
   past the current position reads garbage → corrupted attention. Easy to
   verify and fix (memset on alloc).

3. **Memory pressure.** Each cached graph retains its kernarg blobs. At
   ~37 cached graphs × ~567 kernels × ~100 bytes/kernarg = ~2 MB. Negligible.

4. **Model swap.** The decode graph cache must be invalidated when the
   model changes (same as `verify_graph_cache` invalidation in
   `Gpu::unload_model`).

## Alternatives Considered

### Fused GQA attention (single-launch)

`attention_flash_gqa_fused` (`dispatch.rs:21335`, kernel at
`kernels/src/attention_flash_gqa_fused.hip`) eliminates the partials
buffer and reduce launch by streaming all positions in one kernel per
kv_head. Wired up behind `HIPFIRE_GQA_FUSED=1` in `qwen2.rs:869`.

**Benchmarked 2026-05-26 on gfx1151 Strix Halo:**
- Baseline (split-K): 2.2 tok/s
- Fused: 0.9 tok/s (2.6× slower)

Grid = n_kv_heads only (2 blocks for dots.ocr). With 96 CUs, occupancy
is catastrophically low. The split-K approach with n_chunks × n_kv_heads
blocks is far superior. **Not viable for this model config.**

### hipGraphExecKernelNodeSetParams

ROCm's API for updating individual node params without re-capture.
Avoids re-capture at n_chunks boundaries but requires:
- Tracking node handles from the captured graph
- Per-node param updates between replays
- More complex code for marginal benefit (~37 re-captures)

Deferred. Option D (over-seq) is simpler and sufficient.

### Device-side seq_len buffer

Modify attention kernels to read seq_len from a device pointer instead
of a kernarg scalar. The device buffer is updated before each replay.
This gives exact-seq without re-capture. Requires kernel changes to
all attention variants (flash, gqa, partial, reduce).

Deferred. Option D is simpler and the compute overhead is negligible.
Could be done as a follow-up if the over-seq approximation ever matters.

## Estimated Effort

| Phase | Lines changed | Effort |
|---|---|---|
| Phase 1: Convert 5 dispatch functions | ~150 | 1-2 hours |
| Phase 2: Decode graph cache on Gpu | ~100 | 1 hour |
| Phase 3: forward_step capture/replay | ~80 | 1-2 hours |
| Phase 4: seq_len handling (Option D) | ~20 | 30 min |
| Phase 5: Validation | — | 1-2 hours |
| **Total** | **~350** | **5-8 hours** |

## Blocked On

This work is deferred from the dots.ocr PR to avoid scope creep. It should
land as a standalone follow-up PR after the OCR text prefill kernel and
e2e validation are merged.

Prerequisite: the dots.ocr PR must be merged first so that this work can
be validated against a stable baseline without merge conflicts from the
prefill kernel wiring.
