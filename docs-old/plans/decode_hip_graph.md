# Decode HIP Graph Capture Design

**Status:** Phase 1 shipped (2026-05-26). Phases 2-5 deferred.
**Date:** 2026-05-26 (rev 2026-05-26)
**Context:** dots.ocr decode profiling on gfx1151 (Strix Halo)
**Reviews:** gemini (`decode_hip_graph_plan_rev_gemini.md`),
claude (`decode_hip_graph_plan_rev_claude.md`)

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
| `rmsnorm_f32` | 57 | ✅ | Already used `launch_maybe_blob` |
| `gemv_q8_0` | 197 | ✅ | Already used `launch_maybe_blob` (both wide + narrow) |
| `add_inplace_f32` | 56 | ✅ | Already used `launch_maybe_blob` |
| `silu_mul_f32` | 28 | ✅ | Already used `launch_maybe_blob` |
| `attention_flash_gqa` | 56 | ✅ **Phase 1** | Converted: partial + reduce |
| `attention_flash` | 56 | ✅ **Phase 1** | Converted: partial + reduce |
| `rope_f32` | 28 | ✅ **Phase 1** | Converted |
| `kv_cache_write` | 56 | ✅ **Phase 1** | Converted |
| `bias_add_f32` | 84 | ✅ **Phase 1** | Converted |
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
from a pointer instead of a kernarg. Effort: ~1 attention kernel arg
change + a new device buffer in `Qwen2State`.

**Option B: hipGraphExecKernelNodeSetParams.** ROCm supports updating
individual node parameters on an instantiated graph exec without
re-capture. This lets us update the seq_len scalar in the attention
node between replays. Requires new FFI bindings in hip-bridge + node
handle tracking in Gpu. **Not currently bound** in hip-bridge FFI.

**Option C: Re-capture every token.** Defeats the purpose — capture
overhead would exceed dispatch savings.

**Option D (REJECTED): Over-seq capture.** Capture with `seq_len =
bucket_end`, rely on zeroed KV cache rows for positions beyond actual
seq_len. **Mathematically flawed:** zero K rows give dot=0, score=0,
exp(0-m) ≠ 0. The softmax denominator is inflated by `n_extra ×
exp(-m)` per extra position, attenuating the output. At typical m ≈ 4
(standard-normal scaled attention scores over 5000 positions), 127
extra rows add 127 × exp(-4) ≈ 2.3 to the denominator, corrupting
the output by ~40-60%. This is not a numerical approximation — it
produces wrong results that can trigger token attractors within ~10
tokens. See §Review Assessment for details.

### Recommended approach: Option A (device-side seq_len)

Add a `seq_len_buf: DeviceBuffer` to `Qwen2State` (single i32). Modify
the attention kernels (`attention_flash_gqa_partial`, `attention_flash`)
to read `seq_len` from a device pointer instead of a kernarg scalar.
The graph captures the device pointer; we update `seq_len_buf` contents
via `memcpy_htod_auto` before each replay.

Advantages:
- Exact seq_len — no numerical compromise
- Captures once per n_chunks bucket, replays many times
- Small kernel change (one arg: `int seq_len` → `const int* seq_len_ptr`)
- Re-use existing capture infrastructure

Cost:
- One new device buffer (4 bytes) in `Qwen2State`
- Two attention kernels need a 1-arg signature change
- Dispatch wrappers pass `seq_len_buf.as_ptr()` instead of `seq_len as i32`

### Additional mutable arguments

Other kernel arguments that change per token:
- `pos_buf` (position for kv_cache_write, rope) — already a device buffer,
  updated via `memcpy_htod` before replay. Graph records pointer → works.
- `embedding_lookup_q8` token id — outside graph (1 launch, negligible)
- KV cache pointers — stable addresses, no change
- Weight pointers — stable addresses, no change
- `n_chunks` grid dimension — changes only at bucket boundaries (re-capture)

## Implementation Plan

### Phase 1: Convert dispatch functions to launch_maybe_blob ✅ SHIPPED

Converted 5 dispatch functions from direct `hip.launch_kernel` to
`launch_maybe_blob`. Each follows the same pattern as the existing
`rmsnorm_f32` conversion.

Functions converted:
1. **`rope_f32`** — 1 launch, 7 kernargs
2. **`kv_cache_write`** — 1 launch, 4 kernargs
3. **`bias_add_f32`** — 1 launch, 4 kernargs
4. **`attention_flash_gqa`** — 2 launches (partial + reduce)
5. **`attention_flash`** — 2 launches (partial + reduce)

Validated: identical OCR output and timing on ocr_e2e (2.2 tok/s) with
both normal path and `HIPFIRE_FORCE_BLOB_PATH=1`. Zero behavior change
when `capture_mode` is false.

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

### Phase 4: Device-side seq_len for attention

Modify the attention kernels to read `seq_len` from a device buffer
instead of a scalar kernarg:

1. Add `seq_len_buf: DeviceBuffer` (4 bytes) to `Qwen2State::new`
2. Change `attention_flash_gqa_partial` kernel signature:
   `int seq_len` → `const int* seq_len_ptr`, kernel reads `*seq_len_ptr`
3. Same change for `attention_flash` kernel (reduce kernel doesn't use
   seq_len, no change needed)
4. Update dispatch wrappers to pass `seq_len_buf.as_ptr()` instead of
   `seq_len as i32`
5. Before each graph replay, update via `gpu.memcpy_htod_auto(&state.seq_len_buf, ...)`

### Phase 5: Correctness + perf validation

1. Run `ocr_e2e` with graph capture enabled, verify identical F1 score
2. Run token-attractor detection on graph-captured decode output
   (not just F1 — coherence-gate runs qwen35, not qwen2)
3. Measure decode tok/s: expect ~4× improvement
4. Benchmark re-capture overhead: ~150ms per bucket (warmup + capture +
   instantiate), ~37 buckets ≈ 5.5s total over a 260s decode

## Risks

1. **ROCm graph capture bugs.** hipGraph capture on ROCm has historically
   had issues with async memcpy, shared memory, and certain kernel launch
   configurations. The existing DFlash verify/replay paths have already
   shaken out most of these, but the decode path uses different kernels.

2. **Re-capture stutter.** Each re-capture costs ~150ms (warmup pass +
   capture pass + instantiate). Over a 4633-token decode, ~37 re-captures
   produce ~5.5s overhead and visible ~150ms stutters every 128 tokens.
   Still a massive net win (5.5s vs 260s saved). Background pre-capture
   of the next bucket could hide this in a future iteration.

3. **Argmax sync bottleneck.** Leaving argmax outside the graph introduces
   a host-side sync point every token (~0.1ms). This prevents CPU/GPU
   pipelining but costs only ~4% of the theoretical throughput. Pre-alloc
   async argmax is a follow-up optimization.

4. **Paged KV cache incompatibility.** This design assumes contiguous KV
   cache buffers with stable device pointers. A future vLLM-style paged
   KV cache would break captured graphs (physical addresses change per
   block allocation). Explicitly out of scope.

## Review Assessment

Two external reviews were solicited. Key findings validated or rejected
below.

### Claude review — BLOCKER accepted

**Claim:** Option D "zeros → zero attention weight" is mathematically
false. Zero K rows give dot=0 → score=0 → exp(0-m) ≠ 0, inflating the
softmax denominator by `n_extra × exp(-m)` per extra position. At
typical m ≈ 4 (standard-normal scaled scores over 5000 positions),
127 extra rows add ~2.3 to the denominator — ~40-60% output corruption.

**Verdict: CORRECT.** Verified against the kernel code
(`attention_flash_gqa.hip:47-48`): `dot += q_head[d] * k_t[d]` with
zero K gives score=0, and `expf(scores[t] - max_val)` at line 58 gives
`exp(-m) ≠ 0`. The KV cache IS zero-initialized (`qwen2.rs:612`:
`gpu.zeros()`), but that's what makes it dangerous — zero K ≠ zero
attention contribution. **Option D rejected. Design updated to Option A
(device-side seq_len buffer).**

### Claude review — re-capture cost underestimate accepted

**Claim:** "~5ms/capture, 185ms total" is wrong. Warmup is a full
forward_step (~75ms) + capture pass (~75ms) + instantiate (~5-50ms).
Per re-capture: ~155-200ms. Over 37 buckets: ~5.7-7.4s.

**Verdict: CORRECT.** Verified against existing DFlash capture pattern
in `speculative.rs:692-706`: warmup runs a full forward pass, then
capture runs another, then instantiate. The plan's "~5ms" was off by
~30×. Updated estimates in §Risks and §Phase 5. Conclusion still holds
(5.5s << 260s saved).

### Claude review — coherence gate coverage gap accepted

**Claim:** coherence-gate runs qwen35, not qwen2. Only ocr_e2e (F1)
exercises the qwen2 forward_step. Need token-attractor detection, not
just F1.

**Verdict: CORRECT.** Updated §Phase 5 to mandate attractor detection
on graph output.

### Gemini review — kernarg blob fragmentation rejected

**Claim:** 37 re-captures × 567 blobs per capture creates "thousands of
small heap allocations" causing memory fragmentation.

**Verdict: OVERBLOWN.** The decode graph cache retains blobs per bucket
(HashMap entry). Total live memory: 37 entries × 567 blobs × ~100 bytes
≈ 2 MB of stable, non-churning allocations. jemalloc handles this
trivially. A `KernargArena` is unnecessary complexity for this scale.

### Gemini review — argmax async recommended

**Claim:** argmax forces host-side sync, preventing pipelining of token
N's graph replay with token N-1's argmax retrieval.

**Verdict: VALID but not a blocker.** The impact is ~4% of theoretical
throughput. The existing DFlash paths have the same issue and work fine.
Recommended as a follow-up, not a prerequisite.

### Gemini review — Option B recommended over re-capture

**Claim:** `hipGraphExecKernelNodeSetParams` is the "production-grade"
approach; re-capturing 37 times is "infrastructure weakness."

**Verdict: NOT FEASIBLE.** Claude confirmed and I verified:
`hipGraphExecKernelNodeSetParams` is NOT bound in hip-bridge FFI.
Implementing it requires new FFI bindings + node handle tracking in Gpu.
Option A (device-side seq_len) achieves the same goal (no re-capture
for seq_len changes) with far less infrastructure work — a single
device buffer and a minor kernel arg change. Option B can be evaluated
as a follow-up if Option A proves insufficient.
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

Deferred. Not bound in hip-bridge FFI; Option A (device-side seq_len)
achieves the same goal with less infrastructure work. Can be revisited
if per-node updates are needed for other arguments.

### Device-side seq_len buffer (now recommended — Option A)

**Adopted.** See §"Recommended approach: Option A" above. The Claude
review's BLOCKER on Option D (over-seq) made this the simplest correct
path. Minor kernel change (1 arg: `int seq_len` → `const int* seq_len_ptr`),
one new device buffer in `Qwen2State`.

## Estimated Effort

| Phase | Lines changed | Effort | Status |
|---|---|---|---|
| Phase 1: Convert 5 dispatch functions | ~150 | 1-2 hours | ✅ Shipped |
| Phase 2: Decode graph cache on Gpu | ~100 | 1 hour | Pending |
| Phase 3: forward_step capture/replay | ~80 | 1-2 hours | Pending |
| Phase 4: Device-side seq_len (Option A) | ~40 | 1 hour | Pending |
| Phase 5: Validation | — | 1-2 hours | Pending |
| **Remaining** | **~220** | **4-7 hours** | |

## Next Steps

Phase 1 shipped in this PR. Phases 2-5 remain for a follow-up PR that
adds the actual decode graph capture/replay cycle. The prerequisite
(all dispatch functions using `launch_maybe_blob`) is now complete.
