# Exp #2: hipMemcpyBatchAsync for boundary_copy + model-load

**Date:** 2026-05-07
**Status:** DEFERRED (requires FFI groundwork)

## Lever

`hipMemcpyBatchAsync` (HIP 7.0+) — batches multiple async memcpy operations into a single runtime call, amortizing per-call dispatch and synchronization overhead.

## Why deferred

Three blockers were discovered during scoping that prevent a quick A/B test under the autoresearch contract's "smallest change that exercises it" rule.

### Blocker 1: HIP wrapper has no binding

`crates/rdna-compute/src/ffi.rs` and `crates/rdna-compute/src/hip.rs` do not expose `hipMemcpyBatchAsync`. Adding the FFI requires:

- Define `hipMemcpyBatchAsync` extern "C" prototype in `ffi.rs`.
- Define `hipMemcpyKind`, `hipMemcpyAttributes`, and the parameter struct array contract.
- Add a safe Rust wrapper in `hip.rs` taking a slice of (src, dst, size, kind) tuples.
- Plumb stream + completion event semantics that match our existing `boundary_copy` interface.

This is roughly 100-200 lines of FFI and wrapper code per the HIP 7.0 API surface. It is not a "smallest change."

### Blocker 2: PP boundary copy is single-call

`crates/hipfire-runtime/src/multi_gpu.rs::boundary_copy` issues exactly ONE `hipMemcpyPeerAsync` per layer-band boundary. There is no sequential many-copy pattern in the PP hot path to batch. Per the prior fabric-ablation memory entry, single-card boundary cost is already 0.012% of decode time on 9B PP=2. Batching one call into one call yields zero benefit.

The pre-registered criterion was "≥3% PP=2 wall-clock improvement, or ≥10% reduction in observed boundary_copy time." Neither is achievable on the current PP code path because the lever cannot be exercised against `hipMemcpyPeerAsync`.

### Blocker 3: Hot-path batchable patterns exist but are NOT in PP=2 boundary

Sequential memcpy patterns DO exist in hipfire, but in different code paths:

- `crates/hipfire-runtime/src/dflash.rs:884-887` — 4 sequential `memcpy_dtod_at` per FA layer per DFlash step. Real batching candidate. **But DFlash is refused on pp>1**, and DFlash on single-card RDNA1 is NET NEGATIVE (-41% per `project_gfx1010_5700xt_validated_2026_05_06`). Out of scope for the PP=2 criterion.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:2229-2231, 2442-2444, 4154-4162` — 3-tuple `memcpy_dtod_at` (Q/K/V slice extraction). Repeated per layer in DeltaNet path. Possible target.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3989, 4765, 4769` — per-token loop of `memcpy_dtod_at` in prefill batched extraction. Per the prior research, prefill batched OOMs on 27B PP=2 anyway, and on 9B PP=2 PB=1 the prefill batched cost is small.
- Model-load fan-out — many small h2d copies. Cold-start path, no steady-state hot-path impact. Wall-clock impact on user-perceived load time only.

None of these are the PP=2 boundary path called out in the criterion. Re-targeting the criterion to a different scenario without re-pre-registration would violate immutable rule #4 (no combined experiments with shifting goalposts).

## Action

This experiment requires Kaden's design call on:

1. Whether to invest in the FFI binding + wrapper for `hipMemcpyBatchAsync`.
2. Which target scenario is most worth optimizing: DFlash dtod KV concat (out of scope today; in scope if DFlash gets RDNA1 perf treatment), DeltaNet 3-tuple extraction (modest gain estimate), prefill batched extraction (already infrequent enough to not matter).
3. Whether to re-pre-register a new experiment against a different scenario.

Until a design decision lands, this experiment is parked. The lever is real and may be useful later; it just isn't a same-night drop-in for the PP=2 criterion.

## No code changes

No branch was created. No baseline measurement was run (would have been wasted). Master is unchanged.
