# Redline Phase 2 — HSA dispatch result & strategic re-read

## What we built

`crates/hsa-bridge/` — a thin Rust wrapper around `libhsa-runtime64.so`
parallel to `crates/hip-bridge/`. Loads the library via `dlopen`, exposes
agents, queues, signals, memory pools, executables, and an AQL packet
builder. Total surface ≈ 30 FFI entry points, ~600 lines.

`crates/hsa-bridge/examples/hsa_vs_hip_launch.rs` — the head-to-head
benchmark. Compiles `vector_add` for the local arch (gfx1100), allocates
identical buffers and kernarg layout for both paths, runs 5 000
iterations through HIP and HSA, plus burst-dispatch sweeps for both.

## Measurements (gfx1100, RX 7900 XTX, vector_add 256 elements)

### Single dispatch — sync after every call

| Path | median | mean | p99 | min |
|---|---|---|---|---|
| HIP `hipModuleLaunchKernel + sync` | **21.08 µs** | 21.75 µs | 29.59 µs | 20.16 µs |
| HSA `AQL packet + signal_wait_active` | **15.26 µs** | 15.31 µs | 15.98 µs | 14.57 µs |

HSA wins by 1.38× — saves ~6 µs per dispatch over HIP.

### Burst dispatch — N launches back-to-back, sync once at the end

| Burst size | HIP per-dispatch | HSA per-dispatch |
|---|---|---|
| 10 | 5.11 µs | 5.35 µs |
| 50 | 3.54 µs | 4.20 µs |
| 100 | 3.34 µs | 4.04 µs |
| 200 | **3.22 µs** | **3.97 µs** |

**HIP burst is *faster* than HSA burst.** HIP at 3.22 µs/dispatch beats
our raw HSA AQL submission at 3.97 µs/dispatch (1.23× HIP win).

## Why the result inverts when batched

When the host syncs after each launch (the single-dispatch column), the
critical path is "submit packet → GPU dispatch latency → completion
signal observable to host". Most of that is the GPU's
doorbell-to-completion roundtrip, ~10 µs of irreducible hardware time.
HIP adds another 5 µs of CPU-side wrappers (mutex, error check, stream
selection); HSA only adds ~0.5 µs. Hence HSA wins single-dispatch.

When the host stops syncing per launch (burst column), the critical path
becomes purely the cost of getting each packet onto the queue. The host
can write packets faster than the GPU can drain them, so per-dispatch
cost amortizes to:

- packet build + atomic header store
- doorbell write (one per N packets)
- GPU dispatch + completion (also amortized over N)

Both HIP and HSA bottom out around 3 µs/dispatch. HIP's `hipLaunchKernel`
turns out to do effectively the same thing internally — it's a thin
wrapper around `hsa_queue_*` itself. The 0.75 µs HIP wins by
(3.22 vs 3.97) is HSA bookkeeping we didn't optimize: we still call
`load_write_index_relaxed` per packet, set `completion_signal` only on
the last packet (good), but pay one signal_create and one signal_wait
per burst.

## The strategic implication

The original Phase 1 framing — "HIP costs 10 µs/launch, beat it with HSA"
— is **wrong for production code**. The 10 µs number came from the
bandwidth-ceiling profiler, which calls `hipEventRecord` +
`hipEventSynchronize` after every kernel to attribute time per kernel.
That synchronize is what costs 10+ µs per launch. The profiler is
measuring what *the profiler does*, not what production launches cost.

In the actual Hipfire forward pass:

- `crates/hipfire-rdna/src/dispatch.rs` does NOT call any
  `synchronize`. Confirmed by grep — the only `synchronize` calls in the
  whole tree are in `profile.rs` (gated on profiling-active) and in the
  example runners.
- `crates/engine/src/qwen35.rs` does NOT sync between layers either.

So production already runs HIP back-to-back, paying ~3 µs per launch.
For 36 non-GEMV launches per 9B forward, that's **108 µs total** of
launch overhead — about 0.9% of the 11.85 ms forward time.

The bandwidth-ceiling profiler's "Non-GEMV: 4.55 ms" attribution is
measuring _wall-clock with per-launch sync_. The real production
contribution of these kernels is closer to "compute time + ~3 µs per
launch", which is much less than 4.55 ms.

## Net savings if we ported the engine to HSA

| Path | Per-dispatch | 36 launches | Savings vs HIP burst |
|---|---|---|---|
| HIP burst (current production) | 3.22 µs | 116 µs | — |
| HSA burst | 3.97 µs | 143 µs | **−27 µs (slower)** |

**Porting to HSA would make us slower, not faster, at the
forward-pass-aggregate level.** Bypassing HIP via raw AQL is no longer
the right lever.

## What this leaves on the table

The launch-overhead optimization budget is at most ~108 µs/forward. It's
dwarfed by the actual compute time. The remaining 11.7 ms of forward time
is:

- 7.3 ms in GEMV (88-92% peak bandwidth — close to the ceiling)
- 4.4 ms in everything else (rmsnorm, FWHT, conv1d, attention, gated_norm,
  alpha_gate, etc.) — most of which IS compute, not launch overhead.

The high-leverage paths from here are:

1. **Profile production-mode (no per-launch sync) to find the actual
   bottleneck.** The current profiler over-attributes time to non-GEMV
   kernels because it serializes them via `event_synchronize`. We need a
   profiling mode that uses ONE sync at the end and back-attributes per
   kernel via begin/end events without blocking. (HIP supports this:
   record events on a stream without sync, query elapsed time later.)

2. **hipGraph capture/replay**: capture the entire forward pass as a
   single graph, replay it on each token. This collapses 36+248 launches
   into one submit. Saves the full ~108 µs of launch overhead AND opens
   up multi-stream parallelism inside the graph. HIP supports
   `hipGraphInstantiate` and the existing `hip-bridge` already has the
   stream-capture functions wired up. **This is the highest-ROI lever
   remaining for launch-overhead reduction.**

3. **Direct DRM/PM4 chain dispatch on RDNA3** (Phase 1 path C): same
   idea as hipGraph but bypasses HIP entirely. Requires fixing the
   `RELEASE_MEM`/`WAIT_REG_MEM` packet encoding for gfx1100 (RDNA3 vs
   our current RDNA1 encoding which fails with -62 ETIME on the 7900 XTX).
   Equal-or-better than hipGraph and ROCm-independent, but more
   reverse-engineering work.

4. **Pivot to where the actual bandwidth gap is**: GEMVs are 60.4% of
   *system* peak (960 GB/s) but 88-92% of *achievable* peak. Getting
   from 88% → 95% of peak gets us another 5% on the GEMV time, ~1.5%
   end-to-end. Probably not worth it relative to (1)/(2).

5. **Continue with DFlash speculative decode** as originally planned.
   Algorithmic wins (acceptance rate × draft speedup) typically give
   1.5-3× tokens/sec, far more than launch-overhead optimization can.

## Recommendation for the user

Phase 2 has a clear-but-uncomfortable answer: **don't port to HSA**. The
launch-overhead-bypass premise was based on a profiler artifact, not on
real production cost. The good news is that the existing engine is
already very close to optimal on the launch-overhead axis.

I recommend we **pivot Redline Phase 3 to hipGraph capture/replay** as
the launch-overhead optimization (highest ROI, lowest risk, uses
existing HIP infrastructure), and table further raw-HSA work until we
have a use case that's truly bottlenecked on per-launch CPU overhead
(e.g., multi-stream parallelism on tiny kernels).

OR alternatively: **skip Redline entirely and go straight to DFlash
Phase 4**. The launch-overhead budget on the table is ~108 µs/forward
(<1% of forward time). DFlash has potential for 1.5-3× speedup
(thousands of µs). It's a much bigger lever.

The hsa-bridge work isn't wasted: it's a small, working FFI to ROCr,
useful as a fallback if we ever need a HIP-independent dispatch path
(e.g., for the redline distribution that ships without ROCm). But it's
not the next thing to merge into the engine.

## Files added / changed in this branch

- `crates/hsa-bridge/Cargo.toml`
- `crates/hsa-bridge/src/lib.rs` (~600 LOC)
- `crates/hsa-bridge/src/ffi.rs` (~400 LOC)
- `crates/hsa-bridge/src/error.rs`
- `crates/hsa-bridge/examples/hsa_vs_hip_launch.rs` (~290 LOC)
- `Cargo.toml` workspace member added
- `crates/redline/PHASE1_STATUS.md` (Phase 1 audit)
- `crates/redline/PHASE2_RESULT.md` (this file)
- `crates/redline/examples/bench_dispatch.rs` (auto-detect arch)
- `crates/redline/examples/test_aql_dispatch.rs` (auto-detect arch)
