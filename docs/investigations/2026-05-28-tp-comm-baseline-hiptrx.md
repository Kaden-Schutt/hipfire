# TP comm-cost baseline on hiptrx (4× R9700 / gfx1201)

**Date:** 2026-05-28
**Hardware:** k9lin / hiptrx — 4× Radeon AI PRO R9700 (gfx1201/RDNA4, 34.2 GiB/card), HIP 7.2, ROCm 7.2
**Driver of investigation:** Validate or kill the TP-for-A3B plan
(`docs/plans/multi-gpu-tp-a3b.md` §3.3 comm-cost estimates) **before**
committing to Stage 1 architecture work.
**Bench source:** `crates/hipfire-runtime/examples/tp_comm_smoke.rs`
**Repo commit:** worktree `tp-scoping-ds4` at `62fe152e` (master + PR #352)

## Headline

| Shape | Comm-to-compute ratio | Per-token comm (64 layers) | Verdict |
|---|---:|---:|---|
| TP=4 | **7.29×** | 51.0 ms | **Comm-broken** for single-user decode |
| TP=2 | **1.75×** | 12.3 ms | Salvageable with batching |
| (single-card A3B baseline) | — | 7.0 ms | 143 tok/s reference |

**Decision:** The TP=4 column kills the §3.3 plan as written. Single-user
decode on 4× R9700 cannot exceed ~20 tok/s with TP=4 (comm-bound floor).
TP=2 is the only viable TP shape on this hardware; DP=4 wins for short-
context multi-user fitting workloads; TP=4 is reserved for the
DSv4-doesn't-fit-single-card case and would need a fused all-reduce
kernel + better stream scheduling to beat TP=2.

## Hardware setup

```
peer matrix (can_access_peer):
       -> 0  -> 1  -> 2  -> 3
from 0:  .    Y    Y    Y
from 1:  Y    .    Y    Y
from 2:  Y    Y    .    Y
from 3:  Y    Y    Y    .
peer_access_enabled (global flag): true
```

Full peer mesh; no asymmetric NUMA penalty in the topo matrix. PCIe
appears to be Gen4 x8 per card based on the 14 GB/s ceiling at 512 KB
payload (theoretical Gen4 x8 = 16 GB/s × ~0.87 efficiency).

## Phase A — raw `memcpy_peer_async` + `stream_synchronize`

Median over 100 iterations (10 warmup), sequential pair-by-pair:

| Pair | 4 KB | 32 KB | 128 KB | 512 KB |
|---|---:|---:|---:|---:|
| 0 → 1 | 21.7 µs (0.19 GB/s) | 22.2 µs (1.48 GB/s) | 30.1 µs (4.35 GB/s) | 37.6 µs (13.94 GB/s) |
| 0 → 2 | 21.2 µs | 21.5 µs | 29.9 µs | 37.6 µs |
| 0 → 3 | 20.7 µs | 21.3 µs | 29.9 µs | 37.6 µs |
| 1 → 0 | 24.0 µs | 24.5 µs | 32.3 µs | 40.0 µs |
| 2 → 0 | 24.1 µs | 24.6 µs | 32.3 µs | 40.0 µs |
| 3 → 0 | 24.0 µs | 24.4 µs | 32.3 µs | 40.0 µs |
| (all other pairs) | 23.6-24.0 µs | 24.1-24.4 µs | 31.3-32.3 µs | 37.6-40.0 µs |

**Load-bearing fact:** the floor is **~22 µs per peer copy** regardless
of payload up to ~30 KB. Below that size, comm cost is dominated by
the host-side `memcpy_peer_async` syscall + stream-sync round-trip,
not by PCIe bandwidth. At 512 KB the effective bandwidth is ~14 GB/s.

The §3.3 estimate of "ring all-reduce on 4 ranks = 3× (4 KB / 25 GB/s) =
0.5 µs + 4× event-sync ~10 µs ≈ 40-50 µs" was wrong. The per-step cost
is ~22 µs floor (latency, not bandwidth), and 6 sequential steps
multiplies up.

## Phase B — `Gpus::boundary_copy + wait_boundary` (production path)

Same payload sizes through the orchestrator. Adds `event_create` +
`event_record` + `stream_wait_event` + `event_destroy` overhead.

| Pair | 4 KB | 32 KB | 128 KB | 512 KB |
|---|---:|---:|---:|---:|
| 0 → 1 | 29.2 µs | 29.8 µs | 31.5 µs | 39.1 µs |
| 0 → 2 | 29.4 µs | 30.4 µs | 32.0 µs | 39.7 µs |
| 1 → 0 | 30.6 µs | 31.0 µs | 32.4 µs | 39.2 µs |
| 2 → 0 | 33.4 µs | 34.3 µs | 42.2 µs | 49.7 µs |
| 2 → 1 | 33.1 µs | 34.3 µs | 35.8 µs | 49.9 µs |
| 3 → 0 | 33.2 µs | 34.2 µs | 41.8 µs | 49.8 µs |
| 3 → 1 | 33.0 µs | 33.6 µs | 35.7 µs | 49.9 µs |
| 3 → 2 | 33.5 µs | 34.0 µs | 35.8 µs | 50.2 µs |

**Orchestrator overhead is +7-10 µs vs raw Phase A.** Ranks 2 and 3 as
source are ~5 µs slower at 128 KB+ — consistent with PCIe-pair affinity
asymmetry (likely host bridge layout). Per-pair overhead is acceptable
but **multiplies in any tight comm loop** (Phase C/D below).

## Phase C — 4-rank ring all-reduce (TP=4)

Reduce-scatter 3 steps + all-gather 3 steps. Sequential schedule via
`for step { for src { boundary_copy + wait_boundary } }` — each step
issues 4 src→dst copies on 4 distinct streams.

| Size | Median µs | p10 µs | p90 µs | Per-step µs |
|---|---:|---:|---:|---:|
| 4 KB | **339.7** | 333.7 | 343.4 | 56.6 |
| 32 KB | 333.9 | 329.3 | 339.9 | 55.7 |
| 128 KB | 355.9 | 344.7 | 381.4 | 59.3 |
| 512 KB | 472.5 | 460.9 | 536.4 | 78.8 |

**Floor check:** theoretical = 6 sequential steps × 22 µs/step = 132 µs.
Measured = 340 µs (4 KB), or **2.5× theoretical floor**. The gap is
`wait_boundary`'s cross-stream `stream_wait_event` serializing the
host-issue order — each step's `wait_boundary(evt)` makes dst stream
wait on src stream's event, and the next host iteration's
`boundary_copy` on what was the dst stream now serializes behind that
event. **The existing `boundary_copy` API was designed for PP
(one copy per layer-band, dst stream then runs the next layer's kernels
which naturally wait for the event), not for tight comm loops.**

## Phase C-2 — 2-rank ring all-reduce (TP=2)

Same shape with `n=2`, 2-step ring (reduce-scatter 1 + all-gather 1).

| Size | Median µs | p10 µs | p90 µs | Per-step µs |
|---|---:|---:|---:|---:|
| 4 KB | **72.2** | 71.2 | 73.3 | 36.1 |
| 32 KB | 74.5 | 73.3 | 75.7 | 37.2 |
| 128 KB | 77.3 | 76.3 | 78.6 | 38.7 |
| 512 KB | 103.5 | 102.4 | 104.5 | 51.8 |

**4.7× faster than TP=4** at 4 KB. Two reasons: (a) only 2 ring steps
vs 6, (b) less cross-stream contention with 2 streams vs 4.

## Phase D — full all-to-all (every rank → every other)

Each rank sends `(n-1)` messages to distinct destinations. Issued
sequentially through `boundary_copy + wait_boundary`.

### TP=4

| Size | Median µs | p10 µs | p90 µs | Per-rank µs |
|---|---:|---:|---:|---:|
| 4 KB | 113.8 | 112.8 | 115.0 | 28.5 |
| 32 KB | 117.9 | 116.9 | 119.5 | 29.5 |
| 128 KB | 165.1 | 163.9 | 178.7 | 41.3 |
| 512 KB | 224.9 | 224.0 | 226.1 | 56.2 |

### TP=2

| Size | Median µs | p10 µs | p90 µs | Per-rank µs |
|---|---:|---:|---:|---:|
| 4 KB | 43.7 | 42.9 | 44.8 | 21.8 |
| 32 KB | 47.3 | 46.6 | 48.2 | 23.7 |
| 128 KB | 54.0 | 53.2 | 54.9 | 27.0 |
| 512 KB | 69.4 | 68.6 | 70.1 | 34.7 |

**All-to-all has more cross-stream parallelism** than ring all-reduce
because each rank's `(n-1)` outgoing messages go to distinct
destinations — no event-chain bottleneck. At 4 KB / TP=4 this is
**3× faster** than the ring all-reduce.

## Phase E — synthesis vs §3.3 estimates

### TP=4

| Component | §3.3 estimate | Measured | Ratio |
|---|---:|---:|---:|
| attn all-reduce (4 KB) | ~50 µs | 339.7 µs | **6.8× worse** |
| MoE all-to-all (32 KB) | ~100 µs | 117.9 µs | 1.18× |
| shared-expert all-reduce | ~50 µs | 339.7 µs | 6.8× worse |
| per-MoE-layer comm | ~200 µs | 797.3 µs | **4.0× worse** |
| per-token comm @ 64 layers | ~13 ms | **51.0 ms** | **3.9× worse** |
| Comm-to-compute ratio | — | **7.29×** | — |

### TP=2

| Component | Measured | TP=2 vs TP=4 |
|---|---:|---:|
| attn all-reduce (4 KB) | 72.2 µs | 4.7× faster |
| MoE all-to-all (32 KB) | 47.3 µs | 2.5× faster |
| shared-expert all-reduce | 72.2 µs | 4.7× faster |
| per-MoE-layer comm | 191.7 µs | 4.2× faster |
| per-token comm @ 64 layers | **12.3 ms** | 4.2× faster |
| Comm-to-compute ratio | **1.75×** | — |

## What this means for the TP plan

### `docs/plans/multi-gpu-tp-a3b.md` §3.3 is wrong

The estimate "~50 µs per attn all-reduce on TP=4" is **6.8× too
optimistic**. Re-estimation: each ring step is **floor-bound at ~22 µs**
(R9700 peer-copy latency + stream-sync round-trip), not bandwidth-bound
at small payloads. The §3.3 model assumed bandwidth-bound at 25 GB/s on
PCIe Gen4 x16; actual link is Gen4 x8 (~14 GB/s) and small payloads
never reach bandwidth.

§3.3 also under-modeled the multiplicative effect of cross-stream
event-dependencies in `wait_boundary`. The current API was built for PP
where dst stream then runs the next layer's compute — comm-loop reuse
serializes.

### TP=4 is not viable for single-user decode on hiptrx

Even with perfect software (no `boundary_copy` overhead), the theoretical
floor for a 4-rank ring all-reduce at 4 KB is 6 × 22 µs = 132 µs.
For 64 layers × 2 all-reduces/layer = 128 all-reduces/token = 17 ms
just in attn all-reduce. Single-card decode is 7 ms/token. **TP=4 is
hardware-comm-bound below single-card throughput**, before MoE
all-to-all is even counted.

This holds for **single-user, batch=1 decode**. Multi-user batched
decode amortizes comm across the batch (each all-reduce serves all
concurrent users in that step). Phase 3 prefill comm at 512 KB is
38 µs × 6 steps = 228 µs which is far better relative to the much
larger compute payload — TP=4 prefill is plausible.

### TP=2 is the only viable single-instance TP shape

72 µs all-reduce + 47 µs all-to-all per MoE layer = 191 µs/layer.
64 layers × 191 µs = 12.3 ms/token comm.
Single-card decode 7 ms + comm 12.3 ms = ~19 ms/token = ~52 tok/s for
single user. Slower than single-card decode (143 tok/s), but:

- Halves model memory per rank → enables longer context per concurrent
  user where DP=2 worker would OOM on KV.
- Two TP=2 instances on 4 cards (TP=2 + DP=2) gives the option of 2
  long-context serving slots, each with bigger ctx than DP=4 could fit.

### DP=4 still wins for the dominant A3B-on-hiptrx workload

For Qwen3.6 35B-A3B mq4 (23.5 GB) running 4 independent workers on 4
cards (34 GB each), single-user throughput is **4× 143 tok/s ≈ 572 tok/s
aggregate** with zero comm cost. This dominates TP=2 + DP=2 (~104
tok/s aggregate) and TP=4 (~20 tok/s aggregate) for any short-context
multi-user workload that fits.

### The only path TP justifies its complexity

**Models that don't fit a single 34 GB card.** DSv4 (the original
motivation for this scoping work) is the load-bearing example. The
TP plan should be re-scoped accordingly:

1. **DP=4 ships first** (~1 week, daemon-only, no `Gpus` changes).
   Production multi-user A3B serving on hiptrx runs DP=4 by default.
2. **TP=2 ships second** as the "extended-context single-instance"
   path. Pairs with DP=2 for 2-slot serving. ~6+ weeks. Validates the
   comm primitives + sharding infrastructure for DSv4.
3. **TP=4 only when DSv4 doesn't fit on a single 34 GB card AND
   TP=2 can't either** (DSv4 might fit TP=2 — that's a §6 Stage 10
   discovery, not a Stage 1 assumption). If TP=4 is forced, **invest
   in a fused all-reduce kernel** that bypasses the host-issued
   `boundary_copy + wait_boundary` schedule — single-dispatch peer
   DMA + on-device reduction. Or pull in RCCL.

## Recommended next actions

1. **Update `docs/plans/multi-gpu-tp-a3b.md`** §3.3 and §8 with these
   measurements. §3.3 estimates are wrong; ship-gate should be
   relative to the **measured floor**, not the original estimate.
   §8.1 (DP=4 first) moves from "recommended" to "required."
2. **Ship DP=4 as a separate PR**, not blocked on TP work.
   Path: `crates/hipfire-runtime/src/serve_router.rs` + spawning 4
   worker daemons. ~1 week. Unlocks the actual hiptrx-A3B serving
   need.
3. **Defer Stage 1 (`tp_shard.rs`) of the TP plan** until DSv4 lands
   on master and we know its memory shape on R9700.
4. **Keep the comm microbench in tree** — re-run on DSv4 with its
   actual residual / latent-KV all-reduce shapes when scoping
   DSv4 TP.

## Update — RCCL primitive comparison (same day)

After §Phase E showed the host-driven path was 7.3× comm-bound at TP=4,
checked whether RCCL would close the gap. **RCCL works on gfx1201**
(librccl.so.1.0.70202 in ROCm 7.2 ships with R9700 support;
`ncclCommInitAll` succeeds on all 4 ranks). Two new smoke benches:
`/home/kaden/.claude/jobs/6ea8a1b1/rccl_allreduce_smoke.cpp` and
`rccl_a2a_smoke.cpp`.

### RCCL all-reduce vs host-driven, TP=4

| Size | Host-driven (Phase C) | RCCL `ncclAllReduce` | Speedup |
|---|---:|---:|---:|
| 4 KB | 339.7 µs | **110.7 µs** | **3.07×** |
| 32 KB | 333.9 µs | 108.2 µs | 3.09× |
| 128 KB | 355.9 µs | 110.9 µs | 3.21× |
| 512 KB | 472.5 µs | 148.9 µs | 3.17× |

RCCL is essentially flat at ~110 µs from 4 KB to 128 KB — single-kernel
in-kernel collective using peer-mapped loads + inline reduction. The
2.5× gap between Phase C's 340 µs and the theoretical floor of 132 µs
collapses entirely; RCCL exposes a *better* floor by avoiding the
host-issued cross-stream event-chain.

### RCCL all-to-all vs host-driven, TP=4

| Size | Host-driven (Phase D) | RCCL `ncclAllToAll` | Verdict |
|---|---:|---:|---|
| 4 KB | 113.8 µs | 116.4 µs | tie |
| 32 KB | 117.9 µs | 115.2 µs | tie |
| 128 KB | 165.1 µs | 125.5 µs | RCCL 1.32× |
| 512 KB | 224.9 µs | **399.9 µs** | **host-driven 1.78× faster** |

**RCCL is not a universal win.** All-to-all at small sizes is a tie
(both ~115 µs — both dominated by 22 µs peer-copy floor × 3 hops per
rank in parallel); at 512 KB host-driven actually beats RCCL by 1.78×.
The flip suggests RCCL's all-to-all is bandwidth-suboptimal on
gfx1201 / RCCL 1.0.70202, possibly due to a single-buffer staging
schedule that doesn't pipeline peer DMA.

### Re-synthesis at TP=4 with RCCL all-reduce + host-driven all-to-all

| Component | Original §3.3 estimate | Host-driven measured | RCCL-mixed measured |
|---|---:|---:|---:|
| attn all-reduce (4 KB) | ~50 µs | 339.7 µs | **110.7 µs** |
| MoE all-to-all (32 KB) | ~100 µs | 117.9 µs | 115.2 µs (RCCL) or 117.9 µs (host) |
| shared-expert all-reduce | ~50 µs | 339.7 µs | **110.7 µs** |
| per-MoE-layer comm | ~200 µs | 797.3 µs | **335.5 µs** |
| per-token comm @ 64 layers | ~13 ms | 51.0 ms | **21.5 ms** |
| Comm-to-compute ratio | — | 7.29× | **3.06×** |

### What this changes for the TP plan

**Single-user TP=4 batch=1 decode is still comm-bound** — comm 21.5 ms
+ compute 7 ms = 28 ms = ~36 tok/s vs single-card 143 tok/s. RCCL
narrows the gap from 7.3× to 3× but doesn't close it.

**Batched serving flips the verdict.** Attn all-reduce serves the full
batch in one shot, so:

- **batch=4 TP=4**: comm 21.5 ms, compute 4×7=28 ms → compute-bound,
  ~143 tok/s/user × 4 users = ~570 tok/s aggregate. **Matches DP=4
  aggregate throughput, with shared KV as a bonus.**
- **batch=16 TP=4**: comm 21.5 ms, compute 16×7=112 ms → strongly
  compute-bound (5.2× compute > comm). Throughput scales with batch.

So with RCCL, **TP=4 + batch≥4 is competitive with DP=4** on throughput
and additionally unlocks extended context (shared KV). The DP=4-first
recommendation from §"What this means for the TP plan" softens from
"obviously DP=4 first" to "DP=4 if simplest-acceptable; TP=4 if you
want extended context AND can guarantee batch≥4 on average."

### Revised path forward

1. **Build a `hip-bridge` FFI wrapper around `librccl`** — `ncclCommInitAll`,
   `ncclAllReduce`, `ncclGroupStart/End`, `ncclCommDestroy`. ~1 day.
2. **`Gpus::all_reduce_sum` calls RCCL** instead of building a host-driven
   ring on `boundary_copy`. Falls back to a `boundary_copy` ring path
   when RCCL init fails or the user opts out (`HIPFIRE_TP_USE_RCCL=0`).
3. **All-to-all stays on `boundary_copy`** for now — RCCL doesn't help
   universally and gets worse at 512 KB. Revisit with a custom peer-read
   kernel only if prefill-comm becomes the bottleneck.
4. **Stage 8 ship-gate** uses the RCCL synthesis numbers as the baseline.
   The gate becomes "TP=4 batch=4 ≥ 0.9× DP=4 batch=4" (achievable per
   the math above), not "TP=4 batch=1 ≥ DP=4 batch=1" (impossible).

The TP plan's §8.1 "ship DP=4 first" recommendation is downgraded from
"required" back to "consider"; the bench-driven decision in Stage 8
becomes the actual gate.

## Bench reproduction

```sh
# Build
cargo build --release -p hipfire-runtime --example tp_comm_smoke

# TP=4 (default)
HIP_VISIBLE_DEVICES=0,1,2,3 ./target/release/examples/tp_comm_smoke

# TP=2
HIP_VISIBLE_DEVICES=0,1 HIPFIRE_TP_BENCH_N=2 ./target/release/examples/tp_comm_smoke

# Quick iteration
HIPFIRE_TP_BENCH_ITERS=20 ./target/release/examples/tp_comm_smoke
```

Output is self-explaining; Phase E synthesizes against the
docs/plans/multi-gpu-tp-a3b.md §3.3 estimates and emits a SHIP-GATE
verdict line.

### RCCL comparison benches

```sh
# Build (both)
hipcc -O2 -I/opt/rocm/include -L/opt/rocm/lib -lrccl \
    -o /tmp/rccl_allreduce_smoke /home/kaden/.claude/jobs/6ea8a1b1/rccl_allreduce_smoke.cpp
hipcc -O2 -I/opt/rocm/include -L/opt/rocm/lib -lrccl \
    -o /tmp/rccl_a2a_smoke /home/kaden/.claude/jobs/6ea8a1b1/rccl_a2a_smoke.cpp

# Run TP=4
HIP_VISIBLE_DEVICES=0,1,2,3 /tmp/rccl_allreduce_smoke
HIP_VISIBLE_DEVICES=0,1,2,3 /tmp/rccl_a2a_smoke

# TP=2
HIP_VISIBLE_DEVICES=0,1 HIPFIRE_TP_BENCH_N=2 /tmp/rccl_allreduce_smoke
HIP_VISIBLE_DEVICES=0,1 HIPFIRE_TP_BENCH_N=2 /tmp/rccl_a2a_smoke
```

These should be promoted into the tree once the RCCL FFI wrapper lands —
proposed location: `crates/hip-bridge/examples/rccl_smoke.rs` (Rust
calling our new RCCL bindings) so it stays in sync with the runtime path.
