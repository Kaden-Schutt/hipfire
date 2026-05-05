# 2026-05-04 gfx906 MMQ_X=8 bottleneck attribution

Hardware: AMD MI50 (gfx906), ROCm 6.4.3.
Baseline: commit `39b1eb7`. Prefill 125.2 tk/s on Qwen 3.5 9B pp128
(MMQ_X=8 + screening on), 89% of FP16 wave64 baseline at 140.7 tk/s.

This checkpoint executes step 1 of `plans/gfx906_mmq_l2.md` v2:
**rocprof attribution at MMQ_X=8 with the right counter set, before
any kernel change.** The previous dev log
(`2026-05-04-gfx906-mmq-spill-reduction.md`) collected counters at
MMQ_X=64 where the kernel was VALU-starved; this run collects them at
MMQ_X=8 where wallclock has dropped 5× and the bottleneck has shifted.

## TL;DR

The remaining 91% idle time at MMQ_X=8 is **dominantly VMEM_WR
latency from spill-store traffic** that misses L2 ~35% of the time.
LDS, fetch unit, and instruction-cache pressure are all secondary.

Specifically:
- **MMQ writes 8× more VMEM ops per VALU op than FP16** (0.067 vs
  0.001), and 517× more bytes to HBM per call (517 KB vs 1 KB).
- **L2 hit rate is 65% on MMQ vs 85% on FP16.** Spill traffic is
  spilling out of L2 too.
- **`SQ_WAIT_INST_LDS = 0`.** LDS issue queue is not stalling,
  contradicting one of the candidate axes from the L2 prefetch plan
  reviews.
- **FetchSize is 2.85× FP16's** but absolute volume is small
  (231 KB/call for MMQ vs 81 KB/call for FP16). I-cache pressure is
  real but secondary.

**Picked lever: selective un-unroll of the `j0` loop in
`vec_dot_dp4a`.** Cuts simultaneous live values from 64 → 16 without
reducing dp4a ILP4 or changing the tile shape. Estimated 1 day,
plausible 5–10% gain.

**Rejected levers based on data:**
- L2 prefetch (the plan we were originally going to implement) — wrong
  axis. The kernel isn't HBM-bandwidth-bound; it's spill-write-bound.
- Accumulator transpose — `sum[]` is only 8 floats per thread at
  MMQ_X=8, not the spill source.
- Y-twice barrier collapse — `SQ_WAIT_INST_LDS = 0` says barrier
  waits aren't dominant.

## Counters captured (MMQ_X=8, current commit `39b1eb7`)

64 kernel calls per group (32 K=4096 calls + 32 K=12288 calls), Qwen
3.5 9B pp128, with `HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1`.

### Per-call comparison (K=4096 layer, simpler shape)

| Metric | MMQ_X=8 | FP16 wave64 | MMQ/FP16 |
|---|---|---|---|
| Wallclock per call | **3.41 ms** | **2.16 ms** | **1.58×** (slower) |
| VALUBusy | 8.80% | 61.47% | 0.14× |
| MemUnitBusy | 24.19% | 68.80% | 0.35× |
| MemUnitStalled | 2.91% | 0.85% | 3.4× |
| WriteUnitStalled | 0.018% | 0.000% | — |
| L2 cache hit rate | **65.0%** | **84.7%** | 0.77× |
| FetchSize per call (B) | 230,521 | 80,839 | 2.85× |
| WriteSize per call (B) | 517,844 | 1,002 | **517×** |
| SQ_INSTS_VMEM_RD | 2,615,296 | 5,505,024 | 0.48× |
| **SQ_INSTS_VMEM_WR** | **2,070,528** | **262,144** | **7.90×** |
| SQ_INSTS_FLAT | 212,992 | 5,767,168 | 0.04× |
| SQ_INSTS_VALU | 30,669,824 | 132,612,096 | 0.23× |
| SQ_INSTS_LDS | 524,288,000 (sum across 64 calls) | 83,886,080 | 6.25× |
| **SQ_WAIT_INST_LDS** | **0** | 2,226,933 | — |

### Wall-clock decomposition (K=4096, 3.41 ms per call)

Latency-attributed model (50-cycle average per VMEM op, 64 lanes/wave):

```
4.69 M VMEM ops total per call
÷ 64 lanes = 73K wave-issue per call
× 50 cycles avg = 3.65M cycles = 2.15 ms

VMEM latency attribution: 2.15 / 3.41 = 63% of wallclock
Remainder (VALU + barrier + ds_read + setup): 37%
```

This matches the observed `MemUnitBusy = 24%` (memory units active)
and `VALUBusy = 8.8%` (compute active) summing to ~33% of wallclock —
the rest is *latency*, not *queue depth*. The kernel is round-trip-
latency-bound on VMEM.

## Axis-by-axis verdict

Per the decision tree in `plans/gfx906_mmq_l2.md` §"What we actually
need to do first":

| Axis | Counter signal | Verdict |
|---|---|---|
| **Scratch / spill writes** | VMEM_WR=2.1 M/call (7.9× FP16); WriteSize 517×; L2 hit drops to 65% | **DOMINANT** |
| Spill reads | VMEM_RD=2.6 M/call (0.48× FP16) | not anomalous; FP16 reads more and is fast |
| LDS issue waits | SQ_WAIT_INST_LDS = 0 | rejected |
| LDS bank conflicts | LDSBank% = 0.000% | rejected |
| Barriers | SQ_WAIT_INST_LDS = 0 (covers `__syncthreads()`) | rejected |
| Fetch unit / i-cache | FetchSize 2.85× FP16, but absolute 231 KB/call | secondary, not dominant |
| HBM/L2 miss on weight loads | FLAT=0.21 M/call (0.04× FP16) — global loads are tiny | rejected (HBM/L2 prefetch wouldn't help) |

**Two rejections are particularly important:**

1. **L2 prefetch is wrong axis.** The kernel does very few global
   loads (FLAT=0.21 M/call vs FP16's 5.77 M). HBM→L2 latency for
   weight loads is not the bottleneck. The L2 hit rate problem is on
   the *spill* path (writes to scratch), not the *weight* path —
   confirmed in `docs/plans/gfx906-mmq-prd.md` §3.3.
2. **LDS issue / barrier waits are zero** — the `__syncthreads()`
   pattern in `mmq_body` is not stalling. Restructuring the X-once +
   Y-twice barrier pattern won't help.

## Why the spill writes are excessive

`vec_dot_dp4a` (line 254-322) at MMQ_X=8 unrolls
`#pragma unroll` four loops:

```
k01 ∈ {0, 8, 16, 24}                  // 4 iters
  j0 ∈ {0, 2, 4, 6}                   // 4 iters (MMQ_X=8, step=MMQ_NWARPS=2)
    i0 ∈ {0, 64}                       // 2 iters (MMQ_Y=128, step=WAVE_SIZE=64)
      v ∈ {0..7}                       // 8 dp4a calls (vdr=8)
        sumi = sdot4(x_int, y_int, sumi)
```

= 256 fully-unrolled dp4a sites. Per site, the body needs:
`x_int`, `y_int`, `sumi`, `dm_i.x`, `dm_i.y`, `dsf.x`, `dsf.y`,
`ds_j`, `scale_w`, `zp_eff`, `d_x`, `sum_x`, `idx`, plus loop-invariant
values that the unroll cross-pollinates. With 4 j0 iterations all live
simultaneously, that's roughly 16 unique per-site values × 4 j0 ×
2 i0 = 128 unique live SSA values inside the k01 body. The compiler
can't fit this in 128 VGPRs and resolves the conflict by spilling
~144 VGPRs/thread (564 B private segment).

Crucially, the `sum[]` accumulator itself is **only 8 floats per
thread** at MMQ_X=8 (`(MMQ_X/MMQ_NWARPS) * (MMQ_Y/WAVE_SIZE) = 4 × 2`).
That's 8 VGPRs. The "accumulator transpose" lever from the dev log's
open observation §2 was sized for the MMQ_X=64 case where `sum[]` was
64 floats. **At MMQ_X=8 it's not the source of pressure.**

## Picked lever: selective un-unroll of the j0 loop

Drop `#pragma unroll` from the `j0` loop (line 283) so the compiler
processes one j0 iteration at a time. Keep `#pragma unroll` on the
inner `v` loop (preserves dp4a ILP4 — 4 sequential `v_dot4_i32_i8`
instructions are needed to hit peak throughput per the gfx906 ISA
docs).

Effect on live ranges:
- Before: 4 j0 iters × 2 i0 iters × 8 v iters = 64 simultaneously
  live sites in the inner k01 body.
- After: 1 j0 iter × 2 i0 iters × 8 v iters = 16 simultaneously
  live sites. **4× reduction in live-range pressure.**

Trade-off:
- Loses cross-j0 instruction interleaving (1 j0 at a time means the
  scheduler can't pipeline the `ds_j` half2 load for j0=2 while
  computing j0=0). But: at VALUBusy=8.8%, we have huge ALU headroom
  to absorb a serialized j0 schedule.
- May add a few cycles per j0 iteration for the small dispatch
  overhead. Cheap relative to the spill latency we're killing.

Estimated impact:
- If spill traffic drops ~4× (from 144 spills/thread to ~36) →
  VMEM_WR drops to ~0.5 M per call (matches FP16 ratio of ~2× rather
  than 8×) → ~50% of the 2.15 ms VMEM-attributed time recovered →
  **K=4096 call drops from 3.41 → ~2.3 ms** → prefill rises ~1.4×
  to ~175 tk/s.
- Conservative: even a 2× spill reduction → 25% wallclock recovery
  → ~150 tk/s.
- Pessimistic: scheduler can't recover the latency hiding lost from
  unrolling → flat or small regression. Reverts trivially.

## Implementation steps

1. Edit `kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip:283`:
   change `#pragma unroll` to `#pragma unroll 1` on the `j0` loop.
2. Build, dump ELF: confirm `vgpr_count`, `vgpr_spill_count`,
   `private_segment_fixed_size` deltas. **Hard abort if spill goes
   up.** (Per the L2 plan v2's "front-load VGPR/spill check" rule.)
3. Quick correctness check:
   `./tests/test_gfx906_mmq_correctness 4096 4096 32` (NRMSE ≤ 0.13%)
4. Wallclock bench:
   `bench_qwen35_mq4 ... --prefill 128 --prefill-runs 3` against
   `39b1eb7` baseline 125.2 tk/s.
5. If perf gain: rocprof g0/g2 to confirm VMEM_WR dropped and VALUBusy
   rose.
6. If perf flat or worse: revert; try `#pragma unroll 2` (compromise
   between full unroll and serial) before moving on.

## Fallback ladder if j0 un-unroll doesn't help

In order, gating each on the previous failing:

1. **Selectively un-unroll the `i0` loop instead** (line 295). i0 has
   only 2 iterations; the saving is smaller but cleaner.
2. **Hoist `__half22float2(ds_j)` and `dm_i = x_dm[i]` out of the v
   loop.** These are loop-invariant per (j, i) but the unrolled body
   may be re-emitting them.
3. **`v_dot4_i32_i8` ILP**: try `#pragma unroll 2` on the v loop
   (vdr=4 effective). Loses 2× dp4a throughput per call but may free
   VGPRs from the cross-v live ranges.
4. **Chunk-pipelined X load** (per the L2 plan v2 alternative
   section). Larger restructuring; only if 1–3 leave wallclock
   above ~2.6 ms/call.

## Raw artifacts

- `/tmp/rocprof_mmq/mmqx8_g{0,1,2}.csv` — original captures (3 groups
  × MMQ + FP16, 64 calls each)
- `/tmp/rocprof_mmq/mmqx8_g3_l2hit.csv` — L2CacheHit (this run)
- `/tmp/rocprof_mmq/mmqx8_g3_fetch.csv` — FetchSize (this run)
- `/tmp/rocprof_mmq/mmqx8_g3_write.csv` — WriteSize (this run)

## Cross-reference

- Final outcome (L2 prefetch rejected, redesign supersedes this
  thread): `docs/plans/gfx906-mmq-prd.md` §3.3,
  `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
- Prior dev log (MMQ_X reduction):
  `docs/perf-checkpoints/2026-05-04-gfx906-mmq-spill-reduction.md`
