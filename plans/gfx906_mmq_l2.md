# gfx906 MMQ — fetch-axis investigation plan

Status: design phase, no code yet.
Target kernel: `kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip`
Baseline this builds on: commit `39b1eb7` (MMQ_X=8 spill reduction
landed; prefill 125.2 tk/s on Qwen 3.5 9B pp128 with screening on,
89% of FP16 wave64 baseline at 140.7 tk/s).

**Revision history**
- v1 (2026-05-04): proposed L2 prefetch as the next lever, gated on
  ≥10% prefill gain. Two adversarial reviews (Claude, glm-5-turbo)
  rejected the design — see `gfx906_l2_rev_claude.md` v2 and
  `plans/gfx906_l2_rev_glm5.md`.
- v2 (2026-05-04): plan reframed around the review findings. Prefetch
  downgraded from "next lever" to "diagnostic only," and only after a
  cheap rocprof attribution step rules out cheaper wins.
- v3 (2026-05-04): attribution step done. See
  `docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`.
  Findings:
  - **Spill-write traffic is dominant** (VMEM_WR 7.9× FP16 per call,
    L2 hit 65% vs 85%, WriteSize 517× FP16). Round-trip latency
    accounts for ~63% of K=4096 wallclock.
  - **LDS issue queue is not stalling** (`SQ_WAIT_INST_LDS = 0`).
    Y-twice barrier collapse rejected.
  - **Global loads are tiny** (FLAT 0.04× FP16). HBM/L2 prefetch on
    the weight path would do nothing — confirming the v1 reviews.
  - **`sum[]` is only 8 floats per thread at MMQ_X=8.** Accumulator
    transpose is sized for the wrong era; rejected.
  - **Picked lever:** selective un-unroll of the `j0` loop in
    `vec_dot_dp4a` to cut simultaneous live values 4× without losing
    dp4a ILP4. Estimated 1 day; plausible 5–40% gain depending on
    how much spill the compiler recovers.
- v4 (2026-05-04, this rev): j0 un-unroll **shipped and benched**.
  See `docs/perf-checkpoints/2026-05-04-gfx906-mmq-junroll.md`.
  Outcome:
  - **Prefill 125.2 → 145.5 tk/s (+16.2%).** First time MMQ on
    gfx906 beats the FP16 wave64 baseline (141.3 tk/s); now +3.0%.
  - **Spills fully eliminated** (vgpr_count 128→60, spill 144→0,
    private_segment 564→0).
  - **Spill traffic eliminated** (VMEM_WR 2.07 M→3.5 K per call,
    WriteSize 517 KB→949 B per call).
  - VALUBusy 8.7%→15.6%; still 4× lower than FP16's 61.5%, so
    significant headroom remains.
  - Attribution model validated end-to-end: predicted 5–40% gain,
    delivered 16.2%. The remainder of the predicted ceiling (~63%
    VMEM-attributed) didn't translate because spill writes were
    actually being absorbed by L1, not always reaching HBM, so wall-
    clock latency was masked by L1 hit; nonetheless the compiler
    pressure release was real.
  - **Distance to llama.cpp reference** (~235 tk/s) closes from
    0.53× → 0.62×.

## Why the v1 plan was rejected

Three independent failure modes, any one of which makes the v1 design
unlikely to ship:

1. **Diagnosis is contradicted by data.** `MemStall=2.9%` at MMQ_X=8
   says HBM is not queue-saturated. The ISA dump shows 13,853 scratch
   ops vs 185 `global_load` instructions — global fetches are rounding
   error in this kernel. The plan's own line 14–16 ("SIMDs idle ~67%
   of the time waiting on memory completions") doesn't follow from
   `MemBusy=24%, MemStall=2.9%`. The actual remaining cost is some
   combination of (a) scratch round-trips for the residual 144
   spills/thread, (b) `s_waitcnt lgkmcnt(0)` at the 3 `__syncthreads()`
   per kg = 48 barriers per call, (c) ds_read latency for staged X/Y,
   and (d) possibly fetch-unit stall — not HBM→L2 latency.

2. **Bandwidth math doesn't close.** v1 prefetches 32 lanes × 4 B = 128 B.
   gfx906 has no spatial prefetcher for global memory, so each
   `global_load_dword` brings exactly its 64 B line into L2. The 32
   target rows are 8,704 B apart in the weight matrix → 32 distinct
   cache lines warmed (the plan's "2 cache lines" was wrong by 16×).
   But the X tile is 16,768 B = ~256 lines per kg, so coverage is
   ~12.5%. Even if HBM-fetch were 100% of stall (it isn't — see #1),
   ceiling is ~3% wallclock. Below the v1 ≥10% keep threshold.

3. **VGPR budget is already exhausted.** Kernel sits at 128/128 VGPRs
   with 144 spilled VGPRs/thread. A `global_load_dword` needs a 64-bit
   address VGPR pair plus the destination = **3 VGPRs**, not 1. New
   long-lived VGPRs cascade more spills, not just bump the count to 129.
   And the fallback (drop to occ=1) is a known zero-gain configuration
   per dev log optimization attempt #2: at MMQ_X=64, lifting `(128,2)`
   → `(128,1)` cut spills 16% and moved wallclock 0.1 tk/s.

Plus the strong negative prior: **llama.cpp-gfx906 ships five prefetch
variants (`v1`, `v2`, `v4`, `_second`, `_noop`) and calls none of them.**
The `_noop` stub exists specifically so call sites can be compiled out
without warnings. That's the empirical signature of "tried it,
ripped the call sites out, left helpers as documentation."

Full review trail in `gfx906_l2_rev_claude.md` v2 and
`plans/gfx906_l2_rev_glm5.md`.

## Status of step 1 (attribution): **DONE**

Full results in `docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`.
Summary repeated in v3 revision history above. Outcome: spill-write
dominant, j0 un-unroll picked as next lever.

Original step 1 description preserved below for context.

## What we actually need to do first (1 hour, no kernel changes)

Before any kernel work — prefetch *or* spill-attack *or* anything else
— **redo rocprof at MMQ_X=8 with the right counter set.** The dev log
counters were captured at MMQ_X=64 where the kernel was VALU-starved;
they don't tell us where the remaining 91% of idle time comes from at
MMQ_X=8.

Counters needed (group accordingly to fit HW limit):

| Counter | What it tells us |
|---|---|
| `SQ_WAIT_INST_LDS` | cycles waiting on LDS issue queue |
| `SQ_WAIT_INST_VMEM` | cycles waiting on VMEM completions (HBM/L2/scratch) |
| `SQ_WAIT_INST_SCA` | s_waitcnt at barriers and SMEM |
| `FetchSize` / `FETCH_UNIT_BUSY` | instruction fetch throughput / stall |
| `TCP_TCC_*_HIT_sum` / `TCP_TCC_*_MISS_sum` | L2 hit rate per VMEM access |
| `TCC_HIT_sum` / `TCC_MISS_sum` | L2 hit rate (controller side) |
| `MemBusy` / `MemStall` | already captured, baseline |
| `LDSBank%` / `SQ_INSTS_LDS` / `SQ_INSTS_VALU` | already captured for MMQ_X=64; redo at MMQ_X=8 |

Decision tree after rocprof:

| Dominant axis | Next lever | Plan |
|---|---|---|
| Scratch/spill (VMEM_RD waits high, low L2 miss) | reduce live state | accumulator transpose (dev log open obs §2) |
| LDS issue (SQ_WAIT_INST_LDS high) | restructure ds_read pattern | barrier elimination in Y-twice (dev log §5) |
| Barriers (SQ_WAIT_INST_SCA high) | merge or kill barriers | reorder mmq_body |
| Fetch unit (FETCH_UNIT_BUSY high) | loop compression | selective un-unroll of j-loop (dev log §3) |
| HBM/L2 miss (TCC_MISS high, L2 hit low) | overlap loads with compute | chunk-pipelined X load (§ "Alternative" below) — *not* lane-prefetch |
| Mixed / nothing >30% | step back | we may already be near a real ceiling; reconsider the gate_up port instead |

Cost: 1 rocprof rerun, no code change. Prerequisite for anything below.

## Reordered work queue

Front-load the cheaper, higher-yield items the dev log already
identified, gated on the rocprof attribution above:

1. **rocprof attribution at MMQ_X=8** (1 hour, this section).
2. If scratch/spill dominant: **accumulator transpose** — invert
   per-thread acc from `[32 × 2]` to `[2 × 32]` so the scheduler can
   keep all 32 K-strided columns live across one i-block (dev log
   open obs §2). Estimated 1–2 days, plausible 5–10% gain.
3. If LDS or barrier dominant: **collapse Y-twice pattern** — the
   X-once + Y-twice structure forces 3 barriers per kg. Restructuring
   to load both Y halves up front (or splitting compute across warps
   so each owns one Y half) drops to 1 barrier per kg (dev log open
   obs §5). Estimated 2 days.
4. If fetch dominant: **selective un-unroll of j-loop** (dev log open
   obs §3). Estimated 1 day.
5. **Cheap CSE pass**: per-row `8 * scale_w` fold (dev log open obs
   §4) and `__half22float2` hoisting. Estimated 0.5 day. Already
   tried once and was a no-op (LLVM CSE'd it), but worth retrying
   alongside the structural changes above.
6. **Only if (2)–(5) all leave the kernel HBM-bound** with measurable
   L2 miss → consider chunk-pipelined X load (next section). Skip the
   single-line prefetch entirely.

Prefetch lands last on this list, not first.

## Alternative if HBM/L2 *is* the bottleneck: chunk-pipelined X load

If rocprof attribution lands on "HBM/L2 miss in `load_hfq4_tile_dp4a`"
as the dominant cost, the right structural lever is **interleaving
the X load with compute** — not lane-prefetch. From glm-5 §4:

```
for kg = 0..N-1:
  load_X_chunks(kg, 0..15)     // first half of payload, ~8 KB
  load_Y_half1(kg)
  __syncthreads(); compute_half1()
  load_X_chunks(kg, 16..31)    // second half, overlaps with first compute
  __syncthreads(); compute_half2()  // uses first-half X already in LDS
  __syncthreads()
```

This overlaps **8 KB of payload loads** with compute — 4× more than
v1 prefetch's 2 KB best case — and stays within the existing LDS
budget (no double-buffer). It costs zero new VGPRs (loads go straight
to existing LDS slots, no dummy register).

Caveat: the "compute_half1 only reads first 16 chunks" assumption
must be verified against the actual k01 ∈ {0,8,16,24} mapping inside
`vec_dot_dp4a`. May require refactoring `vec_dot_dp4a` to take a
chunk-range parameter.

Full double-buffered X tile (33 KB × 2 = 66 KB) does **not** fit in
gfx906's 64 KiB LDS cap, so partial pipelining is the only structural
option. Sketch only — not designed in detail until rocprof confirms
the bottleneck.

## If we still want a v1 prefetch (against recommendation): mandatory changes

If after rocprof the team still wants to ship a prefetch experiment,
the v1 design needs these changes before it can land:

1. **Reframe as diagnostic, not perf experiment.** Drop "≥10% to
   keep" → "any directional gain to keep, log result either way."
   Real goal is to confirm or reject the HBM-fetch hypothesis cheaply.
   Wallclock gain from this design ceiling is ≤3% (per §2 of the
   reviews); use L2 hit rate movement as the primary signal, not tk/s.

2. **Pick one regime — not the timid middle.**
   - **Diagnostic regime:** 32 lanes × 1 dword × kg+1 (current v1).
     Goal is to detect *any* L2 hit movement, not to win wallclock.
   - **Best-case regime:** 64+ lanes × kg+2. Issues ahead of the full
     HBM round-trip and warms more of the tile. Higher VGPR risk.
   The current v1 (32 lanes × kg+1) does neither well.

3. **Fix the dummy-discard pattern.** `asm volatile` with `: "memory"`
   may still let LLVM elide the load (the `"memory"` clobber is also
   doing nothing useful — drop it; `volatile` alone is what prevents
   removal). Default to **LDS-bounce**: write `dummy` to a small LDS
   scratch slot the kernel never reads. The compiler can't elide
   stores to memory it doesn't know is dead. Avoids the 64-bit-address
   VGPR-pair problem in #4 (LDS write doesn't need address-VGPR pair
   to be live as long as a global VMEM op).

4. **Front-load the VGPR/spill check.** Before any wallclock bench:
   - Build, dump ELF, capture `vgpr_count`, `vgpr_spill_count`,
     `private_segment_fixed_size`.
   - **Hard abort if `vgpr_spill_count` increases by >5%.** That's the
     spill-cascade failure mode from §3 of the reviews and the most
     likely way this design regresses. Glm-5 §2.5 notes a 64-bit
     `global_load_dword` address needs a VGPR pair, not a scalar — so
     "+1 VGPR" estimate in v1 is wrong; expect +3 VGPRs at the prefetch
     site even before liveness across the compute window.
   - Verify `global_load_dword` instructions appear in the ISA dump
     at the expected location (this is the ISA inspection from v1
     step 2, but elevated to a hard gate — if they're not emitted,
     the prefetch isn't happening regardless of perf number).

5. **Derive the X-row stride from the layout, not the magic 136.**
   v1 uses `* 136` without derivation. Compute it from the HFQ4 group
   layout constants in the kernel — wrong stride means warming wrong
   L2 lines (silent waste at best, segfault at worst).

6. **Don't pre-commit gate_up carry-over.** v1's "carry-over to
   gate_up" section assumes the helper will be reusable. Defer that
   decision until v1 ships a measurable win on residual.

## What this plan is *not* doing

- **Not implementing v1 prefetch as originally specified.** The
  bandwidth math caps it at <3% gain even in the best case, the VGPR
  risk is asymmetric (regression possible, big win impossible), and
  the reference implementation's silence is a negative signal not a
  neutral one.
- **Not committing to chunk-pipelined X load** until rocprof says
  HBM/L2 miss is actually the dominant axis. Sketched above as the
  right structural lever *if* fetch ends up being the bottleneck.
- **Not pursuing two-stage prefetch (kg+2)** as a v2 — the right v2
  is structural overlap (chunk pipeline), not more lanes of single-line
  prefetch. The mistake is doubling down on the wrong abstraction.
- **Not pursuing stream-K work partitioning** — orthogonal axis,
  separately tracked.

## Decision points to track

- **Top of the list:** rocprof attribution at MMQ_X=8. Without this,
  every other lever is guessing.
- After rocprof: which axis (scratch / LDS / barrier / fetch /
  HBM-miss / mixed) dominates → which lever from §"Reordered work
  queue" gets implemented first.
- Whether to revisit prefetch at all after the spill-side and
  barrier-side optimizations land. Likely no — by then the kernel
  shape will have changed enough that the v1 prefetch design is
  outdated regardless.

## Expected outcomes (post-rocprof, before kernel work)

| Rocprof finding | Likelihood | Action |
|---|---|---|
| Scratch dominant (>40% of stall) | medium-high | accumulator transpose (item 2) |
| Barrier / LDS issue dominant | medium | Y-twice collapse (item 3) |
| Fetch unit dominant | low | loop compression (item 4) |
| L2 miss dominant | low-medium | chunk-pipelined X load (alt section) |
| Mixed, no clear winner | medium | stack cheap CSE (item 5) and re-measure |
| Already near ceiling | low | pivot to gate_up port |

For comparison, the v1 plan's expected outcomes table assumed prefetch
was the right axis and listed "≥15% gain, no spill increase" at
"medium" likelihood. Reviews put that probability much lower (~5%);
the threshold is above the design's bandwidth ceiling and the VGPR
risk is asymmetric.

## Carry-over

Whichever lever wins (item 2/3/4/5/alt) will have its own carry-over
analysis to gate_up at port time. Don't pre-commit any specific helper
or shared header until the residual-side work measures.
