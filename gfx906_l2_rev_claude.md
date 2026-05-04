# Adversarial review — `plans/gfx906_mmq_l2.md`

Reviewer: Claude (Opus 4.7), 2026-05-04.
Scope: the L2 prefetch v1 design for `gemm_hfq4g256_residual_mmq_gfx906.hip`,
read against the post-MMQ_X=8 perf checkpoint and the full MMQ plan.

**Revision (2026-05-04 v2):** incorporates the glm-5-turbo review at
`plans/gfx906_l2_rev_glm5.md`. Their §1.3 (occ=1 known-zero), §1.4
(dead-code = negative signal), §2.1 (window-vs-round-trip math), and
§2.5 (64-bit address = VGPR pair) sharpen this review materially —
attributed inline. Their §1.1 ("memory isn't the bottleneck at all")
and §3.4 ("plan's 2-cache-line claim is itself wrong") are accepted
with caveats below.

Bottom line: **the plan is plausible but is selling itself on the wrong
mechanism.** The prefetch helper as written almost certainly will not move
prefill 10%, because the diagnosis it's targeting ("`s_waitcnt vmcnt(0)`
on next-iter X tile from HBM") is contradicted by the very rocprof data the
plan cites. The kernel is not HBM-fetch-bound — it's residual-spill-bound
plus s_waitcnt-on-LDS-bound. A 32-lane × 4-byte prefetch trying to warm
a 16 KB tile is order-of-magnitude undersized for the bandwidth axis
**and** mistargeted for the latency axis. Between the bandwidth ceiling
(<3% per §2 below), the VGPR risk that lands us in a configuration we
already measured as zero-gain (§3), and the fact that llama.cpp shipped
five variants and chose to call none of them (§4), the expected value is
negative.

I'd recommend **not implementing v1 as specified**. Either reframe it as a
diagnostic experiment (prove which axis is the actual stall before
committing 2 days of kernel work) or rescope to attack the larger
remaining cost — operand-fetch latency from LDS, not from HBM.

Concrete defects below, ranked by impact.

---

## 1. The diagnosis doesn't survive the data the plan cites

The plan's premise (top of "Bottleneck recap"):

> the remaining idle time is `s_waitcnt` for VMEM completions on
> global_load (HBM→register) of A weights and remaining scratch loads.
> **L2 prefetch directly attacks the fetch axis.**

But the same table immediately below shows:

| Counter | MMQ_X=8 |
|---|---|
| MemBusy | 24.4% |
| **MemStall** | **2.9%** |
| LDSBank% | 0% |

`MemStall = 2.9%` is the percentage of time the memory units **could not
accept a new request** because they were saturated. **2.9% is not a
stall problem.** It means HBM/L2 has plenty of headroom — we are not
queuing. If we were waiting on HBM for X-tile fetches, we'd see this
counter climbing. It isn't.

What is climbing as we shrink MMQ_X is `MemBusy` (11.2 → 24.4%) — but
that's the memory units doing *useful* work, not waiting. The dev log
itself reaches a different conclusion in the next paragraph
("scratch ops per dp4a is still 0.58:1") and again at the end
("Bottleneck shifted from spill latency to operand fetch latency").

**"Operand fetch latency" is ambiguous.** It could mean:
- (a) global_load HBM→register for A weights (the plan's assumption)
- (b) ds_read LDS→register for staged X/Y operands inside `vec_dot_dp4a`
- (c) remaining scratch loads (residual ~144 spills/thread × N sites)
- (d) `s_waitcnt lgkmcnt(0)` at the 3 `__syncthreads()` per kg
  (glm-5 §1.1 enumerates this; with `groups_per_row=16` for K=4096
  that's 48 barriers per call)
- (e) dp4a→dp4a register dependency chain (4-cycle latency × 8 chained
  per inner loop = 32-cycle chain that the scheduler may not be able
  to interleave under the 564 B/thread spill load — also glm-5 §1.1)

The data points to **(c), (b), and (d)**, not (a):

- ISA composition pre-fix: 185 `global_load` instructions vs 13,853
  scratch ops vs 3,660 `ds_read`. Even a 10× cut in scratch leaves
  ~1,400 scratch ops, 7× more than `global_load`. **Global loads are
  rounding error in this kernel.**
- `SQ_WAIT_INST_LDS = 0` at MMQ_X=64 — but that was when the kernel was
  starving for VALU. Re-measure at MMQ_X=8 before deciding LDS waits
  are still zero; the dev log doesn't show that counter for the new
  config.
- The 16 KB X tile per WG, loaded "fresh every kg, no reuse" — that's
  16 KB × 256 WGs / 32 kg per row ≈ small per-call HBM volume against
  ~1 TB/s peak. HBM is not the constraint.

**Where I'd push back on glm-5:** their §1.1 takes `MemBusy=24%, MemStall=2.9%`
to mean "memory has 75% spare, therefore not the bottleneck" — confident
but slightly overreached. `MemBusy` is "memory units occupied", which
includes time spent draining the scratch traffic. It's not pure HBM
busy. So the right framing is: HBM is not the queue depth bottleneck,
but memory-unit *issue slots* could still be a fraction of the cost.
Either way, the conclusion (HBM prefetch isn't the right lever) holds.

**Required before any prefetch coding:** rerun rocprof at MMQ_X=8 with
`SQ_WAIT_INST_LDS`, `SQ_WAIT_INST_VMEM`, L2 hit rate, and `FetchSize`
(the latter per glm-5 §4 — fetch-unit stall is a real candidate
explanation for the 67% idle that the plan handwaves at).

## 2. The bandwidth math doesn't close — *and* the plan's own count is wrong

Plan's §"Lane count and traffic":
> Total per iter: 32 × 4 = 128 B prefetched, hitting **2 cache lines**

**Two errors in one sentence.** Glm-5 §3.4 catches the second one: the 32
lanes target rows 0, 4, 8, …, 124 — 32 *different* rows, each in a
different region of the weight matrix. Row stride = `groups_per_row × 136`
≈ 2,176 B for K=4096. Adjacent prefetched rows are 8,704 B apart in
memory, far more than a cache line. So the 32 prefetches hit **32
distinct cache lines**, not 2. The plan's own arithmetic is off by 16×.

This is actually *better* for the bandwidth case (more L2 warmed), but
still not enough:

- X tile per WG per kg = 16,768 B = ~256 cache lines (64 B each).
- 32 distinct lines warmed = 32/256 = **12.5% of the tile**. (Glm-5
  reaches 10.9% via a slightly different chunks-per-line accounting;
  same order of magnitude.)
- The remaining 87.5% of cache lines must still be fetched on demand
  during `load_hfq4_tile_dp4a(kg+1)`.

Glm-5 §2.3 adds a sharp point: **gfx906 has no hardware spatial
prefetcher for global memory.** A `global_load_dword` brings exactly the
referenced 64 B line into L2 — adjacent lines are not pulled in. So
the plan's assumption (line 119: "L2 line fills tend to bring adjacent
rows along") is wrong. We need to verify this against the GCN5/CDNA1
arch guide before committing the design, but the default assumption
should be no spatial prefetch.

**Best-case yield** of v1: hide the latency of 32 line fills out of
~256 needed = ~12.5% of the X-tile HBM latency. If X-tile HBM latency
were the whole stall (it isn't — see §1), 12.5% × the X-tile fraction
of MemBusy (which is itself 24%) of wall-clock = realistically 1–3%
gain. The plan's "≥10% to keep" threshold is **above the design's
ceiling**. The experiment is structured to fail.

## 3. The VGPR cost analysis is too optimistic — *and* the wrong VGPR count

Plan §"VGPR cost":
> So we add **1 VGPR** to the kernel's arch_vgpr count. Currently
> 128/128 (occ=2 ceiling), so this could push us to 129 VGPRs and drop
> occupancy to 1.

Two issues.

**(a) Wrong count of VGPRs added.** Glm-5 §2.5: a `global_load_dword`
on gfx906 needs the address in a 64-bit VGPR pair (`v[n]`, `v[n+1]`)
plus the destination VGPR. So the prefetch site costs **3 VGPRs**, not
1. Some of these may be short-lived enough to coalesce with existing
ranges, but it's not "1 VGPR" by inspection — the plan needs ELF
verification before concluding.

**(b) "Pushing to 129" isn't the right model.** We're already at the
spill ceiling: 144 spilled VGPRs/thread at MMQ_X=8. The compiler has
*already* exhausted the 128-VGPR budget and is spilling 564 B/thread to
scratch. Adding any new long-lived VGPR doesn't push us cleanly to 129
— it pushes us to **another 4–8 spills**, because the new VGPR competes
with the existing live ranges and the compiler resolves the conflict by
spilling lower-priority state.

**(c) The fallback occupancy configuration is a known dead end.**
Glm-5 §1.3 makes this point sharply: from the dev log's optimization
attempt #2 (MMQ_X=64):

```
| (128, 2) → 46.7 tk/s, vgpr_spill_count=2121
| (128, 1) → 46.8 tk/s, vgpr_spill_count=1780 (−16%)
```

We *already measured* occ=1 as zero-gain at the larger tile. The 16%
spill reduction was exactly cancelled by the latency-hiding loss.
**If this prefetch pushes us to occ=1, we get the occupancy penalty
with no spill improvement** — a strict regression, not a wash. The
plan's "expected outcomes" table puts "spill increase >50%, perf
neutral" at low likelihood, but the actual risk is "occupancy drop
to 1 with no compensating spill reduction" and that's a guaranteed
≤0% delta.

The plan's "Mitigation 1: Hope LLVM keeps it inside the existing VGPR
budget" is a hope, not a mitigation. Mitigation 3 ("write to LDS
scratch instead") is plausible but the plan dismisses it as "higher
cost." Either mitigation 3 needs to be the default path, or the
experiment must front-load `vgpr_count` and `vgpr_spill_count` deltas
from the ELF before any perf measurement runs.

## 4. The reference code being dead is a *negative* signal, not neutral

The plan says (line 38–45):
> none of them are called from any kernel in their tree — they are
> dead code. … We adopt the *technique* from their helpers but treat
> the *yield* as unknown.

Glm-5 §1.4 sharpens this: llama.cpp-gfx906 ships **five** prefetch
variants (`v1`, `v2`, `v4`, `_second`, `_noop`) plus a no-op stub. The
`_noop` exists specifically so the call sites can be compiled out
without warnings — meaning at one point there were call sites and the
authors put a toggle in. Five variants + a no-op + zero current call
sites is the empirical signature of: **they tried it, it didn't help,
they ripped the call sites out and left the helpers as documentation.**

That is a *negative* prior, not a neutral "yield is unknown" prior. The
plan inherits the helpers without inheriting the negative signal. We
should bias against the design proportionally.

This also affects §5 below: if llama.cpp tried this *with* their full
optimization stack (which beats us at 235 vs 125 tk/s) and still didn't
ship it, the prior over our small variant landing the gain is even
weaker.

## 5. The dummy-discard pattern needs a real test, not "optional"

Plan §"Discard pattern":
> Optional. If ISA dump shows the global_load is correctly emitted,
> this can be skipped.

This is exactly the kind of compiler-behavior assumption that bit the
"`zp_eff = zp + 8·scale` fold" attempt (no-op because LLVM already
CSE'd it). LLVM is *very* aggressive at removing dead loads, even
through `asm volatile` with `: "memory"` if the result is never read
and the asm has no other observable effect.

Glm-5 §3.1 raises a related point: the `: "memory"` clobber is doing
nothing useful here — `asm volatile` already prevents removal, and the
clobber adds a spurious scheduling barrier that may force unrelated
memory ops to be ordered around the prefetch. Drop the `"memory"`
clobber; keep the `volatile`; verify in ISA.

The keep-alive `v_mov_b32 %0, %0` (plan line 146) is also fragile.
Newer LLVM may recognize it as a no-op and elide it. Glm-5 §3.2
suggests `s_nop 0` or a write to a volatile global — but those have
their own costs. The robust pattern is the LDS-bounce mitigation
mentioned in §3 above: write `dummy` to an LDS scratch slot, never
read it. LDS write traffic is cheap (we already write 177 ds_write per
call) and the compiler can't elide a store to memory the kernel
declares as having outside observers.

Plan's "step 2: ISA inspection" needs to be a hard gate, not a
post-hoc check. If we don't see 32 `global_load_dword` instructions
in the disassembly at the expected location, the experiment is over.

## 6. "32 lanes prefetch X only" doesn't match the bottleneck shape

If you accept that prefetch *is* the right axis (I don't, see §1), the
design still picks the wrong target.

Glm-5 §2.1 makes the latency-hiding window argument concretely: at
6.72 ms per call and 16 kg iterations, each kg is ~420 µs. The
prefetch-to-use window (2 compute + 2 barriers + 1 Y load) is 30–50%
of that = 125–210 µs. gfx906 HBM round-trip is ~200–400 cycles ≈
115–230 ns. So the hiding window is **500–1800× longer than the HBM
round-trip we're trying to hide**. Latency is *already* hidden by the
sheer amount of compute between iterations. What's *not* hidden is
the L1/L2 *miss* on first access during `load_hfq4_tile_dp4a(kg+1)` —
and that's a bandwidth question (how many lines are warm), not a
latency question (when did we issue).

This sharpens my §2 conclusion: the actual lever you'd want is "warm
more lines per kg" (full or partial double-buffering — see glm-5 §4),
not "issue earlier." The plan's design hides nothing useful because
nothing useful needs hiding at this granularity.

If we still wanted to spend the VGPR + complexity on prefetch, the
right v1 would be:
- prefetch from kg+2, not kg+1, to actually cover the round-trip
  *and* span more compute
- or issue 2× the lanes to halve the miss count
- but ideally neither — see §10 alternative

The chosen design (kg+1, 32 lanes) is the timid version of both.

## 7. The "≥10% gain to keep" threshold is undermotivated

Where does 10% come from? The dev log shows a 22% headroom remains
between 125 (us) and 140 (FP16) tk/s baselines, and 88% between us
and 235 (llama.cpp). 10% is enough to claim a perf win in isolation but
doesn't move the needle vs. either reference. And it's almost certainly
above what this design can deliver (§2 caps it at <3%).

A useful threshold would be:
- **5% gain**: keep, follow up with v2 (more lanes, kg+2)
- **<2% gain**: revert, rule out prefetch as the lever, move to next
  axis (per-row CSE of zp_w/scale_w, accumulator transpose, etc.)
- **negative gain**: revert, instrument to find why (likely VGPR spill
  cascade per §3)

The dev log already lists 5 untested optimizations from the prior
session (open observations §1–§5). Several look higher-yield: items 2
(accumulator transpose) and 3 (selective un-unroll) directly attack
the residual ~144 spills, which is the actual remaining cost.

## 8. rocprof validation expectations are wrong (per glm-5 §2.4)

The plan's step 6:
> VALUBusy should rise (less idle time)
> MemStall should drop or stay flat

VALUBusy is `compute_time / total_time`. If prefetch saves time on a
non-VALU phase (cache miss latency), `total_time` drops while
`compute_time` stays constant → VALUBusy *rises*. But if the cache
misses are masked by other stalls (spill, barrier), `total_time` may
not drop at all → VALUBusy stays flat. Either is consistent with
"prefetch worked." The counter is ambiguous as a go/no-go signal.

The right go/no-go is wallclock (step 5, already specified). The
counter check should focus on **L2 hit rate** as the leading indicator
that the prefetch is *reaching* L2 at all. If L2 hit rate doesn't
improve, something earlier is broken (compiler elided the load, address
is wrong, line was already warm, etc.).

## 9. Smaller technical issues

- **Address calc bounds**: `(row0 + row_in_tile < M) ? : (M - 1)`.
  When `row0 + row_in_tile >= M`, prefetching `M-1` from the *next* kg
  loads valid data but warms the wrong cache line for the actual use
  (which is bounded-out and never happens). Cheap, but the warmed line
  is wasted L2 capacity. For the M=4096 prefill case this never trips,
  but if the kernel is reused for smaller M it's silent waste.

- **Predication ordering**: `if (kg_next >= groups_per_row) return;`
  before the `threadIdx.x >= 32` check. Either order works, but the
  warp-uniform `kg_next` check should come first to allow the entire
  warp to short-circuit without lane masking — minor codegen win.

- **`actual_row * groups_per_row + kg_next) * 136`**: where does 136
  come from? The dev log says X tile rows are 131 B (HFQ4 group = 128
  nibbles + 8 B scale/zp = 64+8 = 72? doesn't match either). The plan
  uses 136 without derivation. Wrong stride → prefetched address is
  invalid → either segfault on debug build or warming the wrong L2
  line on release. Needs a constant from the actual layout, not a
  magic number. Glm-5 §2.2 corroborates the stride concern (their
  derivation also lands on 136 implying a 4-byte alignment pad).

- **No mention of `s_waitcnt` after the asm**: a `global_load_dword`
  with `: "memory"` clobber may force a fence depending on LLVM
  version. We need `s_waitcnt vmcnt(N)` *before* the line is read by
  compute, but *not* before the prefetch issues — i.e. zero waits
  injected at the prefetch site. Verify this in the ISA dump or the
  prefetch becomes synchronous and pessimizes the kernel.

- **L2 line size assumption**: plan says "gfx906 L2 line = 64 B". MI50
  L2 line is 64 B for most accesses but **128 B for some patterns**
  (sector-cached). Verify against the GCN/CDNA arch guide rather than
  assuming. (Glm-5 §3.3 also flags the missing citation.)

## 10. Alternative direction: chunk-pipelined X load (per glm-5 §4)

Glm-5's alternative is concrete enough to evaluate. Their full
double-buffered X tile fails on LDS budget (33 KB × 2 = 66 KB > 64
KiB cap), but their **chunk-pipelined** variant is interesting:

```
for kg = 0..N-1:
  load_X_chunks(kg, 0..15)     // first half of payload
  load_Y_half1(kg)
  barrier; compute_half1()
  load_X_chunks(kg, 16..31)    // second half, overlaps with first compute
  barrier; compute_half2()     // uses first-half X already in LDS
  barrier
```

This overlaps **8 KB of payload loads** with compute — 4× more than
the prefetch design's best case (2 KB warmed in §2). It requires
restructuring `load_hfq4_tile_dp4a` into two phases and matching the
split inside `vec_dot_dp4a`, which is invasive but stays within the
existing LDS budget and doesn't add VGPR pressure (the loads go
straight to existing LDS slots, no dummy register).

Caveat: the "uses first-half X already in LDS" claim assumes
`vec_dot_dp4a`'s first compute call only reads the first 16 chunks.
That depends on how the kernel's `k01 ∈ {0,8,16,24}` loop maps to
chunks — needs verification before committing the design.

If the diagnostic version of the prefetch (per §11 below) confirms
HBM/L2 *is* the bottleneck, chunk-pipelining is the better v2 than
"more lanes / kg+2 prefetch."

## 11. What the experiment *would* be useful for

If reframed as a diagnostic, this design has value:

1. Build it as specified, ignore perf entirely.
2. Run rocprof groups 1+2+3 with and without the prefetch.
3. Diff the L2 hit rate, `SQ_WAIT_INST_VMEM`, and `MemBusy` counters.
4. If L2 hit rate moves but perf doesn't, prefetch is on the wrong
   axis — confirms the diagnosis in §1. Move to §10 or one of the
   spill-attack items.
5. If L2 hit rate doesn't move, the prefetch isn't even reaching L2 —
   confirms the bandwidth math in §2 and/or §3 (compiler elided it,
   or address is wrong). Cheap to debug from there.
6. Either way, we eliminate prefetch from the search space cheaply.

The "5 days, port to gate_up next" framing in the carry-over section
should not happen until we know the answer.

## 12. Recommended alternative ordering

Given the dev log's 5 untested options and the threshold problem above:

1. **rocprof at MMQ_X=8 with `FetchSize`, `SQ_WAIT_INST_LDS`,
   `SQ_WAIT_INST_VMEM`, and L2 hit rate (1 hour, no kernel changes).**
   This is the cheapest and most informative thing we can do. Per
   glm-5 §4 and my §1, the bottleneck attribution is currently a guess.
2. If the rocprof shows fetch-unit stalls dominate: loop compression
   (selective un-unroll) directly. Estimated 1 day.
3. If LDS waits dominate: barrier elimination in the Y-twice pattern
   (item 5 from the dev log). Estimated 2 days.
4. If scratch round-trips dominate: accumulator transpose (item 2
   from the dev log) — directly attacks the residual ~144 spills.
   Estimated 1–2 days, plausible 5–10% gain.
5. If after (2)–(4) the kernel is *clearly* HBM-bound, then
   chunk-pipelined X load (§10) is the best-shaped lever. Skip
   single-line prefetch entirely.
6. The L2 prefetch as designed is **last** on this list, not first.

The plan's "decision points to track" should include this ordering
question explicitly: **why prefetch first**, given the dev log says the
remaining cost is fetch latency and the prior 4 optimizations on the
spill side delivered 158% perf gain (46.7 → 120.9)?

## 13. What I'd accept the plan with

If we keep prefetch as v1 next step (against my recommendation):

1. Drop "≥10% to keep" → "any directional gain to keep, log result
   either way; reframe as diagnostic if perf-flat".
2. Add a hard prerequisite: rocprof at MMQ_X=8 with LDS counters,
   fetch-unit busy, and L2 hit rate. The plan currently relies on
   counters from the pre-MMQ_X=8 era for the LDS/scratch ratios (the
   perf-checkpoint group 1 and 2 numbers are MMQ_X=64).
3. Pick *one*: either 32 lanes × 1 dword × kg+1 (the diagnostic
   version) or scale to 64+ lanes / kg+2 (the version that could
   actually hide HBM latency). The current middle ground does neither
   well.
4. Make the dummy-discard pattern non-optional. Default to the
   LDS-bounce fallback (write `dummy` to a small LDS scratch slot)
   rather than register keep-alive — robust against future LLVM
   changes and avoids the 64-bit address VGPR-pair problem from §3.
5. Derive the X-row stride (the magic 136) from the layout constants
   in the kernel, not as a literal.
6. Front-load ELF inspection: `vgpr_count`, `vgpr_spill_count`,
   `private_segment_fixed_size` deltas before any wallclock measurement.
   If spills jump >5%, abort before the bench (the spill cascade
   per §3 is the most likely failure mode).
7. Carry-over to gate_up should be **deferred until v1 ships a
   measurable win**, not pre-committed.

Without these, this is a 3-day kernel experiment betting on a
hypothesis the available data already weakens.
