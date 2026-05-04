# Adversarial review — `plans/gfx906_mmq_redesign.md`

Reviewer: Claude (Opus 4.7), 2026-05-04.
Scope: the proposed nwarps=4 redesign of
`gemm_hfq4g256_residual_mmq_gfx906.hip` to close the 3.4× pp128 gap
versus stock llama.cpp.

**Revision history:**
- v1 (2026-05-04): initial review by Claude, 12 sections.
- v2 (2026-05-04): integrated glm-5's review at
  `plans/gfx906_mmq_redesign_pl_rev_glm5.md`. Glm-5 raised six findings
  my v1 missed and one finding sharper than mine.
- v3 (2026-05-04, this rev): integrated gemini's review at
  `plans/gfx906_mmq_redesign_pl_rev_gemini.md`. Gemini caught **five
  findings of substantial impact** that v1+v2 missed — including the
  most important hardware-physical constraint in the entire plan
  (LDS overflow at 2 WGs/CU). Gemini's review is short and surgical.

**Bottom line (revised v3): the plan as written cannot meet its own
`__launch_bounds__(256, 2)` target. Gemini caught the LDS arithmetic:
43.5 KiB × 2 WGs = 87 KiB exceeds gfx906's 64 KiB LDS/CU. Stock fits
in 28.5 KiB × 2 = 57 KiB by holding only 32-K of x_tile in LDS at a
time, not 256-K. Our HFQ4-G256 layout was designed around 256-K-
group-resident x_tile; matching stock's LDS budget at nwarps=4
requires either restructuring the X-tile loader to a streaming
pattern or accepting 1 WG/CU.** This is a substantive architectural
finding that changes the scope from "topology change" to "X-tile
restructure."

Plus the broader v1+v2 findings: the plan oversells prediction,
conflates per-call vs coverage gaps, and underestimates the rewrite
scope. Glm-5 sharpened that gate_up is the real headline opportunity
(81% of GEMM time vs residual's 18%). And the locked-in mmq_x=128 is
moot because stock caps at 64 on gfx906.

I'd **not ship the plan as written**. Three load-bearing fixes must
land before coding starts:
1. **Resolve the LDS-vs-occupancy contradiction** (gemini #1).
2. **Confirm VGPR margin** at exactly 128 isn't a soft cliff
   (gemini #2).
3. **Fix `ds_read_b128` alignment** by padding X_STRIDE to 68 ints
   (gemini #4) — and re-verify the LDS budget after that.

Plus the v1+v2 fixes: gate_up co-design (glm-5 H3), drop mmq_x=128
(glm-5 C2), greedy step-8 dispatch (glm-5 C1), Phase 2a spill probe
(my §2), HFQ4 K-iter cadence doc (my §6).

Defects below, ranked by impact. Source attribution noted inline:
[gemini], [glm-5], [my v1], or combinations.

---

## 0a. LDS allocation contradicts `__launch_bounds__(256, 2)` [gemini #1]

**This is the most important finding in any of the three reviews.**

The plan §3 calculates per-WG LDS at mmq_x=64:
- x_qs: 33,280 B (128 rows × 65 ints)
- x_dm: 1,024 B
- tile_y: 9,216 B (64 × 36 ints)
- **Total per-WG: 43,520 B = 42.5 KiB**

The plan §1 commits to `__launch_bounds__(256, 2)` (2 WGs/CU).

**gfx906 LDS/CU = 64 KiB.** Per Vega 20 ISA reference, skyne98 wiki,
ROCm docs.

**42.5 KiB × 2 WGs = 85 KiB.** Exceeds 64 KiB by 33%. The hardware
scheduler will hard-clamp to **1 WG/CU**, immediately halving the
intended occupancy.

**I verified this against stock empirically.** From rocprof CSV:
stock's actual runtime LDS allocation at mmq_x=64 = **29,184 B per
WG = 28.5 KiB**. 28.5 × 2 = 57 KiB ≤ 64 KiB cap. Stock fits 2 WGs/CU.

How does stock fit in 28.5 KiB when ours wants 43.5 KiB? Decomposing
stock's tile_x_sizes for Q4_K with mmq_y=128:

```
MMQ_DP4A_TXS_Q4_K = {mmq_y*MMQ_TILE_NE_K + mmq_y,         // qs:  4224 ints = 16,896 B
                     mmq_y*MMQ_TILE_NE_K/QI4_K,            // dm:   128 ints =    512 B
                     mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}      // sc:   528 ints =  2,112 B
                                                            // x_total:        19,520 B
```

Plus tile_y (9,216 B) and 256 B of ids_dst_shared = **28,992 B**.
Matches the observed 29,184 B.

**Stock's x_tile is half the size of ours: 19.5 KiB vs our 33.3 KiB.**
Why? **Stock holds only 32 K-elements (one MMQ_TILE_NE_K) of x_qs in
LDS at a time**, iterating 8× per HFQ4-group's worth of K. Our HFQ4
layout holds the entire 256-K group of unpacked nibbles in LDS once,
trading larger LDS for less HBM thrashing.

This is an **architectural choice we made when we designed the HFQ4
MMQ kernel**. It's incompatible with stock's 2 WGs/CU at nwarps=4
mmq_x=64 unless we restructure the X-tile loader to a streaming
pattern (load 32-K, compute, evict, load next 32-K).

Three options to resolve:

**Option A: Accept 1 WG/CU.** Change `__launch_bounds__(256, 2)` to
`__launch_bounds__(256, 1)`. Major perf hit — half the latency-hiding
capacity. Probably loses most of the redesign's gain.

**Option B: Restructure X-tile loader to 32-K streaming.** Match
stock's pattern. Adds 8 X-tile loads per kg (vs current 1) but each
is 8× smaller. Requires rewriting `load_hfq4_tile_dp4a` and
restructuring `mmq_body`'s kg loop. Adds significantly to Phase 2
scope (estimate +2–3 days on top of current Phase 2 budget).

**Option C: Reduce mmq_y to 64 at nwarps=4.** mmq_y=64, mmq_x=64,
nwarps=4: x_qs = 64×65×4 = 16,640 B, ½ of the 33,280 B at mmq_y=128.
Total per-WG ~22 KiB; 2 WGs = 44 KiB, fits. But mmq_y=64 vs 128
halves the per-WG output rows, doubling WG count. Could be net win
(more parallelism) or loss (smaller arithmetic intensity).

**Recommended: prototype Option B in a Phase 2a stub** alongside the
spill probe. If 32-K streaming X-load works without spilling, that's
the architecturally-correct match to stock. If it doesn't, fall back
to Option C (reduce mmq_y) before considering Option A.

**Defect [gemini, validated, critical]:** Plan §1 commits to
`__launch_bounds__(256, 2)` without verifying the LDS budget per-CU.
Hardware-impossible at the planned tile sizes. Add Phase 2a probe
**before** committing to the topology choice.

## 0b. VGPR budget is exactly at the cliff edge [gemini #2]

gfx906 VGPRs/CU = **65,536** (64K × 32-bit registers). With 256
threads/WG × 2 WGs/CU × 128 VGPRs/thread = **65,536 VGPRs/CU = 100%
exactly**.

The plan's "non-negotiable" §6: "Accept `vgpr_count ≈ 128`."

If the compiler emits **129 VGPRs**, occupancy hard-drops to 1 WG/CU.
At 2 WGs/CU we'd need ≤ 128 VGPRs/thread, which is exactly what stock
hits and exactly what we have to hit too.

**The j0 full-unroll decision (locked-in Q7) is the most likely thing
to push us over.** Stock's Q4_K vec_dot uses table-lookup dequant
(fewer intermediates) and lands at exactly 128. Our HFQ4 vec_dot has
to hit 128 too — not 129, not 130. Stock is already at the cliff;
we have no margin.

Combined with §0a: **if we hit 1 WG/CU due to either LDS *or* VGPRs,
the redesign delivers ~0 vs the current kernel.** Both constraints
must be satisfied simultaneously.

**Defect [gemini, validated, critical]:** Plan's "accept 128 VGPRs"
is not a margin — it's a ceiling. Phase 2a spill probe must verify
**both** spill count *and* VGPR count don't exceed 128. If either
condition fails, choose between Option B (restructure) or Option C
(smaller tile) before proceeding.

## 0c. `ds_read_b128` requires X_STRIDE=68, not 65 [gemini #4]

**Critical correctness/performance bug in the plan's locked-in Q8.**

`ds_read_b128` requires **16-byte aligned LDS pointers**. Otherwise the
compiler silently emits a slower fallback (or, on some hardware
configurations, traps).

X_STRIDE = 65 ints = **260 bytes**. 260 % 16 = 4. **Not 16-byte
aligned.** Adjacent rows in our x_qs LDS layout are 4 bytes off the
16-byte boundary.

Result: int4 reads from x_qs would silently fall back to two int2 or
four int1 reads, defeating the entire optimization. This is exactly
what gemini caught — a bug we wouldn't have noticed except in ISA
inspection (which the plan does require, but only after writing the
code).

**Fix:** Pad X_STRIDE from 65 to 68 ints (272 B = 17 × 16). Adds 1.5
KB to x_qs LDS budget per WG (128 rows × 12 extra B = 1,536 B).

**But this worsens the LDS overflow per §0a.** New x_qs = 128 × 68 ×
4 = 34,816 B. Total per-WG at mmq_x=64 = 34,816 + 1,024 + 9,216 = **45
KiB**. 45 × 2 = 90 KiB, even further over the 64 KiB cap.

This makes Option A/B/C from §0a more pressing — even if we found a
way to live within 43.5 KiB × 2 = 87 KiB (which we can't), the
b128-required padding pushes us further over.

**Defect [gemini, validated, critical]:** Plan's locked-in Q8 (emit
ds_read_b128 from day one) is incompatible with X_STRIDE=65. Either:
- Pad X_STRIDE to 68 and accept higher LDS pressure, *or*
- Drop ds_read_b128 from v1 (which glm-5 M2 also recommended).

If we want b128, the LDS layout in §0a's Option B (streaming 32-K
x_qs) becomes more attractive: a 32-K x_qs is 128 × 32 × 4 = 16,384
B, which is naturally 16-byte aligned at every row boundary.

## 0d. X-load memory access pattern is uncoalesced [gemini #5]

**Q4 (locked: 1 thread per row, 128 idle threads during X load) has a
worse failure mode than the idle-thread waste my §4 and glm-5 M3
called out.**

With 1 thread per row and X_STRIDE=65 ints/row, thread 0 reads from
`base + 0`, thread 1 reads from `base + 260 B`, thread 2 reads from
`base + 520 B`, etc. Adjacent threads access addresses 260 B apart.

**The HBM/L1 hardware coalesces adjacent threads accessing adjacent
addresses.** With strides of 260 B (>> 64 B cache line, >> 128 B HBM
burst), each thread issues its own transaction. We get **128
transactions per warp wavefront** instead of 1 coalesced. Bandwidth
efficiency drops by ~64×.

This is a separate issue from the idle-thread count. Even if we used
all 256 threads (option (b) from Q4: 2 threads/row), they'd still
stride by 130 B per pair — same coalescing problem.

The fix is to **transpose the load**: distribute *adjacent threads*
across *the same row's* chunks, not across rows. Stock's
`load_tiles_q4_K` does this:

```cpp
for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
    // tid maps to (chunk, row) such that adjacent tids share row, differ in chunk
}
```

Adjacent threads now access addresses 4 B apart (consecutive chunks
within the same row). Coalesces into 4–8 transactions per warp.

**Defect [gemini, validated, important]:** Q4's locked-in option (a)
is not just "wasted parallelism" — it's catastrophically uncoalesced
HBM access. Phase 2 must rewrite `load_hfq4_tile_dp4a` with a
chunk-major thread layout. Reject Q4's locked-in (a); commit to a
properly-coalesced load up-front in v1.

## 0e. N-remainder handling not specified [gemini #3]

The plan's `_full_*` entry symbols require `batch_size % mmq_x == 0`.
With mmq_x runtime-dispatched ∈ {8, 16, 24, 32, 40, 48, 56, 64} (per
glm-5 C1), what happens for batch_size = 65?

The current dispatch (line 6207 of dispatch.rs) uses
`batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X` which computes the
*number of column tiles* assuming each tile covers `mmq_x` columns
(last tile may be partial). The bounds-checked entry handles the
partial last tile.

But the plan's locked-in entry-symbol naming (`_full_add`, `_full_set`,
bounds-checked) requires the dispatcher to **pick the right symbol per
batch shape**. For batch_size = 65 and mmq_x = 64:
- One tile of 64 cols covers cols [0..64) → use `_full_*` (clean)
- One tile of 1 col covers col 64 → use bounds-checked (need to
  handle 1-col tile in the kernel, which the plan doesn't address)

Or the dispatcher could pick mmq_x=8 for the whole 65 → 9 tiles, last
is partial → bounds-checked everywhere.

**Defect [gemini, validated]:** Phase 3 dispatch logic must specify
**how the host loops over N**. Three options:
- (i) Always use bounds-checked entry → loses the `_full` perf benefit
  but handles arbitrary N cleanly.
- (ii) Use `_full` for the bulk, bounds-checked for the last partial
  tile only.
- (iii) Pick mmq_x such that batch_size % mmq_x == 0 always (chunked
  dispatch).

Stock uses (ii). Plan should specify; (ii) is the right choice.

The plan claims:

> Current measured: 147.8 tk/s prefill on Qwen 3.5 9B pp128.
> Target: stock llama.cpp at 500.5 tk/s pp128 → ~3.4× headroom.

But the **comparison checkpoint itself** (which the plan cites as
authoritative) says explicitly:

> The *real* gap on the residual kernel is 1.5×, not 4×.

There are two gaps, not one:
- **Per-call gap on the residual kernel:** 1.3 ms (hipfire) vs ~1.07 ms
  (stock's larger Q4_K calls — the comparable shape). Per-call ratio
  ≈ **1.22×**. (My doc said 1.5×; rerunning the data on the actual
  bucket of long-duration calls says 1.22.)
- **Coverage gap:** hipfire dispatches MMQ for residual + gate_up only.
  qkvza, qkv, attention all go through FP16 wave64 or other paths. Even
  if the residual+gate_up MMQ became infinitely fast, those other
  kernels still run on the slower path.

A nwarps=4 redesign of the MMQ kernel only attacks the per-call gap,
which by the data is ≤ 1.5×. **Going from 148 tk/s to 500 tk/s requires
*also* solving the coverage gap, which the plan defers to §10 carry-over
("separate commits").** The headline "3.4× headroom" is achievable; the
"3.4× from this redesign" is not.

Glm-5 H3 [validated, sharpens this]: glm-5 quantified the coverage gap
specifically — gate_up at 12 ms × 64 calls = 768 ms is **far more** of
the GEMM time than residual at 1.3 ms × 128 = 166 ms. Residual is only
~21% of GEMM time. Even fully closing the residual per-call gap saves
<100 ms; halving gate_up saves 384 ms. **Gate_up is the headline
opportunity, not residual.** This is a stronger finding than my §1
framing — the plan's structure (residual first, gate_up Phase 5) has
the priorities inverted.

If only the residual+gate_up calls run 1.22× faster per call, hipfire
at pp128 moves from 148 to roughly **170–190 tk/s** — a real win, but
not "≥400 tk/s." The plan's End-to-end Target line should be downgraded
to that range, with the larger goal split into a multi-commit roadmap.

**Defect:** The "≥400 tk/s" target in §Goal is not derivable from the
per-call ratio actually measured. Either restate as "≥180 tk/s after
this redesign + matching gate_up port", or commit to porting
qkvza/qkv/attention before declaring the gap closed.

Glm-5 L5 [validated]: the comparison vs stock isn't apples-to-apples —
stock measured Q4_K_M, hipfire uses HFQ4-G256. Format overhead (extra
4 B zp per group, heavier dequant) costs an unknown 5–10%. The "within
20% of stock" target should explicitly acknowledge that even matching
stock's per-call latency at the kernel level may not yield 500 tk/s
end-to-end.

## 2. "Stock and hipfire spill the same 144 VGPRs" is a coincidence, not a license to ship spills

The plan repeatedly leans on the observation that stock's Q4_K mmq_x=64
ELF reports `vgpr_spill_count = 144` to justify accepting spills in the
new hipfire kernel ("Accept the spills the unroll causes"). This frames
spills as a benign feature of the design.

That framing is sloppy on three counts:

a) **The spill counts are 144 by happenstance, not target.** Stock didn't
   pick mmq_x=64 to land on 144; they picked mmq_x=64 because that's
   where the *arithmetic intensity vs latency* trade-off lands best, and
   144 is what their compiler emits at that point. A hipfire kernel with
   different inner-loop body (HFQ4 unpack, zp_eff fold, our specific
   accumulator pattern) may emit 100, 200, or 400 spills — only 144
   would mean we got lucky.

b) **Hipfire's compute at MMQ_X=64 + j0 unroll *did* emit 144 spills**
   in the prior session (`docs/perf-checkpoints/2026-05-04-gfx906-mmq-spill-reduction.md`).
   So we have one data point at nwarps=2, mmq_x=64 = 144 spills. We have
   *no* data points at nwarps=4. The Risk register acknowledges
   "compiler can't fit nwarps=4 acc + intermediates" with "Probability:
   Medium" — that should drive the Risk to severity High and gate the
   work behind a build-and-ELF-check **before** committing to the rewrite.

c) **The "fall back to nwarps=2 if spills >250" mitigation is a no-op
   in practice.** The Phase 2 work to write the nwarps=4 kernel is days;
   if it spills 350 VGPRs we don't go back to the current shipped kernel
   (which is what the user already has), we go back to the design-
   board, which is the same place we'd be without doing the work. The
   real mitigation is to pre-flight a *minimal* nwarps=4 prototype (just
   the inner loop body, dummy LDS) and check spill count *first*, before
   doing the full rewrite.

Glm-5 C3 [partial validate]: glm-5 amplifies (a) — the HFQ4 dequant has
more intermediates (n0..n7, int_a, int_b) than Q4_K's table-lookup
dequant, so the compiler may spill more. **One correction to glm-5's
claim:** the n0..n7 intermediates are scoped to `load_hfq4_tile_dp4a`,
not `vec_dot_dp4a`. They live in the load function's stack frame and
go out of scope before the j0-unrolled compute starts. So they don't
amplify j0-loop spills directly. They *do* affect the X-load phase's
own register pressure, but that's a separate measurement. Glm-5's
spill threshold of 200 (vs my 250) is more conservative; the right
gate is whatever signals "we've left the regime stock is in."

**Defect:** The plan's spill argument is "stock has 144, so 144 is fine
for us." That's post-hoc reasoning. Strengthen the Risk register
mitigation: **Phase 2a (new): minimal nwarps=4 ISA probe (1 day) — write
the inner-loop dp4a body only, in a stub kernel, build, dump VGPR count
and spill at both `#pragma unroll` and `#pragma unroll 1` on j0. Hard
gate on `vgpr_spill_count ≤ 200` before proceeding to Phase 2b (full
rewrite).**

## 3. LDS budget calculation is wrong at mmq_x=128 — and mmq_x=128 doesn't exist on gfx906 anyway

§3 says:

> For nwarps=4 mmq_x=128:
> - `tile_y`: 128 × 36 = 18,432 B
> - Total ~52.7 KiB, still under 64 KiB

Glm-5 C2 [validated, critical]: **Stock's `get_mmq_x_max_device()` returns
64 on gfx906** (`mmq.cuh:124-125`):
```cpp
#if defined(GGML_USE_HIP)
    return 64;
```
Stock never dispatches mmq_x=128 on gfx906. The Q1 question I asked
("include 128 from day one?") and the user's locked-in "yes" answer
are both based on a false premise. **The answer is no, mmq_x ≤ 64.**
The 52.7 KiB LDS estimate at mmq_x=128 is moot.

This is a substantive plan correction, not a nit. Glm-5 caught it; I
missed it. The plan needs to be edited to:
- Drop mmq_x=128 from the candidate set
- Cite mmq.cuh:124 as the source of the cap
- Reduce the symbol count (was 5×3=15, now 4×3=12 if we keep
  {8,16,32,64} — and per glm-5 C1, more like 8×3=24 if we go
  greedy-by-8)

Adding it up for the (correct) max mmq_x=64: 33,280 (x_qs) + 1,024
(x_dm) + 9,216 (tile_y at mmq_x=64) = 43,520 B = **42.5 KiB**, well
under 64 KiB cap. Comfortable.

The remaining LDS concerns:

**3a. Y-twice barrier collapse interaction (still valid):**
The existing `Y_STRIDE=36` includes 4 ints of inlined ds plus 32 ints
of qs — for **one Q8_1 block of 128 K-elements**. Our kg loop iterates
256 K-elements per kg = 2 Q8_1 blocks. The current code re-loads
tile_y for each Q8_1 half (Y-twice pattern); the 36-int tile holds one
block at a time.

If we ever decide to try Y-twice barrier collapse again (the
follow-up 3 from the junroll log) at the new mmq_x sizes, the budget
becomes `2*MMQ_X*Y_STRIDE`. At mmq_x=64: 2×64×36 = 4,608 ints =
18,432 B (same as my mmq_x=128 single-half calc). Total LDS at
mmq_x=64 with Y-doubled tile_y: 33,280 + 1,024 + 18,432 = 52,736 B =
51.5 KiB. Still under 64 KiB cap. So Y-twice collapse stays available
even at the larger mmq_x — slightly different from my v1 conclusion.

**Add to plan §3: "tile_y is sized for one Q8_1 block half (=
MMQ_TILE_NE_K=32 K-elements). Y-twice barrier collapse experiment
would double tile_y to 2*MMQ_X*Y_STRIDE; at mmq_x=64 that pushes total
LDS to 51.5 KiB. Still feasible. Re-evaluate only if the redesigned
kernel shows barrier latency as a bottleneck."**

**3b. LDS bank-conflict padding (my finding, glm-5 silent):**
§3 doesn't mention bank-conflict padding for the new tile_y access
pattern. The current kernel has `X_STRIDE = 65` (= 64 + 1 padding) for
0% bank conflict at mmq_x=8. At mmq_x ∈ {16, 32, 64}, the wider
tile_y access pattern from `vec_dot_dp4a`'s nwarps=4 j-stride could
hit bank conflicts. With wave64 hitting 32 banks (inherent 2× lane-to-
bank ratio), the access pattern needs care.

**Defect:** Add a sub-step to compute, per (nwarps, mmq_x), how the
j-stride aligns with the 32-bank LDS at wave64 = 2× (since 64 lanes
hit 32 banks). Pre-emptively pad if needed. Don't rely on "verify with
rocprof later."

## 4. The nwarps=4 X-load thread distribution is a perf trap, not just a "v1 simplicity" trade

Q4 (locked: option (a), 1 thread per row, 128 idle threads during X
load): the plan rationalizes that "load phase is ~250 chunks × 4 B per
row × 128 rows = 128 KB of HBM read per kg, dwarfed by the 4 ms of
compute. Idle threads during load are cheap."

That math is wrong on two axes:

a) Per the existing `load_hfq4_tile_dp4a`, **each thread loads 32 chunks
   per row** (one loop iteration loads one uint = 4 bytes = 8 nibbles =
   2 ints into x_qs). 128 threads × 32 iters/thread = 4096 chunks loaded
   per kg. But the comment in the kernel source itself says "lots of
   dependent iterations per thread" — these load are *serialized within
   each thread*, not parallel. So the load phase is 32 *serial* HBM
   round-trips per active thread, not the rosy "128 KB / HBM bandwidth"
   the plan quotes.

b) **Half-WG idle during load means half the WG's HBM-issue slots are
   wasted.** On gfx906, the memory system has fixed issue-rate per WG
   (TC/SQ contention). If 128 threads share the issue rate, doubling
   the WG to 256 threads with 128 idle doesn't slow each thread down —
   but it doesn't speed it up either. So we pay *2× the WG occupancy
   cost* for the same load throughput. With `__launch_bounds__(256, 2)`
   we're now requesting 2 WGs/CU × 256 threads = 512 threads/CU, vs
   stock's same setup. Fine in steady state. But during the load phase,
   half of those 512 threads are doing nothing while 128 plough through
   32 serial loads each.

c) **Stock's load_tiles_q4_K does NOT use 1 thread per row.** At
   `mmq.cuh:1129` and elsewhere, stock uses `for (int i0 = 0; i0 <
   mmq_y; i0 += nwarps * rows_per_warp)` — they parallelize across all
   256 threads. Our v1 will issue HBM loads at half the parallelism,
   then sit idle on the second half of the WG.

Glm-5 M3 [validated, sharpens]: glm-5 quantifies — 549 KB HBM read for
X across 16 kg iters at K=4096, ~69 µs total at MI50's HBM bandwidth.
Currently 5% of per-call time. At target per-call time of ~0.85 ms,
load grows to ~8% of wallclock. Small but real.

The plan's "revisit (b) only if rocprof shows the load phase is a
hotspot" is reasonable as a triage *order*, but the framing
underestimates what we're conceding. The right way to phrase Q4 is:

> v1 ships with a known suboptimal load-phase, accepting ~5–8% longer
> load-phase wallclock as a debt to be paid in v2 if profiling shows
> load is non-trivial. Set a TODO: measure VALUBusy during the first
> barrier after X load. If load contribution ≥15% of per-call time,
> prioritize (b) for v2.

**Defect:** Q4's recommendation is fine as triage order, but the plan
should commit to *measuring* load contribution explicitly, not "verify
later when convenient."

## 5. The compile-time / binary-size estimate is wrong (and will get worse with glm-5 C1)

§Q9 promises 12 entry symbols (3 add modes × 4 mmq_x). The user's
locked-in Q1 added mmq_x=128 → 15. But per glm-5 C2, mmq_x=128 is
moot, so we're back to 12. Glm-5 C1 then says we need {8, 16, 24, 32,
40, 48, 56, 64} (8 values, multiples of 8 per stock's greedy dispatch)
→ **24 entry symbols**.

Empirically:
- Current single-entry kernel (3 entry symbols, MMQ_X=8): hsaco = 56 KB.
- Stock's per-type-per-mmq_x kernel ELFs in the fatbin: largest are
  ~1 MB each. With 8 mmq_x values × ~3 hsaco variants = ~24 MB of GPU
  code per kernel — and we'd be doing this for the residual + gate_up
  + qkvza + qkv kernel paths. Per-kernel hsaco at mmq_x=64 hits the
  same compile-time cost stock pays (their fatbin extracted to 42 MB).
- Hipfire's `compile-kernels.sh` builds in ~30 s today; with 8×
  template instances and a per-mmq_x variant, expect closer to
  **2–4 minutes** for just this kernel, not "10 s longer."

That's not catastrophic, but the Risk severity estimate is too low. It
also affects iteration speed during the rewrite: every spill-check
build cycle gets slower. Worth budgeting for.

**Defect:** Risk register understates compile-time cost by ~10–30×.
Severity should be Medium, with concrete mitigation: **compile only
mmq_x ∈ {64} during the inner-loop dev cycle (single template instance);
add the other mmq_x values only after the dominant case lands.**

## 6. The Q4_K vs HFQ4 K-iter cadence mismatch is presented as "verify in correctness test" but it's actually a design constraint

Plan §Risk register row 2:

> HFQ4 group size (256 K) doesn't map cleanly to stock's MMQ_TILE_NE_K=32
> iter cadence | Severity: Medium | Probability: Medium | Mitigation:
> Keep our 8-half-iters-per-group structure; verify K-indexing in
> correctness test.

This handwaves the actual issue. Concretely:

- Stock's `MMQ_ITER_K = 256`. Their kg-loop iterates one full `qk` per
  iter for K-quants (Q4_K has `qk = 256`), and 8 sub-32-K iters for
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 (which have `qk = 32`).
- Our HFQ4-G256 has 256-K groups, mapping to Q4_K's iter cadence.

So the *cadence* is fine for K-quants, but the *internal sub-block
structure* differs:
- Q4_K: 256 K-elements with **8 sub-scales** (one per 32-K sub-block),
  scales are i6 packed into a 12-byte block-side mask.
- HFQ4-G256: 256 K-elements with **one f32 scale + one f32 zp**, no
  sub-scales.

This means our `vec_dot_dp4a` *cannot* mirror stock's
`vec_dot_q4_K_q8_1_dp4a` directly — Q4_K's vec_dot reads one of 8
sub-scales per (k01) iteration; ours uses a single (scale, zp_eff) for
all 8 sub-iters. **The right reference is `vec_dot_q4_0_q8_1_dp4a`
(line 460), which has one (scale, zero) per qk-block — matching our
HFQ4 structure.**

But Q4_0's `qk = 32`, so its tile_x layout is `mmq_y * MMQ_TILE_NE_K +
mmq_y` ints for x_qs (= 4224 ints at mmq_y=128). Our x_qs is 8× larger
(33,280 B = 8320 ints) because we hold a full 256-K group of unpacked
nibbles. **Adapting Q4_0's vec_dot to our 256-K x_qs requires 8 inner
k01 iterations per kg, not 1.**

The plan acknowledges this in §3 (LDS section, "8× MMQ_TILE_NE_K") but
doesn't connect it to the vec_dot redesign. The Phase 2 task list says
"Update `vec_dot_dp4a` to nwarps=4 indexing, full j0 unroll, int4 reads"
— that doesn't capture the actual restructuring needed.

Concrete: in our kernel, `vec_dot_dp4a` has an inner k01 loop:
```cpp
for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += vdr) {  // 4 iters, vdr=8
```
covering 32 K-elements per call, called twice (k_offset=0 and 32) per
kg = 64 K-elements total. But our X tile has 256 K-elements. **The
remaining 192 K-elements per kg are processed by inheriting the kg
loop's outer iteration**, which the plan handles correctly today.

In the new nwarps=4 kernel, this is fine *as long as we keep our 8-
half-iters-per-group structure*. But if we naively port stock's
mul_mat_q_process_tile that assumes Q4_0-style `qk = 32` cadence, our
inner loop's K-bounds break.

**Defect:** §Risk register row 2's "verify K-indexing in correctness
test" is post-hoc validation, not pre-coding design. Add a Phase 2
sub-task: **before writing vec_dot_dp4a in the new shape, write a
1-page document that maps {kg, k_offset, k01, kx, ky} indices between
hipfire's HFQ4-G256 layout and stock's Q4_0 vec_dot reference.** This
is the most likely place to introduce a silent correctness bug.

Glm-5 H1 [validated]: glm-5's microbench recommendation is good — time
`load_hfq4_tile_dp4a` alone before commitment. If VALU is saturated by
the dequant chain (7 shifts + 7 masks + packs per 8 elements), the
nwarps=4 redesign won't help as much as predicted. Fold this into the
Phase 2a stub work: **measure dequant-only VALUBusy at nwarps=4. If
≥40%, consider fusing the nibble unpack into the sdot4 source operand
to avoid materializing int_a/int_b.**

## 7. The "12 entry symbols" promise vs how many actually need to compile

§Q9 plus the locked-in Q1 = 5 mmq_x values × 3 add modes was originally
**15 entry symbols**. Per glm-5 C2 we drop mmq_x=128 → 12. Per glm-5 C1
if we go greedy-by-8 we need {8,16,24,32,40,48,56,64} → **24 entry
symbols** (3 modes × 8 mmq_x). But:

- Each entry symbol is a separately-named `extern "C" __global__`
  function, but the *underlying kernel* is one templated body. Stock
  emits 257 mul_mat_q symbols per ELF (per the readelf dump earlier in
  the session) because they're templates instantiated by the build
  system.
- Hipfire's compile-kernels.sh produces one .hsaco per .hip file. We'd
  need either (a) 24 separate .hip files, each with one
  `extern "C" __global__` wrapper around the template, or (b) a single
  .hip file with 24 `extern "C"` wrappers and let dispatch pick by name.

Option (b) is what stock does (single `mmq.cuh` with template
instantiations). Option (a) is what hipfire's compile-kernels.sh
expects today. **The plan needs to choose** — and it doesn't.

Also: hipfire's `dispatch.rs:6207` currently picks between `_full_add`
and the bounds-checked entry by `m % 128 == 0 && batch_size % 8 == 0`.
With mmq_x parameterized, the bounds-check predicate becomes `m % 128
== 0 && batch_size % mmq_x == 0`, varying per dispatch. The dispatch
helper needs to know mmq_x to pick the predicate. Currently it doesn't.

**Defect:** §Q9 is settled-by-fiat ("keep the 3-symbol pattern") but
doesn't address the real question: one .hip file with 24 wrappers, or
24 .hip files, or change compile-kernels.sh to consume templates? The
answer affects how big the rewrite is.

## 8. Phase time estimates are optimistic by ~2–3×

Plan claims:
- Phase 2 (kernel rewrite): 1–2 days
- Phase 3 (dispatch update): 0.5 day
- Phase 4 (validation): 0.5 day
- **Total: 2–3 days**

That's a clean-slate budget assuming everything works the first time.
Reality on this codebase, based on this session's experience:

- This session, three "trivial" experiments (MMQ_X sweep, ds_read_b128
  conversion, Y-twice barrier collapse) each took ~30 minutes of code
  + 30 minutes of stale-cache debugging + 1 hour of bench/rocprof =
  ~2 hours each. All three regressed.
- The j0 un-unroll change (1 line!) took ~3 hours including rebuild,
  hot-cache invalidation gotcha, ELF check, bench, coherence gate.
- Each rebuild after a kernel edit is ~10s for the .hip but ~30s+ for
  the dependent .rs (hipfire embeds source via `include_str!`, so any
  .hip change triggers a partial Rust rebuild of every binary that
  uses `GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_SRC`).
- Coherence gate is 5+ minutes per run.
- 24 mmq_x template instances (per glm-5 C1) × 3 add modes = 24 binaries
  to compile in parallel at every iteration (or 12 if we accept the
  utilization loss of the {8,16,32,64} subset).

Glm-5 M5 [validated]: glm-5 estimates 3–4 days; my estimate is 5–8
days. Both larger than the plan's 2–3. Glm-5 M4 [validated]: also
notes the test infrastructure (test_gfx906_mmq_correctness,
test_gfx906_mmq_realdata, dispatch.rs hardcoded MMQ_X=8) needs
updating in the same phase. Plan's 0.5-day Phase 3 estimate doesn't
account for this.

Realistic estimate: **Phase 2 = 3–5 days, Phase 3 = 1 day, Phase 4 =
1–2 days. Total: 5–8 days, not 2–3.** And that's if no design pivots
are needed — which is unlikely given the open questions, several of
which (LDS layout, Q4_K vs Q4_0 vec_dot reference, dispatch formula)
have non-trivial implications.

**Defect:** Time budgets should be doubled. State the realistic budget
explicitly so calendar planning doesn't underdeliver.

## 9. The dispatch.rs `next_pow2(batch_size).clamp(8, 64)` formula is wrong (glm-5 C1 strengthens this)

§Phase 3 says:

> Pick mmq_x at runtime based on batch_size:
> `mmq_x = next_pow2(batch_size).clamp(8, 64)`

Two issues my v1 raised:

a) The clamp upper bound is 64, but Q1 was locked-in to include
   mmq_x=128. (Per glm-5 C2 this is moot — gfx906's max is 64 — so the
   formula's clamp(8, 64) is correct for max, but wrong for the
   stepping.)

b) `next_pow2(batch_size)` may *over-allocate*. If batch_size = 96, we
   get next_pow2 = 128, dispatching mmq_x=128 with 32 lanes idle in
   the j-direction of every WG.

Glm-5 C1 [validated, sharpens this critically]: glm-5 produced a
concrete table showing stock's actual dispatch (`mmq.cuh:4069-4082`)
uses a **greedy step-by-granularity loop** (granularity=8 on gfx906),
picking the largest `mmq_x ≤ ncols_max` that minimizes the column tile
count. This produces:

| ncols_max | Stock picks | next_pow2 | Tile utilization |
|---|---|---|---|
| 48 | **24** | 32 | 100% vs 67% |
| 40 | **40** | 64 | 100% vs 63% |
| 56 | **56** | 64 | 100% vs 88% |
| 96 | **96** | 128 (clamped to 64) | varies vs varies |

For N=48 (the alpha/beta matrices in qkvza), stock picks mmq_x=24,
which isn't even in our locked-in Q1 set {8, 16, 32, 64, 128}. Dropping
to {8, 16, 32, 64} doesn't help — N=48 still maps to mmq_x=32 with
33% padding waste. **Glm-5's recommendation: compile multiples of 8
from 8 to 64, total 8 mmq_x values.**

This is a substantive change to the plan — the symbol count goes from
12 to 24 (or 18 if we drop 40 and 56 since they're rarely the optimal
pick in a Qwen 3.5 9B prefill). Each variant adds binary size and
build time per glm-5 §M5 / §5 above.

**Defect:** Phase 3's dispatch formula is too simple. Replace with
glm-5's recommendation:
- Compile mmq_x ∈ {8, 16, 24, 32, 40, 48, 56, 64} (or a measured
  subset based on which values our actual batch sizes hit)
- Dispatch via greedy loop matching stock's mmq.cuh:4069-4082

## 10. Gate_up should be co-designed with residual, not deferred (glm-5 H3)

This is glm-5's most important finding and outranks anything in my v1.

§Q10 / §Phase 5 commit 2: gate_up port is "Commit 2: re-port gate_up
to use the new dispatch."

Glm-5 H3 [validated, critical]: per the comparison doc:
- Residual MMQ: 1.3 ms/call vs stock 0.85 ms → gap = 0.45 ms × 128
  calls = ~58 ms savings if we close it.
- gate_up FP16 wave64: 12 ms/call vs what stock's MMQ achieves on
  the same shape.
- Total hipfire GEMM time: 944 ms (256 calls); residual contributes
  166 ms (~18%); gate_up 768 ms (~81%).

**Even halving gate_up saves 384 ms — 6.7× more than the residual
fix.** Closing the residual gap to stock's ratio saves only 58 ms.

Furthermore: dispatch.rs:6304-6383 for `gemm_hfq4g256_mmq_set_gfx906`
shares the same MMQ_X, MMQ_Y, and LDS constants as the residual
dispatcher. **If the residual redesign changes these constants (it does),
the set-mode dispatcher breaks silently unless updated simultaneously.**
Shipping residual without the gate_up dispatch update would land us
in a broken intermediate state where gate_up MMQ is wired to the new
kernel binary with old constants.

**Defect:** §Phase 5 commit 2 is misordered. **Move gate_up dispatch
update to Phase 3, alongside the residual dispatch update.** The kernel
itself is shared (same `mmq_body` with `add=0`), so the kernel work is
done once. The dispatch helpers (`gemm_hfq4g256_residual_mmq_gfx906`
and `gemm_hfq4g256_mmq_set_gfx906`) need parallel updates.

This also means **the Phase 4 bench should measure both residual+gate_up
together**, since gate_up is where the real win lives.

## 11. The carry-over plan to qkvza glosses over the structural mismatch

§Q10 / §Phase 5 commit 3:

> qkvza: re-attempt qkvza port now that small-mmq_x is supported.

But the actual qkvza problem identified earlier in this session wasn't
"small mmq_x not supported" — it was **the FP16 wave64 kernel fuses 4
output matrices into 1 launch, and our MMQ replacement uses 4 separate
launches.** That's an architectural mismatch that won't be fixed by
adding mmq_x=8.

Specifically:
- `gemm_qkvza_hfq4g256_fp16_wave64` routes rows: `if gid < qkv_m: A=A_qkv`
  etc. Single launch covers qkv (M=4096) + z (M=4096) + beta (M=16) +
  alpha (M=16).
- Our MMQ approach calls `gemm_hfq4g256_mmq_set_gfx906` 4 times. Even
  with mmq_x adapted per call, the 2 small (M=16) calls dispatch
  ~1 row-tile each, finishing in microseconds but paying full launch
  overhead.

The redesigned MMQ kernel doesn't change this. To fix qkvza properly we
need either:
- A **fused 4-output MMQ kernel** that dispatches once and routes rows
  internally (same pattern as the FP16 wave64 version), OR
- Accept that qkvza stays on FP16 wave64 and only ship the residual +
  gate_up wins.

**Defect:** §Phase 5 commit 3 is misleading. Either rewrite as "qkvza
port deferred — requires separate fused-4-output design" or commit to
the fused-output kernel as a Phase 6.

## 12. Phase 0 nwarps contradiction is unresolved (glm-5 H4)

Glm-5 H4 [validated, my v1 missed this]: the original
`plans/gfx906_mmq_plan.md` Phase 0 deliverable explicitly stated:

> Switch to llama.cpp-gfx906 topology (`nwarps=2`, block `(64, 2, 1)`).

That was approved as "GO" and shipped. The redesign reverses
this — adopting nwarps=4 based on the runtime observation that
**stock's** kernel uses 4 warps, not the unverbraucht fork. The
unverbraucht fork is stale (per our session conversation about its
divergence) and its `gfx906-config.h` may not reflect what stock or
its own runtime actually does.

The redesign plan doesn't acknowledge this contradiction. Future
readers reading both plans will be confused about which is current.

**Defect:** Add a §"Relationship to Phase 0 plan" section to the
redesign plan stating:
- Phase 0's nwarps=2 recommendation came from
  llama.cpp-gfx906/unverbraucht's config file, which is stale.
- Stock llama.cpp's runtime uses nwarps=4 on gfx906 (per
  `2026-05-04-llamacpp-stock-comparison.md` and the ELF metadata
  showing `max_flat_workgroup_size=256`).
- Phase 0's other conclusions (wave64 indexing, LDS bank conflict
  padding observations) remain valid.

## 13. mmq_x=128 is locked-in but doesn't exist on gfx906 (glm-5 C2)

Already covered in §3. Worth highlighting separately because **the
user's review locked Q1 to "include 128"** based on my framing, which
turns out to be wrong. Stock caps at 64 on gfx906 per `mmq.cuh:124`.
The user should be informed of this finding so they can re-decide Q1.

**Defect:** Plan needs an explicit revisit of Q1 with the correct
information.

## 14. j0 full-unroll reversal lacks empirical analysis (glm-5 M1)

Glm-5 M1 [partial validate]: the user's locked-in Q7 ("full unroll on
j0") reverses commit `17c05f3` which delivered +16.2% by going to
`#pragma unroll 1`. The plan justifies the reversal with "stock does
it" but doesn't account for the spill-cost difference between Q4_K's
table-lookup dequant and HFQ4's shift-mask dequant.

**Where I'd push back on glm-5:** the live-range pressure that drove
the j0 un-unroll win in commit 17c05f3 was *per-(i,j)-site live values
inside vec_dot_dp4a*, not load-side intermediates. The HFQ4 vs Q4_K
load-side complexity (n0..n7 etc.) is irrelevant to vec_dot_dp4a's
compute body's spill pressure. The relevant comparison is:
- Stock's vec_dot_q4_0 body at j0 full-unroll (mmq_x=64, nwarps=4):
  144 spills (measured in ELF).
- Hipfire's vec_dot_dp4a body at j0 full-unroll (mmq_x=64, nwarps=4):
  unknown — that's what Phase 2a probes.

So glm-5's recommendation (test both unroll modes in the stub) is
right, but the *reasoning* (HFQ4 dequant heavier than Q4_K) is
slightly off-target. The test is still worth running because we don't
have data; the rationale is "we don't know, measure both" not "HFQ4
dequant amplifies vec_dot spills."

**Defect:** Phase 2a stub should test j0 full-unroll vs j0 unroll-1
side-by-side, ship the lower-spill variant. (Same as my §2 fix, just
adding the unroll-mode dimension.)

## 15. ds_read_b128 prior negative result underweighted (glm-5 M2)

Glm-5 M2 [validated]: we regressed 2.7% on this in the small-tile
kernel earlier this session. The plan argues the larger nwarps=4
mmq_x=64 body changes the bottleneck, but this is untested.

**Defect:** Implement ds_read_b128 as a separate commit, not "from day
one." Ship the redesign without it first, benchmark, then add and
benchmark again. The 2.7% regression on the small kernel doesn't prove
b128 hurts on the larger kernel; but it doesn't prove it helps either.
Layered commits are safer.

## 16. Smaller technical issues

- **Plan doesn't specify the bench harness for validation.** Phase 4
  says "bench at pp32, pp64, pp128, pp256, pp512." Hipfire's
  `bench_qwen35_mq4` only takes `--prefill N`, single-batch. To match
  stock's `llama-bench` we need either a multi-batch hipfire harness
  or an external comparison via the daemon. Add to Phase 4: "If
  hipfire's bench harness only does pp128, augment with a script that
  varies --prefill or accept that we measure pp128 only."

- **Coherence gate timing.** Per CLAUDE.md, every kernel commit must
  pass the coherence gate. With 12–24 entry symbols on first try,
  there's a real risk one of them dispatches differently for some
  shape and silently produces gibberish. Glm-5 H2 partially validated:
  coherence gate catches catastrophic bugs but not subtle per-mmq_x
  numerical regressions. **Add: coherence gate after each mmq_x
  variant lands, plus per-mmq_x synthetic NRMSE tests.**

- **No `dispatch.rs` line numbers cited.** This is a 13K-line file;
  the plan's "Update `gemm_hfq4g256_residual_mmq_gfx906` and
  `_set_gfx906`" should cite line ranges so the rewriter doesn't
  hunt-and-peck through git diffs. (Glm-5 M4 cited 6202-6297 and
  6304-6383, useful starting points.)

- **The "1.22× per-call gap" in §1 assumes equivalent shapes.** Stock's
  Q4_K and our HFQ4-G256 are both ~4.5 bpw, but Q4_K has sub-scales
  and HFQ4 doesn't, which changes per-byte arithmetic intensity. Glm-5
  L5 makes the same point — measure stock-with-q4_0 to remove format
  variance from the comparison.

- **gfx906 has 60 CUs (MI50) or 64 CUs (MI60).** Plan doesn't say
  which the 32 GiB MI50 in `mi50_benchmark.txt` is — the device id
  `0x66a1` in the rocm-smi output is the MI50 (60 CU). Total threads
  in flight: 60 CUs × 4 SIMDs × 256 threads/WG × 2 WGs/CU = **122,880
  active threads**. If the kernel dispatch only fills, say, 30 WGs
  (because mmq_x=64 + small batch), we under-utilize the GPU. Stock
  uses stream-K elsewhere for this reason; on gfx906 they use
  conventional tiling instead. **Add: at small batch_size, smaller
  mmq_x is preferred to keep WG count high (e.g., batch=32 should pick
  mmq_x=8 → 4 column-tiles → 4× more WGs to fill the GPU).** Note:
  this aligns with glm-5 C1 — stock's greedy dispatch *already* picks
  smaller mmq_x for smaller batches.

- **Q8_1 tile loader edge case at mmq_x=8 with nwarps=4 (glm-5 L1)**
  [validated]: 256 threads available, only 32 have data. Need
  `if (tid < 32)` guard equivalent to current code. Add to Phase 2
  task list.

- **Risk register additions (glm-5 L3) [validated]:**
  - Symbol name typo / link-time assertion that all 12–24 expected
    symbols exist
  - ROCm version sensitivity (specify minimum: ROCm 6.4.3 since that's
    what we measured on)
  - Occupancy regression at scratch — verify via rocprof, not just
    ELF (e.g., 144 spills × 4 B × 64 lanes × 256 threads × 2 WGs/CU =
    9.4 MB scratch per CU, well within MI50's 16 MB scratch per CU).
  - Correctness regression from template-instance edge cases at
    mmq_x values not exercised by current bounds-checking code.

- **Fate of current kernel (glm-5 L4)** [validated]: compile-kernels.sh
  selects per-arch variants; the plan needs to be explicit about
  whether the current `gemm_hfq4g256_residual_mmq_gfx906.hip` is
  replaced (overwritten), kept as fallback (behind config flag), or
  deleted (file removed). Recommend: replace in place. The current
  kernel is not preserved as a fallback because the dispatch contract
  changes.

## 17. Glm-5 findings I rejected

- **Glm-5 H2 (coherence gate doesn't test prefill):** partially
  rejected. The coherence gate runs prompts through the daemon, which
  exercises both prefill and decode. The MMQ kernel runs in prefill,
  so the gate *does* catch catastrophic prefill bugs. Glm-5's
  observation that the gate doesn't exhaustively test every mmq_x
  variant is correct and worth adding to validation, but the framing
  "tests model loading + decode, not prefill" is wrong.

- **Glm-5 C3 specific claim that HFQ4 dequant intermediates spill the
  vec_dot body:** rejected. The n0..n7 / int_a / int_b intermediates
  live in `load_hfq4_tile_dp4a` (the X-load function), not in
  `vec_dot_dp4a` (the compute function). They affect the X-load
  phase's register pressure but not the j0-unrolled compute body.
  Glm-5's higher-order point (HFQ4 is heavier than Q4_K, may spill
  differently) is valid; the specific spill-amplification mechanism
  is wrong. The Phase 2a stub probe still resolves the question
  empirically.

- **Glm-5 L2 (MMQ_TILE_Y_K terminology):** rejected. Both glm-5 and I
  used "MMQ_TILE_Y_K" loosely. Stock has `MMQ_TILE_Y_K` defined as
  the K-dim of the Y tile (32 elements per sub-block), and the per-col
  byte stride is 36 ints (which I called Y_STRIDE in our code, glm-5
  identified as the same). My plan's wording "Adopt stock's
  MMQ_TILE_Y_K = 36 ints" is sloppy but the underlying intent
  (per-col stride = 36 ints) is right. The fix is to use Y_STRIDE
  consistently.

## 18. What I'd accept the plan with (revised v3)

If we still want to proceed with this redesign (we should — the
direction is right), I'd add these gates/fixes to the plan, in priority
order. **The v3 critical fixes are gemini's** — they're hardware-
physical constraints, not soft optimization questions.

**Critical-blocker (cannot ship without these — gemini):**

1. **Resolve LDS-vs-occupancy contradiction.** [gemini #1] Phase 2a
   stub must measure actual per-WG LDS *and* verify the chosen
   `__launch_bounds__(256, N)` is achievable. Three options to pick
   from: (A) accept 1 WG/CU, (B) restructure X-tile loader to 32-K
   streaming, (C) reduce mmq_y to 64. Recommended: prototype B in
   stub.

2. **Verify VGPR ≤ 128 in stub.** [gemini #2] Hard gate. If 129+,
   topology is broken. Combined with §1 above.

3. **Pad X_STRIDE to 68 (16-byte alignment) for `ds_read_b128`.**
   [gemini #4] Or drop b128 from v1 (matches glm-5 M2). The plan's
   locked-in Q8 is incompatible with the existing X_STRIDE=65.

4. **Coalesce X-tile load.** [gemini #5] Reject Q4's locked-in (a).
   Use chunk-major thread layout (stock's pattern). v1.

5. **Specify N-remainder dispatch.** [gemini #3] Recommend stock's
   pattern: `_full` for bulk, bounds-checked for last partial tile.

**Critical (must fix before coding — glm-5 / my v1):**

6. **Drop mmq_x=128 from Q1.** Per glm-5 C2: stock caps at 64 on
   gfx906. Re-confirm with user — the user's locked-in answer to Q1
   was based on incorrect framing.

7. **Replace `next_pow2` dispatch with greedy step-8.** Per glm-5 C1.
   Compile mmq_x ∈ {8, 16, 24, 32, 40, 48, 56, 64} (8 values, 24
   symbols), or measured subset.

8. **Move gate_up dispatch update to Phase 3.** Per glm-5 H3. Residual
   and set-mode dispatchers share constants — must update together.
   **This is the single biggest end-to-end-perf finding.** Gate_up
   accounts for 81% of GEMM time.

9. **Phase 2a (new): nwarps=4 stub probe.** Test both unroll modes,
   measure VGPR count, spill count, *and* required LDS. Hard gate
   on all three. ≤ 1 day. Replaces "build then check" with "probe
   then commit."

**Strongly recommended:**

10. **Doc the K-iter mapping** (HFQ4 256-K vs stock's Q4_0 32-K)
    before Phase 2b. ½ day.

11. **Pick: one .hip with N wrappers, or N .hip files** for Q9.

12. **Pre-empt LDS bank conflicts at wider tile_y** with explicit
    padding analysis.

13. **Restate Phase 5 commit 3 (qkvza) as out-of-scope** for this
    redesign; track separately as "fused-4-output MMQ kernel."

14. **Realistic Phase budget: 6–10 days, not 2–3.** (Bumped up
    from v1's 5–8 due to the X-tile restructure now likely required
    per gemini #1.)

15. **Add Phase 0 reconciliation section.** Per glm-5 H4: explicitly
    note nwarps=4 supersedes Phase 0's nwarps=2 recommendation.

**Helpful:**

16. **ds_read_b128 in a separate commit, not v1.** Per glm-5 M2 and
    the X_STRIDE alignment fight in gemini #4. v1 ships without b128;
    v2 adds it once we've decided on X_STRIDE padding.

17. **Update test harnesses** (test_gfx906_mmq_correctness etc.) in
    Phase 3, not later. Per glm-5 M4.

18. **Downgrade headline target to "≥185 tk/s after this commit +
    matching gate_up port."** Reserve "≥400 tk/s" as a multi-commit
    target after qkvza/qkv ports also land.

Without these, the plan would ship and hit at least 4–5 of the open
defects in the first week. With these (especially gemini's hardware-
physical fixes #1–#5), this is a credible multi-week project.

The two single most important fixes:

- **#1 (LDS vs occupancy):** without resolving this, the redesign
  delivers ~0 over the current kernel. Hardware-physical.
- **#8 (gate_up co-design):** without this, even a successful
  residual redesign moves prefill <5% end-to-end.

A revised plan that addresses both becomes a credible 1.5–2 week
project for a 1.8–2.5× prefill improvement.
