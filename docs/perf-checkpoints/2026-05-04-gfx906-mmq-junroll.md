# 2026-05-04 gfx906 MMQ j0 un-unroll: spills eliminated, MMQ overtakes FP16

Hardware: AMD MI50 (gfx906), ROCm 6.4.3.
Baseline: commit `39b1eb7` (MMQ_X=8 spill reduction).
Prior session prefill: 125.2 tk/s on Qwen 3.5 9B pp128 (89% of FP16
wave64 baseline at 140.7 tk/s).

Executes step 2 of `plans/gfx906_mmq_l2.md` v3 — picked lever from
the attribution checkpoint
(`docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`).

## TL;DR

One-line edit: `#pragma unroll` → `#pragma unroll 1` on the `j0` loop
in `vec_dot_dp4a` (kernel line 283). Effect:

| Metric | Pre (MMQ_X=8 + full unroll) | Post (j0 un-unroll) | Δ |
|---|---|---|---|
| **Prefill (Qwen 3.5 9B pp128)** | 125.2 tk/s | **145.5 tk/s** | **+16.2%** |
| **vs FP16 wave64 baseline (141.3)** | 0.89× | **1.03×** | first time MMQ > FP16 |
| arch_vgpr | 128 | 60 | −53% |
| vgpr_spill_count | 144 | **0** | eliminated |
| private_segment_fixed_size | 564 B | **0** | eliminated |
| WriteSize per call | 517 KB | 949 B | −99.8% |
| VMEM_WR per call | 2.07 M | 3.5 K | −99.8% |
| VMEM_RD per call | 2.62 M | 191 K | −92.7% |
| VALUBusy | 8.7% | 15.6% | +79% relative |
| MemUnitBusy | 24.0% | 5.6% | bandwidth freed |
| MemUnitStalled | 2.9% | 0.04% | gone |
| Synthetic NRMSE (4096×4096×32) | 0.12% | 0.12% | identical |
| Synthetic NRMSE (4096×12288×128) | 0.04% | 0.04% | identical |
| ELF size (mmq_gfx906.hsaco) | 73 KB | 56 KB | −23% |
| Decode tk/s | 50.5 | 52.8 | +5% (within noise) |

5 bench runs at the new config: 145.4 / 145.5 / 145.6 / 145.5 / 145.4
tk/s. Stddev <0.1.

## The change

`kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip:283`:

```diff
-        #pragma unroll
+        // j0 un-unroll: serializes 4 j0 iterations to cut live-range
+        // pressure 4×. At MMQ_X=8, fully-unrolled this loop holds 64
+        // live (x_int, y_int, sumi, dm_i, dsf, ds_j, ...) sites in flight
+        // simultaneously, forcing 144 spilled VGPRs and 0.067 VMEM_WR/VALU
+        // (vs 0.001 for FP16). Serializing j0 cuts that to ~16 live sites
+        // per iter. Keeps inner v loop fully unrolled to preserve dp4a ILP4.
+        #pragma unroll 1
         for (int j0 = 0; j0 < MMQ_X; j0 += MMQ_NWARPS) {
```

Inner `v` loop (8 sequential `v_dot4_i32_i8`) stays fully unrolled —
that's the dp4a ILP4 we need for arithmetic throughput. Outer `k01`
and `i0` loops also stay unrolled (4 and 2 iters respectively).

## Why this worked

The attribution checkpoint identified spill-write traffic as the
dominant cost (VMEM_WR 7.9× FP16, L2 hit 65% vs 85%, WriteSize 517×
FP16). At MMQ_X=8 the unrolled body had `4 × 4 × 2 × 8 = 256` dp4a
sites in flight; the per-site live set (x_int, y_int, sumi, dm_i.x,
dm_i.y, dsf.x, dsf.y, ds_j, scale_w, zp_eff, d_x, sum_x, idx) ×
4 j0 iters × 2 i0 iters = ~64 simultaneous SSA values. The compiler
couldn't fit this in 128 VGPRs and resolved the conflict by spilling
144 VGPRs/thread to scratch.

Serializing the j0 loop drops the simultaneous-live count to 16 (one
j0 iteration's worth). The compiler now schedules within a 60-VGPR
budget with zero spills. Scratch is fully eliminated.

## Why we didn't see this earlier

Two confounders:

1. **Hot-path kernel cache invalidation.** The first bench attempt
   showed 125.2 tk/s (no change) because `.hipfire_kernels/gfx906/`
   contained the stale 73 KB blob from before the source edit, plus
   a `.hash` sidecar that made `seed_hot_from_cold` skip the copy
   from `kernels/compiled/`. Per `compiler.rs:42-47`, the seed code
   short-circuits if both `.hsaco` and `.hash` exist in the hot dir.
   Removing the stale `.hsaco` + `.hash` triggered a JIT recompile,
   producing the new 56 KB blob.

2. **Source baked into binary via `include_str!`.** The bench binary
   embeds the kernel source at Rust compile time (line 218 of
   `crates/rdna-compute/src/kernels.rs`). After editing the `.hip`
   file we have to rebuild *both* the kernel artifacts (`compile-kernels.sh`)
   *and* the Rust binary (`cargo build --release --example
   bench_qwen35_mq4`). Otherwise the old source's hash mismatches the
   new blob in `.hipfire_kernels/`, the JIT recompiles, and the cached
   `kernels/compiled/` blob is ignored.

Build/cache hygiene checklist for future kernel edits:
1. Edit `.hip` file.
2. `JOBS=N ./scripts/compile-kernels.sh gfx906`.
3. `cargo build --release --example <bench> --features deltanet`.
4. `rm -f .hipfire_kernels/<arch>/<kernel>.hsaco .hipfire_kernels/<arch>/<kernel>.hash`.
5. Run bench.

## What attribution looked like, post-fix

| Counter | MMQ_X=8 baseline | j0 un-unroll | FP16 wave64 |
|---|---|---|---|
| arch_vgpr | 128 | 60 | 64 |
| scr (private segment) | 564 | 0 | 0 |
| VALUBusy | 8.7% | 15.6% | 61.5% |
| MemUnitBusy | 24.0% | 5.6% | 68.8% |
| VMEM_RD per call | 2.62 M | 191 K | 5.50 M |
| VMEM_WR per call | 2.07 M | 3.5 K | 0.26 M |
| L2 hit rate | 65.0% | 62.8% | 84.7% |
| WriteSize per call | 517 KB | 949 B | 1 KB |

Post-fix MMQ now writes only 949 B/call to HBM — within noise of FP16
wave64 (1 KB/call), and 545× less than the pre-fix MMQ. The L2 hit
rate is *slightly* lower (62.8% vs 65.0%) because the residual
weight-load traffic doesn't reuse L2 as well as the spill-store
traffic did, but absolute miss volume is far smaller.

## What's next

VALUBusy is now 15.6% — up from 8.7%, but still 4× lower than FP16's
61.5%. The next bottleneck axis is one of:

1. **ds_read latency in the inner v loop** — each dp4a needs an int
   from `x_qs` and an int from `y_qs_base`. Two `ds_read_b32`s per
   site × 256 sites per (j, i) call. SQ_WAIT_INST_LDS was 0 at
   MMQ_X=8 baseline; need to recapture post-junroll to confirm.
2. **Barriers** — `mmq_body` still has 4 `__syncthreads()` per kg.
3. **Kernel launch overhead** — at 4 ms/call across 32 layers × 2
   shapes × N tokens, launch latency could be visible.
4. **Tile size** — at MMQ_X=8 we have 256 WGs across 60 CUs (~4
   WGs/CU at full grid; could be undersaturated near tail). Now that
   spills are gone, the larger MMQ_X (16, 32) experiments from the
   prior session may need revisiting — the negative result there was
   driven by spill cascade, not by tile-shape itself.

Recommended next experiment: rerun rocprof groups 1+2 to nail down
the new dominant axis before picking the next lever.

## Conclusion

Single-line `#pragma unroll 1` on the j0 loop:
- Eliminates 100% of VGPR spills (144 → 0).
- Eliminates 99.8% of VMEM_WR traffic (2.07 M → 3.5 K per call).
- Brings prefill from 125.2 → 145.5 tk/s (+16.2%).
- For the first time, MMQ on gfx906 is faster than FP16 wave64
  (145.5 vs 141.3 = +3.0%).
- Decode unaffected (within ±2% noise).

Distance to llama.cpp-gfx906 reference (~235 tk/s) closes from
0.53× → 0.62×. Still 38% gap to close, but with VGPR pressure
released and 84% of cycles still idle, the headroom is structural —
not blocked by any axis we can't address with further loop/barrier
restructuring.

## Post-commit follow-up (same session)

Two additional levers attempted post-commit, both **negative results**.
Source unchanged from the committed state after each. The negative
results are themselves informative for future work on this kernel.

### Follow-up 1: MMQ_X sweep (8, 16, 32) with j0 un-unroll active

Hypothesis: now that spills are gone, larger MMQ_X amortizes the 4
barriers/kg across more output cells. The prior session's MMQ_X sweep
saturated at 8 because of spill cascade — that constraint is now lifted.

Method: each MMQ_X tested with the v2 sweep harness (`/tmp/sweep_mmqx_v2.sh`)
that hard-clears `.hipfire_kernels/<arch>/<kernel>.{hsaco,hash}` between
runs and verifies the JIT-cached blob via md5 + ELF metadata read from
the cached file (not the precompiled artifact). Three distinct cache
md5s confirmed each bench used its own binary.

| MMQ_X | Cache md5 | Size | JIT ELF | Prefill (3 runs) |
|---|---|---|---|---|
| **8** | 3ce0de83… | 56.9 KB | 60 VGPR, 0 spill, 0 priv | **145.6 / 145.6 / 145.6** |
| 16 | fb94b017… | 51.9 KB | 73 VGPR, 0 spill, 0 priv | 141.4 / 141.1 / 141.1 (−2.9%) |
| 32 | 213306f6… | 57.2 KB | 57 VGPR, 0 spill, **144 priv** | 141.2 / 141.2 / 141.2 (−3.0%) |

**MMQ_X=8 is genuinely the optimum.** Both larger tiles regress ~3%.
The barrier-amortization hypothesis is wrong: barrier overhead is not
currently dominant. Larger tiles cost more in WG-count (TLP) than they
save in barriers. MMQ_X=32 also re-introduces 144 B/thread of private
segment — the compiler keeps VGPR count low (57!) by selectively
spilling rather than blowing up vgpr_count, so the spill cascade lurks
just beyond MMQ_X=16 even with j0 un-unrolled.

Note vs prior session: the same MMQ_X=16 that ran at 92 tk/s with full
unroll now runs at 141 tk/s — the j0 un-unroll change itself is worth
+53% at MMQ_X=16, even though the optimum is still at 8. The change
helps across all MMQ_X, not just the 8 case we kept.

### Follow-up 2: ds_read_b128 LDS read pattern

Hypothesis from ISA inspection of the post-junroll binary:

```
=== ISA breakdown (current MMQ_X=8 binary) ===
  Total instructions: 6,931
  v_dot4_i32_i8:      384
  ds_read*:           324
  s_waitcnt lgkmcnt:  132   ← ≈3 dp4a per LDS-issue wait
  s_barrier:          12
```

132 `s_waitcnt lgkmcnt` instructions suggested LDS-issue throughput
might be the next bottleneck. Per the wiki, `ds_read_b128` runs at
9.5–11.2 TB/s vs 1.9–3.9 TB/s for `ds_read_b32` — 5× the LDS bandwidth
per issue. The inner v loop reads 8 contiguous ints (32 B) per (i, j)
site; rewriting as 2× int4 reads (16 B each, kx ∈ {0,8,16,24} so
16-byte aligned) should halve LDS issues.

Implementation: replaced the int-by-int v loop with explicit int4
casts:

```cpp
const int4 x4_lo = *reinterpret_cast<const int4*>(&x_qs[i*X_STRIDE+kx]);
const int4 x4_hi = *reinterpret_cast<const int4*>(&x_qs[i*X_STRIDE+kx+4]);
const int4 y4_lo = *reinterpret_cast<const int4*>(&y_qs_base[j*Y_STRIDE+ky]);
const int4 y4_hi = *reinterpret_cast<const int4*>(&y_qs_base[j*Y_STRIDE+ky+4]);
int sumi = 0;
sumi = __builtin_amdgcn_sdot4(x4_lo.x, y4_lo.x, sumi, false);
... (8 sdot4s using x4_lo/x4_hi.{x,y,z,w}, y4_lo/y4_hi.{x,y,z,w})
```

Result: structural shift worked, wallclock regressed.

| Counter | j0 baseline | + ds_read_b128 |
|---|---|---|
| **Prefill** | **145.5** | **141.5 (−2.7%)** ❌ |
| ds_read_b128 | 0 | 144 (wide reads emitted ✓) |
| ds_read_b32 | many | 24 (mostly gone ✓) |
| ds_read2_b32 | many | 0 ✓ |
| s_waitcnt lgkmcnt | 132 | 84 (**−36%**) ✓ |
| arch_vgpr | 60 | 62 |
| spill | 0 | 0 |
| Correctness NRMSE | 0.12% | 0.12% |

**Interpretation:** the compiler did exactly what we asked, dropping
LDS-issue waits by 36%. But wallclock didn't follow. Two likely
causes:

1. **132 lgkmcnt waits weren't actually serializing the kernel.** At
   ~10 cycles each = ~1,300 cycles per call ≈ 0.7 µs out of a 3.4 ms
   call. Killing 36% saves ~0.3 µs — well below noise. The compiler
   was already absorbing the waits via interleaved compute (visible in
   the disasm: `lgkmcnt(3)` → dp4a → `lgkmcnt(2)` → dp4a → ... pattern).
2. **`ds_read_b128` may have higher per-issue latency** than
   `ds_read_b32` on gfx906. The wiki cites bandwidth, not latency.
   Fewer issues × longer in-flight time → final `lgkmcnt(0)` waits
   slightly longer.

Reverted. Source/binary back to committed state (md5 `3ce0de83…`
restored and verified post-revert).

### Follow-up 3: Y-twice barrier collapse

Hypothesis: `mmq_body` has 4 barriers per kg (load → sync → compute →
sync → reload Y → sync → compute → sync). Doubling `tile_y` to hold
both Q8_1 halves at once collapses to 2 barriers/kg. LDS budget grew
from 35,456 B → 36,608 B — well under the 64 KiB cap.

Implementation:
- New `load_q8_1_tile_both()` that writes both halves into a doubled
  tile_y (32 active threads each load 18 ints — 9 per half × 2 halves).
- `mmq_body` issues one barrier after the combined load, two
  back-to-back `vec_dot_dp4a` calls (one with the half-0 base pointer,
  one with the half-1 base pointer offset by `MMQ_X * Y_STRIDE`),
  then one final barrier per kg.
- `dispatch.rs` LDS allocation doubled for the tile_y portion.

| Counter | j0 baseline | + Y-twice |
|---|---|---|
| **Prefill** (10 runs) | 145.4–145.6 (mean 145.5) | 145.3–145.6 (mean 145.5) **statistically identical** |
| **s_barrier** | **12** | **6** (**−50%**) ✓ as designed |
| s_waitcnt lgkmcnt | 132 | 129 |
| ds_read* | 324 | 318 |
| v_dot4_i32_i8 | 384 | 384 |
| arch_vgpr | 60 | **70** (+10) ❌ |
| spill | 0 | 0 ✓ |
| LDS allocation | 35,840 B | 36,992 B (within cap) |
| Correctness | 0.12% / 0.04% | 0.12% / 0.04% ✓ |

**Result: structural change worked, wallclock recovered nothing.**
Cutting barriers in half cost +10 VGPRs and bought 0 tk/s. The "barriers
are 30% of wallclock" theory is wrong on this kernel — barriers in a
2-wave64 WG synchronize only 2 waves and the hardware overlap is
already efficient enough that the visible cost per barrier is near zero.

Reverted. Cache md5 `3ce0de83…` (the committed j0 baseline) restored
and verified.

### What three negative results teach

- ds_read_b128: cut LDS-issue waits 36%, regressed wallclock 2.7%.
- MMQ_X=16: relieved barrier amortization concern, regressed 2.9%.
- Y-twice barrier collapse: cut barriers 50%, recovered 0%.

Three structural hypotheses, all empirically ruled out as the dominant
remaining cost. The 78% idle time genuinely is not (a) inner-loop dp4a
chain dependency, (b) ds_read latency, (c) LDS-issue throughput, (d)
WG-count amortization, or (e) `s_barrier` cost.

What remains as the most likely candidate is **kernel launch /
dispatch overhead**:
- 64 kernel calls per prefill (32 layers × 2 shapes per layer).
- The K=4096 call at 3.4 ms total — if launch overhead is ~1 ms (HIP
  launches on AMD typically cost 50–500 µs each, even more under
  rocprof — verified via the prior "rocprof reports lower
  prefill_tok_s than non-rocprof bench" observation).
- 64 × ~100 µs = 6.4 ms of pure dispatch latency in a ~880 ms prefill
  → ~0.7%. Probably not the *whole* explanation either.

Or it could be **per-WG init/teardown**: the 256 WGs/call each spend
some cycles on workgroup setup, register init for `sum[8] = {0.0f}`,
and the final `write_back_residual` phase. That's a fixed cost per WG
that doesn't scale with the dp4a count.

### Follow-up 4: HIP+HSA trace overturns the "78% idle" framing

`rocprof --hip-trace --hsa-trace` captured per-kernel timestamps WITHOUT
counter sampling, revealing that the prior "78% idle" was largely an
artifact of counter-instrumentation overhead.

**Per-call durations from the trace (real wallclock, no counter
overhead):**

| Layer | MMQ (j0 baseline) | FP16 wave64 | MMQ vs FP16 |
|---|---|---|---|
| K=4096 (32 calls) | **~1.3 ms** | ~2.1 ms | **MMQ −38%** |
| K=12288 (32 calls) | ~6.6 ms | ~6.7 ms | essentially equal |

The `rocprof --pmc` numbers in this and the prior checkpoints reported
3.4 ms / 6.7 ms for MMQ — **inflated by ~50% by the act of profiling**.
Counter-sampling stalls the kernel between transitions; the ratios
between counter values are still meaningful, but absolute wallclock
attribution from `rocprof --pmc` durations is wrong.

**End-to-end prefill breakdown (MMQ + screen ON, ~880 ms prefill):**

| Component | Total time | Calls | Notes |
|---|---|---|---|
| Residual MMQ kernel | 256 ms | 109 | includes 32 small screening probes |
| Q8_1 quantize (MMQ-only kernel) | 2 ms | 173 | dispatch overhead, cheap |
| All other kernels | ~620 ms | ~5,000 | gate_up, qkvza, rmsnorm, etc. |

| Component (FP16 baseline) | Total time | Calls | Notes |
|---|---|---|---|
| Residual FP16 kernel | 282 ms | 64 | 26 ms slower than MMQ residual |
| (no quantize) | — | 0 | |
| All other kernels | ~611 ms | ~5,000 | identical to MMQ |

**Screening cost:** measured directly via `HIPFIRE_MMQ_SCREEN={0,1}`:

| Config | Prefill (5 runs) | vs FP16 baseline (141.3) |
|---|---|---|
| MMQ + screen=0 | 146.7 / 146.7 / 146.8 / 146.7 / 146.7 | **+3.8%** |
| MMQ + screen=1 | 145.5 / 145.4 / 145.4 / 145.4 / 145.4 | +2.9% |

Screening costs **−1.3 tk/s** (32 extra short kernel dispatches per
prefill). Cost of correctness — required because some weight rows have
distributions where dp4a produces gibberish, so screen routes them to
the FP16 kernel. Hard to remove without either (a) per-weight
quantization quality improvements or (b) caching the screen result
across forward passes.

### What this changes about "next lever"

The MMQ residual kernel is **already running near-optimally** for
what it does:
- 38% faster per K=4096 call than FP16 wave64.
- ~equal at K=12288.
- The 78% "idle" was profiler overhead, not real headroom.
- All five inner-loop micro-architectural levers attempted (j0
  un-unroll, MMQ_X sweep, ds_read_b128, dual-accumulator dp4a, Y-twice
  barrier collapse) have been tried — only j0 un-unroll won, and that
  win is now committed.

Further inner-loop optimization on `gemm_hfq4g256_residual_mmq_gfx906`
will not move prefill meaningfully because the residual GEMM is only
~30% of total prefill, and even halving that 30% only buys ~3% overall.

**The real opportunity is now elsewhere:**

1. **Port MMQ to `gemm_gate_up_hfq4g256`** — gate_up takes ~33% of
   prefill (`mmq.stats.csv`: 32 calls × ~12 ms avg = 389 ms total). If
   MMQ saves 38% there too, that's **~12% prefill end-to-end**.
   Bigger than anything left in the residual kernel. This was Path A
   §A2 of the original L2 plan and the original gfx906 MMQ plan §3.4.
2. **Port MMQ to `gemm_qkvza_hfq4g256`** (~17% prefill share per the
   plan).
3. **Cache the screening result** across forward passes — weights
   don't change, so the screen verdict is invariant. Saves 1.3 tk/s
   without giving up correctness.
4. **Operator fusion**: combine residual MMQ with the next op in the
   model (RMSNorm/Add) to cut dispatch count.

The original plan's Path A §A2 (port to gate_up) was the right next
move all along — the inner-loop work overshot what was achievable on
the residual kernel.

None of these are one-line changes. The "easy" levers on this kernel
shape are now exhausted; further gains need architectural work.

## Cross-reference

- Final outcome (this work was superseded by the redesign):
  `docs/plans/gfx906-mmq-prd.md`,
  `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
- Attribution that picked this lever:
  `docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`
- Prior session (MMQ_X reduction):
  `docs/perf-checkpoints/2026-05-04-gfx906-mmq-spill-reduction.md`
