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

## Cross-reference

- Plan: `plans/gfx906_mmq_l2.md` v3
- Attribution that picked this lever:
  `docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`
- Adversarial reviews of the original L2 prefetch v1:
  `gfx906_l2_rev_claude.md` v2, `plans/gfx906_l2_rev_glm5.md`
- Prior session (MMQ_X reduction):
  `docs/perf-checkpoints/2026-05-04-gfx906-mmq-spill-reduction.md`
