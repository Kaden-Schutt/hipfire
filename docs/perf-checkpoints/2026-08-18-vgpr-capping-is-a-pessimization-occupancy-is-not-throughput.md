# Capping VGPRs to "fix" occupancy is a 12–27% pessimization on a bandwidth-bound GEMV

**Date:** 2026-08-18 · **Lifecycle:** `historical`, fixture-bound. Not a default,
baseline, or admission decision.

## What happened

Porting the v1.5 / `mq4c` residual GEMV (132 B/group, per-256 fp16 header) produced a
kernel at **97 VGPR** — exactly one over the 96 needed for 16 waves/SIMD (1536/16) —
so reported occupancy fell to **12**. That looks like a defect worth fixing, and
`__attribute__((amdgpu_num_vgpr(n)))` does fix it.

Fixing it made the kernel **slower than the thing it was supposed to beat.**

gfx1201, `crates/rdna-compute/examples/bench_vgpr_cap_sweep.rs`, m=k=5120 (o_proj
shape), 64 warmups, 200 launches per sample, arms interleaved sample-by-sample:

| arm | VGPR | occ | spill | µs (med) | GB/s | vs v1 |
|---|---|---|---|---|---|---|
| v1 control (136 B) | 93 | 16 | 0 | 15.82 | 880.2 | 1.0000 |
| **v1.5 uncapped** | **97** | **12** | **0** | **14.54** | **929.4** | **0.9192** |
| v1.5 cap 96 | 96 | 16 | **3** | 17.93 | 753.8 | 1.1334 |
| v1.5 cap 84 | 84 | 16 | 0 | 17.80 | 759.5 | 1.1249 |
| v1.5 cap 80 | 79 | 16 | 0 | 16.95 | 797.3 | 1.0715 |
| v1.5 cap 76 | 76 | 16 | 0 | 16.48 | 820.4 | 1.0414 |
| v1.5 cap 72 | 70 | 16 | 0 | 19.95 | 677.7 | 1.2607 |
| v1.5 cap 64 | 63 | 16 | 0 | 20.12 | 672.0 | 1.2714 |

**The configuration that looks worst on every static metric is the fastest**, and
every occupancy-16 variant is slower than v1. Confirmed across three fresh processes;
run-to-run thermal drift is wide (samples 15.5–25.5 µs over 7 runs), so the robust
statistic is the per-arm minimum, which gives uncapped/v1 = 0.937 and 0.933 —
consistent with the 3-run medians.

## Why this matters beyond one kernel

1. **Occupancy and spill count are inputs to throughput, not proxies for it.** The
   extra registers buy deeper prefetch; on a bandwidth-bound GEMV that outweighs wave
   count. Any kernel in this tree sitting just over an occupancy boundary should be
   *measured* before being "corrected" — a cap that improves the static report can
   cost double-digit throughput.
2. **The register allocator is non-monotonic.** Caps 96 and 88 spill (3 and 5 VGPRs)
   while 84, 80, 76, 72 and 64 do not. A small cap violation gets patched by spilling;
   a large one makes the machine scheduler re-plan and genuinely need fewer registers.
   So "tighter cap" is not monotonically worse, and the spill-free region sits *below*
   the spilling one. Any future cap must re-check spills at that exact value.
3. **The no-spill rule was never in tension here.** The fastest arm (uncapped) also
   has zero spills, so the project's hard requirement is satisfied without a cap. The
   temptation to cap came from the occupancy *number*, not from spilling.
4. `-mllvm -amdgpu-waves-per-eu` / `-amdgpu-num-vgpr` / `-amdgpu-vgpr-limit` are all
   **rejected** by this hipcc (HIP 7.14); the source attribute is the only lever. Note
   `__launch_bounds__` in these kernels is `#if`-guarded per arch — on gfx12 the
   active line is `__launch_bounds__(32, 32)`, and patching the inactive `#else`
   silently does nothing.

## Incidental result: v1.5 beats v1 by ~8% on the shipping decode kernel

The uncapped v1.5 arm is **0.9192× v1** (14.54 vs 15.82 µs; 929.4 vs 880.2 GB/s).
v1.5 moves 2.94% fewer weight bytes, so pure traffic predicts a floor of ~0.971. Coming
in at 0.919 means the halved header load — one packed fp16 dword instead of two f32
dwords — is doing real work beyond the byte saving, as
[`docs/quant-formats/mq4-v2.md`](../quant-formats/mq4-v2.md) § 1b predicted
qualitatively.

This is the first v1.5 measurement on a kernel that is actually on the dense decode
path. § 1b's 0.9847 / 0.9773 figures came from `gemv_hfq4g256_multirow`, the same proxy
that wrongly reported v2 as throughput-neutral.

## Where the +4 VGPR comes from

Bisected on gfx1201:

| variant | VGPR | occ |
|---|---|---|
| v1 | 93 | 16 |
| v1 with only the stride changed 136 → 132 | 93 | 16 |
| v1.5 layout with the fp16 convert **removed** | 99 | 12 |
| v1.5 full | 97 | 12 |

So it is neither the stride, nor 132's loss of 8-byte group alignment (a 136-stride
variant with the v1.5 header is also 97), nor the fp16 conversion — removing the
conversion is *worse*. It is the loss of v1's two adjacent header loads at `gp` and
`gp+4`, which the compiler merges into one `global_load_dwordx2`. v1.5's single dword
cannot merge with the nibble load, and the resulting schedule needs 4 more registers.

## Limits

- One kernel, one shape (m=k=5120), one arch (gfx1201), synthetic weights. Correctness
  is not tested here — that is `mq4v2_parity` and the mq4c repack probe.
- Host timing amortised over 200 launches/sample (~22.1 µs sync tax ≈ 0.4%), not
  device hipEvent timing.
- Thermal drift is the dominant noise source at 7 runs. Prefer minimums, or fewer runs
  in a fresh process.
- Says nothing about whether v1.5 is worth shipping: that still needs the other 12
  translation units, and v1.5 remains unscoreable for KLD until they exist.
