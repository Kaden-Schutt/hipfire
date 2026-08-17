# MQ4-GL vs HFQ4-G256 GEMV — a row-count mismatch manufactured a 57% deficit that does not exist (gfx1201)

**Lifecycle:** `historical`. Fixture-bound measured evidence. It is **not** a
current default, an automatic baseline, a product claim, or an admission
decision.

**Disposition:** MQ4-GL's single-row GEMV was compared against HFQ4-G256 for six
consecutive optimization attempts under the belief that GL was **57% slower**.
That figure was an artifact of comparing **GL at R=1 against HFQ4 at R=2**.

`ArchCaps::gemv_rows_default()` (`crates/rdna-compute/src/arch_caps.rs:145`) is:

```rust
if is_wave64_native || is_rdna2 || is_rdna3_dgpu { 1 } else { 2 }
```

gfx1201 is RDNA4 — not wave64-native, not RDNA2, not RDNA3-dGPU — so it falls
through to **2**. `Gpu::gemv_hfq4g256` consults that default and silently ran
multirow R=2. `Gpu::gemv_mq4g256gl` never consults it and always ran single-row.

At **matched** R, the deficit is **5–14% of time**, and GL's real weakness is
achieved bandwidth, not latency:

| R | GL µs | GL GB/s | HFQ4 µs | HFQ4 GB/s | GL time penalty | GL/HFQ4 GB/s |
|---|---|---|---|---|---|---|
| 1 | 39.20 | 272.6 | 36.88 | 303.1 | +6.3 % | 0.899 |
| 2 (production) | 21.72 | 492.0 | 19.12 | 584.6 | +13.6 % hot / +5.6 % stable | 0.842 / 0.905 |
| 4 | 21.28 | 502.2 | 19.92 | 561.1 | +6.8 % | 0.895 |

GL moves **4.4% fewer** weight bytes (130 vs 136 B/group) yet takes ~6% longer,
so it holds a consistent **0.84–0.90 of HFQ4's achieved bandwidth** across every
R. That ratio — roughly 10–16% — is the real and much smaller target.

**Source:** branch `quant/quality`, gfx1201, m=4096, K=5120, device-side timing
via `rdna_compute::profile::start`/`stop`.

---

## Measurement-protocol findings (these generalize beyond this record)

**1. `HIPFIRE_GEMV_ROWS` must be set explicitly for every GEMV measurement.**
Relying on the arch default is what produced the phantom 57%. The default is
arch-dependent and differs between the two kernels being compared, so an
unstated row count silently compares different configurations.

**2. Absolute GEMV timings on this stack are thermally soft to a degree that
invalidates cross-session comparison.** The *same* GL R=2 kernel measured:

| warmup samples discarded | GL R=2 median |
|---|---|
| 32 | 21.72 µs |
| 8 | 30.00 µs |

A **38% swing from warmup count alone**. The GL/HFQ4 *ratio* moved far less
(1.136 hot vs 1.056 stable), so **interleaved ratios are trustworthy where
absolute microseconds are not**. Any A/B must alternate arms sample-by-sample
inside one thermal window and report the ratio as primary.

**3. Do not pin clocks to stabilize this.** Pinning HIGH measures worse than
auto on this stack and real deployments run under auto, so the pinned number is
the unrepresentative one.

---

## Five mechanisms falsified while chasing the phantom

Each was a plausible explanation for a 57% gap that did not exist. They remain
useful as negative results, since each is a mechanism now ruled out for the real
10–16% bandwidth residual.

| hypothesis | test | outcome |
|---|---|---|
| Memory-level parallelism / narrow loads | `_wide`: `global_load_b128` 2 → **22** (HFQ4 has 14), `max_consecutive_vmem` 3 → 8 | 30.88 → 30.68 µs. **FLAT** |
| Stall density | `_perm` has the lowest `s_wait` density of any GL kernel | 8.7 % vs HFQ4's 8.2 %, and it was the **slowest** at 34.07 µs |
| ILP / accumulator count | `_wide` used `acc0..acc3` in HFQ4's 4-group unroll and replaced 4× `u8` loads with `uint4` | **FLAT** |
| Per-row prologue amortization | K-sweep at gpr = 10 / 20 / 40 / 80 | ratio 0.969 / 0.909 / 0.892 / 0.970 — **flat across 8× gpr**, so no fixed per-row cost |
| Register-resident codebook | `v_perm_b32` (attempt 5), int8 + `v_dot4` (attempt 3), polynomial (attempt 4) | all separately **NO-GO** |

Verified instruction counters (disassembled locally, not self-reported):

| kernel | instrs | `s_wait` | density | `global_load_b128` | µs |
|---|---|---|---|---|---|
| `gemv_mq4g256gl` | 369 | 54 | 14.6 % | 2 | 30.9 |
| `gemv_mq4g256gl_wide` | 916 | 106 | 11.6 % | 22 | 30.7 |
| `gemv_mq4g256gl_perm` | 321 | 28 | 8.7 % | 0 | 34.1 |
| `gemv_hfq4g256` | 573 | 47 | 8.2 % | 14 | 19.6 |

One framing error worth recording: HFQ4's 14 `b128` loads are the **activations**
(`x[base..base+7]`, 8 consecutive f32 = 32 B/lane). Its *weight* load is a single
`b32` per lane per group. Both kernels already read weights as a perfectly
coalesced 128 B per wave per group, so "widen the weight loads" was never a
lever. A corollary: HFQ4 needs only 4-byte alignment for its weight load, so odd
group strides are not disqualifying.

---

## Structural gap that remains open

`Gpu::gemv_mq4g256gl_multirow` exists (`crates/rdna-compute/src/gemv.rs:12349`)
and **does** consult `gemv_rows_default()` (line 12357), but it has **zero
references** in `crates/hipfire-dispatch/src` or
`crates/rdna-compute/src/dispatch.rs`. It is unreachable from dispatch.

qt=40 / `DType::MQ4G256GL` is still encode-only ("no kernel consumes this yet",
`dispatch.rs:120`), so nothing is broken in production today. But when GL is
wired, it **must** be wired to the multirow path, or it will run single-row
against an HFQ4 that defaults to R=2 — reproducing the phantom gap as a *real*
one, for purely structural reasons.

---

## What has not been tested

The layout. GL is **SoA**: indices at `row*gpr*128`, fp16 scales at
`M*gpr*128 + (row*gpr+g)*2` — a second stream ~10 MB away.
`gemv_mq4g256gl.hip`'s own comment names the consequence: the scale load "cannot
be claused with the index load and each iteration stalls on it," which is why the
kernel hoists all scales into LDS per row behind a `__syncthreads()`.

HFQ4 is **AoS**: `[f32 scale][f32 zero][128 B nibbles]` contiguous, one stream,
no hoist, no LDS, no barrier, and its f32 header lets the affine fold lower to
VOPD `v_dual_fmac_f32`.

Note the prologue *amortization* hypothesis is already falsified by the flat
K-sweep — so if AoS helps, it will be through **stream locality**, not through
removing a fixed per-row cost. Those are different mechanisms and the K-sweep
only ruled out the latter.

---

## Caveat on the reference numbers

The "19.56 µs HFQ4" figure used throughout the six attempts does not reproduce
cleanly as *cold* R=2. Measured HFQ4 R=1 ranges 29.76–39.2 µs and R=2 ranges
19.12–28.4 µs depending on warmup, with R=4 at 19.92 µs. The ~19.5 µs value
corresponds to hot R=2 or to R=4. This is the same thermal softness recorded
above, and it is why only matched, interleaved ratios are quoted in the
disposition.
