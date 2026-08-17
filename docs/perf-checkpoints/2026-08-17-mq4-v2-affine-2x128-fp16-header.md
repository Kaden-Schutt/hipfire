# MQ4 v2.0: affine 2×128 with fp16 headers — 16% better MSE, 40% better tail, at byte-identical 136 B

- **Date:** 2026-08-17
- **Lifecycle:** `historical`
- **Quality fixture:** 278,528 real post-FWHT 256-blocks, Qwen3.8-27B bf16 parent, layers
  0 / 20 / 40, engine FWHT sign seeds 42/1042. Tail threshold = 99th pct \|w\| = 2.869166e-02.
  Harness [`tools/quant-design/`](../../tools/quant-design/).
- **Throughput fixture:** gfx1201 (RX 9070), m=4096 k=5120, 32 warmups, 100 iters/run,
  3 runs, arms interleaved sample-by-sample, device-side HIP events.
  `crates/rdna-compute/examples/bench_gemv_paired_throughput.rs`
- **Disposition:** design proposal with codec + throughput evidence. **No KLD yet** — and
  codec MSE is a confirmed non-predictor of KLD, so the quality case here rests on the
  tail metric, which is the only one that reproduces our byte-comparable KLD inversion.

## The proposal

Affine's header is over-provisioned for FWHT-rotated data. Two measured facts:

- Post-rotation asymmetry is negligible: `(max+min)/(max−min)` has mean **+0.0004**, mean
  absolute **0.0757**.
- fp16 headers cost **0.03%** versus f32 (tail 5.6621e-07 vs 5.6606e-07).

So spend the same 8 header bytes on **granularity instead of precision**:

| | payload | header | total | bpw |
|---|---|---|---|---|
| today qt=1/6/13 | 128 B nibbles | f32 scale + f32 zero = 8 B | 136 B | 4.2500 |
| **v2** | **128 B nibbles, byte-identical** | 2 × (fp16 scale + fp16 zero) = 8 B | **136 B** | **4.2500** |

## Codec quality — data-free, no calibration

| variant | B | bpw | overall MSE | tail-1% MSE |
|---|---|---|---|---|
| affine 1×256 f32 hdr (**today**) | 136 | 4.2500 | 1.4415e-06 | 9.4635e-07 |
| **affine 2×128 fp16 hdr (v2)** | **136** | **4.2500** | **1.2089e-06** (−16.1%) | **5.6621e-07** (−40.2%) |
| affine 2×128 f32 hdr | 144 | 4.5000 | 1.2089e-06 | 5.6606e-07 |
| affine 4×64 fp16 hdr | 144 | 4.5000 | 9.7617e-07 | 3.2450e-07 |
| affine 8×32 fp16 hdr | 160 | 5.0000 | 7.4343e-07 | 1.8091e-07 |
| affine 1×256 fp16 hdr | 132 | 4.1250 | 1.4415e-06 | 9.4643e-07 |
| GL codebook qt=40 | 130 | 4.0625 | 1.1441e-06 | 2.0147e-05 |
| SEL 64-profile qt=43 | 132 | 4.1250 | 1.0338e-06 | 4.8843e-06 |

The 1×256-fp16 row isolates header precision: identical quality at 132 B, so **4 bytes are
free right now** independent of everything else. v2 spends them on granularity instead.

`max_rel` is **0.000%** for every affine row — min/max fitting makes block extremes exactly
representable. That is the property both codebook formats lack (GL 0.1076) and it is what
the tail column is measuring.

## Throughput — and a premise this overturned

Measured at R=2, `gemv_rows_default()` on gfx1201:

| format | B | median µs | GB/s | ratio vs hfq4g256 | VGPR / SGPR | spills |
|---|---|---|---|---|---|---|
| **gemv_hfq4g256** | 136 | **17.00** | **657.5** | 1.00× | 97 / 20 | 0 |
| gemv_mq4g256gl | 130 | 20.88 | 511.8 | 1.23× slower | 94 / 24 | 0 |
| gemv_mq4g128 (qt=7) | 144 | 29.40 | 402.5 | 1.73× slower | 14 / 14 | 0 |
| gemv_mq4g256sel (qt=43) | 132 | **155.76** | 69.7 | **9.16× slower** | 122 / 94 | 0 at R=2 |

### The "G128 is slow" premise is wrong — it is a missing kernel

G128's timings are **29.34 / 29.40 / 29.36 µs at R=1 / 2 / 4** — dead flat — while
hfq4g256 goes 28.00 → 17.00 µs. Cause: `gemv_hfq4g256_multirow` has **6** multirow entry
points; `gemv_hfq4g128` and `gemv_mq4g128` have **zero**. G128 runs single-row regardless
of R.

At matched R=1 the per-128 header costs **4.8%** (28.00 → 29.34 µs), and G128 posts *higher*
GB/s (403.3 vs 399.2) because it moves more bytes in the same time. The 1.73× is entirely
the absent multirow kernel.

### v2 is not qt=7, and that matters

qt=7 is a true G128 format: separate 72 B groups per 128 weights. **v2 keeps the 256-group
payload byte-for-byte identical** — the same 128 B of contiguous nibbles, the same lane
mapping — and only reinterprets the same 8 header bytes as two halves. On wave32 each lane
covers 8 weights, so sub-block 0 is lanes 0–15 and sub-block 1 is lanes 16–31: the header is
**uniform per half-wave**, costing two scalar loads plus a select rather than per-lane
traffic. v2 therefore inherits hfq4g256's access pattern, not qt=7's.

**Remaining work for v2: a multirow kernel with half-wave header selection.** Until that
exists, v2 has no throughput number, and this document makes no throughput claim for it.

## Verdict on qt=43 (MQ4-G256-SEL): retire

- **9.16× slower** than hfq4g256 at R=2 (155.76 vs 17.00 µs; 69.7 vs 657.5 GB/s, 10.6%).
  This is the cause of the 2 tok/s observed while scoring it — not a fallback path, the
  kernel itself.
- **Spills at R≥4**: `sel_multirow_r4` vspill 13 / priv 56; `r8` vspill 83 / priv 336.
  Zero spills is a hard requirement, so R=2 is the only legal configuration.
- SGPR 94 at R=2 versus GL's 24 — the wave-uniform scalar-loaded 64-profile table consumed
  the scalar register file. The design that made the table cheap in theory made it
  expensive in registers.
- Its quality advantage is **overall MSE**, the metric confirmed not to predict KLD, while
  it loses **tail-1% MSE** by 8.6× against affine — the metric that does.

Failing the perf gate on three counts while winning only the discredited metric, qt=43
should not be pursued further.

## Caveat on absolute microseconds

This session measures hfq4g256 at 17.00 µs / 657.5 GB/s where an earlier session measured
19.12 µs / 584.6 GB/s at the same shape. Different thermal state, different day. **Ratios
are the primary result**; absolutes are only comparable within one interleaved run. The
same kernel has previously swung 38% (21.72 vs 30.00 µs) on warmup count alone.

---

## Amendment — v2 throughput measured on five architectures; gfx1030 regresses 14–18%

The original document claimed no throughput number for v2 because no kernel existed. One now
does: the header substitution was applied to both `gemv_hfq4g256_multirow.hip` (multirow,
R=2/4/8) and `gemv_hfq4g256.hip` (single-row, R=1), preserving every byte offset so both
header dwords remain **lane-invariant scalar loads**. The diff per group is 5 lines:

```c
const unsigned int hA = LOAD_WEIGHT_HEADER(gp,     goff);      // unchanged address
const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u); // unchanged address
const unsigned int hs = (tid < 16) ? hA : hB;                  // half-wave select
float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
```

Net cost: one `v_cndmask` plus two `v_cvt_f32_f16` per group per row.

### Why the single-row port was mandatory

`ArchCaps::gemv_rows_default()` (`arch_caps.rs:145-151`) is
`if is_wave64_native || is_rdna2 || is_rdna3_dgpu { 1 } else { 2 }`. **gfx1100 and gfx1030
therefore default to R=1 and never touch the multirow kernel.** The earlier gfx1201-only
result rested entirely on a path those parts do not take.

### Results — hipx, 4 arches, interleaved hipEvent, 32 warmups, 100 iters, 3 runs

Each arch's **shipping default R** is the only row that matters. Ratio = v2 / today; >1 is slower.

| arch | GPU | default R | ratio @ 4096×5120 | ratio @ 4096×17408 | VGPR today→v2 | spills | verdict |
|---|---|---|---|---|---|---|---|
| gfx1201 | RX 9070 | 2 | 0.9645 | — | 94 → 94 | 0 | **free** |
| gfx1100 | RX 7900 XTX | **1** | 0.97–0.99 | 0.97–0.99 | 72 → 76 | 0 | **free** |
| gfx1151 | Radeon 8060S | 2 | 0.9846 | 0.9978 | 94 → 94 | 0 | **free** |
| gfx1010 | RX 5700 XT | 2 | 1.0020 | 0.9984 | 36 → 36 | 0 | **free** |
| **gfx1030** | **RX 6950 XT** | **1** | **1.1386** | **1.1758** | 61 → 61 | 0 | **REGRESSION** |

gfx1030 was confirmed across **6 independent invocations**: 5120-shape ratios 1.1386, 1.1287,
1.1226, 1.1594; 17408-shape 1.1758, 1.1810, 1.1709, 1.1775. Stable, not noise.

Device binding was verified empirically per arch rather than assumed — **KFD node order is not
HIP device ordinal** (KFD 1/2/3/4 = gfx1100/1151/1010/1030; HIP 0/1/2/3 =
gfx1100/gfx1151/gfx1010/gfx1030 only after `HIP_VISIBLE_DEVICES` remap). Each run printed
`gcnArchName` and it matched its compile target.

### Mechanism: RDNA2's Infinity Cache makes the kernel VALU-bound

gfx1030 does the 5120 shape in **17.37 µs for 11.18 MB = 643 GB/s**, against the RX 6950 XT's
**576 GB/s** VRAM peak. Exceeding VRAM peak means it is not reading VRAM: the 11.18 MB weight
matrix fits entirely inside the **128 MB Infinity Cache**, and the 17408 shape (~38 MB) does
too (46.34 µs → 820 GB/s).

So on RDNA2 this kernel is **compute-bound, not bandwidth-bound**, and v2's three extra VALU
ops per group are fully exposed. VGPR count is unchanged at 61 with zero spills, confirming
the cost is instruction latency rather than occupancy. It is consistent that the regression
*grows* on the larger shape (13.9% → 17.6%) instead of amortising away.

**This inverts the intuition recorded earlier in this campaign.** The expectation was that a
bandwidth-saturated part (gfx1100, 192 SIMDs, ~60% of peak at R=1) would be least able to
hide the extra ops. The opposite holds: **saturation is exactly what makes the change free**,
and the vulnerable target is the one with enough cache to have become compute-bound.

### Consequence for shipping

v2 is throughput-neutral on gfx1201, gfx1100, gfx1151 and gfx1010 at their default R, and
costs 14–18% on gfx1030. Two clean options, both idiomatic for this codebase, which already
carries five gfx1030-specific `gemv_hfq4g256.gfx1030.v*.hip` variants:

1. **Arch-gate the format**: gfx1030 keeps qt=1. Costs a dtype branch at load.
2. **Raise gfx1030 to R=2**, where v2 measures 0.999 — but that changes a tuned default for
   every 4-bit format on RDNA2, not just this one, and needs its own validation.

Not yet resolved, and it should not be decided on this data alone: the R=2 option is a
one-line `ArchCaps` change with wide blast radius, while the arch-gate is contained.

### Still owed before v2 ships

- **A real KLD run.** Tail-1% MSE is the only proxy that reproduced our one byte-comparable
  KLD inversion, and it remains a proxy. Unlike qt=43, v2 *can* be scored at full speed
  because affine already has the WMMA GEMM family.
- **Encoder, qt number, and GEMM wiring.** All throughput above used synthetic blobs; nothing
  is wired into the quantizer or dispatch.
- **The other ~8 arch-specific HFQ4-G256 kernel variants** would each need the same
  5-line substitution: `gemv_hfq4g256.gfx1100.hip`, `.gfx1201.hip`,
  `_multirow.gfx1100.hip`, and `gemv_hfq4g256.gfx1030.v{1..5}.hip`.
- The `HIPFIRE_HFQ4G256_XBATCH_KERNEL` path in the single-row file was **excluded** from this
  measurement, not ported.
