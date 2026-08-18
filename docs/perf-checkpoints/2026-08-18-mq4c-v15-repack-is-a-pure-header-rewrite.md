# v1 → v1.5 (mq4c) repack is a pure header rewrite — 0 nibble flips in 236 M weights

**Date:** 2026-08-18
**Lifecycle:** `historical`. Fixture-bound measured evidence, not a current default,
automatic baseline, or admission decision.

## Question

Can an existing qt=13 artifact be converted to v1.5 / `mq4c` (qt=45) **in flight** —
at load time, with no re-quantization and no parent model?

`docs/quant-formats/mq4-v2.md` § 7 says a 136 B → 132 B repack is safe "because it
only discards unused f32 header precision — there is no grid realignment." But § 3
also states that round-tripping the header through fp16 **before** quantizing is
mandatory, and that the measured 0.008% cost "is only valid with this discipline."
Those two statements are in tension for a repack, which by definition inherits
nibbles that were fitted against the f32 header. So: is a naive header truncation
equivalent to a proper re-fit, or not?

## Answer: they are the same operation

Measured on `/home/kaden/qcal/q38.ctl.mq4` — the qt=13 baseline arm (15,662,615,552 B,
WT2 KLD 0.043776) — by `tools/quant-design/mq4c_repack_probe.py`, sampling six qt=13
tensors spread across depth:

| | |
|---|---|
| groups probed | 923,840 |
| weights probed | **236,503,040** |
| max \|delta\| | **0.011076 steps** (a nibble flips at 0.5) |
| **nibble flips required** | **0** |
| min f32 scale | 7.609e-04 (fp16 min normal 6.104e-05) |
| groups with denormal scale | 0 |
| max \|zero\| / scale | 10.80 |
| repack relative RMS drift | 9.093e-04 (**0.0909%**) |

`delta` is how far v1's reconstruction `w = z32 + q·s32` sits from an integer on the
v1.5 grid, in units of the new step:

    delta(q) = ( (z32 - z16) + q·(s32 - s16) ) / s16

Predicted bound before measuring, from fp16's 11-bit significand:

    |delta| <= |z|/(s·2^11) + 15/2^11 ≈ 10.8/2048 + 0.00732 = 0.0126

Measured max **0.011076**. Since rounding boundaries are at ±0.5, the drift is ~45×
too small to move any weight to a different level. Hence `q' == q` everywhere, and
"naive truncation" and "proper re-fit" are literally the same output.

### The drift reconciles with the spec's 0.008%

A 0.0909% relative RMS drift adds `(9.093e-4)^2` to a relative quantization MSE of
about 1.19% (absolute codec MSE 1.4415e-06 against weight RMS 0.011), i.e. a **0.007%
MSE increase** — matching the 0.008% measured independently in the codec sweep. Two
different methods, same number.

## Therefore the repack is trivial

    read  136 B: [f32 scale][f32 zero][128 B nibbles]
    write 132 B: [fp16 scale][fp16 zero][128 B nibbles verbatim]

Payload offset moves 8 → 4, stride 136 → 132, nibbles copied unchanged. No decode, no
re-encode, no parent, no imatrix, no search. Works offline as a file transform or
in-flight on the way into VRAM.

On the `ctl` arm: 12,936,232,960 B of qt=13 payload is 95,119,360 groups, so

    95,119,360 × (136 - 132) = 380,477,440 B = 380.5 MB

**2.94% of the quantized payload, 2.43% of the whole file**, for 0.007% MSE.

## What this does NOT resolve

1. **The third layout is still the whole cost.** v1.5 is not byte-compatible with v1
   or v2 (payload at offset 4, stride 132), so it needs its own kernel family — all
   13 translation units, the launchers, the keys, and the same container-aware key
   sites that took four rounds to get right for v2. The repack being free does not
   make v1.5 free.
2. **v1.5's throughput advantage is still unmeasured on the shipping path.** § 1b's
   0.9847 / 0.9773 ratios came from `gemv_hfq4g256_multirow`, the same proxy that
   wrongly reported v2 as throughput-neutral (v2 actually costs 3.5% decode; see
   `2026-08-18-mq4-v2-qt44-kld-beats-v1-at-equal-bytes.md` amendment 2). The
   mechanism favours v1.5 — 2.94% less weight traffic, one packed scalar header load
   instead of two, and no `cndmask` — but on a bandwidth-bound decode path that
   predicts roughly +2.9%, which is a prediction, not a measurement.
3. **No KLD for v1.5.** Cannot be scored until the kernels exist, since nothing can
   decode a 132 B group today.
4. **Sampling.** Six tensors, 200 k groups each, one artifact. The `min scale`
   7.609e-04 sits an order of magnitude above the fp16 denormal threshold on this
   model; a model whose per-group ranges are far smaller could in principle push a
   scale into denormals, where fp16 relative precision collapses and the bound above
   fails. The probe reports `min f32 scale` and `groups w/ denormal s` precisely so
   that is checked per model rather than assumed.

## Bearing on the decision

v1.5 and v2 do not compete. v2 dominates v1.5 for anything re-quantizable — strictly
better KLD at the *same* 136 B. v1.5's entire value is that it needs no parent: it is
a legacy-artifact optimisation, and this record shows its conversion cost is
effectively zero. The open question is therefore only whether 380 MB and a predicted
~3% decode on already-shipped `.mq4` files justifies maintaining a third kernel
layout.

Recommended gate before porting 13 translation units: port ONE — the residual decode
GEMV — at 132 B and measure it against its v1 twin with
`crates/rdna-compute/examples/bench_gemv_paired_throughput.rs`. That is one file of
work and it replaces the discredited proxy with a real number on the shipping kernel
shape. If v1.5 does not actually beat v1 there, the third layout is not worth it.
