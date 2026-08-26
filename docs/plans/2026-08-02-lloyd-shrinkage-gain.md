<!-- SPDX-License-Identifier: Apache-2.0 -->
# MQ*-Lloyd energy shrinkage: fold the gain into the codebook

Date: 2026-08-02 · Status: proposed, not started · Owner: unassigned

## Summary

The MQ2-Lloyd codec loses **11% of weight energy** on every round-trip. On
DeepSeek V4 this has been silently compensated by `route_scale` since May. On
every other model using an MQ*-Lloyd tier there is no equivalent knob, so those
tensors are simply short.

The fix is one constant folded into the codebook at quantization time. The
measurement is done; the work is not.

## Evidence

Measured with the production codec (`quantize_mq2g256_lloyd` →
`dequantize_mq2g256_lloyd_to_f32`) on 512 groups of post-FWHT-domain Gaussian
input. Test: `mq2_lloyd_shrinkage_on_routed_expert_tier` in
`crates/hipfire-quantize/src/main.rs`.

| tier | codec | retained `E[ŵ·w]/E[w²]` | `‖ŵ‖/‖w‖` | gain to restore |
|---|---|---|---|---|
| routed experts (qt=19) | MQ2-Lloyd | **0.8877** | 0.9422 | **1.1265** |
| shared expert + `ffn.gate` (qt=35) | MFP4G32E8SOA | 0.9998 | 1.0093 | ~1.00 |

Per-group spread across 512 groups: mean 0.8877, **sd 0.0111 (1.25%)**,
p05 0.8694, p95 0.9049, p95−p05 = 3.99% of mean.

### This is inherent, not a coding error

0.8877 matches the textbook 2-bit Lloyd-Max Gaussian value (0.8825) to three
digits. Lloyd-Max centroids are conditional means, so by the orthogonality
principle `E[ŵ·w] = E[ŵ²]`, which makes the reconstruction *provably* shorter
than the source — retained energy is exactly `1 − MSE/E[w²]`. The codec
minimizes MSE; preserving energy is a different objective.

### The FWHT rotation is NOT the cause

Checked and cleared. The scale is `1/16` (= `1/√256`) in all three places it
appears, with matching sign seeds (42, 1042):

- offline weight rotation — `e8_gptq.rs:57`
- online activation rotation — `gemv.rs:44`
- MQ4/MQ6 dequant inverse — `qwen35.rs:2481`

So `R = diag(s₂)·(H/16)·diag(s₁)` is orthogonal and `(Rw)·(Rx) = wᵀx` cancels
exactly. A standalone round-trip of that convention is orthonormal to 6.7e-16.

## Why it matters beyond DS4

DeepSeek V4 has `route_scale`, a **routed-branch-only** multiplier, which is
exactly the tier that shrinks — so the loss has been absorbed there and is
invisible. The shipped values are 1.8 (`.mq2r`) and 2.2 (other DS4 builds)
against a checkpoint value of 1.5, and the PyTorch reference scores PPL 4.693
*at 1.5*, confirming 1.5 is correct for the model and our routed branch is weak.

| build | shipped ratio | shrinkage accounts for | residual |
|---|---|---|---|
| `.mq2r` @ 1.8 | 1.2000 | 1.1265 | 1.065 |
| other DS4 @ 2.2 | 1.4667 | 1.1265 | 1.302 |

**Every other architecture on an MQ*-Lloyd tier has no such knob.** The defect
is invisible where it was found and live everywhere else.

## Proposal

Fold a constant gain of `1/retained` into the Lloyd codebook at quantization
time — i.e. scale the fitted centroids so the reconstruction preserves energy
rather than minimizing MSE.

**Not** a per-group gain. The spread is 1.25% sd, so a global constant captures
98.75% of the available correction; a per-group mechanism adds machinery for
almost nothing. This was proposed and withdrawn on the measurement.

Consequences:

- `route_scale` for DS4 can move back toward the checkpoint's 1.5.
- Models with no compensating knob gain the 11% outright.
- One constant per Lloyd format, applied at quantize time; no runtime change,
  no format change, no re-plumbing.

## Open questions — close these before changing any codebook

1. **Real weights, not synthetic.** The measurement used Gaussian input in the
   post-FWHT domain, which is what the codec is tuned for, but actual expert
   weights should be confirmed. Dequantize a routed expert from the shipped
   `.mq2r` and compare against the same tensor in the original checkpoint.
2. **Per-format constants.** Only MQ2-Lloyd was measured. MQ3-Lloyd and
   MQ4-Lloyd each fit their own codebook and need their own numbers — the
   shrinkage shrinks as bit-width rises, so MQ4's may be negligible.
3. **Does energy-preserving actually beat MSE-optimal for inference?** This is
   the real question and it is not obvious. Rescaling *increases* MSE by
   construction. The hypothesis is that a systematic magnitude bias compounds
   across 43 layers in a way that random error does not, but that must be shown
   end-to-end on PPL, not argued. Use the torch teacher (PPL 4.693) as the
   reference rather than the quants.
4. **The residual on the 2.2 builds.** Shrinkage explains 1.1265 of 1.4667,
   leaving ~1.30 from a second, unidentified effect.

## How to verify a change

Do **not** accept a codebook change on reconstruction MSE — it will get worse by
construction. Accept on end-to-end PPL and KLD against the torch teacher:

- teacher baseline `/mnt/scratch/quantization/deepseek-v4-flash-0731-teacher/`,
  `ref_fp8_1024.plog` sha256 `deb6f4b4…`, PPL 4.693
- teacher self-consistency floor: KLD mean 0.0404, top-1 ceiling **0.9297**
- current MQ2R at `route_scale` 2.0: PPL 9.254 on the same tokens

A win is PPL below 9.254 *with* `route_scale` at 1.5, which would show the
compensation was genuinely replaced rather than duplicated.

## Suggested test expansion

The two existing measurements are inline in the crate they test
(`e8::tests::e8_shrinkage_on_fwht_gaussian`,
`mq2_lloyd_shrinkage_on_routed_expert_tier`), both informational with
sanity-only assertions. If this work proceeds, they should become a proper
per-tier sweep covering every shipped quant format, reporting retained energy
and per-group spread for each, so the constants are derived from one table
rather than one-off probes.
