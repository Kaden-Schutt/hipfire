<!-- SPDX-License-Identifier: Apache-2.0 -->
# Where hipfire's quantization has math headroom left

Date: 2026-08-02 · Status: analysis, no work started

## The one-line answer

**2-bit is where the headroom is. 4-bit is saturated.** And hipfire already owns
the machinery to close the 2-bit gap — it is currently spent on the tier that
has nothing left to win.

## Measured position

Retained energy `E[ŵ·w] / E[w²]` through the production codecs, measured on
post-FWHT-domain Gaussian input (`crates/hipfire-quantize`, tests
`e8_shrinkage_on_fwht_gaussian` and
`mq2_lloyd_shrinkage_on_routed_expert_tier`):

| tier | rate | codec | retained |
|---|---|---|---|
| routed experts, qt=19 | 2 bpw | scalar Lloyd-Max | **0.8877** |
| shared expert + `ffn.gate`, qt=35 | ~4.25 bpw | E8 lattice + per-32 scale | **0.9998** |

The routed tier is the overwhelming majority of the model: 33,024 tensors,
277,025,390,592 elements.

## Reading it against theory — with the caveats that matter

For a Gaussian source under MSE, the Shannon bound is `D/σ² = 2^(-2R)`.
Practical quantizers pay a space-filling loss above it: scalar ≈1.53 dB, E8
lattice ≈0.65 dB better than scalar, Leech (24-D) ≈1.03 dB better, trellis/TCQ
≈1.2 dB better.

**Caveat 1 — the 1.53 dB figure is a high-rate asymptote and overstates at
2 bits.** The asymptote predicts retained 0.9111 for scalar at 2 bpw; the true
Lloyd-Max 4-level value is 0.8825 (MSE 0.1175). Our measured 0.8877 sits on the
*true* value. **MQ2-Lloyd is achieving near-optimal scalar quantization — it is
not broken**, it is simply scalar.

**Caveat 2 — the 4-bit row cannot be compared to the Gaussian bound.** 0.9998
exceeds the Shannon bound (0.9961), which is impossible for a stationary
Gaussian. `MFP4G32E8SOA` carries a per-32 scale, so effective rate exceeds
4 bpw and group scaling exploits local magnitude structure the bound assumes
away. The honest conclusion is only that this tier is *saturated*, not that it
beats theory.

## Why 2-bit is the target

Low rate is exactly where lattice and trellis VQ beat scalar by the most. The
QuIP# / QTIP line of work reports its largest gains at 2 bits, not 4 — the
space-filling advantage is a bigger share of a smaller budget.

And the asymmetry in what we deploy is stark:

- **E8 lattice VQ is already in-tree** (`crates/hipfire-quantize/src/e8.rs`) and
  is spent on the 4-bit shared tier, which measures 0.9998 and has nothing left.
- **Scalar Lloyd runs on the 2-bit routed tier**, which is ~all of the model.

## Proposal

Move the 2-bit routed tier from scalar Lloyd to lattice or trellis VQ. It splits
cleanly across the two resources:

**Reasoning / derivation.** Choose the construction. E8 at rate 2 (32 codewords
over 8 dimensions), a Leech-derived code, or a trellis code — each trades
codebook size, decode cost, and achievable distortion differently. Decode cost
matters: the routed tier is read every token for six of 256 experts, so a
construction that wins on distortion but loses on GEMV throughput is not a win.
This is derivation work with a right answer.

**MI300X time.** Fit codebooks on real post-FWHT expert weights, sweep
constructions, and measure end to end.

## What makes this newly tractable

Until today there was no trustworthy reference to measure quality against — the
quants were compared to each other, or to a broken parent. There is now a
verified teacher:

- `/mnt/scratch/quantization/deepseek-v4-flash-0731-teacher/`
- `ref_fp8_1024.plog` sha256 `deb6f4b4…`, **PPL 4.693**
- self-consistency floor: KLD mean **0.0404**, top-1 ceiling **0.9297**
- current MQ2R at `route_scale` 2.0: PPL 9.254 on the same tokens

So any quantization idea becomes falsifiable in one run: fit, quantize, capture
logits, KLD against the teacher. Distortion numbers tie to actual quality
instead of being proxied.

**Acceptance for any new construction:** beat MQ2R's 9.254 PPL at equal or
lower bpw, measured on the pinned tokens against the teacher. Do not accept on
reconstruction MSE alone — see the sibling plan on Lloyd shrinkage for why MSE
and end-to-end quality diverge.

## Related, and a prerequisite to keep honest

`docs/plans/2026-08-02-lloyd-shrinkage-gain.md` — the same 2-bit tier loses 11%
of its energy per round-trip, uniformly (sd 1.25%). That is a *bias*, separate
from the *space-filling* question here. A new construction should fix or avoid
the bias rather than inherit it, and the two effects must not be conflated when
attributing an improvement.

## Second candidate, different flavour

**DDTree tree-shape optimization.** Given a draft model's per-position accept
distribution, choosing the tree that maximizes expected accepted tokens per
target forward under a node budget is a clean constrained-optimization problem.
`AGENTS.md` records DDTree on gfx1100 as a structural perf regression with
Path C (trained custom draft) and Path D (stale-context overlap) unresolved.

MI300X time would measure accept distributions; the tree derivation is pure
reasoning. Lower expected value than the quantization work because it affects
throughput rather than quality, but it is well-posed and currently stuck.
