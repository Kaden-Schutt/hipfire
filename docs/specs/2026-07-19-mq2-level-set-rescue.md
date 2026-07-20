<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# MQ2 quality rescue: level-set geometry first, per-block codebook second

| Field | Value |
|---|---|
| State | **planned** |
| Date | 2026-07-19 |
| Author | Kaden Schutt (drafted with agent assistance) |
| Supersedes/extends | `docs/plans/mq-sub4bit-research-queue.md` Q0/Q1 ordering (queue statuses stale; MQ3-Lloyd has since been implemented and ppl-validated) |
| Validation route | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Key external input | arXiv:2606.02823 (Qift), arXiv:2605.26339 (QAM-W), arXiv:2606.23406 (HyperQuant) |

## 1. Background

The 2026-04-30 sweep established that MQ2 fails at every tested size (symbol
soup / mojibake) and MQ3 collapses below 9B, with engine wiring proven
correct (32 generations, 0 hard errors, monotonic quality vs size). The
queue's Q1 proposes a per-block 4-entry Lloyd-Max codebook (qt 19,
72 B/group, kernel `gemv_mq2g256_lloyd.hip`, est. 1–2 weeks). Q1.5 has since
validated the same algorithm at 3 bits: MQ3-Lloyd halves the perplexity gap
to MQ4 across sizes (9B: 18.52 vs MQ4 10.34, a 1.79× gap vs uniform-MQ3's
4.07×).

Qift (arXiv:2606.02823) changes the order of operations. Studying
Hadamard-rotated W2 pipelines — MagnumQuant's exact setting — it shows:

1. Pretrained weights are already near-zero-centered across all linear
   modules; the rotation primarily Gaussianizes (excess kurtosis drops by
   orders of magnitude).
2. The W2 collapse is a **reconstruction-level problem**, not only a
   bit-width problem. Conventional asymmetric W2 (scale + zero) recovers a
   large fraction of quality without any codebook.
3. A fixed **no-zero level set** `{±0.5, ±1.5}` (equivalently `{±1, ±3}`
   half-scale; power-of-two variant `{±1, ±4}` for sign-and-shift decode) is
   training-free, codebook-free, group-grid-free, and zero-point-free.
4. Its scale-invariant ratio analysis (inner/outer centroid ratio 0.25–0.33)
   explains *why* Lloyd-Max, NF2, and mirror-no-zero level sets all work —
   they all land near the Gaussian-optimal 2-bit reconstruction levels
   (Max 1960), while `{±1, ±2}` does not.

Consequence: after FWHT + sign randomization, per-block weights are
approximately Gaussian, so the Gaussian-optimal fixed levels are available
**as a constant change in the dequant mapping** — no new kernel shape, no
header-format change, no quantizer iteration loop. Per-block Lloyd then
captures only the residual non-Gaussianity. The A/B between the two
measures exactly how much that residual is worth on our fixtures.

## 2. Work items (ordered)

### W0 — Queue-doc reconciliation (day 0, docs-only)

Update `docs/plans/mq-sub4bit-research-queue.md`: mark Q1.5 implemented
(qt 20, gated by `HIPFIRE_ALLOW_MQ3_LLOYD=1`), insert this spec as the Q0.5
arm, note the Qift/QAM-W/HyperQuant citations, and correct PRD §5.4 framing
per the queue's own out-of-band note. No claim-state promotions: everything
here stays **planned** until sweep-measured.

### W1 — Qift level-set MQ2 arm (day-scale)

1. Quantizer: `quantize_mq2g256_qift` in `crates/hipfire-quantize/src/main.rs`
   — identical to uniform MQ2 except: (a) drop the zero-point; (b) quantize
   to nearest level in `{-1.5, -0.5, +0.5, +1.5}`; (c) per-group scale
   fitted as `E[|w|] / 1.0`-class moment match (or least-squares against the
   rotated block; record the choice). New qt id; do not overload qt 19.
2. Kernel: no new shape. The uniform `gemv_hfq2g256`-family decode line
   `recon = scale * q + zero` becomes `recon = scale * L(q)` with `L` as a
   4-entry fp16 constant (immediate or `__constant__`). If the existing
   header layout is retained, zero is simply ignored — **bytes/param
   identical to uniform MQ2, bit-exact same bandwidth budget**.
3. Optional second variant: PoT `{±1, ±4}` sign-and-shift decode for the
   dp4a path if the fp16 multiply shows up in kernel time.
4. CPU reference + kernel-parity check per the queue's Q0 plan (the Q0
   bit-exactness harness is reused, not skipped).

### W2 — Per-block Lloyd MQ2 arm (week-scale, per queue Q1)

Implement qt 19 exactly as specced in the queue (4 fp16 centroids per
256-block, sorted at quantize time; `gemv_mq2g256_lloyd.hip` with
register-resident `half4 cb` lookup; +2 VGPRs). Fall back to uniform for
low-variance blocks if the outlier-block risk materializes (queue §Risks).

### W3 — Sweep and decision gate

Fixtures (frozen): Qwen3.5 2B / 4B / 9B artifacts; wikitext2-test ppl at
ctx=2048, warmup=8, scored=2039 (the Q1.5 protocol); the 4-prompt fluency
battery from the 2026-04-30 sweep; byte-identical prompts from
`benchmarks/prompts/`.

| Arm | Question |
|---|---|
| MQ2 uniform (current, refused-by-default per #109) | baseline |
| MQ2-Qift (W1) | does level-set geometry alone rescue W2? |
| MQ2-Lloyd (W2) | what is residual non-Gaussianity worth? |
| MQ4 | quality ceiling / gap reference |

**Decision gate:**

- If MQ2-Qift closes ≥ 70% of the uniform→Lloyd ppl gap at any size, ship
  MQ2-Qift (cheaper path, no per-block codebook maintenance) and keep
  MQ2-Lloyd experimental.
- If neither arm produces fluent 9B output and ppl gap vs MQ4 ≤ 6%
  (Q1 acceptance), **kill MQ2 with evidence**: keep refuse-by-default,
  record the negative result in the queue doc, redirect effort to the
  MQ3-Lloyd sub-9B stack (Q2 GPTQ escalation / Q4 mixed precision per the
  queue).
- MQ3-Lloyd ungate (decode-perf recovery, 44 → ~140 tok/s on gfx1100) is a
  **separate, already-specced item**; do not couple it to this A/B.

## 3. Non-goals and constraints

- No activation-weighted (GPTQ-style) fitting in W1/W2 — that is Q2, and
  breaks the deterministic data-free property MQ formats currently have.
- No change to MQ4/MQ6/HFQ4 paths; new qt ids only; existing artifacts
  decode unchanged.
- QAM-W's own conclusion applies: at strict 4 bpw the rotated-codebook
  frontier (MQ4-Lloyd's family) is the right band; this spec does not touch
  4-bit formats. Its paired-KL↔ΔPPL correlation (Spearman 0.99 over 37 rows)
  licenses continuing to use `kld_logits`-style KL gating as the cheap
  quality proxy in the sweep.
- HyperQuant's lattice/Rice construction is out of scope for runtime
  kernels (decode-cost mismatch with the RDNA GEMV shape); its KV bias-
  correction is routed to the cross-arch campaign spec's long-context lever
  queue, not here.

## 4. Acceptance summary

| Gate | Criterion |
|---|---|
| W0 | Queue doc reflects implemented/stale states; no present-tense claims without measured fixtures |
| W1 | Kernel parity ≤ 1e-3 max-abs-err vs CPU reference; artifact bytes/param == uniform MQ2 |
| W3 | Decision recorded per the gate table with dated fixture manifests; sweep numbers land as a `docs/perf-checkpoints/` (measured) or queue-doc (historical) entry per INDEX lifecycle |
