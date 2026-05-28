# Devlog 2026-05-28 — Hetero MTP perf: was a bug in MY peer-copy, not RDNA2

**This entry supersedes my earlier analysis on the same day.** I claimed
RDNA2 numerical drift caused τ to collapse from 3.25 → 2.24 in hetero
MTP. The investigation (bit-diff of capture points in chain forward at
cycle 0/1) **falsified that hypothesis**: the MTP head produces
bit-identical outputs on gfx906 and gfx1031. Looking further, the bug
was in MY hetero spec function, not in RDNA2 kernels.

## The actual bug

Cycle-exit handoff (`spec_step_mtp_compressed_serial_hetero`) computed
`prev_hidden_row = advance - 1` correctly, but the peer copy IGNORED
it — it always copied from byte offset 0 of `verify_hidden` (= row 0 =
`last_committed`'s post-output-norm hidden), instead of row
`advance - 1` (the bonus token's hidden, which IS what next cycle's
chain should read).

Single-gpu's `capture_prev_hidden_from_verify_row` correctly uses
`memcpy_dtod_at(dst, 0, src, row*dim*4, dim*4)`. My hetero variant
called `memcpy_peer` (no offset param) and added `let _ = prev_hidden_row;`
to silence the unused warning. Compiler did not catch this; the only
symptom was τ collapse.

## How the diagnostic worked

Built tooling: `HIPFIRE_HETERO_DIFF=<prefix>` env that dumps three
points of every K step of every cycle into `<prefix>.{single|hetero}.posXXXX.kY.{prev_hidden,t_mtp_out,logits_compressed}.bin`.
Plus a `diff_f32_bin` example that reports bit-equal count, RMS, top-10
diverging indices.

Ran both single-gpu and hetero on the same prompt, then `diff_f32_bin`
on each cycle. Findings:
- **Cycle 0 (cur_pos=23) all K steps: 100% bit-equal** (5120/5120 +
  16384/16384 elements match across 4 chain steps)
- **Cycle 1 (cur_pos=24) all K steps: 100% bit-equal**
- **Cycle 2 (cur_pos=29) k=0 prev_hidden: 0% bit-equal** (RMS=2.6,
  max diff=27, argmax differs)

That was the smoking gun — the MTP block forward is bit-equal across
the two archs, but the INPUT to it (`prev_hidden`) diverged at cycle 2.
prev_hidden is set by the cycle-exit handoff. Reading my hetero
handoff code revealed the row-0 bug.

## Numbers after the fix

| metric                  | single-gpu (gfx906)  | hetero (gfx906+gfx1031) | delta  |
| ---                     | ---                  | ---                     | ---    |
| τ                       | 3.25                 | **3.25**                | match  |
| cycles for 66 tok       | 20                   | **20**                  | match  |
| accepted_mtp_total      | 45                   | **45**                  | match  |
| bonus_total             | 20                   | **20**                  | match  |
| replay_skipped          | 7 (35%)              | 7 (35%)                 | match  |
| **tok/s**               | **20.10**            | **18.13**               | **-10%** |
| output text             | identical            | identical               | ✓      |

## Real cross-device overhead

The 10% tok/s gap (3.28 s → 3.64 s decode wall over 20 cycles =
164 ms/cycle → 182 ms/cycle = +18 ms/cycle) is the actual cost of:

- 2 peer copies per cycle (20 KB prev_hidden + the per-step chain
  control flow that involves D2H token args)
- Drafter-side stream/event setup
- The 1.29 GiB token_embd mirror (one-shot at init, not per-cycle)

Microbench predicted ~112 µs/cycle for the cycle-exit handoff alone,
which would be 0.7% of cycle. We see 11%. The gap is probably the
per-step embedding D2H token argument (4 B but a host roundtrip per
chain step, K=4 per cycle = 4 round-trips × 30 µs each ≈ 120 µs/cycle)
plus the multiple cross-device active_stream operations. Still
reasonable — not a +12% gain over single-gpu, but not blocking. Hetero
MTP frees ~800 MB on gfx906 (the head + scratch + KV) at a -10%
tok/s cost.

## What I retract from the earlier devlog

Specifically: the claim "this re-orients the RDNA2-kernel-work
question: the direction is NUMERICAL EQUIVALENCE with gfx906's
output." That's wrong. **RDNA2 IS numerically equivalent on this
workload — every kernel called from the MTP head produces bit-equal
output to gfx906.** I had no evidence for the drift hypothesis; I
inferred it from circumstantial perf data without instrumenting.

The lesson for future hetero-class bugs: **don't speculate about
kernel-level numerical drift without first bit-comparing the
captures**. The diagnostic tooling we built (HIPFIRE_HETERO_DIFF +
diff_f32_bin) is now permanent; reach for it BEFORE forming a
kernel-perf hypothesis.

## What this means for the multi-GPU MTP project

Goes from "needs RDNA2 numerical-equivalence work" to "ships at -10%
tok/s, frees ~800 MB gfx906 VRAM, plumbing is correct." Whether to
keep paying that 10% in production depends on whether the freed VRAM
unblocks something else (longer context, larger model, parallel
PFlash decoder co-resident with MTP-on-drafter). Plumbing is done;
v1 sync split is genuinely viable.
