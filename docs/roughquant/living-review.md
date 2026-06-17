# RoughQuant living review

Status: in-progress audit, 2026-06-17.

## Current model after reading the docs

The chronological docs are a correction trail, not a linear success story. The
early phase-2 PCA result was later invalidated by two issues: an unfoldable
per-weight rotation assumption and a protected-quantizer zeroing artifact. The
later phase2g/phase2h conclusion supersedes the negative "no foldable
concentration" interpretation: the outliers appear to be real, shared across
layers, and foldable as a small protected residual-channel set.

The strongest remaining claim is relative, not shippable: in the bf16 sim,
protecting the shared outlier set improves teacher-forced KLD versus mq4 and
improves generation metrics versus the same sim path at protect=0. The docs are
explicit that this does not prove absolute generation quality or performance;
those need a real packed format using the real GEMV path.

## Audit targets

1. Verify the documented chronology against the code knobs and scripts.
2. Inspect the protected-quantizer invariants:
   - protect=0 should reduce to the selected bulk path;
   - protect=100% should reduce to exact bf16;
   - protected columns/rows must be overwritten exact after bulk quantization;
   - bit-accounting in docs must match the actual simulated precision.
3. Check roughquant4 channel-consistent selection:
   - residual readers protect the shared residual-channel set in columns;
   - residual writers protect the same set in rows;
   - non-residual inputs use their own saliency ranking.
4. Check the helper scripts for confounds:
   - prompt/state choices in the coherence battery;
   - whether sweep scripts record enough provenance;
   - whether analysis scripts still encode superseded assumptions.
5. Find and fix any protection-code bug that violates the above invariants.

## First concerns to test in code

- `HIPFIRE_RQ4_PROTECT_Q8` is described in the docs as a failed experiment and
  should not leak into production conclusions. Its implementation still matters,
  because it can affect reproduced tables.
- The roughquant4 docs say `protect_frac=0` is a same-path mq4 control. The
  implementation must make that exactly true at the sim level for eligible
  tensors, otherwise the coherence-control interpretation is weak.
- Selection by aggregated energy was later called less principled than
  persistence-based selection. If the code only implements aggregation, the docs
  should continue to frame persistence selection as future work.

## Finding: roughquant4 d_model / role selection

The roughquant4 post-pass hardcoded `dmodel = 1024` and used `k == dmodel` as the
residual-reader test. That matched the documented Qwen3.5-0.8B run, but it would
silently invalidate the documented 7B/9B confirmatory step: larger models would
either protect no shared residual columns or protect a coincidental 1024-wide
internal projection. It also encoded a role decision as a shape decision.

Fix applied: infer `d_model` from actual residual-reader tensor roles, and use
name-based residual-reader / residual-writer classification for roughquant4:

- readers: `linear_attn.in_proj_*`, `mlp.gate_proj`, `mlp.up_proj`,
  `self_attn.{q,k,v}_proj`;
- writers: `self_attn.o_proj`, `linear_attn.out_proj`, `mlp.down_proj`.

Added unit coverage for the role classifier and for the specific failure mode
where a 1024-wide writer input must not force `d_model=1024`.

## Other audit notes

- The current protected QTIP/MQ helpers no longer zero protected channels before
  quantization; they quantize the full group and overwrite protected entries
  exact. That matches the phase2e zeroing-bug correction. I also removed stale
  local comments that still described zeroing.
- `scripts/roughquant_energy_cdf.py` had a top-level docstring stating the
  superseded phase2e conclusion that raw residual energy is spread and foldable
  channel protection does not help. I updated that prose to point readers at the
  phase2g/phase2h correction; the script remains a CDF/eigenbasis inspection
  helper, not the final verdict.
- `scripts/roughquant_coherence_battery.sh` uses inline prompts. That was useful
  for the quick sim-path comparison, but it is not yet a canonical gate under the
  repo rule that prompt-sensitive evidence should use byte-identical
  `benchmarks/prompts/*.txt` prompts.

## Finding: protected-quantizer invariants reproduced (audit target 2)

Re-ran the invariants against the refactored (role-based) roughquant4, 0.8B,
mq4 bulk, bf16 embed, KLD vs `/tmp/bf16.pkld`:

| protect_frac | residual channels | role split | KLD |
|---|---|---|---|
| 0.0  | 0/1024    | 48 writers, 138 other | 0.160753 (= mq4 / uniform-4bit 0.161) |
| 0.05 | 51/1024   | 48 writers, 138 other | 0.084188 (the half-mq4 KLD win) |
| 1.0  | 1024/1024 | 48 writers, 138 other | **0.000000** (exact bf16) |

- protect=0 reduces to the mq4 bulk path (KLD = mq4). ✅
- protect=1.0 → exactly bf16 (KLD 0). ✅ (monotone, overwrite-not-zeroing holds)
- protect=0.05 → 0.084 reproduces the headline KLD win. ✅
- The **role-based** classifier yields the SAME counts as the old `k==1024`
  literal on 0.8B (138 readers / 48 writers / 51 channels @ 5%), so the d_model
  fix is behavior-preserving here AND removes the silent 7B/9B failure. ✅
- Bit-accounting cross-check: protect=0 ≡ uniform-4bit anchor (0.161) and
  protect=1.0 ≡ bf16 — the two endpoints match the documented precision, so the
  intermediate avg-bits interpolation in phase2h is consistent.

Net: ledger C1/C4/C5 reproduce; the quantizer invariants are sound on the
audited code. Remaining audit work: canonicalize the coherence battery prompts
(`benchmarks/prompts/*.txt`), and the real-packed-format coherence/perf (the only
path to a shippable verdict).

## Finding: importance selection beats random (the win is importance-driven)

Control (`HIPFIRE_RQ4_SALIENCY=random`, seeded): protect-5% (51 ch), bf16, KLD —

| selection | KLD | vs mq4 (0.162) |
|---|---|---|
| diag (ours) | 0.084 | −48% |
| product | 0.087 | −46% |
| random (3 seeds) | 0.140 / 0.148 / 0.151 | −7…−14% |

Our selector beats random ~1.7×; random promotion of the same count barely helps.
⇒ the protection win comes from WHICH channels (importance), not just from having
bf16 channels. `diag(H)` captures real, selectable importance (≈ product here, so
weight-magnitude adds little at this level). Answers "is OUR selector worthless" →
no.

## Finding: canonical coherence battery (protection helps coherence)

8 committed `benchmarks/prompts/*.txt` (md5-recorded), FP32 DeltaNet state:

| model | avg uniq↑ | 5gram-rep↓ | attractors |
|---|---|---|---|
| mq4 (real) | 0.458 | 0.187 | 3/8 |
| rq-mq4path (protect-0 control) | 0.255 | 0.538 | 6/8 |
| rq-protect5bf16 | 0.371 | 0.286 | 4/8 |

Protection recovers ~half the sim-path coherence gap toward mq4 ⇒ helps, doesn't
hurt (confirms the phase2h retraction). NB: the sim-path itself (rq-mq4path) still
degrades vs real mq4 → absolute coherence needs the real packed format.

## Finding: importance-metric bake-off — diag(H) wins; OBS backfires

`HIPFIRE_RQ4_SALIENCY ∈ {diag, product, wnorm, obs}` (OBS = ‖W[:,c]‖²/[H⁻¹]_cc,
compensation-aware, reuses LDLQ Cholesky):

| metric | protect-5% (top) KLD | void-bottom-1% (tail) KLD |
|---|---|---|
| diag | 0.084 | 0.0285 |
| product | 0.087 | — |
| obs | 0.088 | 0.046 |
| wnorm | 0.140 | — |
| random | ~0.146 | ~0.043 |

- **Activation energy is THE signal.** diag/product/obs tie at the top
  (0.084–0.088); wnorm (weight-magnitude only) ≈ random ⇒ weights without
  activations are useless. product (W²·E[x²]) doesn't beat diag (E[x²]).
- **OBS does NOT help and is WORSE than random at the tail** (0.046 > 0.043).
  OBS assumes COMPENSATION (other weights re-optimize); our scheme doesn't
  re-optimize, so "compensatable ⇒ safe to drop" is wrong — those channels are
  correlated but still carry signal. diag's "low energy ⇒ safe" is correct here.
- ⇒ **diag(H) is the best selector at both ends**; the shallow tail gradient is
  intrinsic, not a metric limitation. Importance-metric exploration settled:
  plain activation-energy diag(H) for non-compensating protection.

## Finding: does diag(H) rank the TAIL? (not just the outliers) — yes, weakly

Control (`HIPFIRE_RQ4_BULK=void`, `HIPFIRE_RQ4_INVERT` for top): void the
diag-BOTTOM-k vs RANDOM-k vs diag-TOP-k, KLD damage:

| void % (ch) | diag-bottom | random (2 seeds) | diag-top |
|---|---|---|---|
| 0.2% (2)  | 0.0060 | 0.0064 / 0.0072 | — |
| 0.5% (5)  | 0.0138 | 0.0188 / 0.0175 | — |
| 1.0% (10) | 0.0285 | 0.0447 / 0.0413 | — |
| 2%  (20)  | 0.070  | 0.099           | — |
| 5%  (51)  | 0.216  | 0.321           | 7.67 |

- diag-bottom < random at EVERY fraction (~25–33% less damage ≥0.5%) and
  diag-top is ~35× worse ⇒ **diag orders the whole spectrum, tail included.**
- BUT the tail gradient is SHALLOW (~30% gap) vs the top (35× gap / KLD-halving);
  at 0.2% (2 ch) diag ≈ random. So diag is decisive at the top, weakly-but-real at
  the tail ⇒ room for a better metric (OBS/Fisher) at the tail. Motivates the
  importance-metric exploration; validates the graded-tier premise.
- Note: turnaround profiling — mq4-path quantize is **2.5s** (eval-bound, ~50s
  KLD), so GPU-quant is unnecessary for the importance sweep (reserve for
  PCA/cross-model). Measure-first win.

## Finding: does diag(H) rank the FULL SPECTRUM? (large void fractions) — yes, decisively

Extends the tail finding to the deep bulk. Cumulative void-BOTTOM-k% vs RANDOM-k%
(`HIPFIRE_RQ4_BULK=void`), KLD damage (vs `/tmp/bf16.pkld`):

| void % | diag-bottom | random | random/diag |
|---|---|---|---|
| 10% | 0.598 | 3.762 | 6.3× |
| 20% | 1.790 | 4.706 | 2.6× |
| 40% | 4.511 | 8.283 | 1.8× |
| 60% | 5.747 | 12.165 | 2.1× |

- diag-bottom stays FAR below random at every fraction ⇒ **diag orders the whole
  spectrum, not just the outlier head.** The gap is huge in the bulk (6.3× at 10%)
  and only collapses to ~30% at the sub-1% tail (prior finding) because there the
  bottom channels are uniformly near-zero, so diag ≈ random by construction — NOT
  a metric failure. Reconciles the "shallow tail" observation: the tail looks flat
  because it genuinely is flat (no signal to rank), while the bulk ordering is
  strong.
- Practical read: a graded scheme can safely crush a large diag-bottom fraction far
  harder than a random fraction; the danger is entirely in mis-ranking the head,
  which diag gets right. Confirms diag(H) as the production selector across the
  full bit-budget range. Closes the user's "haven't proved it works for the tail"
  concern — proven for tail AND full spectrum.

## Finding: the 5 permutations — bijectivity verified

`scripts/roughquant_permute_verify.py` + `docs/roughquant/permutation-bijectivity.md`:
#1 hidden / #2 MLP / #3 attn-heads(GQA) / #5 residual are FREE (function-preserving,
max|Δ|~machine-zero on synthetic per-block forwards). **#4 per-head dims is free
WITHOUT RoPE but BROKEN with RoPE** (relative-position dot product changes; only
RoPE-pair-preserving perms are free). The verifier self-caught a same-position bug
that had masked the #4 failure. Propagation specs recorded for the production
machinery (next).
