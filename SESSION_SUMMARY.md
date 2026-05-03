# PFlash Movie-Night Session Summary

Session start: 2026-05-02 ~16:52
Session end: 2026-05-02 ~18:55
Branches: 3 deliverable branches off feat/89-llama-batched-prefill (master is the merge target after Kaden reviews).

## Deliverables completed

### 1. Auto-pick FullAttention layer (committed)

- Branch: `feat/pflash-auto-fullattn-layer`
- Commit: `0814e22` "feat(pflash): auto-pick first FullAttention layer for scoring"
- What changed:
  - Added `DrafterModel::score_layer_idx() -> Option<usize>` returning the
    smallest layer where `LayerType::FullAttention` (layer 0 for plain Qwen3,
    the first FullAttn slot for Qwen3.5/3.6 hybrid layer pattern).
  - `compute_scores_batched_gpu` and `compute_scores_batched` now default to
    this auto-picked index instead of `n_layers - 1`.
  - `HIPFIRE_PFLASH_SCORE_LAYER` env var preserved as an in-range escape hatch.
  - `pflash_load_demo` prints the picked layer per drafter.
- Per-family auto-pick: Qwen3.5-0.8b -> 3, Qwen3.5-27b -> 3 (matches the
  hand-set value previously known to unblock 32K).
- End-to-end verified, no env var:
  - 8K NIAH on 27B: PASS, total 4238 ms (parity with prior 4266 ms median).
  - 32K NIAH on 27B: PASS, total 25567 ms (parity with prior L=3 by hand).

### 2. 3-fresh-process statistical backing (committed)

- Branch: `feat/pflash-phase5-statistical-backing`
- Commit: `94fbd76` "docs(pflash): 3-fresh medians for Phase 5 perf table"
- What changed: ran `scripts/pflash-niah-bench.sh --runs 3` against all 12
  (fixture × mode) rows and updated PFLASH_LOG.md with medians + spread.
- Per-row median totals (qwen3.5-27b.mq3 + qwen3.5-0.8b.mq4 drafter):
  - niah_8k:        baseline 11768 ms -> PFlash 4278 ms  (-64%, 3/3 PASS)
  - niah_16k:       baseline 26333 ms -> PFlash 9106 ms  (-65%, FAIL -> PASS)
  - niah_multi_16k: baseline 26931 ms -> PFlash 10209 ms (-62%, 0/3 -> 3/3)
  - longcode:       baseline 33683 ms -> PFlash 12450 ms (-63%, FAIL -> PASS)
  - longprose:      baseline 18689 ms -> PFlash 6259 ms  (-67%, 3/3 PASS)
  - niah_32k:       baseline 64850 ms -> PFlash 25818 ms (-60%, FAIL -> PASS)
  - geomean speedup: -64%
- Spread observation: multi-16K PFlash had a single 5115 ms compress
  outlier in run 3 producing 27.8% total spread; verdict still 3/3 PASS.
  All other rows under 1% spread. Documented in the log; not a blocker.
- Spot-check: 8K (4278 vs prior 4266, +0.3%) and 16K (9106 vs prior 9071,
  +0.4%) confirm deliverable 1 didn't drift the perf claim.

### 3. Regression gate (committed)

- Branch: `feat/pflash-regression-gate`
- Commit: `af79784` "feat(pflash): regression gate with ±10% wall-clock and verdict match"
- What shipped:
  - `scripts/pflash-gate.sh`: runs the 12-row matrix against a baseline
    JSON, asserts verdict-match + ±10% wall-clock per row, exits nonzero
    on any regression with per-row diagnostics.
  - `scripts/pflash-baselines/gfx1100-2026-05-02.json`: seeded from the
    deliverable-2 medians; documents arch, ROCm, target/drafter pair,
    `keep_ratio`, `block_size`, runs-per-row, capture commit, tolerance.
  - `scripts/coherence-gate.sh`: invokes pflash-gate.sh as a follow-up
    stage after the daemon-coherence battery; skipped when target/drafter
    are absent or `HIPFIRE_SKIP_PFLASH_GATE=1`.
- Smoke result: 12/12 rows clean, max abs drift 3.3% (multi-16K PFlash),
  every row's verdict matched baseline, gate returns exit 0.

## Stretch deliverable not entered

### 4. Sparse drafter kernel (deferred)

Per contract escalation rule: with all three primary deliverables
committed at ~18:55 and ~5 minutes of buffer remaining before the
3-hour cap, the multi-day sparse-kernel scaffold was not entered.
Item is unchanged: scaffold + first-compile is the spike target,
and the existing dispatch swap inside `compute_scores_batched_gpu`
remains a one-liner once the kernel ships. PRD reference: §6 Phase 3
"Fall back to dense drafter attention below a configurable threshold,
initially 32K"; the `sparse_threshold` field is already plumbed in
PflashConfig from prior session work.

## Items 5-8 status (out of scope tonight, recorded for handoff)

- 5. 64K / 128K NIAH: still gated on item 4. Today's auto-FullAttn-layer
  unblock (item 1) likely makes 64K viable without sparse, but
  pretokenizing 64K at the engine's O(N²) tokenizer is ~24 min and
  128K is ~96 min; one-shot work that fits a future loop.
- 6. Deep-layer NaN root cause: NOT investigated. Auto-FullAttn-layer
  papers over by scoring from the shallowest layer; the deep-layer
  RoPE-OOD on small drafters at long source remains the underlying
  bug. Either YaRN-style RoPE scaling on the drafter or a longer-
  context drafter retrain would fix it. MANUAL_REVIEW.md has the
  bisect record from the prior session.
- 7. PFlash + DFlash composability: unchanged. Daemon still emits
  `pflash_bypass{dflash_decode_active}` when both modes are requested.
- 8. Per-genre keep_ratio sweep: unchanged. 0.30 holds across NIAH,
  multi-needle, longcode, and longprose; 0.10 / 0.15 sweeps remain a
  future experiment.

## Branch / commit ledger

- master  d1506d0 (unchanged this session)
- feat/89-llama-batched-prefill  cbd6bac at session start; +5 commits
  for the long-code fixture work this same session, then deliverables
  branched off:
- feat/pflash-auto-fullattn-layer    0814e22  (deliverable 1)
- feat/pflash-phase5-statistical-backing  94fbd76  (deliverable 2)
- feat/pflash-regression-gate        af79784  (deliverable 3, current HEAD)

The three deliverable branches are stacked: auto-fullattn -> phase5-stats
-> regression-gate. Merge in order. The pflash-gate baseline JSON
points at the post-94fbd76 numbers and will need a refresh whenever a
later kernel / dispatch / quant change shifts the medians by more than
±10%.

## Notes for next agent

- The regression-gate stage in coherence-gate.sh runs against the same
  target/drafter pair, so it's pinned to qwen3.5-27b.mq3 + qwen3.5-0.8b.mq4.
  Different arches need their own baseline JSONs (e.g. gfx1151, MI300X);
  the file naming convention `<arch>-<date>.json` should keep that clean.
- The multi-16K compress jitter (one 5115 ms outlier vs ~2300 ms typical)
  is the only soft signal in this session's perf data. If a future gate
  run flips that row to a regression, look at GPU thermal / DPM state
  before assuming a code change caused it.
- `HIPFIRE_PFLASH_SCORE_LAYER` is now strictly a debug knob. If a future
  drafter family has an unusual layer pattern, set it temporarily to
  bisect, but the auto-pick should be the source of truth in production.
