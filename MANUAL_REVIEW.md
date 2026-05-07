# Manual Review Queue

Items that require Kaden's judgment. Sorted by what unblocks the most
downstream work.

## REQUIRES-KADEN-DECISION: ZAYA1 port recurrent-state shape (Phase 6)

- **What's escalated:** First per-layer recurrent state in hipfire that
  lives across decode steps. CCA (Compressed Convolutional Attention)
  carries two per-ATT-layer per-sequence state buffers across decode
  steps (~370 KB total per sequence at fp16 for ZAYA1-8B):
   - `conv_states[40, B, 1280, 2]` fp16 - causal Conv1d-along-time
     circular buffer (40 = ATT-layer count, not 80; ZAYA1 alternates
     ATT and MLP sub-layers).
   - `prev_hs[40, B, 2048]` fp16 - 1-step lag of input hidden state.
- **Why escalated:** the contract for the overnight ZAYA1 port intake
  forbids autonomous structural runtime changes. Three coupled
  decisions need your call before the implementing PR can land.
- **Decisions needed:**
   1. **State location.** Option A (per-arch State carries the
      buffers, zero hipfire-runtime churn, fits in arch crate) vs
      Option B (first-class recurrent-cache primitive in
      hipfire-runtime parallel to KV pager). Recommendation: A first,
      migrate to B when a second recurrent arch arrives. Cost delta
      for B: roughly +1 week.
   2. **First-ship spec-decode policy.** AR-only or recurrent-spec-decode
      from day one. Recommendation: AR-only first; ZAYA1 loses the
      1.5-3x DFlash multiplier on first ship in exchange for
      structural simplicity. Recurrent spec-decode added in a later
      PR with proper state fork/commit primitives.
   3. **Paging policy.** HBM-pinned recurrent state vs paged.
      Recommendation: HBM-pinned. ~370 KB per sequence at fp16; even
      256 sequences only consumes 95 MB pinned per device.
- **Predicates already landed on this branch:**
   - `00-cca-disambiguation.md` (verdict: RECURRENT, with verbatim code
     evidence).
   - `02-mod-design.md` (no runtime changes needed).
   - `03-eda-identification.md` (no recurrent state).
   - Phase 1 arch crate scaffold (compiles, 5 unit tests pass).
- **Spec doc:** `docs/investigations/2026-05-07-zaya1-port-intake/01-recurrent-state-design.md`
- **Effort to implement Phase 6.A** (Option A path): ~2 weeks calendar.
  End-to-end ZAYA1 first coherent decode: ~5-6 weeks.
- **Sequencing:** This PR (`feat/zaya1-port-intake`) is intake-only.
  No code touches the recurrent state primitives until you land this
  decision. Phases 1-5 (arch crate scaffold, free components, MoD
  design, EDA design) can proceed independently and have begun.

## Full coherence-gate / speed-gate run hangs in this session

- **Why escalated:** Both `scripts/coherence-gate.sh` (no flags) and
  `scripts/speed-gate.sh --fast` ran past 20 min in this session
  without producing a fresh report or registering GPU activity. Single-
  shot Qwen3.5-4B.mq4 generate via the daemon works fine end-to-end
  (verified: prefill 535 tok/s, decode 167 tok/s, no `pflash` field on
  off-default), so it doesn't look like a PFlash regression. More
  likely a local env quirk (sequential ~73 GB load disk thrash, stale
  pool state, or background process holding the GPU).
- **What was tried:** Started both gates in background, polled for
  reports, watched ROCm-SMI (GPU idle 0%). Killed and confirmed no
  daemon process still holding the lock.
- **Suggested next step:** Re-run from a fresh shell after the next
  session reset; if it still hangs, bisect by running the gate's per-
  model curl one at a time to isolate which load stalls.
- **Files touched:** none (gate is shipped; environment issue only).
- **Commits:** Phase 5 partial documented in PFLASH_LOG.md; PFlash off-
  default smoke on Qwen3.5-4B did pass.

## ~~Drafter availability for Qwen3.5/3.6 targets~~ RESOLVED

Original concern: qwen3-0.6b (vocab 151743) and Qwen3.5 family targets
(vocab 248320) have incompatible tokenizers.

Resolution: qwen3.5-0.8b is the smallest matched-vocab member of the
Qwen3.5 family and was already on disk. The blocker was that
`pflash::load_drafter` only supported plain LLaMA-family loading.
Adding a `DrafterModel::{Plain, Hybrid}` enum routes hybrid drafters
through `qwen35::load_weights` + `qwen35::forward_prefill_batch`
without changing the Q8 K cache layout (so the GPU score kernel is
unchanged).

Verified end-to-end via daemon stdio: qwen3.5-4b target +
qwen3.5-0.8b drafter → tokenizer_compat=true, 565→181 (32%),
target prefill 3118 tok/s on compressed, decode 157 tok/s. Done.

## PFlash score kernel produces NaN at ~21K source tokens (32K NIAH)

- **Why escalated:** PRD §6 Phase 5 requires NIAH PASS at 32K. With
  qwen3.5-4b target + any drafter (qwen3.5-0.8b, 2b, or 4b self), the
  bench cleanly bypasses with
  `ScoringDegenerate { non-finite scores: 337 NaN, 0 inf }` on the
  21551-token niah_32k.jsonl source. 16K (10881 tokens) and 8K (5487
  tokens) are clean and PASS.
- **Diagnostic surface:** every block (337 of 337) returns NaN, so the
  issue is global, not edge-block. Suggests:
   1. drafter forward producing NaN K cache at long source length
      (RoPE / softmax / DeltaNet recurrent state numerical issue), OR
   2. the pflash_score_q8_kv.hip kernel has a numerical-instability
      regime that triggers above ~16K source tokens.
- **What works:** baseline (full prefill, no PFlash) on niah_32k.jsonl
  PASSes in 13.4s with the same 4B target via the same forward path.
  So target arithmetic at 32K asym3 KV is fine. The drafter path is
  the isolation point.
- **Suggested next step:** add a debug dump in
  `pflash::compute_scores_batched_gpu` that downloads the drafter K
  cache contents at e.g. block 100 and inspects for NaN at the source.
  If K is finite but scores aren't, the score kernel has a long-context
  numerical bug. If K itself is NaN, the issue is in the drafter
  forward at long context.
- **Files touched:** none (debug dump is the next investigative step).
- **Commits:** Phase 5 NIAH PASS documented in PFLASH_LOG.md for 8K +
  16K only. 32K listed as known-bypass-degenerate.
