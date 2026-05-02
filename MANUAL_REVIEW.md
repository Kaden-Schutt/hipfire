# PFLASH Manual Review Queue

Items that require Kaden's judgment. Sorted by what unblocks the most downstream work.

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

## Drafter availability for Qwen3.5/3.6 targets

- **Why escalated:** Qwen3-0.6B (vocab 151743) and Qwen3.5-4B target
  (vocab 248144) have different tokenizers, so `tokenizers_compatible`
  returns false and `decide_bypass` returns `TokenizerMismatch`. This
  matches PRD §3.4 ("Refuse compression if tokenizer compatibility
  fails") so the MVP is correct, but it means we cannot demonstrate
  end-to-end PFlash on the available HFQ pair.
- **What was tried:** `pflash_load_demo qwen3.5-4b.mq4 qwen3-0.6b.hf4` →
  load OK in 358 ms, drafter VRAM estimate 439 MB, but tokenizer_compat
  false. The available local artifacts are `qwen3-0.6b.{hf4,mq4}` and
  `qwen3.5-{0.8b,2b,4b,9b,27b}.mq4` plus the 35B-A3B variants.
- **Hypothesis:** Qwen3.5/3.6 use an extended vocab (added thinking/tool/
  image special tokens) on top of Qwen3 base. The smallest member of the
  matched family is qwen3.5-0.8b — but that is a hybrid (DeltaNet+FA)
  model, so its drafter loader cannot be the plain `hfq::load_weights_hfq`
  path; it would need `qwen35::load_weights` and would put Q/K capture
  hooks inside `qwen35.rs`, conflicting with "do not break Qwen3.5".
- **Suggested next step:** Pick one of three:
  1. Quantize a Qwen3.5-tokenized small dense model from HF (e.g.
     Qwen2.5-0.5B if its tokenizer happens to match) as the drafter.
  2. Allow a hybrid drafter through pflash by lifting Q/K capture on the
     existing `qwen35::forward_prefill_batch` (additive, no behavior
     change for non-PFlash callers).
  3. Implement cross-tokenizer retokenization at the span boundary —
     drafter scores text-byte spans, then target-tokenizes the kept
     spans. Closer to the Lucebox cross-family approach but more work.
- **Files touched:** `crates/engine/src/pflash.rs`, `crates/engine/examples/pflash_load_demo.rs`.
- **Commits:** [pending — Phase 1.1].
