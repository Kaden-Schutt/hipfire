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
