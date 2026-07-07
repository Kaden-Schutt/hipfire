---
title: EP/minimax daemon overcounts conversation_tokens by 1 on stop-sequence exit
date: 2026-07-07
tags: [daemon,multi-turn,kv-cache,minimax,462]
---

**Latent, pre-existing, minor. Not yet fixed.** Found 2026-07-07 while validating dense TP/PP multi-turn KV reuse (device-mesh item B).

## Symptom
The EP serve loop (`generate_ep`, minimax arm) can bake one token into `m.conversation_tokens` that was never written to the KV cache, on a **stop-sequence** exit. On the next turn `plan_prompt_cache`'s LCP can then extend through that un-materialized slot → the reused KV prefix attends over one stale/unwritten entry. This is the same #462 class ("`conversation_tokens` must exactly mirror the tokens materialized in the KV") as the dense-path bug fixed in item B.

## Mechanism (`crates/hipfire-runtime/examples/daemon.rs`, minimax decode loop ~3702-3768)
- `~3725 m.conversation_tokens.push(next)` runs **before** the KV write.
- `~3726 ep_emit_token(...)` returns true on a **stop-sequence** match → `break` at `~3727`, which is **before** `forward_ep(next, pos)` at `~3739`. So on stop-match, `next` is in `conversation_tokens` but not in the KV.
- `max_tokens` exit is **safe** here: the `while generated < max_tokens` check is at the loop top, *after* `forward_ep`, so the last counted token was materialized. EOS (`~3720`) breaks before the push — also safe.
- Net exposure: **stop-sequence exits only** (narrower than the dense path's, which skewed on both max_tokens and stop).

## Fix pattern (mirror the dense-path fix)
Track a `materialized`/committed list (push right after a successful `forward_ep`) and bake that into `conversation_tokens`, OR move the `push` to after the KV write. See the dense-path fix (commit `25f9046f`, `generate_dense` returns `materialized` not `history`) for the exact shape.

## Notes
- Only bites when: EP/minimax serve + a client `stop` sequence + a following turn that shares the prefix (multi-turn reuse). Stop-sequences are absent from the coherence/serve gates, so it hasn't surfaced in CI.
- The single-GPU **dflash** path does NOT have this (its `emitted` = spec-committed tail = KV-safe; can only undercount).
- Same review that found this proved the dense-path fix closed a real *new* gap (not a shared-with-reference skew).
- Related: device-mesh item B (dense TP/PP multi-turn KV reuse) in `device-mesh-pivot-execute-steps-spine.md`.
