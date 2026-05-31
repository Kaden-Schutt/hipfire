# Outstanding: re-integrate DeepSeek4 daemon serving after the PR #352 merge

**Status:** OPEN follow-up. Branch `merge/master-pr352` (merge of
origin/master incl. PR #352 into the Stage-2b branch).
Created 2026-05-31.

## What happened

The merge brought origin/master (which includes PR #352 — Qwen MTP
device-resident token chain + GPU greedy-accept — AND the DeepSeek4
bring-up from other PRs #316/#318) into our Stage-2b branch.

`crates/hipfire-runtime/examples/daemon.rs` conflicted hard: master's
daemon and our Stage-2b daemon are two large parallel rewrites. Master's
daemon carries the DeepSeek4 serving glue; our daemon carries the Qwen
MTP / PpMtp dispatch (`pick_path`, `SpecPath`, `generate_mtp`,
`generate_multi`, `MtpSpecState` load-time alloc).

**Resolution chosen:** keep OUR daemon.rs as the base (commit
`40b1fe59`). Rationale: it preserves the path we actively bench (Qwen
27B single-GPU MTP + PpMtp), and #352's device-resident chain lives in
`mtp_spec.rs`/`mtp_head.rs` — which merged cleanly — so it is reachable
from our `generate_mtp` regardless of daemon shape. Verified:
single-GPU Qwen MTP-vs-AR = 1.18× (median of 3, 19.5→23.0 tok/s,
τ=3.15), up from 1.15× pre-merge.

## What is therefore MISSING from this branch's daemon

Our daemon.rs predates DeepSeek4, so the merge dropped master's
DeepSeek4 daemon-serving glue. Confirmed absent in `daemon.rs` on this
branch:

- `fn generate_deepseek4(...)` — the DS4 generate leaf (grammar-guided
  DSML tool-call masking, decoded-vocab cache, prefix-cache replay).
- The `if m.arch_id == 9 { generate_deepseek4(...); return; }`
  short-circuit at the top of the generate dispatch.
- `think_mode: ThinkMode` threading through the generate signature and
  the DS4 chat-template path.
- DS4-specific load params: `mtp_mode` / `mtp_k` / `mtp_weights_present`
  on `LoadedModel` and their wiring in the `load` handler (these are
  DS4's own MTP gating, NOT our Qwen `m.mtp`).

NOTE: the DeepSeek4 *crate* (`hipfire-arch-deepseek4`), its kernels, and
the arch itself ARE present and build on this branch — only the daemon
**serving** entry point is missing. DS4 inference via other entry points
(e.g. `deepseek4_chat` example) is unaffected.

## Re-integration plan (when picked up)

Source of truth for the DS4 daemon glue: `origin/master`'s
`crates/hipfire-runtime/examples/daemon.rs` (snapshot of the
#352-spine resolution also saved at `/tmp/daemon_352spine.rs` during
the merge session, but prefer `git show origin/master:...` — it is
authoritative and persists).

Steps:
1. Port `fn generate_deepseek4(...)` verbatim from origin/master into
   this branch's daemon.rs (it's self-contained — reads DS4 state, no
   dependency on the Qwen dispatch refactor).
2. Add the `arch_id == 9` short-circuit. In our `GenerateCtx`-shaped
   `generate_qwen35`, mirror the existing `arch_id == 7 → generate_qwen2`
   short-circuit: add `if m.arch_id == 9 { generate_deepseek4(...); return; }`
   BEFORE `pick_path`. The DS4 leaf takes flat args, not GenerateCtx, so
   unpack from ctx at the call.
3. Thread `think_mode`: our daemon's request loop must parse `think_mode`
   from the load/generate params (see origin/master's parsing) and pass
   it into `generate_deepseek4`. Our Qwen paths can `let _ = think_mode;`
   if they don't consume it (our think handling uses the
   `<think>`/`</think>` byte detector + `max_think_tokens`).
4. Add the DS4 load params (`mtp_mode`/`mtp_k`/`mtp_weights_present`) to
   `LoadedModel` and the `load` handler. Keep them disjoint from our
   Qwen `m.mtp` field — they are different MTP subsystems.
5. Build + run `scripts/coherence-gate-deepseek4-mtp.sh` (shipped by the
   merge) to confirm DS4 serving works.

Estimated: contained, ~1 focused session. No conflict with the Qwen MTP
path — the two dispatch families coexist (arch_id gates them apart).

## Cross-references

- PR #352 actual scope (Qwen MTP host-sync, NOT DeepSeek4): the device
  chain is in `mtp_spec.rs` (`mtp_gpu_greedy_accept_enabled_from_env`,
  `greedy_accept_from_argmax_i32`, `argmax_token_chain_f32`).
- Bench confirming the merge's Qwen MTP uplift: 1.18× median, recorded
  below the daemon resolution commit `40b1fe59`.
