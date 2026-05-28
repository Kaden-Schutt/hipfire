# Plan: `generate_mtp_multi` via unified generate-path refactor (Stage 2b)

Status: planning. Stage 2a foundation already shipped in commit
[upcoming]; this doc scopes Stage 2b — the daemon-serve dispatch for
PP+MTP — as a unified refactor of `generate_mtp` and `generate_multi`.

## Background

The Stage 2 audit (`docs/plans/pp_plus_mtp_audit.md`) projected ~450
LOC for the PP+MTP combo. Stage 2a built the foundation:
- `spec_step_mtp_compressed_serial_hetero` learned a same-device
  shortcut for the cycle-exit handoff (D2D memcpy when
  `target_gpu.device_id == drafter_gpu.device_id`).
- `peer_clone_tensor` / `seed_prev_hidden` got the same shortcut.
- `MtpState.drafter_state: Option<MtpHeteroDrafterState>` added.
- `load_model_pp` extended with MTP head load on `output_device`,
  token_embd mirror, and full state allocation.

Remaining: the daemon needs a serve dispatch that, when
`m.pp > 1 && m.mtp.is_some()`, runs the MTP spec loop with the
trunk verify on multi-gpu (PP path) and the chain on the drafter
gpu (which is `output_device` in our layout).

The naïve scope (a new `generate_mtp_multi` function) is ~600 LOC of
duplicated state plumbing. Option Y (chosen): unify `generate_mtp`
and `generate_multi` into ONE function parameterized on `Option<&mut
Gpus>` so the dispatch matrix collapses. Larger one-time touch
(~1000 LOC), but much smaller surface for the future fourth combo
(e.g. PP + DFlash, or PP + PFlash + MTP) and far less drift over
time.

## Current state of the four functions

| function | LOC | dispatch | trunk path | spec path |
| --- | --- | --- | --- | --- |
| `generate`        | ~600 | AR / DFlash / PFlash gate at the top | `forward_scratch` / `forward_prefill_batch` | n/a or DFlash spec |
| `generate_multi`  | ~486 | PP version of `generate` | `forward_prefill_batch_multi` + per-tok `forward_scratch_multi` | n/a |
| `generate_mtp`    | ~500 | MTP single-gpu | `forward_prefill_batch` | `spec_step_mtp_compressed_serial` |
| `generate_mtp_multi` (planned) | 0 | MTP PP+drafter-on-output_device | `forward_prefill_batch_multi` | `spec_step_mtp_compressed_serial_hetero` |

Total existing: ~1586 LOC across three functions, all with overlapping
prelude (prompt encoding, chatml framing, capacity check) and overlapping
postlude (token-emit format, done-event format, decode_tok_s
computation). Estimated duplication: ~300 LOC.

## Refactor target

Single `generate_qwen35` function with this control flow:

```
fn generate_qwen35(
    m: &mut LoadedModel,
    gpu: &mut Gpu,                              // dev 0 for both pp=1 and pp>1
    drafter_gpu: Option<&mut Gpu>,              // hetero MTP sibling, NOT pp_gpus
    stdout, id, prompt, system_prompt,
    temp, top_p, max_tokens, repeat_penalty, repeat_window,
    budget_alert_at_tok, budget_alert_text, max_think_tokens,
    assistant_prefix,
    pflash_state, pflash_cfg,
    tools, messages_history,
) {
    // 1. Encode prompt + ChatML / Jinja frame (common, ~100 LOC)
    // 2. Capacity check + auto-reset (common, ~30 LOC)
    // 3. Pick path:
    //    let mode = match (m.pp > 1, m.mtp.is_some(), m.dflash.is_some()) {
    //        (false, false, false) => Path::AR,
    //        (false, true,  false) => Path::MtpSingle,
    //        (false, false, true)  => Path::DFlash,
    //        (true,  false, false) => Path::PpAr,
    //        (true,  true,  false) => Path::PpMtp,         // ← Stage 2b new
    //        (true,  false, true)  => Path::PpDflash,      // already refused at load
    //        (_,     true,  true)  => Path::Refused,       // already refused at load
    //    };
    // 4. Prefill (path-dispatched, ~80 LOC each, 4 paths)
    //    AR/MtpSingle/DFlash → forward_prefill_batch on `gpu`
    //    PpAr/PpMtp → forward_prefill_batch_multi on `m.pp_gpus`
    // 5. Decode loop (path-dispatched, the heaviest divergence):
    //    AR/PpAr → per-token forward_scratch / forward_scratch_multi
    //    MtpSingle → spec_step_mtp_compressed_serial
    //    PpMtp → spec_step_mtp_compressed_serial_hetero(target_gpu=output_device,
    //                                                   drafter_gpu=output_device)
    //    DFlash → spec_step_dflash
    // 6. Common postlude: KV write of final token, done event, conversation_tokens push (~60 LOC)
}
```

The wins:
1. Common prelude/postlude written once. ~300 LOC removed.
2. Path-decision matrix is explicit and exhaustive. Adding a 5th path
   (e.g. PP+PFlash) becomes one match arm + one prefill/decode chunk
   instead of a new ~500 LOC function.
3. Stage 2b's MTP-under-PP dispatch is just the new match arm at #5.
4. Refusal contracts (DFlash+CASK, DFlash+pp>1 in v1, etc.) are
   collapsed into a single decision point at #3 instead of scattered
   guards in each function.

## Risks

- **Touching ~1500 LOC of working serve code.** The existing
  functions have subtle behavior (multi-turn KV reuse, eviction
  triggers, PFlash decision branching, etc.) that's easy to drop
  during a refactor. Mitigation: keep all four behaviors testable
  via `scripts/coherence-gate.sh` + `scripts/coherence-gate-dflash.sh`
  + (new) a coherence test for PP-AR + PP-MTP.
- **Pflash integration.** `generate_multi` is the only function
  currently with PFlash support. The unified function inherits that;
  but PFlash + MTP is not a v1 target. Add explicit refusal at the
  load handler, similar to DFlash+CASK.
- **Prefill batching paths diverge.** `forward_prefill_batch_multi`
  takes `Gpus + Qwen35ScratchSet`; single-gpu takes `Gpu + Qwen35Scratch`.
  The unified function needs to thread both shapes — either through
  separate code paths inside the prefill chunk, or by introducing a
  `PrefillContext` enum. Latter is cleaner but +200 LOC; former is
  faster to land but keeps duplication.
- **DFlash co-residence with MTP.** The current code refuses MTP +
  DFlash at load. That stays.

## Suggested sequence

1. **Setup**: Read all four function bodies; build a per-section
   comparison table (prelude, frame, prefill, decode, postlude). ~1 hour.
2. **Common prelude/postlude extraction**: extract pure-helper
   functions for: prompt encoding + chatml, ChatML token cache,
   capacity check, done-event emission. Land as standalone refactor
   commits that change NO behavior (validated by coherence-gate
   passing). ~250 LOC reduction, ~500 LOC touched. Ship as a series
   of small commits.
3. **PrefillContext enum**: introduce `enum PrefillCtx<'a> { Single(&'a mut
   Gpu, &'a mut Qwen35Scratch), Multi(&'a mut Gpus, &'a mut Qwen35ScratchSet) }`.
   Move existing `forward_prefill_batch` calls behind a `ctx.prefill_batch(...)`
   method on PrefillCtx that dispatches internally. Both single-gpu and
   PP paths now use the same call site. ~150 LOC. Validated by
   coherence-gate (no behavior change).
4. **Path enum**: define `enum SpecPath { Ar, MtpSingle, MtpHetero, MtpPp,
   Dflash }`. Wire load handler to compute it from `m.{pp, mtp, dflash,
   pflash}`. Each generate function calls the same match-on-path. ~100
   LOC. Validated by coherence-gate.
5. **The unification**: collapse `generate / generate_multi /
   generate_mtp` bodies into the new `generate_qwen35`. Validate
   against coherence-gate AND coherence-gate-dflash. ~500 LOC of
   moved-around code, ~200 LOC genuinely deleted. This is the
   surgery commit.
6. **The new arm**: Stage 2b proper. Add the MtpPp match arm that
   calls `spec_step_mtp_compressed_serial_hetero` with the right
   gpus and threads its result back into the common postlude. ~80
   LOC. **This is the small payoff after the refactor.**
7. **Validation**: serve qwen3.6-27b at pp=2 + mtp + 64k ctx; confirm
   coherent output, VRAM math matches the audit projection,
   τ matches single-gpu MTP, tok/s within ~10% of single-gpu baseline.

Total estimated effort: **2-3 days of focused work** across 5-7
commits.

## Alternative if 2-3 days is too much

Build Option X instead (a separate `generate_mtp_multi` of ~600 LOC
that duplicates state plumbing). Lands in ~1 day. Pays back the
duplication debt the FIRST time we want a 5th combo. Default to this
only if there's a near-term deadline that conflicts with the refactor
timeline.

## Open questions

- **PFlash + MTP under PP.** Currently PFlash is generate_multi-only
  and MTP is generate_mtp-only. The unified function would naturally
  allow PFlash + MTP; should we explicitly refuse, or test that
  combination? PFlash compresses the PROMPT; MTP accelerates
  DECODE; they should compose. But neither has been tested together
  yet. Suggest: enable the combo silently, add a coherence test in
  the validation step.
- **Eviction.** generate_multi doesn't currently handle eviction
  (load_model_pp refuses it at load). MTP-single supports eviction;
  if MTP+PP gets eviction "for free" via the unified function, we'd
  need to either disable it or test it. Suggest: disable at load,
  same as today.

## Decision gates within the refactor

After step 3 (PrefillContext): if the diff is too noisy, abort and
ship the existing functions with light cleanup. The refactor only
pays off if the unification gets to step 5 cleanly.

After step 5: if any of {AR, MtpSingle, DFlash} regress on coherence,
revert the unification commit and ship Option X (separate
`generate_mtp_multi`) as a hotfix. The intermediate-commit series
(steps 2-4) is the safety harness — each is independently revertable
without losing the refactor work.
