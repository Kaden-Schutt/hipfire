# Task 7 Report — Cohere2MoeDispatch + Inject arm + cohere2moe dual-run

**Status:** DONE (build-only). **Commit:** (see git head). daemon builds clean; lib 342 passed.

## (A) Cohere2MoeDispatch (daemon.rs, after MinimaxDispatch)
Full ArchDispatch modeled on MinimaxDispatch. Confirmed: `m.cohere2moe()`/`cohere2moe_mut()`
accessors; `Cohere2MoeBundle.eos_tok` = END_OF_TURN; `state.n_tokens`/`state.logits` (GPU-
resident, decode_step downloads+returns it — same as minimax); `state.reset(gpu)->Result`;
`config.vocab_size`; `forward::{decode_step, forward_batch, forward_batch_supported}`.
- forward hooks: batched forward_batch (chunks-of-64) when forward_batch_supported else
  per-token decode_step; sample = download state.logits + sample_cpu.
- Struct holds `tools: Option<Vec<Value>>` (constructed per-request) so `stream_parser()` can
  build the parser's known_tools/tool_params.
- `stream_parser(cfg)` OVERRIDE → `Cohere2MoeStreamParser::new(tokenizer, tools, cfg.max_tokens,
  cfg.max_think_tokens)`. No eos_filter_config (eos consumed by on_eos, never emitted).

## (B) Driver on_eos→Inject fix (ar_generate)
Moved the eos decision to PRE-COMMIT (loop top, after abort). `eos_commit_and_stop` flag falls
through to commit+forward+emit_only+break (byte-identical to the prior post-forward
CommitAndStop). `Inject(v)` → `for t in v { parser.enqueue(t) }` + `next_token = next_forced();
was_forced=true; continue` — eos NOT committed. `Stop` → break (no commit). Removed the old
post-forward eos block. DefaultStreamParser (always CommitAndStop) path verified unchanged.

## (C) Dual-run in generate_cohere2moe
`__parity`/`__old_tape` before the loop; `__old_tape.push(next_tok)` at the single commit site
(13182-13190, every marker/forced/text token); parity re-run after the done event
(model_reset_context + Cohere2MoeDispatch{m, tools} + ar_generate(prompt_ids full render,
tape=Some) + assert_token_parity). Legacy loop intact (T9 deletes).

## What T8 MUST validate (GPU, north-mini-code.mq4.hfq)
1. Token-parity FLOOR: HIPFIRE_ARCHDISPATCH_PARITY=1 temp0 — no PARITY FAIL (but it is BLIND to
   tool_calls/reasoning events; do NOT accept green parity as sufficient).
2. EVENT-EQUIVALENCE (the real bar): `scripts/coherence-gate-cohere2moe.sh` (hard-fails on
   `<|MARKER|>` leak + error event) + the 4 guard fixtures from Task 6, live-tuned:
   - empty_turn → on_eos Inject fires (may need subtler prompt / verify it reason-only-stops)
   - think_budget → SET max_think_tokens (~100) so the force-close trips
   - toolcall / toolcall_as_text → SUPPLY a tool schema at run time (else North can't call)
   Diff the prod-path (flag off) event stream vs a legacy-path capture of the same inputs.
3. Concern: the empty-turn on_eos Inject re-entry (pre-commit) is a structural change to the
   shared driver — confirm DefaultStreamParser arches (qwen35/minimax/lfm2moe) still byte-
   identical (re-run a quick T4-style check if paranoid; the CommitAndStop path is unchanged
   by construction but the eos check moved location).
