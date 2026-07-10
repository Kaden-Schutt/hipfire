---
title: "ArchDispatch migration: eos_filter_config() per-arch check is MANDATORY (or the eos marker leaks into visible output)"
date: 2026-07-10
tags: [archdispatch, ar_generate, eos, eos_filter, eos_filter_config, migration, checklist, cohere2moe, lfm2moe, deepseek4, minimax, regression, device-mesh, marker-leak]
---

**Rule (learned Inc 4, minimax):** when migrating an arch's AR path onto `ar_generate`
(the ArchDispatch absorption program — see [[daemon-god-struct-archdispatch-design]]),
you MUST check what the arch's **eos token decodes to** and override
`ArchDispatch::eos_filter_config()` if it decodes to a **literal** string.

**Why:** the legacy separate-fn decode loops break on `next_tok == eos_tok` BEFORE
emitting. `ar_generate` does the opposite — it commits+emits the token through its
`EosFilter`, THEN checks `is_eos` and breaks. So the eos token's decoded text reaches
the visible stream unless the filter strips it. The default `EosFilterConfig::default()`
is an **empty pass-through**. It only "works" for ChatML arches (qwen35/qwen2) because
their eos `<|im_end|>` decodes to EMPTY text. minimax's eos IS `[e~[` and decodes to that
literal → it LEAKED into visible output on eos-terminated turns until fixed.

**Fix pattern (arch owns its markers):** `ArchDispatch::eos_filter_config()` hook
(default = `EosFilterConfig::default()`, byte-identical to before → ChatML arches
unchanged). Override per-arch, e.g. `MinimaxDispatch`:
`EosFilterConfig { stop_at: vec![b"[e~[".to_vec()], ..Default::default() }`.

**Two traps that make this easy to miss:**
1. **temp0 dual-run parity CANNOT catch it** — parity prompts that hit `max_tokens`
   never emit eos, so the eos path is untested. ALWAYS validate one **eos-terminated**
   turn (`finish_reason":"stop"`) on the PROD path, not just parity. (A short factual Q
   with history primes a concise answer that hits eos; a fresh single prompt tends to
   over-reason and hit the cap instead.)
2. Grep `resolve_eos_tok` / the arch's loader eos-candidate list (carriers.rs) and decode
   the winning token to see if it's literal-or-empty BEFORE flipping.

**Per-arch status:** minimax `[e~[` DONE; lfm2moe stop-id-set DONE. **cohere2moe (arch 12)
DONE (2026-07-10)** — NOT via eos_filter_config: its `<|MARKER|>` machine is owned by
`Cohere2MoeStreamParser` (the StreamParser output layer), which suppresses markers by
token-id in `feed()` and consumes the `<|END_OF_TURN_TOKEN|>` eos via `on_eos()` (never
emitted). `coherence-gate-cohere2moe.sh` PASS, 0 marker leaks. deepseek4 (9) = Axis B
(deferred, EP). See [[daemon-god-struct-archdispatch-design]].

**SECOND per-arch migration lesson (Inc-8, cohere2moe): `model_reset_context` MUST reset
the migrated arch.** It was missing a cohere2moe reset (had qwen2/ds4/lfm2moe/minimax) —
the dual-run parity harness `model_reset_context`+re-run then ran the ar_generate arm on
the legacy arm's STALE state → a deterministic argmax flip ~19 tokens in (temp0). Also a
latent #462 reset-command multi-turn bleed. The dual-run parity CAUGHT it (fix cf754c74).
**Before flipping any arch: confirm `model_reset_context` resets it.** Also: batched
`forward_batch` is NOT numerically batch-size-invariant — a `Dispatch.prefill_forward`
chunk size that differs from the legacy prefill's chunk size shifts logits and fails temp0
parity (match the legacy chunk size, e.g. cohere2moe 256).

**THIRD lesson (Axis B, deepseek4-EP): `on_eos()` discipline + a grammar gotcha.**
(1) An arch whose legacy loop BREAKS on the sampled eos *before* feeding/forwarding it
(ep_serve_ds4: `if next==eos_tok { break }`) needs `on_eos() = EosDecision::Stop`, NOT
`CommitAndStop` — the eos must never enter KV/tape/parser. CommitAndStop would forward +
emit_only the eos (extra forward pass + spurious emit) → parity divergence. (2) A
grammar-capable dispatch MUST override `ensure_decoded_vocab` — the driver calls it the
moment tools activate; the trait default `unimplemented!()` panics. This is INVISIBLE to
every non-grammar dual-run row (they never call it) — only a tool-call prompt exposes it.
Always include a tool-call row in the arch's dual-run. (3) `model_reset_context` still
didn't reset `m.ep` after this fold — BENIGN only because generate_ep resets EP per-turn
(EP has no LCP); a non-generate EP path would bleed. See [[daemon-god-struct-archdispatch-design]].
