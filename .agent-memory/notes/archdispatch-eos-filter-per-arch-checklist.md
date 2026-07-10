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

**Per-arch status:** minimax `[e~[` DONE. **cohere2moe (arch 12) — HIGH RISK**: its
decode is a `<|MARKER|>` state machine (`<|START_THINKING|>`/`<|START_TEXT|>`/
`<|START_ACTION|>`/`<|END_OF_TURN_TOKEN|>`); `coherence-gate-cohere2moe.sh` HARD-FAILS on
a marker leak — budget an `eos_filter_config` override + eos-turn validation. lfm2moe
(11) + deepseek4 (9): check eos decode before flip.
