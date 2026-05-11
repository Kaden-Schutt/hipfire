# AR Think-Policy Patch — Bug Report

**Date:** 2026-05-10
**Branch:** feature/ar-think-policy (should have been, NOT master — reverted from drbearjew/master)
**Files changed:** `sampler.rs`, `daemon.rs`

## Summary

Added think-block policy state machine to the AR `generate()` path, matching the DFlash `ban_token_id` mechanism. Three sample sites unified under centralized helpers `merge_think_bans()` and `update_think_state()`.

## What was implemented

### 1. `sampler.rs` — new public API

**`ThinkState` struct** — state machine tracking:
- `in_think: bool`
- `think_blocks_seen: usize`  
- `visible_answer_started: bool`

**`merge_think_bans(ban_tokens, state, thinking_allowed, think_start_id, think_end_id)`** — merges structural think-block bans into any existing ban list. Rules:
- thinking=off → ban `<think>` + `</think>` always
- Ban `<think>` if already in think, or think_blocks_seen >= 1, or visible answer started
- Ban `</think>` when not inside think

**`update_think_state(state, tok, think_start_id, think_end_id)`** — updates state machine from raw sampled token.

### 2. `daemon.rs` — AR path wiring

**Prompt injection** (line ~3165): When `thinking_allowed`, injects `<think>\n` token IDs into `new_tokens` BEFORE prefill. This ensures the model sees `<think>` as part of its input prefix.

**State initialization** (line ~3320): Scans `conversation_tokens` for unmatched `<think>` in the prefix, seeds state with `in_think=true, think_blocks_seen=1` when found.

**Three sample sites unified:**

| Site | Location | Purpose |
|------|----------|---------|
| First token | ~line 3344 | Initial sample after prefill |
| Budget-alert skip | ~line 3561 | Resample when budget alert skipped |
| Main loop | ~line 3642 | Every subsequent token |

All three call `merge_think_bans()` before `SamplerConfig`, and `update_think_state()` after `sampler::sample()`.

### 3. Environment

- `HIPFIRE_NGRAM_LOOP_THRESHOLD=4` — n-gram loop guard
- `HIPFIRE_GRAPH=0` — hipGraph disabled
- No Jinja template required (manual prefix injection handles `<think>`)

## What works

- ✅ No think-reopen loops on AR path
- ✅ `<think>` in visible answer is correctly banned
- ✅ Nested `<think>` is banned
- ✅ `</think>` when not in think is banned
- ✅ All 3 sample sites covered — no bypass path exists
- ✅ Prefix injection works — model sees `<think>` before generation
- ✅ thinking=OFF: clean direct answers, 10-14 tok/s at 50K ctx

## What does NOT work — model limitation

**Qwen 3.6 27B MQ4 does NOT use `<think>` blocks natively.**

With thinking=ON + `<think>` prefix injected:
- Model immediately emits `</think>` then answers directly → 0 visible think content
- Without prefix: model writes "Here's a thinking process:" as visible monologue

This is a **training data issue**, not a code bug. The Qwen 3.6 27B was not fine-tuned for the structural `<think>`/`</think>` convention. It treats `<think>` as a decorative XML tag to close immediately, not as a reasoning container.

## Expected behavior (with think-capable model)

A model trained for `<think>` blocks (DeepSeek, Qwen 35B, etc.) should produce:

```
<think>
reasoning about the chess transcript...
</think>
1. 2015 European Women's Championship held in Chaki, Georgia.
2. WGM Sandu defeats four higher-rated opponents.
...
```

Visible output stripped of think content:
```
1. 2015 European Women's Championship held in Chaki, Georgia.
2. WGM Sandu defeats four higher-rated opponents.
...
```

## Test results (chess transcript, 5.5K words, 7.2K tokens)

| Config | Speed | `<think>` blocks | Output |
|--------|-------|-----------------|--------|
| AR 50K, thinking=ON, Jinja template | 18 tok/s | 0/0 | "Here's a thinking process…" monologue |
| AR 50K, thinking=ON, manual injection | 8 tok/s | 0/0 | Direct answer (closed empty think) |
| AR 50K, thinking=OFF | 10-14 tok/s | 0/0 | Clean direct answer ✅ |
| DFlash 12K, thinking=OFF | 94 tok/s (short) → 11 tok/s (7K) | 0/0 | Clean ✅ |

## Recommendations

1. **For Qwen 3.6 27B production use:** thinking=OFF, AR at 50K for documents, DFlash at 12K for coding/short-QA
2. **For `<think>` enabled production:** Deploy with Qwen 35B or DeepSeek model that uses structural think blocks
3. **Future work:** Add an `enable_thinking` config toggle that the AR policy state machine reads to skip prefix injection when the model doesn't support it
4. **Git:** Push to feature branch, NOT master
