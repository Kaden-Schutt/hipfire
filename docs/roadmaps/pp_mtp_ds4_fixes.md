# DS4 Re-integration Critical Review — Findings

**Branch:** `merge/master-pr352`
**Reviewer:** Kevin Read
**Date:** 2026-06-02
**Status:** All findings fixed (commit `TBD`)
**Comparison:** `crates/hipfire-runtime/examples/daemon.rs` (ours) vs `/tmp/daemon_master.rs` (origin/master, 6832 lines)

Master's DS4 touchpoints were extracted verbatim into our daemon, but the
two codebases diverged in the request-parsing layer and the bench handler.
This review catalogs every divergence that could break DS4 functionality
on merge, ordered by severity.

---

## Finding 1 — CRASH: `bench_prefill` panics on arch_id=9

**Severity:** 🔴 Hard crash (panic on `unwrap()` of `None`)
**Location:** `daemon.rs` line ~1868 (the `else` branch of bench_prefill's arch dispatch)

**What happens:** `bench_prefill` dispatches on `arch_id` but only handles
5/6 (qwen3.5), 7 (qwen2). Every other arch falls into an `else` that
unwraps `m.llama_config.as_ref().unwrap()` — which is `None` for DS4.

Master has an explicit `arch_id == 9` branch at master:1696 that does a
per-token `deepseek4::forward::decode_step` warm-pass instead.

**Fix:** Add `arch_id == 9` branch before the `else`, matching master's
decode_step loop.

---

## Finding 2 — CRASH: Missing `gpu.invalidate_graph_state()` in reset handler

**Severity:** 🔴 Hard crash on second generate after reset ("illegal memory access")
**Location:** `daemon.rs` line ~1712 (DS4 reset block)

**What happens:** Our reset handler calls `s.reset()` on the DS4 state but
does NOT call `gpu.invalidate_graph_state()`. Master's reset handler (line
1538-1541) calls both — the comment explains that the captured hipGraph
bakes session-1's device-buffer pointers into kernarg memory; without
invalidation, session-2's replay fires against stale pointers.

The `DeepseekV4State::reset()` sets `ar_forward_warmed_up = false` but
that alone doesn't tear down the captured `graph_exec` on the GPU side.

**Fix:** Add `gpu.invalidate_graph_state();` after the DS4 state reset,
matching master's pattern.

---

## Finding 3 — INCORRECT DEFAULTS: DS4 temp/top_p defaults not applied at request-parsing level

**Severity:** 🟡 Incorrect sampling behavior (quality, not crash)
**Location:** `daemon.rs` lines 1470-1472 (request parsing) vs 5498-5501 (dispatch override)

**What happens:** Master conditionally sets `(temp, top_p)` defaults to
`(1.0, 1.0)` for `arch_id == 9` at the request-parsing layer (master:1274).
Our daemon hard-codes `(0.3, 0.8)` for all arches.

Our dispatch-side override (`ds4_temp` / `ds4_top_p` at line 5498) is a
partial workaround:
- When user sends no `temperature`, we get 0.3 → our override maps ≤1e-6
  to 1.0, so the effective default is **still 0.3** (not near zero), so
  **the override doesn't fire**. DS4 runs at 0.3 instead of 1.0.
- When user explicitly sends `temperature: 0`, we override it to 1.0 —
  correct behavior but for the wrong reason (silently overriding explicit 0).
- `top_p` default is **never overridden** for DS4 — it stays at 0.8
  instead of master's 1.0.

**Impact:** DS4 runs at temp=0.3 / top_p=0.8 instead of the recommended
1.0 / 1.0. This can trigger block-level attractors on quantized DS4 models
(the exact issue master's comment warns about).

**Fix:** Move the arch-conditional defaults to the request-parsing layer
(where `temp` and `top_p` are first resolved from the message), matching
master's pattern. Remove the dispatch-side override.

---

## Finding 4 — MISSING FEATURE: `think_mode` not parsed from generate request

**Severity:** 🟡 Missing feature (thinking modes don't work)
**Location:** `daemon.rs` lines 5494-5497 (hardcoded `ThinkMode::NonThink`)

**What happens:** Master parses `reasoning_effort` / `thinking_mode` from
the generate request (master:1297-1302) and threads it through `GenerateCtx`
or directly to `generate_deepseek4`. Our daemon hardcodes `ThinkMode::NonThink`.

The `ThinkMode` enum and its logic are correctly ported inside
`generate_deepseek4` — the function body does the right thing when given
`High` or `Max`. But the dispatcher never passes anything other than
`NonThink`.

**Impact:** Extended reasoning is inaccessible via the daemon. Model always
uses the non-thinking frame.

**Fix:** Parse `reasoning_effort` / `thinking_mode` from the generate
request, resolve to `ThinkMode`, and thread through the dispatch call.
This requires either adding a field to `GenerateCtx` or passing it as a
separate argument alongside the existing DS4 dispatch.

---

## Finding 5 — COSMETIC: Missing `9 => "deepseek4"` in arch name mapping

**Severity:** 🟢 Cosmetic (wrong string in `loaded` event)
**Location:** `daemon.rs` line ~1239 (arch name match block)

**What happens:** When a DS4 model loads, the `"loaded"` event reports
`"arch": "qwen3"` (the `_` fallback) instead of `"deepseek4"`. Master
has `9 => "deepseek4"` (master:1037).

**Impact:** Downstream tooling that keys on `arch` to route DS4-specific
logic (e.g. Pi's client-side think-mode toggle) would misidentify the
model. No correctness impact on inference itself.

**Fix:** Add `9 => "deepseek4",` to the arch name match block.

---

## Finding 6 — COSMETIC: `(dim, layers, vocab)` reports `(0, 0, 0)` for DS4

**Severity:** 🟢 Cosmetic (zero values in `loaded` event)
**Location:** `daemon.rs` lines ~1245-1250 (dim/layers/vocab extraction)

**What happens:** The `(dim, layers, vocab)` tuple is extracted from
`q35_config`, `qwen2_config`, or `dots_ocr_config` — none of which are
populated for arch_id=9. The fallback returns `(0, 0, 0)`.

Master has a `deepseek4_config` branch at master:1244 that reads
`c.hidden_size, c.num_layers, c.vocab_size`.

**Impact:** The `"loaded"` event reports `"dim":0,"layers":0,"vocab":0`.
No functional impact on inference, but clients that log or gate on these
values (e.g. context-window calculation) will be confused.

**Fix:** Add an `else if let Some(ref c) = m.deepseek4_config` branch
reading `c.hidden_size, c.num_layers, c.vocab_size` (verify field names
against `DeepseekV4Config`).

---

## Finding 7 — BEHAVIORAL: `mtp_k` hard-coded to 3 instead of read from load params

**Severity:** 🟢 Low (env var override exists; most deployments use default 3)
**Location:** `daemon.rs` line 6852 (`unwrap_or(3)`)

**What happens:** Master stores `mtp_k` on `LoadedModel` (set from the
load request's `params.mtp_k`, default 3) and reads it in
`generate_deepseek4` via `m.mtp_k`. Our `LoadedModel` doesn't have the
field, so `generate_deepseek4` uses a hard-coded fallback of 3.

The DS4 `generate_deepseek4` function still reads `HIPFIRE_DEEPSEEK4_SPEC_K`
and `HIPFIRE_MTP_K` env vars before falling back to the default, so
operators can override at runtime. But per-request `mtp_k` from the load
message is silently ignored.

**Impact:** `{"type":"load","params":{"mtp_k":5}}` is ignored for DS4.
The env var escape hatch covers production use. The field was not added
to avoid the ripple of adding it to all 8+ `LoadedModel` construction sites
in our daemon (master has a different set of sites).

**Fix (optional):** Add `mtp_k: usize` to our `LoadedModel` struct,
default it to 3 at all construction sites, and read from it in
`generate_deepseek4`. Low priority since the env var override exists.

---

## Finding 8 — COSMETIC: Unload order differs from master (functional equivalent)

**Severity:** ⚪ No impact
**Location:** `daemon.rs` lines 3474-3477

**What happens:** Master frees `deepseek4_state` → `deepseek4_pbs` →
other weights → `deepseek4_weights`. Our code frees other weights →
`deepseek4_pbs` → `deepseek4_state` → `deepseek4_weights`.

All four DS4 allocations are independent GPU tensors freed through the
pool. Order doesn't affect correctness — `drain_pool()` at the end
handles actual `hipFree` calls regardless of queue order.

**No fix needed.**

---

## Summary

| # | Severity | Finding | Crash? | Fix complexity |
|---|----------|---------|--------|----------------|
| 1 | 🔴 Critical | `bench_prefill` panics on DS4 | Yes (panic) | Low — add arch branch |
| 2 | 🔴 Critical | Missing `invalidate_graph_state` in reset | Yes (illegal mem access) | Trivial — add 1 line |
| 3 | 🟡 Incorrect | temp/top_p defaults not applied | No — quality issue | Medium — refactor request parsing |
| 4 | 🟡 Missing | `think_mode` not parsed from request | No — feature gap | Medium — parse + thread |
| 5 | 🟢 Cosmetic | Wrong arch name in `loaded` event | No | Trivial — add match arm |
| 6 | 🟢 Cosmetic | Zero dim/layers/vocab for DS4 | No | Low — add config branch |
| 7 | 🟢 Low | `mtp_k` hard-coded to 3 | No (env var exists) | Medium — struct change |
| 8 | ⚪ None | Unload order differs | No | No fix needed |

**Recommendation:** Fix 1 and 2 before any DS4 testing (both are crash
bugs). Fix 3 and 4 before claiming DS4 feature parity with master. Fix
5–7 as follow-up polish.
