<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: Generate

## Broken

### 1. EP MoE (arch 6) generate never wired — routes to dense TP (#683)
- **path_line:** `crates/hipfire-generate/src/qwen.rs:228-252` (`5 | 6 => ep_serve_qwen35_dense_tp`)
- **verified:** true (open issue #683 + code read)
- **how known:** `generate_ep` sends arch 5 and 6 to `ep_serve_qwen35_dense_tp`, which pattern-matches only `EpArch::Qwen35DenseTp` (`qwen.rs:373-390`) and errors `EP arch mismatch (expected dense Qwen TP)`. MoE loads as `EpArch::Qwen35 { batch, .. }`. The only `EpArch::Qwen35` arm in generate is reset (`qwen.rs:740-747`). Loader admits tp≥2 MoE; generate cannot run it.
- **sibling:** `select_generation_route` EP short-circuit only names arch 9/10 (`ar.rs:882-887`); Qwen EP becomes `GenerationRoute::Unknown`. Dispatch still serves via `Unknown if m.ep.is_some()` (`ar.rs:1305-1330`). `supports_tools` excludes Unknown, so tools are refused while generation is still attempted on a broken MoE path.

### 2. EP GPU error exits skip `ep_reset_after_abort`
- **path_line:** `qwen.rs:404-414`, `499-508`, `1014-1026`, `1193-1201`, `1464-1473`, `1579-1587` (representative)
- **verified:** true
- **how known:** Prefill/decode/download failures call `emit_active_attempt_error` then `return` with no `ep_emit_abort` / `ep_reset_after_abort`. Client-cancel paths do reset. DS4 mitigates with unconditional start-of-turn zero (`qwen.rs:882-894`); dense_tp resets at entry (`qwen.rs:320-365`). **MiniMax EP uses LCP reuse** (`qwen.rs:1410-1427`) without full per-turn reset — a mid-prefill/decode error leaves rank KV/cursors dirty for the next LCP hit (same contamination class DS4 start-of-turn already fixed).

### 3. Vision cancel: wire terminal without GPU/DN/spec rollback
- **path_line:** `vision.rs:1007-1010`, `1380-1382`, `1418-1422` (and dots/lfm2 VL cancel sites)
- **verified:** true
- **how known:** `check_abort` / `ClientTerminalDecision::Abort` call `emit_qwen_ar_cancelled` only. Comment claims next dispatch non-zero-seq_pos reset reclaims state (`vision.rs:1005-1008`) — that is **not** attested `production_fail_closed_rollback` used by Qwen AR/DFlash/LFM (`ar.rs:3531-3532`, `dense.rs:6152-6154`). Partial VL turn leaves `seq_pos`, `conversation_tokens`, DN, KV compact, checkpoints live until a later capacity path happens to fire.

### 4. Vision context-full DN reset is manual memset; AR uses canonical reset
- **path_line:** `vision.rs:565-617` vs `ar.rs:1948-1966`
- **verified:** true
- **how known:** AR switched to `b.dn_state.reset(gpu)` so Q8 `s_ef_residual` cannot leak (`ar.rs:1948-1952`). VL still field-wise memsets `s_matrices`/`s_scales`/`conv_states`/`s_ef_residual` and ignores reset Result — the exact incomplete pattern AR comments call unsafe.

### 5. Bring-up dense routes: terminal Abort is empty (no GPU rollback)
- **path_line:** `dense.rs:6699-6700` (minimax), `7430-7431` (cohere), `7661-7662` (qwen2), `8363-8364` (maple); contrast LFM `6268-6270`
- **verified:** true
- **how known:** `ClientTerminalDecision::Abort => {}` suppresses success `done` but leaves GPU session state. LFM/DS4/Glimmer call `production_fail_closed_rollback`. AR bring-up comment admits the gap (`ar.rs:4531-4533`).

### 6. `fail_closed_reset_target_and_spec` omits Maple; `reset_core_arch_key` maps 15→unknown
- **path_line:** `common.rs:1075-1182`; `ar.rs:4609-4621`
- **verified:** true
- **how known:** Reset walk covers qwen35, llama, qwen2, ds4, lfm, minimax, cohere, gemma4, glimmer — **no `m.maple_mut()`**. `reset_core_arch_key(15)` falls through to `"unknown"`. Shared fail-closed epilogue will not reset MapleState if ever reached (maple Abort currently empty so less hot, inventory still wrong).

### 7. Grammar sample path still swallows logits download into zeros (FIX #4 sibling)
- **path_line:** `ar.rs:3432-3433`, `3909-3910`, `4079-4080`; `qwen.rs:4644-4645`, `4959-4960`, `5091-5092`
- **verified:** true
- **how known:** EP comments forbid zero-logits fallback (`qwen.rs:1091-1093`). Single-GPU AR and PP multi grammar branches still `unwrap_or_else(|_| vec![0.0; vocab_size])` → silent token-0 corruption instead of fail-closed error.

### 8. DS4 AR/spec dispatch discards `max_think_tokens`
- **path_line:** `ar.rs:1445-1482` (`let _ = (… max_think_tokens …)` before `generate_deepseek4` / `_spec`)
- **verified:** true
- **how known:** ThinkMode is threaded; numeric think cap is not. DS4 EP decode loop also has no think-cap enforcement (parser priming only).

### 9. Stale Maple docs claim max_think discarded; code now enforces
- **path_line:** `dense.rs:7679-7682` comment vs `dense.rs:8187-8191` + `ar.rs:1418-1441`
- **verified:** true (doc/code contradiction)
- **how known:** Module doc says dispatch discards max_think like Qwen2; MapleAr threads it and `MapleThoughtRouter` force-closes. Contributor trap.

### 10. Qwen2 path has no cancel poll and no think/tools
- **path_line:** `dense.rs:7605-7663`; dispatch drops max_think at `ar.rs:1361-1395`
- **verified:** true
- **how known:** Decode loop never calls `check_abort`; Abort arm empty. Documented bring-up limits, still a live serve route for arch 7.

## Missing

### M1. No `GenerationRoute::QwenEp` / MoE EP serve
- **path_line:** `ar.rs:882-887`, `qwen.rs:240`; `tests/generation_route_matrix_tests.rs` lacks Qwen+ep row
- **verified:** true
- **summary:** Matrix covers MiniMaxEp/Deepseek4Ep only. Arch crate has `forward_ep` / batch readiness (peer AuditArchQwen); generate never calls them for single-stream MoE EP. #683 tracks wire-up; admission refusal still missing on master.

### M2. EP reset does not clear `asst_turn_cache` / host checkpoint rings / speculator
- **path_line:** `qwen.rs:787-798` vs `common.rs:1082-1083`, `1156-1167`
- **verified:** true
- **summary:** `ep_reset_after_abort` clears seq_pos + conversation_tokens + EP GPU state only. Single-GPU production epilogue also clears asst_turn_cache and frees prefill/dflash checkpoints. EP MiniMax LCP + shared host cache can diverge after abort.

### M3. Think-cap policy drift across routes
- **path_line:** dense_tp fail-closed `qwen.rs:561-575`; VL force-close `vision.rs:1167-1240`; AR/PP force-answer latch `qwen.rs:4777+`; Glimmer/Gemma/Maple routers force-close
- **verified:** true
- **summary:** Same user control means hard validation error (dense TP EP), force `</think>` and continue (VL/AR/Maple), or strength-primary (Glimmer strength ignores cap at `dense.rs:2638`). No single contract table.

### M4. Gemma4 lowered/MoE generate refused at runtime only
- **path_line:** `ar.rs:1158-1170`
- **verified:** true
- **summary:** Loader can publish lowered state; generate errors after load. Same #683-class admission gap.

### M5. Continuous-batch eligibility reads then discards max_think
- **path_line:** `batch.rs:204-208` (`let _ = max_think`)
- **verified:** true (intentional dead read / unfinished gate)
- **summary:** Comment says 0/1/ordinary budgets are valid batch controls; gate never uses the value (budget_alert still blocks).

### M6. No mid-decode `check_abort` on maple/qwen2/minimax/cohere bring-up loops
- **path_line:** maple/`dense.rs` decode loops; qwen2 `7605+`
- **verified:** true
- **summary:** Cancel only at commit_ready (often empty Abort). Long max_tokens burns GPU after client gone.

## Would change

1. **Refuse or wire Qwen MoE EP (close #683)** — `qwen.rs:240` + loader admission. Admission refuse now (**hours**); full `ep_serve_qwen35_moe` over `EpArch::Qwen35` (**days–week**).
2. **Fail-closed all EP GPU error exits through `ep_reset_after_abort` before emit** (**hours**); MiniMax post-error or start-of-turn reset when LCP unsafe (**hours**).
3. **VL cancel → `production_fail_closed_rollback` + attested cancel terminal** (match AR) (**half-day**); unify VL context-full onto `dn_state.reset` (**hours**).
4. **Extend `fail_closed_reset_target_and_spec` + `reset_core_arch_key` for maple (15)**; fill Abort arms on maple/qwen2/minimax/cohere (**day**).
5. **Replace grammar zero-logits swallow with emit+rollback** on AR/PP (copy EP FIX #4) (**hours**).
6. **Thread or explicitly refuse DS4 `max_think_tokens`** (**hours**).
7. **Add `GenerationRoute::QwenEp` (DenseTp vs Moe)** so tools/route matrix stay honest (**half-day** once serve exists).
8. **Single think-cap policy matrix test** per route (**day**).
9. **Fix stale maple module docs** (`dense.rs:7679`) (**minutes**).
10. **EP `ep_reset`: clear `asst_turn_cache`** parity with single-GPU (**hours**).

## Confidence

- No GPU/hw-gate/builds/tests run (read-only).
- Did not fully trace `redline.rs` capture paths or every batch lane teardown.
- Did not verify maple GPU reset API beyond missing callsite in fail_closed.
- #683 confirmed open — not re-claimed as novel; added siblings (Unknown route, tools gate, reset-only MoE arm).
- PP abort double path (`reset_pp_uncommitted_state` then `production_fail_closed_rollback`) looks intentional; not proven redundant.
- Peers own loader admission / arch EP readiness / runtime `reset_core` internals.
