// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Helpers shared by more than one generation family.
//!
//! Single owner for the tail that made the first three-way split fail.
//! Families depend on this module; they never copy from it and never
//! depend on each other.

use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_engine::emit::*;
use hipfire_engine::prompt::*;
use hipfire_engine::redline::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_loader::{AsstTurnCache, LoadedModel};
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::spec::{
    ClientEvent, EvictRetain, FinishSummary, SpecTarget, Speculator, StopReason,
};
use std::any::Any;
use std::io::Write;

/// Permanently latch a per-request think cap once its numeric budget is hit.
///
/// The close-token injection alone is insufficient: a model may reopen
/// `<think>` or continue an unbounded visible answer. The latch blocks future
/// openers and activates the existing bounded answer tail.
pub(crate) fn latch_request_think_cap(
    budget_hit: bool,
    generated: usize,
    force_answer_latched: &mut bool,
    latch_gen_mark: &mut Option<usize>,
) -> bool {
    if !budget_hit {
        return false;
    }
    let newly_latched = !*force_answer_latched;
    *force_answer_latched = true;
    latch_gen_mark.get_or_insert(generated);
    newly_latched
}

/// Stable fingerprint over an assistant turn — pair of (text content,
/// tool_calls canonical JSON). Output is identical for two messages
/// that have the same content+tool_calls regardless of how the
/// surrounding bytes (e.g. whitespace inside JSON args) were rendered
/// upstream. Used by the V4F prefix-cache to identify "this is the
/// same assistant turn the model previously emitted, so reuse the
/// emitted token IDs verbatim instead of re-encoding via the DSML
/// renderer + BPE (which is not bijective)."
pub fn asst_turn_fingerprint(
    content: &str,
    tool_calls: &[hipfire_runtime::prompt_frame::ToolCall],
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    "assistant".hash(&mut h);
    if tool_calls.is_empty() {
        // Pure-text turn — content IS the message. Trim whitespace
        // to absorb minor formatting drift between store (model's
        // verbatim emission) and lookup (whatever the client preserved).
        content.trim().hash(&mut h);
    } else {
        // Mixed turn (text + tool_calls) or pure tool_call. Hash ONLY
        // the tool_calls — pi-coding-agent (and most OpenAI-compat
        // clients) sends `content: null` on assistant messages that
        // carry tool_calls, even when the model originally emitted
        // prose ahead of the tool block (e.g. "Let me check the
        // structure first.<｜DSML｜tool_calls>…"). The store-side
        // sees the prose in `emit_text_buf`; the lookup-side sees
        // content=`""`. Excluding content from the fingerprint when
        // tool_calls is non-empty matches the client's effective
        // identity for tool-call turns and lets the cache hit.
        //
        // Collision risk: two distinct turns with identical
        // tool_calls hash to the same key; the later store wins,
        // and a replay of the earlier turn replays the later turn's
        // tokens. In practice this only matters when the model emits
        // the SAME tool_call twice with different surrounding prose
        // in the same conversation — uncommon for agent flows, and
        // the worst-case effect is the model seeing slightly altered
        // prose in its own history.
    }
    for tc in tool_calls {
        tc.name.hash(&mut h);
        // Serialize args in a CANONICAL form: walk the Value tree and
        // emit objects with keys sorted lexically (recursively). The
        // upstream `serde_json::Map` uses insertion order — fine for
        // round-tripping a single payload, but two clients (or two
        // parser passes on the same payload) can yield different
        // insertion orders for the same logical args. Without
        // canonicalization those two turns hash to DIFFERENT keys,
        // dropping cache hit rate on otherwise-identical tool calls.
        let args = canonical_json(&tc.arguments);
        args.hash(&mut h);
    }
    h.finish()
}

/// Build the fingerprint-key string for an emitted assistant turn so
/// it matches `msg.content` as the CLI sends it back next turn.
/// Mirrors the *visible-content* transformation the bun CLI's HTTP
/// serve applies between SSE-relay and `messages[].content`:
///
///   1. Strip paired `<think>…</think>` blocks plus any trailing
///      whitespace (`cli/index.ts:1656-1658`).
///   2. Strip an unclosed `<think>…$` tail (same site).
///   3. Strip an orphan `</think>` opener — when the daemon's prompt
///      ends with `<think>\n` the model resumes inside think mode and
///      never emits an opening tag; the CLI's `inThink` state machine
///      (`cli/index.ts:2334-2365`) treats every token until `</think>`
///      as `reasoning_content` and only emits content from after the
///      close. We match that by stripping `text-up-to-and-including-
///      first-</think>` + trailing whitespace when no `<think>`
///      preceded it.
///   4. Strip the literal `<|im_end|>` substring (the CLI relay
///      removes it at `cli/index.ts:2366`).
///
/// Without (3) and (4) the fingerprint stored after turn N would
/// include reasoning + the ChatML terminator that the CLI strips
/// before sending back as `msg.content` on turn N+1, dropping the
/// cache hit rate to ~zero for thinking-on Qwen models.
pub fn strip_think_for_fingerprint(s: &str) -> String {
    let mut out = s.to_string();
    // (1) + (2): paired/unclosed `<think>` blocks.
    loop {
        let open = match out.find("<think>") {
            Some(i) => i,
            None => break,
        };
        match out[open..].find("</think>") {
            Some(close_rel) => {
                let close_end = open + close_rel + "</think>".len();
                let mut tail = close_end;
                let bytes = out.as_bytes();
                while tail < bytes.len() {
                    let c = bytes[tail];
                    if c == b' ' || c == b'\n' || c == b'\t' || c == b'\r' {
                        tail += 1;
                    } else {
                        break;
                    }
                }
                out.replace_range(open..tail, "");
            }
            None => {
                out.truncate(open);
                break;
            }
        }
    }
    // (3): orphan `</think>` closer with no preceding opener (model
    // resumed inside think mode from the prompt's `<think>\n` prefix).
    if let Some(close_idx) = out.find("</think>") {
        let after_close = close_idx + "</think>".len();
        let mut tail = after_close;
        let bytes = out.as_bytes();
        while tail < bytes.len() {
            let c = bytes[tail];
            if c == b' ' || c == b'\n' || c == b'\t' || c == b'\r' {
                tail += 1;
            } else {
                break;
            }
        }
        out.replace_range(0..tail, "");
    }
    // (4): strip the literal `<|im_end|>` substring (CLI relay strips
    // it from every chunk before forwarding as content).
    while let Some(idx) = out.find("<|im_end|>") {
        out.replace_range(idx..idx + "<|im_end|>".len(), "");
    }
    out
}

/// Apply the same visible-content transformation before every assistant-turn
/// cache lookup and store. Keeping this centralized prevents the model's raw
/// emission (which can contain thinking markers and prompt-normalized
/// whitespace) from hashing differently than the history content returned by
/// an OpenAI-compatible client on the next turn.
pub fn normalize_asst_turn_for_fingerprint(s: &str) -> String {
    let stripped = strip_think_for_fingerprint(s);
    hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned()
}

/// Cancel terminal after production fail-closed rollback attestation.
///
/// Attested rollback (`rolled_back`) keeps the fold-compatible
/// `aborted`+`done(finish_reason=aborted)` pair. Unattested rollback emits
/// exactly one correlated nonretryable fail-closed error (with epilogue
/// context) and **no** `done` — never a second terminal.
pub fn emit_spec_cancel_after_rollback(
    stdout: &mut impl std::io::Write,
    id: &str,
    completion_tokens: usize,
    epilogue: &RollbackEpilogue,
) {
    if epilogue.rolled_back {
        emit_qwen_ar_cancelled(stdout, id, completion_tokens);
        return;
    }
    emit_fail_closed_error(
        stdout,
        Some(id),
        "client cancelled; fail-closed rollback could not be attested",
        "validation",
        false,
        epilogue,
    );
}

/// Outcome of the single production fail-closed rollback epilogue.
///
/// `rolled_back` is true only after every required host/target/spec/graph
/// reset AND a successful GPU `device_synchronize`. Callers must not claim
/// rollback before this struct is returned.
///
/// Fallible reset steps (`bind_thread`, `memset`, arch `reset` Results) are
/// accumulated. Synchronize is still attempted after an earlier failure so the
/// device is drained, but any reset or sync error keeps `rolled_back=false`.
#[derive(Debug, Clone)]
pub struct RollbackEpilogue {
    pub rolled_back: bool,
    /// Present when `rolled_back` is false (sync/reset could not be attested).
    pub context: Option<String>,
}

/// One production rollback epilogue for spec-step / forced-advance / grammar /
/// malformed / open-think / cancel failure paths.
///
/// Order (single epilogue — no caller-side duplicate):
/// 1. clear host cursors/history (`seq_pos`, `conversation_tokens`)
/// 2. reset target KV/recurrent/compact (slot or host bundle path)
/// 3. free host checkpoint rings; reset drafter/speculator
/// 4. invalidate captured HIP graphs + request-local replay observations
/// 5. `device_synchronize`
///
/// `rolled_back` is true only when every required reset and synchronize succeeds.
///
/// When the caller already holds a live `&mut dyn Speculator` derived from
/// `m.speculator`, prefer [`production_fail_closed_rollback_live`] so host
/// fields can be reborrowed disjointly without aliasing `m.speculator`.
pub fn production_fail_closed_rollback(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    slot: Option<&mut dyn SpecTarget>,
    spec: Option<&mut dyn Speculator>,
) -> RollbackEpilogue {
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    fail_closed_reset_target_and_spec(m, gpu, slot, spec)
}

/// Live-slot form used inside `generate_spec` while `spec` is borrowed from
/// `m.speculator` and `slot` from the RAII guard — host fields only.
///
/// Takes the host counters and checkpoint rings as disjoint reborrows so the
/// RAII target guard's `&mut m.state` can remain live without aliasing
/// `LoadedModel`. Same reset ordering as [`production_fail_closed_rollback`].
pub fn production_fail_closed_rollback_live(
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    gpu: &mut rdna_compute::Gpu,
    slot: &mut dyn SpecTarget,
    spec: &mut dyn Speculator,
) -> RollbackEpilogue {
    *seq_pos = 0;
    conversation_tokens.clear();
    asst_turn_cache.clear();
    // Host-side AR/vestigial rings (speculator owns the live DFlash ring).
    free_checkpoints(prefill_checkpoints, gpu);
    free_checkpoints(dflash_checkpoints, gpu);
    let mut first_err: Option<String> = None;
    if let Err(e) = slot.reset_recurrent(gpu) {
        push_reset_err(&mut first_err, "reset_recurrent", e);
    }
    if let Err(e) = spec.reset(gpu) {
        push_reset_err(&mut first_err, "spec.reset", e);
    }
    fail_closed_invalidate_graphs_and_replay(gpu);
    let prior = match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    };
    let epilogue = fail_closed_epilogue_after_sync(prior, fail_closed_device_sync(gpu));
    if epilogue.rolled_back {
        gpu.replay.begin_replay_observation_window();
    }
    epilogue
}

/// Emit one correlated fail-closed error (no done). Appends epilogue context
/// when rollback could not be attested.
pub fn emit_fail_closed_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    epilogue: &RollbackEpilogue,
) {
    let full = match &epilogue.context {
        Some(ctx) if !epilogue.rolled_back => format!("{message} ({ctx})"),
        _ => message.to_string(),
    };
    emit_active_attempt_error(stdout, id, &full, class, retryable, epilogue.rolled_back);
    let _ = stdout.flush();
}

/// DS4 does not advertise semantic contract v2 (`gen_start.contract_version`).
/// Task 6 capability denial relies on this remaining unset until a later
/// producer registration proves full semantic-v2 shape.
///
/// Production gen_start selection MUST call [`gen_start_contract_version_for_arch`]
/// (or this DS4 helper) rather than a test-only constant.
pub fn ds4_gen_start_contract_version() -> Option<u32> {
    gen_start_contract_version_for_arch(9)
}

// helper gen_start_contract_version_for_arch 7104..7109
pub fn gen_start_contract_version_for_arch(arch_id: u32) -> Option<u32> {
    hipfire_loader::carrier_for(arch_id)
        .map(|c| c.caps().semantic_contract_version)
        .unwrap_or(None)
}

/// Pure terminal decision shared by DS4 AR, EP, and speculative DSML paths.
///
/// When DSML finishes with an unclosed/truncated tool protocol, production
/// paths must emit exactly one typed validation error, suppress `done`,
/// suppress assistant-cache store, and expose no tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ds4MalformedTerminalAction {
    /// Request-scoped error message (includes parser detail when available).
    pub message: String,
    /// Typed error class for retry gates.
    pub class: &'static str,
    /// Malformed model output is never automatically retried.
    pub retryable: bool,
    /// Fail-closed turns are not rolled back for retry.
    pub rolled_back: bool,
    /// Never emit `{type:"done",...}` after this action.
    pub emit_done: bool,
    /// Never write `asst_turn_cache` for this turn.
    pub store_cache: bool,
    /// Never release executable tool calls on the wire/fingerprint.
    pub expose_tool_calls: bool,
}

/// Build the shared fail-closed terminal action for malformed DSML.
///
/// `detail` is the parser/spec detail string (may be empty). Spec path may pass
/// a fixed detail when `FinishSummary` only carries the magic finish_reason.
pub fn ds4_malformed_terminal_action(detail: &str) -> Ds4MalformedTerminalAction {
    let message = if detail.is_empty() {
        "malformed DSML tool protocol".to_string()
    } else if detail.starts_with("malformed DSML tool protocol") {
        detail.to_string()
    } else {
        format!("malformed DSML tool protocol: {detail}")
    };
    Ds4MalformedTerminalAction {
        message,
        class: "validation",
        retryable: false,
        rolled_back: false,
        emit_done: false,
        store_cache: false,
        expose_tool_calls: false,
    }
}

/// Live wireability of a DSML stream event before the turn terminal.
/// Structured calls and malformed status are held turn-wide.
pub fn ds4_stream_event_wireable(ev: &hipfire_arch_deepseek4::dsml::StreamEvent) -> bool {
    use hipfire_arch_deepseek4::dsml::StreamEvent;
    matches!(ev, StreamEvent::Token(_) | StreamEvent::Reasoning(_))
}

/// Outcome of finishing a DS4 AR/EP DSML turn after turn-wide call buffering.
#[derive(Debug, Clone)]
pub enum Ds4ArEpRouteTerminal {
    /// Fail-closed: discard every buffered call; emit typed error; no done/cache.
    Malformed(Ds4MalformedTerminalAction),
    /// Tool-safe or plain terminal. `wire_tool_calls` is empty unless safe.
    Safe {
        finish_reason: &'static str,
        wire_tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall>,
        store_cache: bool,
    },
}

/// Absorb one DSML stream event into turn-local buffers.
/// `ToolCalls` are retained only; never released here.
pub fn ds4_absorb_stream_event(
    ev: &hipfire_arch_deepseek4::dsml::StreamEvent,
    text_buf: &mut String,
    tool_calls_buf: &mut Vec<hipfire_runtime::prompt_frame::ToolCall>,
    malformed: &mut Option<String>,
) {
    use hipfire_arch_deepseek4::dsml::StreamEvent;
    match ev {
        StreamEvent::Token(t) => text_buf.push_str(t),
        StreamEvent::Reasoning(_) => {}
        StreamEvent::ToolCalls(calls) => {
            for c in calls {
                tool_calls_buf.push(hipfire_runtime::prompt_frame::ToolCall {
                    id: None,
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                    rendered_body: None,
                });
            }
        }
        StreamEvent::Malformed { detail } => {
            *malformed = Some(detail.clone());
        }
    }
}

// helper ds4_ar_ep_finish_route 7203..7232
pub fn ds4_ar_ep_finish_route(
    malformed: Option<String>,
    tool_calls_buf: Vec<hipfire_runtime::prompt_frame::ToolCall>,
    hit_length_cap: bool,
) -> Ds4ArEpRouteTerminal {
    if let Some(detail) = malformed {
        return Ds4ArEpRouteTerminal::Malformed(ds4_malformed_terminal_action(&detail));
    }
    if hit_length_cap {
        return Ds4ArEpRouteTerminal::Safe {
            finish_reason: "length",
            wire_tool_calls: Vec::new(),
            store_cache: false,
        };
    }
    if !tool_calls_buf.is_empty() {
        Ds4ArEpRouteTerminal::Safe {
            finish_reason: "tool_calls",
            wire_tool_calls: tool_calls_buf,
            store_cache: true,
        }
    } else {
        Ds4ArEpRouteTerminal::Safe {
            finish_reason: "stop",
            wire_tool_calls: Vec::new(),
            store_cache: true,
        }
    }
}

/// Cache-store action for DS4 AR/EP/spec — production and tests share this seam
/// so `store_cache` is never asserted in isolation from the sink mutation.
/// Fingerprint text + wire tool_calls match `build_deepseek4_dsml_prompt` lookup.
#[derive(Debug, Clone)]
pub struct Ds4CacheAction {
    pub store: bool,
    pub fingerprint_text: String,
    pub tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall>,
}

/// Derive cache action from the AR/EP pure route terminal.
pub fn ds4_ar_ep_cache_action(
    terminal: &Ds4ArEpRouteTerminal,
    visible_for_cache: &str,
) -> Ds4CacheAction {
    match terminal {
        Ds4ArEpRouteTerminal::Malformed(_) => Ds4CacheAction {
            store: false,
            fingerprint_text: String::new(),
            tool_calls: Vec::new(),
        },
        Ds4ArEpRouteTerminal::Safe {
            store_cache,
            wire_tool_calls,
            ..
        } => Ds4CacheAction {
            store: *store_cache,
            fingerprint_text: normalize_asst_turn_for_fingerprint(visible_for_cache),
            tool_calls: wire_tool_calls.clone(),
        },
    }
}

/// Apply a DS4 cache action through a shared insert seam (production
/// `AsstTurnCache` or a test `HashMap`). Returns the fingerprint when stored.
/// `cached_seq` must be the verbatim raw `streamed_tokens` body expected by
/// `build_deepseek4_dsml_prompt` replay (no surround EOS/Assistant markers).
pub fn ds4_apply_cache_action<F>(
    mut insert: F,
    action: &Ds4CacheAction,
    cached_seq: Vec<u32>,
) -> Option<u64>
where
    F: FnMut(u64, Vec<u32>),
{
    if !action.store || cached_seq.is_empty() {
        return None;
    }
    // Empty trimmed content AND no tool_calls collides on the hash of
    // ("assistant", "") — skip the same dead-weight empty turns AR/EP skip.
    let have_replayable_payload =
        !action.fingerprint_text.trim().is_empty() || !action.tool_calls.is_empty();
    if !have_replayable_payload {
        return None;
    }
    let fp = asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls);
    insert(fp, cached_seq);
    Some(fp)
}

/// Pure Commit/Abort side-effect gate for DS4 AR / EP / spec successful
/// terminals. Commit preserves the intended release/store/done flags; Abort
/// suppresses all three so producers never store or release after cancel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ds4ClientCommitEffects {
    pub release_tool_calls: bool,
    pub store_cache: bool,
    pub emit_done: bool,
}

// helper ds4_client_commit_effects 7398..7416
pub fn ds4_client_commit_effects(
    decision: ClientTerminalDecision,
    intended_release: bool,
    intended_store: bool,
) -> Ds4ClientCommitEffects {
    match decision {
        ClientTerminalDecision::Commit => Ds4ClientCommitEffects {
            release_tool_calls: intended_release,
            store_cache: intended_store,
            emit_done: true,
        },
        ClientTerminalDecision::Abort => Ds4ClientCommitEffects {
            release_tool_calls: false,
            store_cache: false,
            emit_done: false,
        },
    }
}

/// Pure EP abort wire shape (attempt-correlated aborted + done). Used by
/// production `ep_emit_abort` and no-GPU seam tests.
pub fn ds4_ep_abort_wire_events(
    id: &str,
    completion_tokens: usize,
    attempt_id: u64,
) -> (serde_json::Value, serde_json::Value) {
    (
        hipfire_runtime::semantic::wire_aborted(id, "client_cancelled", attempt_id),
        hipfire_runtime::semantic::wire_aborted_done(id, completion_tokens, attempt_id),
    )
}

/// Resolve whether a pure length-cap was hit (no semantic stop / decoded EOT).
/// Semantic stop (StopSequence / EOS / ThinkCap) or decoded EOT on the final
/// budget token is tool-safe stop/tool_calls, not length.
pub fn qwen_dflash_hit_length_cap(
    generated: usize,
    max_tokens: usize,
    decoded_eot: bool,
    semantic_stop: bool,
) -> bool {
    generated >= max_tokens && !decoded_eot && !semantic_stop
}

/// Extract held ToolCalls from a FinishSummary (generate_spec holds them).
pub fn finish_summary_held_tool_calls(
    finish: &FinishSummary,
) -> Vec<hipfire_runtime::prompt_frame::ToolCall> {
    finish
        .events
        .iter()
        .find_map(|ev| match ev {
            ClientEvent::ToolCalls(c) => Some(c.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Apply a pre-built malformed action (shared AR/EP/spec decision path).
pub fn emit_ds4_malformed_action(
    stdout: &mut impl std::io::Write,
    id: &str,
    action: &Ds4MalformedTerminalAction,
) {
    debug_assert!(!action.emit_done);
    debug_assert!(!action.store_cache);
    debug_assert!(!action.expose_tool_calls);
    debug_assert!(!action.retryable);
    emit_active_attempt_error(
        stdout,
        Some(id),
        &action.message,
        action.class,
        action.retryable,
        action.rolled_back,
    );
    let _ = stdout.flush();
}

/// Emit a parsed `deepseek4::dsml::StreamEvent` to the JSONL stream.
/// Maps:
///   - Token(text)        → `{type:"token",   id, text}`
///   - Reasoning(text)    → `{type:"reasoning", id, text}`
///   - ToolCalls(calls)   → **held** (return); staged onto commit_ready/done
///     via [`stage_terminal_tool_calls`] after a tool-safe terminal
///   - Malformed {..}     → no wire event (caller terminalizes with typed error)
///
/// The CLI / OpenAI HTTP layer translates these into the corresponding
/// SSE chunks (`content`, `reasoning_content`, `tool_calls.delta`).
pub fn emit_stream_event(
    stdout: &mut impl std::io::Write,
    id: &str,
    ev: hipfire_arch_deepseek4::dsml::StreamEvent,
) {
    use hipfire_arch_deepseek4::dsml::StreamEvent;
    // The request id is user-supplied. Build the envelope through
    // `serde_json` so any embedded `"` / `\` / control chars are
    // escaped — otherwise a malformed id corrupts every subsequent
    // line of the JSONL stream and the cli/serve loop dies with a
    // `JSON Parse error: Expected '}'`.
    if !ds4_stream_event_wireable(&ev) {
        // ToolCalls: buffered turn-wide. Malformed: terminalized by caller.
        return;
    }
    let attempt_id = active_attempt_id();
    let envelope = match ev {
        StreamEvent::Token(text) => serde_json::json!({
            "type": "token",
            "id": id,
            "text": text,
            "attempt_id": attempt_id,
        }),
        StreamEvent::Reasoning(text) => serde_json::json!({
            "type": "reasoning",
            "id": id,
            "text": text,
            "attempt_id": attempt_id,
        }),
        StreamEvent::ToolCalls(_) | StreamEvent::Malformed { .. } => return,
    };
    let _ = writeln!(stdout, "{}", envelope);
}

// helper emit_committed_event 8603..8630
pub fn emit_committed_event(
    stdout: &mut (impl std::io::Write + ?Sized),
    id: &str,
    tok_id: u32,
    pos: usize,
    t_ms: u64,
) {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var("HIPFIRE_EMIT_TOKEN_IDS").ok().as_deref() == Some("1"));
    if !*ENABLED {
        return;
    }
    // Build through `serde_json::json!` for the same reason
    // `emit_error_with_id` does: `id` is user-supplied and a single `"`
    // or `\` in it would corrupt the line, breaking the client's JSONL
    // parser for every subsequent event on the same connection.
    let envelope = serde_json::json!({
        "type": "committed",
        "id": id,
        "tok_id": tok_id,
        "pos": pos,
        "t_ms": t_ms,
        "attempt_id": active_attempt_id(),
    });
    let _ = writeln!(stdout, "{}", envelope);
}

/// Drain + free a DeltaNet checkpoint ring. `DeviceBuffer` has no `Drop`, so a
/// bare `Vec::clear()` orphans each snapshot's GPU buffers — the per-reset leak
/// that OOMs long-lived serves (hipMalloc-OOM after ~N independent requests).
/// Routes every drop through `DeltaNetSnapshot::free_gpu`.
pub fn free_checkpoints(
    cks: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    gpu: &mut rdna_compute::Gpu,
) {
    for (_, snap) in cks.drain(..) {
        snap.free_gpu(gpu);
    }
}

/// Zero qwen3.5 (arch 5/6) recurrent DeltaNet + KV state IN PLACE for a fresh
/// turn. The state lives in the bundle (ModelState::Qwen35), NOT the always-None
/// direct fields m.dn_state/m.kv_cache — sourcing from the bundle (no hoist →
/// no double-free). Mirrors the per-LA-device memset for pp>1. No-op off qwen35.
///
/// Accumulates every `bind_thread` and HIP `memset` failure. Clears
/// `s_matrices`, `s_scales`, `conv_states`, and `s_ef_residual` on both
/// multi-GPU and single-GPU paths. Returns `Err` with the joined failures
/// (still attempts every surface — does not short-circuit).
pub fn reset_qwen35_recurrent(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    let mut first_err: Option<String> = None;
    if m.pp > 1 {
        if let (Some(b), Some(gpus), Some(la)) = (
            m.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
            }),
            m.pp_gpus.as_mut(),
            m.pp_dn_la_to_device.as_ref(),
        ) {
            let dn = &b.dn_state;
            for (i, s) in dn.s_matrices.iter().enumerate() {
                let g = &mut gpus.devices[la[i] as usize];
                if let Err(e) = g.bind_thread() {
                    push_reset_err(&mut first_err, "pp s_matrices bind_thread", e);
                }
                if let Err(e) = g.hip.memset(&s.buf, 0, s.buf.size()) {
                    push_reset_err(&mut first_err, "pp s_matrices memset", e);
                }
            }
            for (i, s) in dn.s_scales.iter().enumerate() {
                let g = &mut gpus.devices[la[i] as usize];
                if let Err(e) = g.bind_thread() {
                    push_reset_err(&mut first_err, "pp s_scales bind_thread", e);
                }
                if let Err(e) = g.hip.memset(&s.buf, 0, s.buf.size()) {
                    push_reset_err(&mut first_err, "pp s_scales memset", e);
                }
            }
            for (i, s) in dn.conv_states.iter().enumerate() {
                let g = &mut gpus.devices[la[i] as usize];
                if let Err(e) = g.bind_thread() {
                    push_reset_err(&mut first_err, "pp conv_states bind_thread", e);
                }
                if let Err(e) = g.hip.memset(&s.buf, 0, s.buf.size()) {
                    push_reset_err(&mut first_err, "pp conv_states memset", e);
                }
            }
            // multi-GPU currently leaves s_ef_residual empty; loop is a no-op then,
            // but keeps single-GPU parity if EF is ever wired per-device.
            for (i, s) in dn.s_ef_residual.iter().enumerate() {
                let g = &mut gpus.devices[la[i] as usize];
                if let Err(e) = g.bind_thread() {
                    push_reset_err(&mut first_err, "pp s_ef_residual bind_thread", e);
                }
                if let Err(e) = g.hip.memset(&s.buf, 0, s.buf.size()) {
                    push_reset_err(&mut first_err, "pp s_ef_residual memset", e);
                }
            }
        }
    } else if let Some(b) = m.qwen35() {
        let dn = &b.dn_state;
        for s in &dn.s_matrices {
            if let Err(e) = gpu.hip.memset(&s.buf, 0, s.buf.size()) {
                push_reset_err(&mut first_err, "s_matrices memset", e);
            }
        }
        for s in &dn.s_scales {
            if let Err(e) = gpu.hip.memset(&s.buf, 0, s.buf.size()) {
                push_reset_err(&mut first_err, "s_scales memset", e);
            }
        }
        for s in &dn.conv_states {
            if let Err(e) = gpu.hip.memset(&s.buf, 0, s.buf.size()) {
                push_reset_err(&mut first_err, "conv_states memset", e);
            }
        }
        for s in &dn.s_ef_residual {
            if let Err(e) = gpu.hip.memset(&s.buf, 0, s.buf.size()) {
                push_reset_err(&mut first_err, "s_ef_residual memset", e);
            }
        }
    }
    if let Some(b) = m.qwen35_mut() {
        b.kv_cache.compact_offset = 0;
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// helper deepseek4_reasoning_prefix 23525..23560
pub fn deepseek4_reasoning_prefix(mode: ThinkMode) -> &'static str {
    match mode {
        ThinkMode::High => DEEPSEEK4_REASONING_HIGH_PREFIX,
        ThinkMode::Max => DEEPSEEK4_REASONING_MAX_PREFIX,
        ThinkMode::NonThink | ThinkMode::Low => "",
    }
}

// helper build_deepseek4_dsml_prompt 23561..23820
pub fn build_deepseek4_dsml_prompt(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    system_prompt: Option<&str>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    live_prompt: &str,
    think_mode: ThinkMode,
    deepseek4_eos_tok: u32,
    asst_turn_cache: &mut AsstTurnCache,
) -> Vec<u32> {
    // DeepSeek V4 non-thinking chat template (per HF encoding/README.md):
    //   <｜begin▁of▁sentence｜>{system?}<｜User｜>{msg}<｜Assistant｜></think>
    //
    // The `</think>` immediately after `<｜Assistant｜>` is REQUIRED in
    // non-thinking mode — it tells the model "skip the reasoning block,
    // go straight to the response." Without it the model is in
    // undefined-behavior territory. Raw prompts (no chat-template wrap)
    // also collapse to attractor garbage on this quantized instruct
    // model. Multi-turn / thinking-mode plumbing is a follow-up; this
    // emits a single non-thinking turn per /generate call.
    let lookup = |s: &str| -> Option<u32> {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    };
    let bos_tok = lookup("<｜begin▁of▁sentence｜>");
    let user_tok = lookup("<｜User｜>");
    let asst_tok = lookup("<｜Assistant｜>");

    // Build the effective system message: optional user-supplied system
    // text + (if request has tools) the DSML "## Tools" preamble.
    //
    // HF reference render: the system role is rendered as `{content}`
    // (raw, no role prefix), then appended with `"\n\n" + render_tools`
    // when tools are present. For an empty system + tools this becomes
    // `"" + "\n\n" + tools_block` = `"\n\n" + tools_block` — the model
    // was trained to see two newlines BEFORE `## Tools` even with no
    // system content. Omitting them puts the model in off-distribution
    // territory; observed 2026-05-23 to drive the V4F MQ2-Lloyd
    // checkpoint into `<｜DSML｜tool_cin> / <｜DSML｜-cin>` attractor
    // loops on no-system + 4-tools requests. The leading `\n\n` is
    // load-bearing — do not drop.
    let tools_block: Option<String> = tools
        .filter(|t| !t.is_empty())
        .map(|t| deepseek4::dsml::tools_prompt_block(t));
    let effective_system: Option<String> = match (
        system_prompt.filter(|s| !s.is_empty()),
        tools_block.as_deref(),
    ) {
        (Some(sys), Some(tb)) => Some(format!("{sys}\n\n{tb}")),
        (Some(sys), None) => Some(sys.to_string()),
        (None, Some(tb)) => Some(format!("\n\n{tb}")),
        (None, None) => None,
    };

    let mut prompt_ids: Vec<u32> = Vec::new();
    if let Some(b) = bos_tok {
        prompt_ids.push(b);
    }
    let effort_prefix = deepseek4_reasoning_prefix(think_mode);
    if !effort_prefix.is_empty() {
        prompt_ids.extend(tokenizer.encode(effort_prefix));
    }
    if let Some(ref sys) = effective_system {
        prompt_ids.extend(tokenizer.encode(sys));
    }

    // Multi-turn history. Each prior message gets rendered as a turn:
    //   user → `<｜User｜>{content}{tool_results?}`
    //   assistant → `<｜Assistant｜>{content_or_dsml}<｜end▁of▁sentence｜>`
    // Tool result messages (role=tool) attach to the previous user turn
    // wrapped in `<tool_result>…</tool_result>` per HF encoding/README.md.
    // The CURRENT user prompt is appended last (outside this loop).
    if let Some(history) = messages_history {
        // Skip the leading system message (if any) — already handled.
        // Skip the trailing user prompt — we add it explicitly after, BUT only
        // when a non-empty `live_prompt` actually carries it. The OpenAI
        // messages API (no separate `prompt` field) puts the live user turn as
        // the LAST history message with `live_prompt == ""`; trimming it then
        // drops the user's question entirely (model greets instead of answering
        // — observed on ds4 EP tp4). So only trim when live_prompt is non-empty.
        use hipfire_runtime::prompt_frame::Role;
        let trim_end = if !live_prompt.is_empty()
            && matches!(history.last().map(|m| m.role), Some(Role::User))
        {
            1
        } else {
            0
        };
        let end = history.len().saturating_sub(trim_end);
        // Track whether the previous emission was already a tool_result
        // wrapped in a user turn — when YES, the next consecutive tool
        // message MUST NOT open a new `<｜User｜>` marker; instead it
        // stacks its `<tool_result>` body into the existing user turn.
        // Matches the reference imatrix dataset renderer in
        // `gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py:196-201`
        // — OpenAI's parallel-tool-call flow produces consecutive tool
        // messages (one per parallel call), and a fresh `<｜User｜>`
        // between them isn't what V4F was trained on.
        let mut pending_tool_result = false;
        for msg in &history[..end] {
            match msg.role {
                Role::System => {
                    // Already handled via effective_system; skip.
                }
                Role::User => {
                    if let Some(u) = user_tok {
                        prompt_ids.push(u);
                    }
                    prompt_ids.extend(tokenizer.encode(&msg.content));
                    pending_tool_result = false;
                }
                Role::Tool => {
                    // Wrap as `<tool_result>{escaped}</tool_result>`. Open
                    // a new user turn ONLY if the prior message wasn't
                    // already a tool_result.
                    if !pending_tool_result {
                        if let Some(u) = user_tok {
                            prompt_ids.push(u);
                        }
                    }
                    prompt_ids.extend(
                        tokenizer.encode(&deepseek4::dsml::render_tool_result(&msg.content)),
                    );
                    pending_tool_result = true;
                }
                Role::Assistant => {
                    // Daemon-emitted surround tokens that bracket every
                    // assistant turn in V4F format:
                    //   <｜Assistant｜>{</think> when not in think-replay}
                    //     {turn body — content + tool_calls}
                    //   <｜end▁of▁sentence｜>
                    //
                    // The cache stores ONLY the inner turn body (the
                    // tokens the model itself emitted during decode).
                    // The surround tokens are deterministic functions
                    // of `msg.content` and `think_mode` and must be
                    // emitted IDENTICALLY on both hit and miss paths so
                    // the prompt-cache LCP can extend through every
                    // prior assistant turn.
                    if let Some(a) = asst_tok {
                        prompt_ids.push(a);
                    }
                    let starts_with_think_tag =
                        msg.content.starts_with("<think>") || msg.content.starts_with("</think>");
                    if !starts_with_think_tag {
                        prompt_ids.extend(tokenizer.encode("</think>"));
                    }

                    // Prefix-cache fast path: if we previously emitted
                    // this exact assistant turn, replay the model's
                    // verbatim token sequence instead of re-rendering
                    // via DSML + BPE encode (which is not bijective —
                    // multi-char DSML special tokens picked greedily
                    // during decode can come back out of
                    // `tokenizer.encode(render(...))` as a longer
                    // sequence with different boundaries, capping the
                    // LCP at the assistant-turn boundary).
                    // Match store-side stripping (see qwen35 path comment).
                    let stripped = strip_think_for_fingerprint(&msg.content);
                    let normalized =
                        hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
                    let fp = asst_turn_fingerprint(&normalized, &msg.tool_calls);
                    if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        eprintln!(
                            "[asst-cache lookup] fp={:#018x} content.len={}/stripped.len={} tool_calls={} hit={}",
                            fp, msg.content.len(), normalized.len(),
                            msg.tool_calls.len(),
                            asst_turn_cache.contains_key(&fp),
                        );
                    }
                    // DSML builds a flat token stream; project the content slot out of the
                    // per-channel cache value (DeepSeek turns are content-only).
                    if let Some(cached) = asst_turn_cache.get(&fp).and_then(|t| t.content.as_ref())
                    {
                        prompt_ids.extend_from_slice(&cached.token_ids);
                    } else {
                        // Cache miss — render the turn the long way.
                        if !msg.content.is_empty() && msg.content != "null" {
                            prompt_ids.extend(tokenizer.encode(&msg.content));
                        }
                        if !msg.tool_calls.is_empty() {
                            let dsml_calls: Vec<hipfire_arch_deepseek4::dsml::ToolCall> = msg
                                .tool_calls
                                .iter()
                                .map(|c| hipfire_arch_deepseek4::dsml::ToolCall {
                                    name: c.name.clone(),
                                    arguments: c.arguments.clone(),
                                })
                                .collect();
                            let dsml = hipfire_arch_deepseek4::dsml::render_assistant_tool_calls(
                                &dsml_calls,
                            );
                            prompt_ids.extend(tokenizer.encode(&dsml));
                        }
                    }

                    // If the replayed turn body opened a `<think>` block but
                    // the model premature-stopped without closing it (EOS inside
                    // the think, no tool call), close it here with a `</think>`.
                    // Otherwise the dangling `<think>…<EOS>` drifts the next turn
                    // (more premature stops, a leaked `</think>`). This is a
                    // deterministic surround token — a pure function of
                    // msg.content, NOT part of the cached turn body or the
                    // asst_turn_fingerprint (which strips think anyway) — so it
                    // is emitted identically on hit and miss paths and the
                    // prefix-cache LCP + asst_turn_cache stay effective.
                    if msg.tool_calls.is_empty()
                        && msg.content.starts_with("<think>")
                        && !msg.content.contains("</think>")
                    {
                        prompt_ids.extend(tokenizer.encode("</think>"));
                    }

                    // Close the assistant turn with the EOS marker so
                    // the next turn starts cleanly.
                    prompt_ids.push(deepseek4_eos_tok);
                    pending_tool_result = false;
                }
            }
        }
    }

    // Append the live user turn ONLY when `prompt` carries one. When the
    // serve has handed us a structured `messages` history that already
    // ends in a tool result (mid-conversation, model is meant to continue
    // generating the next assistant turn) it sends `prompt=""` — in that
    // case we MUST NOT emit an empty `<｜User｜><｜Assistant｜>` wrapper,
    // because the empty-user turn is off-distribution and the V4F MQ2-
    // Lloyd checkpoint drifts into invented paths / repeated wrong tool
    // calls when fed one.
    if !live_prompt.is_empty() {
        if let Some(u) = user_tok {
            prompt_ids.push(u);
        }
        prompt_ids.extend(tokenizer.encode(live_prompt));
    }
    if let Some(a) = asst_tok {
        prompt_ids.push(a);
    }
    // Thinking-mode signal token immediately after `<｜Assistant｜>`:
    //   NonThink → `</think>`   (skip reasoning, respond directly)
    //   Low|High|Max → `<think>` (open a reasoning block)
    match think_mode {
        ThinkMode::NonThink => prompt_ids.extend(tokenizer.encode("</think>")),
        ThinkMode::Low | ThinkMode::High | ThinkMode::Max => {
            prompt_ids.extend(tokenizer.encode("<think>"));
        }
    }

    prompt_ids
}

// helper DEEPSEEK4_REASONING_HIGH_PREFIX 23514..23518
pub const DEEPSEEK4_REASONING_HIGH_PREFIX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);

// helper DEEPSEEK4_REASONING_MAX_PREFIX 23519..23524
pub const DEEPSEEK4_REASONING_MAX_PREFIX: &str = concat!(
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n",
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n",
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n",
);

// helper fail_closed_reset_target_and_spec 6656..6759
pub fn fail_closed_reset_target_and_spec(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    slot: Option<&mut dyn SpecTarget>,
    spec: Option<&mut dyn Speculator>,
) -> RollbackEpilogue {
    let mut first_err: Option<String> = None;
    // Authoritative cold reset: drop assistant-turn semantic cache.
    m.asst_turn_cache.clear();
    if let Some(slot) = slot {
        // Live SpecTarget path: arch hook owns KV/recurrent/compact (and
        // deepseek4's zero_decode_caches). Graph teardown is always daemon-side.
        if let Err(e) = slot.reset_recurrent(gpu) {
            push_reset_err(&mut first_err, "reset_recurrent", e);
        }
    } else {
        // Post-guard / AR paths: reset via the bundle still on `m`.
        if let Err(e) = reset_qwen35_recurrent(m, gpu) {
            push_reset_err(&mut first_err, "reset_qwen35_recurrent", e);
        }
        if let Some(b) = m.llama_mut() {
            b.kv.compact_offset = 0;
        }
        if let Some(b) = m.qwen2_mut() {
            b.state.reset();
        }
        if let Some(b) = m.deepseek4_mut() {
            // Mirror generate_deepseek4 cache-miss teardown: cursor reset alone
            // leaves position-indexed SWA/full/compressed/indexer residue.
            b.state.reset();
            b.state.zero_decode_caches(gpu);
        }
        if let Some(b) = m.lfm2moe_mut() {
            if let Err(e) = b.state.reset(gpu) {
                push_reset_err(&mut first_err, "lfm2moe.reset", e);
            }
        }
        if let Some(b) = m.minimax_mut() {
            b.state.reset();
        }
        if let Some(b) = m.cohere2moe_mut() {
            if let Err(e) = b.state.reset(gpu) {
                push_reset_err(&mut first_err, "cohere2moe.reset", e);
            }
        }
        if let Some(bundle) = m.gemma4_mut() {
            // Gemma4State::reset is cursor-only (n_tokens = 0); the
            // captured decode hipGraph stays valid across resets (position
            // is re-staged via pos_host on every replay and attention
            // geometry is sized for max_seq).
            bundle.state.reset();
        }
        if let Some(bundle) = m.muse_glimmer_mut() {
            bundle.reset_session_state();
        }
        if let Some(ad) = m.kv_adaptive.as_mut() {
            if let Some(b) = m.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
            }) {
                ad.reset_with_cache(gpu, &mut b.kv_cache);
            } else {
                ad.reset();
            }
        }
        if let Some(b) = m.qwen35_mut() {
            if let Some(bs) = b.qwen35_decode_batch.as_mut() {
                if let Err(e) = bs.reset(gpu) {
                    push_reset_err(&mut first_err, "qwen35_decode_batch.reset", e);
                }
            }
        }
        if let Some(b) = m.lfm2moe_mut() {
            if let Some(bs) = b.lfm2_decode_batch.as_mut() {
                if let Err(e) = bs.reset(gpu) {
                    push_reset_err(&mut first_err, "lfm2_decode_batch.reset", e);
                }
            }
        }
    }
    // Always free host checkpoint rings. When a Speculator is present its own
    // ring is freed by `reset` below; these are AR/vestigial rings on `m`.
    free_checkpoints(&mut m.prefill_checkpoints, gpu);
    free_checkpoints(&mut m.dflash_checkpoints, gpu);

    if let Some(spec) = spec {
        if let Err(e) = spec.reset(gpu) {
            push_reset_err(&mut first_err, "spec.reset", e);
        }
    } else if let Some(s) = m.speculator.as_mut() {
        if let Err(e) = s.reset(gpu) {
            push_reset_err(&mut first_err, "spec.reset", e);
        }
    }

    fail_closed_invalidate_graphs_and_replay(gpu);
    // Always attempt sync after resets; preserve aggregate reset failure.
    let prior = match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    };
    let epilogue = fail_closed_epilogue_after_sync(prior, fail_closed_device_sync(gpu));
    if epilogue.rolled_back {
        gpu.replay.begin_replay_observation_window();
    }
    epilogue
}

// helper push_reset_err 6645..6655
pub fn push_reset_err(acc: &mut Option<String>, label: &str, err: impl std::fmt::Display) {
    let msg = format!("{label}: {err}");
    match acc {
        None => *acc = Some(msg),
        Some(prev) => {
            prev.push_str("; ");
            prev.push_str(&msg);
        }
    }
}

/// Drop captured HIP graphs and mark the request-local replay observation
/// window failed so a fail-closed turn cannot retain stale graph/replay state.
///
/// Infallible-by-contract (destroys host-side graph handles + sets a flag).
pub fn fail_closed_invalidate_graphs_and_replay(gpu: &mut rdna_compute::Gpu) {
    gpu.invalidate_graph_state();
    gpu.replay.invalidate_replay_observation_window();
}

// helper fail_closed_device_sync 6572..6584
pub fn fail_closed_device_sync(gpu: &mut rdna_compute::Gpu) -> RollbackEpilogue {
    match gpu.hip.device_synchronize() {
        Ok(()) => RollbackEpilogue {
            rolled_back: true,
            context: None,
        },
        Err(e) => RollbackEpilogue {
            rolled_back: false,
            context: Some(format!("device_synchronize failed: {e}")),
        },
    }
}

/// Merge a prior reset-step error with a device_synchronize result.
/// Sync is always attempted; attestation requires both prior Ok and sync Ok.
pub fn fail_closed_epilogue_after_sync(
    prior: Result<(), String>,
    sync: RollbackEpilogue,
) -> RollbackEpilogue {
    match (prior, sync.rolled_back, sync.context) {
        (Ok(()), true, _) => RollbackEpilogue {
            rolled_back: true,
            context: None,
        },
        (Ok(()), false, ctx) => RollbackEpilogue {
            rolled_back: false,
            context: ctx,
        },
        (Err(e), true, _) => RollbackEpilogue {
            rolled_back: false,
            context: Some(e),
        },
        (Err(e), false, Some(s)) => RollbackEpilogue {
            rolled_back: false,
            context: Some(format!("{e}; {s}")),
        },
        (Err(e), false, None) => RollbackEpilogue {
            rolled_back: false,
            context: Some(e),
        },
    }
}

/// Model-independent emitter recipe a `generate_spec` wrapper supplies so the
/// arch's carrier can build the matching [`SpecEmit`] *after* `generate_spec` has
/// acquired the slot (the emitter needs `slot.eos_token()` and the tokenizer,
/// both derived from `m` *inside* `generate_spec`). Every field is neutral —
/// tool definitions are the request's raw JSON, which each arch's emitter parses
/// into its own grammar schema; `generate_spec` builds a `SpecEmitCtx` from this
/// + the slot's eos + the tokenizer and calls `carrier.make_spec_emitter`.
pub struct SpecEmitRequest {
    pub im_end: Option<u32>,
    /// Raw tool definitions (OpenAI-shape JSON); `None`/empty ⇒ no tool grammar.
    pub tools: Option<Vec<serde_json::Value>>,
    pub grammar: bool,
    pub stop: Vec<String>,
    pub max_think: usize,
    pub assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    /// Reasoning-effort level (consumed by the ds4 emitter; ignored by ChatML).
    pub think_mode: ThinkMode,
    /// Pre-decoded vocab for arches whose grammar masks per-token (ds4). The
    /// wrapper builds/caches the Arc so this struct carries no `LoadedModel` ref.
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
}

/// Summary returned by `generate_spec` to its arch wrapper, which writes the
/// arch-specific `done` envelope and (qwen35/deepseek4) the asst-turn cache
/// store from it. `None` is returned on the abort/error early-exits, which
/// already wrote their own `done`/`error`; the wrapper then does nothing.
pub struct SpecRun {
    pub generated: usize,
    pub spec_cycles: usize,
    pub spec_accepted: usize,
    /// Full committed stream (from the emitter) for the wrapper's cache store.
    /// Empty for emitters that don't track it. Deepseek4 and qwen35 both expose
    /// streamed tokens for asst-turn cache replay on safe terminals.
    pub streamed_tokens: Vec<u32>,
    /// Newly-prefilled token count (the suffix actually fed through the model).
    pub prefill_tokens_len: usize,
    /// The terminal flush summary (tool-call count drives the wrapper's
    /// `finish_reason`; events were already rendered inside `generate_spec`).
    pub finish: FinishSummary,
    /// Emitter reported a grammar violation (wrapper may suppress cache/calls).
    pub grammar_violated: bool,
    /// Emitter semantic stop (Eos / StopSequence / ThinkCap) observed during
    /// begin, normal observe, or forced observe. Independent of
    /// `finish.decoded_eot` — wrappers OR both so stop-at-max_tokens wins over
    /// length. GrammarViolation is fail-closed, not carried here.
    pub semantic_stop: Option<StopReason>,
    /// Truthful rollback attestation when this turn ended fail-closed
    /// (grammar / open-think / malformed). `None` on safe Done paths.
    pub fail_closed_rollback: Option<RollbackEpilogue>,
    pub prefill_s: f64,
    pub total_s: f64,
    pub decode_s: f64,
}

/// Fire one-shot after-prefill fault on qwen AR (host LoadedModel path).
/// Returns true when the fault was taken (caller must return immediately).
///
/// Spec is reset via `crate::common::production_fail_closed_rollback(..., None)` which
/// reborrows `m.speculator` internally — do not pass both `m` and a
/// split `m.speculator` borrow.
#[cfg(feature = "serve-fault-inject")]
use crate::ar::take_fault_after_prefill;

#[cfg(feature = "serve-fault-inject")]
pub fn maybe_inject_fault_after_prefill_ar(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
    id: &str,
) -> bool {
    if !take_fault_after_prefill() {
        return false;
    }
    // Only qwen35 AR/DFlash are fault-inject eligible.
    if !matches!(m.arch_id, 5 | 6) {
        return false;
    }
    let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
    crate::common::emit_fail_closed_error(
        stdout,
        Some(id),
        "injected fault after prefill",
        "gpu",
        true,
        &ep,
    );
    true
}

/// Fire one-shot after-prefill fault on qwen DFlash (live slot/spec path).
///
/// Takes host counters/rings as disjoint reborrows so the RAII target guard's
/// `&mut m.state` (via `slot`) and `m.speculator` (via `spec`) stay live —
/// same pattern as [`crate::common::production_fail_closed_rollback_live`].
/// Returns true when the fault was taken (caller must return immediately).
#[cfg(feature = "serve-fault-inject")]
pub fn maybe_inject_fault_after_prefill_dflash(
    arch_id: u32,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
    id: &str,
    slot: &mut dyn SpecTarget,
    spec: &mut dyn Speculator,
) -> bool {
    if !take_fault_after_prefill() {
        return false;
    }
    if !matches!(arch_id, 5 | 6) {
        return false;
    }
    let ep = crate::common::production_fail_closed_rollback_live(
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
        dflash_checkpoints,
        asst_turn_cache,
        gpu,
        slot,
        spec,
    );
    crate::common::emit_fail_closed_error(
        stdout,
        Some(id),
        "injected fault after prefill",
        "gpu",
        true,
        &ep,
    );
    true
}

// ── test-support helpers, moved with the daemon test modules ──

/// Pure attestation combiner for unit tests / failure injection: every required
/// reset class must succeed AND sync must succeed for `rolled_back=true`.
/// Sync is modeled as always attempted (callers pass its outcome regardless).
#[allow(dead_code)]
pub fn attest_rollback_steps(
    steps: &[(&str, Result<(), String>)],
    sync: Result<(), String>,
) -> crate::common::RollbackEpilogue {
    let mut errs: Vec<String> = Vec::new();
    for (name, r) in steps {
        if let Err(e) = r {
            errs.push(format!("{name}: {e}"));
        }
    }
    if let Err(e) = sync {
        errs.push(format!("device_synchronize failed: {e}"));
    }
    if errs.is_empty() {
        crate::common::RollbackEpilogue {
            rolled_back: true,
            context: None,
        }
    } else {
        crate::common::RollbackEpilogue {
            rolled_back: false,
            context: Some(errs.join("; ")),
        }
    }
}

/// Write one Qwen DFlash Done terminal via the production envelope builder.
pub fn emit_qwen_dflash_done_terminal(
    stdout: &mut impl std::io::Write,
    id: &str,
    generated: usize,
    tok_s: f64,
    prefill_tokens: usize,
    prefill_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
    tau: f64,
    cycles: usize,
    cached_tokens: usize,
    finish_reason: &str,
    pflash: Option<(&str, f32)>,
) {
    let mut done_env = crate::qwen::qwen_dflash_done_value(
        id,
        generated,
        tok_s,
        prefill_tokens,
        prefill_ms,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms,
        tau,
        cycles,
        cached_tokens,
        finish_reason,
        active_attempt_id(),
    );
    if let Some((reason, alpha)) = pflash {
        done_env["pflash"] = serde_json::json!({
            "bypass_reason": reason,
            "alpha": alpha,
        });
    }
    let _ = writeln!(stdout, "{}", done_env);
    let _ = stdout.flush();
}

/// Whether a crate::common::SpecRun None early-exit may enter the wrapper epilogue.
/// Production contract: None already wrote error/aborted; epilogue is skipped.
pub fn qwen_dflash_epilogue_after_spec_run(run_present: bool) -> bool {
    run_present
}

/// Pure speculative-route terminal decision after `Deepseek4Emit::finish`.
/// Returns `Some(action)` when the emitter reported malformed protocol.
pub fn ds4_spec_finish_route(
    finish_reason: &str,
    tool_calls: usize,
) -> Option<crate::common::Ds4MalformedTerminalAction> {
    if finish_reason == "malformed_protocol" {
        debug_assert_eq!(
            tool_calls, 0,
            "spec malformed must report tool_calls=0 (buffered calls discarded)"
        );
        Some(crate::common::ds4_malformed_terminal_action(
            "unclosed DSML tool_calls block at end of output",
        ))
    } else {
        None
    }
}

/// Apply [`crate::common::ds4_malformed_terminal_action`] to the active attempt writer.
/// Returns after writing the error envelope (caller must `return` from generate).
pub fn emit_ds4_malformed_terminal(stdout: &mut impl std::io::Write, id: &str, detail: &str) {
    let action = crate::common::ds4_malformed_terminal_action(detail);
    debug_assert!(!action.emit_done);
    debug_assert!(!action.store_cache);
    debug_assert!(!action.expose_tool_calls);
    debug_assert!(!action.retryable);
    crate::dense::emit_active_attempt_error(
        stdout,
        Some(id),
        &action.message,
        action.class,
        action.retryable,
        action.rolled_back,
    );
    let _ = stdout.flush();
}

/// Build the `logprob` / `top_logprobs` fields for one token event.
///
/// Returns `None` when logprobs were not requested, so a caller can attach
/// nothing rather than attach nulls. The gateway omits the whole `logprobs`
/// object when no token carried one, and a null here would defeat that.
///
/// Cheap on the paths that call it: `logits` is already a host slice at sampling
/// time — `hipfire_arch_deepseek4::sampling::sample_token` takes `&[f32]` — so
/// this adds a log-sum-exp pass and a bounded top-K selection with no
/// device-to-host copy. Do not call it from a path that would have to download
/// logits for the purpose; a prior optimisation removed exactly that round-trip
/// from the decode loop.
///
/// `bytes` is emitted per OpenAI's schema so a client can reconstruct tokens that
/// are not valid UTF-8 on their own — a multi-byte character split across two
/// tokens renders as replacement characters in `token` but is exact in `bytes`.
pub fn token_logprob_fields(
    logits: &[f32],
    sampled: u32,
    top_k: Option<usize>,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
) -> Option<(f64, serde_json::Value)> {
    let k = top_k?;
    let sampled_lp = saddle_core::logprobs::logprob_of(logits, sampled)?;
    let top = saddle_core::logprobs::top_k_logprobs(logits, k)
        .into_iter()
        .map(|t| {
            let text = tokenizer.decode(&[t.token_id]);
            serde_json::json!({
                "token": text,
                "logprob": t.logprob,
                "bytes": tokenizer.decode_bytes(&[t.token_id]),
            })
        })
        .collect::<Vec<_>>();
    Some((f64::from(sampled_lp), serde_json::Value::Array(top)))
}

#[cfg(test)]
mod tests {
    use super::latch_request_think_cap;

    #[test]
    fn numeric_think_cap_latches_once_and_keeps_first_position() {
        let mut latched = false;
        let mut mark = None;

        assert!(!latch_request_think_cap(
            false,
            100,
            &mut latched,
            &mut mark
        ));
        assert!(!latched);
        assert_eq!(mark, None);

        assert!(latch_request_think_cap(true, 4096, &mut latched, &mut mark));
        assert!(latched);
        assert_eq!(mark, Some(4096));

        assert!(!latch_request_think_cap(
            true,
            5000,
            &mut latched,
            &mut mark
        ));
        assert!(latched);
        assert_eq!(mark, Some(4096));
    }
}
